using Test

module AtlasEngineTests
using Test, TOML, SHA, LinearAlgebra
include(joinpath(@__DIR__, "..", "scripts", "atlas", "engine.jl"))

function discovery_job(index, chart, boundary)
    Dict{String,Any}("kind" => "discovery", "queue" => "discovery", "index" => index,
        "chart" => chart, "boundary" => boundary)
end

function empty_state()
    Dict{String,Any}("jobs" => Dict{String,Any}[], "discovery_index" => 0,
        "queue_seconds" => Dict(queue => 0.0 for queue in ATLAS_QUEUES),
        "elapsed_seconds" => 0.0, "setup_seconds" => 0.0, "setup_failures" => String[])
end

@testset "Atlas execution contracts" begin
    config = atlas_config()
    @test config["benchmark_seconds"] == 1200
    @test_throws ArgumentError validate_atlas_config(merge(config, Dict("max_jobs" => 0)))
    @test_throws ArgumentError validate_atlas_config(merge(config, Dict("job_timeout_seconds" => Inf)))
    invalid_science = deepcopy(config)
    invalid_science["science"]["max_evaluations"] = 0
    @test_throws ArgumentError validate_atlas_config(invalid_science)
    invalid_science["science"]["max_evaluations"] = 100
    invalid_science["science"]["balance_atol"] = NaN
    @test_throws ArgumentError validate_atlas_config(invalid_science)
    for key in ("balance_atol", "event_atol")
        invalid_boolean = deepcopy(config)
        invalid_boolean["science"][key] = true
        @test_throws ArgumentError validate_atlas_config(invalid_boolean)
    end
    @test_throws ArgumentError run_atlas("run", "unused-output", config)
    @test_throws ArgumentError atlas_main(["benchmark", "output", "--budget-seconds", "1"])
    @test_throws ArgumentError atlas_main(["report", "output", "--config", "file"])
    @test_throws ArgumentError atlas_main(["resume", "output", "--config", "file"])

    state = empty_state()
    for _ in 1:16
        next_discovery!(state)
    end
    payloads = getindex.(state["jobs"], "payload")
    @test count(p -> p["chart"] == "parameter", payloads) == 8
    @test count(p -> p["chart"] == "equilibrium", payloads) == 8
    @test count(p -> p["boundary"], payloads) == 4
    @test Set(p["chart"] for p in payloads if p["boundary"]) == Set(("parameter", "equilibrium"))
    @test [p["index"] for p in payloads if p["boundary"] && p["chart"] == "parameter"] == [1, 2]
    @test register_atlas_job!(state, payloads[1]) === nothing

    state = empty_state()
    register_atlas_job!(state, Dict("kind" => "event", "queue" => "events"))
    @test select_atlas_job!(state, config)["queue"] == "discovery"
    state["queue_seconds"]["discovery"] = 1.0
    @test select_atlas_job!(state, config)["queue"] == "events"
    @test_throws ArgumentError register_atlas_job!(state, Dict("kind" => "bad", "queue" => "bad"))

    # A cold nexus can dominate the event allocation. Benchmark calibration
    # still attempts newly available continuation/section work before cheap
    # discovery exhausts the job cap; normal pilot allocation remains weighted.
    calibrated = empty_state()
    calibrated["mode"] = "benchmark"
    register_atlas_job!(calibrated, Dict("kind" => "nexus", "queue" => "events"))["status"] = "failed"
    register_atlas_job!(calibrated, Dict("kind" => "discovery", "queue" => "discovery", "index" => 0))["status"] = "completed"
    register_atlas_job!(calibrated, Dict("kind" => "discovery", "queue" => "discovery", "index" => 1))
    continuation = register_atlas_job!(calibrated, Dict("kind" => "continuation", "queue" => "events"))
    calibrated["queue_seconds"]["events"] = 60.0
    @test select_atlas_job!(calibrated, config) === continuation
    continuation["status"] = "failed"
    @test select_atlas_job!(calibrated, config)["queue"] == "discovery"
    section = register_atlas_job!(calibrated, Dict("kind" => "section", "queue" => "events"))
    @test select_atlas_job!(calibrated, config) === section
    calibrated["mode"] = "run"
    @test select_atlas_job!(calibrated, config)["queue"] == "discovery"

    mktempdir() do directory
        path = joinpath(directory, "checkpoint.toml")
        atomic_toml(path, Dict("value" => 1))
        atomic_toml(path, Dict("value" => 2))
        @test TOML.parsefile(path)["value"] == 2
        @test !isfile(path * ".tmp")
        victim = joinpath(directory, "victim")
        write(victim, "preserved")
        symlink(victim, path * ".tmp")
        atomic_toml(path, Dict("value" => 3))
        @test read(victim, String) == "preserved"
        lock = acquire_atlas_lock(directory)
        try
            @test_throws ArgumentError acquire_atlas_lock(directory)
            @test_throws ArgumentError atlas_main(["report", directory])
        finally
            release_atlas_lock(lock)
        end
        lock = acquire_atlas_lock(directory)
        release_atlas_lock(lock)
    end
end

@testset "CLI coordinator does not load numerical dependencies before its deadline" begin
    mktempdir() do fixture
        scripts = joinpath(fixture, "scripts")
        assets = joinpath(scripts, "atlas")
        mkpath(assets)
        source = joinpath(@__DIR__, "..", "scripts")
        cp(joinpath(source, "zero_drive_atlas.jl"), joinpath(scripts, "zero_drive_atlas.jl"))
        for filename in ("core.jl", "engine.jl", "defaults.toml")
            cp(joinpath(source, "atlas", filename), joinpath(assets, filename))
        end
        cp(joinpath(source,"atlas","storage.jl"),joinpath(assets,"storage.jl"))
        write(joinpath(assets,"pilot.jl"),"# no finite pilot needed for report fixture\n")
        write(joinpath(assets, "events.jl"), "# no numerical handlers needed\n")
        write(joinpath(assets, "science.jl"), "# no numerical handlers needed\n")
        write(joinpath(assets, "report.jl"), """
        function write_report(output)
            atomic_toml(joinpath(output, "lightweight.toml"), Dict(
                "package_loaded" => isdefined(@__MODULE__, :FailureOfInhibition2025),
                "seed_count" => length(initial_jobs(Dict("mode" => "benchmark")))))
        end
        """)
        write(joinpath(fixture, "Project.toml"), "[deps]\n")
        depot = joinpath(fixture, "empty-depot")
        mkpath(depot)
        output = joinpath(fixture, "output")
        mkdir(output)
        log = open(joinpath(fixture, "bootstrap.log"), "w")
        command = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$fixture $(joinpath(scripts, "zero_drive_atlas.jl")) report $output`,
            "JULIA_DEPOT_PATH" => depot, "JULIA_LOAD_PATH" => "@:@stdlib", "JULIA_PKG_OFFLINE" => "true")
        process = run(pipeline(command; stdout=log, stderr=log); wait=false)
        close(log)
        try
            deadline = time() + 30.0
            while !process_exited(process) && time() < deadline
                sleep(0.02)
            end
            @test process_exited(process)
            if process_exited(process)
                success(process) || println(stderr, read(joinpath(fixture, "bootstrap.log"), String))
                @test success(process)
                if success(process)
                    record = TOML.parsefile(joinpath(output, "lightweight.toml"))
                    @test !record["package_loaded"]
                    @test record["seed_count"] == 4
                end
            end
        finally
            process_exited(process) || kill(process, Base.SIGKILL)
            wait(process)
        end
    end
end

@testset "Hard worker timeout and checkpoint preservation" begin
    mktempdir() do directory
        script = joinpath(directory, "mock_worker.jl")
        engine = joinpath(@__DIR__, "..", "scripts", "atlas", "engine.jl")
        open(script, "w") do stream
            println(stream, "module MockAtlas")
            println(stream, "using TOML, SHA, LinearAlgebra")
            println(stream, "include(", repr(engine), ")")
            println(stream, "function discovery_job(index, chart, boundary)")
            println(stream, "index == 99 && sleep(3.0)")
            println(stream, "Dict(\"kind\"=>\"generated\", \"queue\"=>\"discovery\", \"index\"=>index)")
            println(stream, "end")
            println(stream, "function execute_job(job, config)")
            println(stream, "sleep(get(job, \"sleep\", 0.0))")
            println(stream, "if get(job, \"busy\", false); while true; end; end")
            println(stream, "get(job, \"fail\", false) && error(\"deliberate worker failure\")")
            println(stream, "get(job, \"unsupported\", false) && return Dict(\"status\"=>\"completed\",\"evidence\"=>Dict(\"unsupported\"=>(\"bad\",\"tuple\")),\"followups\"=>Any[])")
            println(stream, "get(job, \"malformed\", false) && return Dict(\"status\"=>\"completed\",\"followups\"=>[Dict(\"wrong\"=>1)])")
            println(stream, "Dict(\"status\"=>\"completed\",\"evidence\"=>Dict(\"value\"=>7,\"job_id\"=>get(job,\"id\",\"\")),\"followups\"=>Any[])")
            println(stream, "end\nend")
            println(stream, "MockAtlas.atlas_worker_main(ARGS[2])")
        end
        mkpath(joinpath(directory, "workers"))
        worker = start_atlas_worker(directory, 30.0; script=script)
        try
            config = atlas_config()
            result, status, seconds, message = execute_atlas_worker!(worker, Dict("kind" => "test"), config, 20.0)
            @test status == "completed"
            @test result["evidence"]["value"] == 7
            @test haskey(result, "worker_timing")
            generated = Dict("kind"=>"discovery", "generate"=>true, "index"=>1, "chart"=>"parameter", "boundary"=>false, "id"=>"unique-id")
            result, status, seconds, message = execute_atlas_worker!(worker, generated, config, 20.0)
            @test status == "completed"
            @test result["executed_job"]["kind"] == "generated"
            @test result["executed_job"]["id"] == "unique-id"
            result, status, seconds, message = execute_atlas_worker!(worker, Dict("kind" => "unsupported", "unsupported" => true), config, 20.0)
            @test status == "failed"
            @test occursin("serialization failed", message)
            @test result["evidence"]["failure_stage"] == "result_serialization"
            @test result["evidence"]["job_kind"] == "unsupported"
            @test !process_exited(worker.process)
            result, status, seconds, message = execute_atlas_worker!(worker, Dict("kind" => "after_serialization_failure"), config, 20.0)
            @test status == "completed"
            result, status, seconds, message = execute_atlas_worker!(worker, Dict("fail" => true), config, 20.0)
            @test status == "failed"
            @test occursin("deliberate worker failure", message)
            result, status, seconds, message = execute_atlas_worker!(worker, Dict("sleep" => 3.0), config, 0.15)
            @test status == "timed_out"
            @test result === nothing
            @test seconds < 1.0
            @test process_exited(worker.process)
        finally
            stop_atlas_worker!(worker)
        end
        worker = start_atlas_worker(directory, 30.0; script=script)
        try
            generated = Dict("kind"=>"discovery", "generate"=>true, "index"=>99, "chart"=>"parameter", "boundary"=>false, "id"=>"bounded-factory")
            result, status, seconds, message = execute_atlas_worker!(worker, generated, atlas_config(), 0.15)
            @test status == "timed_out"
            @test process_exited(worker.process)
        finally
            stop_atlas_worker!(worker)
        end
        # The coordinator cap leaves the second job pending and preserves the first result.
        capped = joinpath(directory, "capped")
        mkpath(joinpath(capped, "workers"))
        state = empty_state()
        state["queue_seconds"]["discovery"] = 100.0
        register_atlas_job!(state, Dict("kind" => "first", "queue" => "events"))
        register_atlas_job!(state, Dict("kind" => "second", "queue" => "events"))
        config = atlas_config()
        config["max_jobs"] = 1
        config["warmup_seconds"] = 30.0
        drive_atlas!(capped, state, config; additional_seconds=10.0, worker_script=script)
        @test state["stop_reason"] == "job_limit"
        @test state["jobs"][1]["status"] == "completed"
        @test state["jobs"][2]["status"] == "pending"
        @test TOML.parsefile(joinpath(capped, state["jobs"][1]["result_file"]))["evidence"]["job_id"] == state["jobs"][1]["id"]
        checkpoint = TOML.parsefile(joinpath(capped, "state.toml"))
        @test checkpoint["jobs"][1]["result_hash"] == bytes2hex(sha256(read(joinpath(capped, state["jobs"][1]["result_file"]))))

        # A smaller total budget controls even a job whose individual timeout is larger.
        limited = joinpath(directory, "limited")
        mkpath(joinpath(limited, "workers"))
        state = empty_state()
        state["queue_seconds"]["discovery"] = 100.0
        register_atlas_job!(state, Dict("kind" => "slow", "queue" => "events", "sleep" => 3.0))
        config["max_jobs"] = 20
        drive_atlas!(limited, state, config; additional_seconds=0.15, worker_script=script)
        @test state["stop_reason"] == "budget_exhausted"
        @test state["jobs"][1]["status"] == "timed_out"
        @test state["elapsed_seconds"] < 1.0

        # A restarted worker gets its setup allowance, rather than the much
        # smaller handler timeout that caused the preceding worker to stop.
        restart_root=joinpath(directory,"restart-setup")
        mkpath(joinpath(restart_root,"workers"))
        delayed_script=joinpath(directory,"delayed_worker.jl")
        write(delayed_script,replace(read(script,String),
            "MockAtlas.atlas_worker_main(ARGS[2])"=>"sleep(0.25); MockAtlas.atlas_worker_main(ARGS[2])"))
        restart_state=empty_state()
        restart_state["queue_seconds"]["discovery"]=100.0
        register_atlas_job!(restart_state,Dict("kind"=>"first_slow","queue"=>"events","sleep"=>3.0))
        register_atlas_job!(restart_state,Dict("kind"=>"second_slow","queue"=>"events","sleep"=>3.0))
        restart_config=atlas_config()
        restart_config["max_jobs"]=2
        restart_config["job_timeout_seconds"]=0.05
        restart_config["warmup_seconds"]=30.0
        drive_atlas!(restart_root,restart_state,restart_config;additional_seconds=15.0,worker_script=delayed_script)
        @test restart_state["stop_reason"]=="job_limit"
        @test restart_state["jobs"][1]["status"]=="timed_out"
        @test restart_state["jobs"][2]["status"]=="timed_out"
        @test isempty(restart_state["setup_failures"])

        broken = joinpath(directory, "broken")
        mkpath(joinpath(broken, "workers"))
        state = empty_state()
        state["queue_seconds"]["discovery"] = 100.0
        register_atlas_job!(state, Dict("kind" => "broken", "queue" => "events", "malformed" => true))
        @test_throws ArgumentError drive_atlas!(broken, state, config; additional_seconds=10.0, worker_script=script)
        @test state["jobs"][1]["status"] == "unresolved"
        @test TOML.parsefile(joinpath(broken, "state.toml"))["jobs"][1]["status"] == "unresolved"
        lock = acquire_atlas_lock(broken)
        release_atlas_lock(lock)

        # Killing the coordinator also kills a worker that never yields to Julia tasks.
        orphan_test = joinpath(directory, "parent-death")
        mkpath(joinpath(orphan_test, "workers"))
        coordinator = joinpath(directory, "coordinator.jl")
        open(coordinator, "w") do stream
            println(stream, "module Coordinator; using TOML, SHA, LinearAlgebra; include(", repr(engine), "); end")
            println(stream, "lock = Coordinator.acquire_atlas_lock(ARGS[1])")
            println(stream, "worker = Coordinator.start_atlas_worker(ARGS[1], 30.0; script=ARGS[2])")
            println(stream, "Coordinator.atomic_toml(joinpath(ARGS[1], \"info.toml\"), Dict(\"worker_pid\"=>getpid(worker.process)))")
            println(stream, "Coordinator.execute_atlas_worker!(worker, Dict(\"busy\"=>true), Dict(), 120.0)")
        end
        process = run(pipeline(`$(Base.julia_cmd()) --startup-file=no $coordinator $orphan_test $script`;
            stdout=devnull, stderr=devnull); wait=false)
        worker_pid = 0
        try
            deadline = time() + 30.0
            info = joinpath(orphan_test, "info.toml")
            while !isfile(info) && time() < deadline && !process_exited(process)
                sleep(0.02)
            end
            @test isfile(info)
            if isfile(info)
                worker_pid = TOML.parsefile(info)["worker_pid"]
                sleep(0.1)
                kill(process, Base.SIGKILL)
                wait(process)
                running() = isfile("/proc/$worker_pid/stat") && !occursin(r"\) Z ", read("/proc/$worker_pid/stat", String))
                deadline = time() + 5.0
                while running() && time() < deadline
                    sleep(0.02)
                end
                @test !running()
                lock = acquire_atlas_lock(orphan_test)
                release_atlas_lock(lock)
            end
        finally
            process_exited(process) || kill(process, Base.SIGKILL)
            wait(process)
            worker_pid > 0 && ccall(:kill, Cint, (Cint, Cint), worker_pid, 9)
        end
    end
end

@testset "Resume validates immutable evidence" begin
    mktempdir() do root
        mkpath(joinpath(root, "src"))
        mkpath(joinpath(root, "scripts", "atlas"))
        write(joinpath(root, "Project.toml"), "name = \"Fixture\"\n")
        write(joinpath(root, "Manifest.toml"), "julia_version = \"1.10.12\"\n")
        write(joinpath(root, "scripts", "zero_drive_atlas.jl"), "# fixture\n")
        write(joinpath(root, "src", "fixture.jl"), "# model\n")
        mktempdir() do output
            hashes = atlas_source_hashes(root)
            snapshot_atlas_source(output, hashes; root=root)
            config = atlas_config()
            atomic_toml(joinpath(output, "config.toml"), config)
            state = empty_state()
            merge!(state, Dict("schema_version" => 1, "source_hashes" => hashes, "config_hash" => toml_digest(config)))
            register_atlas_job!(state, Dict("kind" => "test", "queue" => "events"))
            state["jobs"][1]["status"] = "running"
            save_atlas_state(output, state)
            recovered, _ = validate_atlas_resume(output; root=root)
            @test recovered["jobs"][1]["status"] == "interrupted"
            modified = deepcopy(state)
            modified["jobs"][1]["result_file"] = "../../victim"
            save_atlas_state(output, modified)
            @test_throws ArgumentError validate_atlas_resume(output; root=root)
            modified = deepcopy(state)
            push!(modified["jobs"], deepcopy(modified["jobs"][1]))
            save_atlas_state(output, modified)
            @test_throws ArgumentError validate_atlas_resume(output; root=root)
            modified = deepcopy(state)
            modified["jobs"][1]["payload"]["kind"] = "changed"
            save_atlas_state(output, modified)
            @test_throws ArgumentError validate_atlas_resume(output; root=root)
            save_atlas_state(output, state)
            mkpath(joinpath(output, "results"))
            symlink(joinpath(output, "config.toml"), joinpath(output, "results", "job-000001.toml"))
            @test_throws ArgumentError validate_atlas_resume(output; root=root)
            rm(joinpath(output, "results", "job-000001.toml"))
            write(joinpath(output, "source", "src", "fixture.jl"), "changed\n")
            @test_throws ArgumentError validate_atlas_resume(output; root=root)
            cp(joinpath(root, "src", "fixture.jl"), joinpath(output, "source", "src", "fixture.jl"); force=true)
            write(joinpath(root, "src", "fixture.jl"), "changed\n")
            @test_throws ArgumentError validate_atlas_resume(output; root=root)
        end
    end
end
end

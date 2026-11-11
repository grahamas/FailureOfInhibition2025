module AtlasStorageTests
using Test,TOML,SHA,LinearAlgebra
include(joinpath(@__DIR__,"..","scripts","atlas","storage.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","engine.jl"))

function storage_fixture(root;cap=4*1024^2)
    atomic_toml(joinpath(root,".storage.toml"),storage_policy(cap))
    storage_budget(root)
end

@testset "Pilot operational guards reject invalid values before creating output" begin
    mktempdir() do root
        for (index,invalid) in enumerate((true,NaN,Inf,0.0))
            config=atlas_config()
            config["pilot_guard_seconds"]=invalid
            output=joinpath(root,"guard-$index")
            @test_throws ArgumentError run_pilot(output,config;storage_bytes=5_000_000_000)
            @test !ispath(output)
        end
        for (index,invalid) in enumerate((true,1.5,0))
            config=atlas_config()
            config["pilot_max_jobs"]=invalid
            output=joinpath(root,"jobs-$index")
            @test_throws ArgumentError run_pilot(output,config;storage_bytes=5_000_000_000)
            @test !ispath(output)
        end
    end
end

@testset "Reserved atomic storage writes" begin
    mktempdir() do root
        budget=storage_fixture(root)
        snapshot=storage_snapshot!(budget)
        @test snapshot["retained_bytes"]>0
        @test snapshot["cap_bytes"]==4*1024^2
        path=joinpath(root,"report.txt")
        storage_atomic_write(path,io->write(io,"original");budget=budget)
        @test read(path,String)=="original"
        @test_throws StorageBudgetExceeded storage_atomic_write(path,io->write(io,repeat("x",budget.report+1));budget=budget)
        @test read(path,String)=="original"
        @test all(name->name in (".storage.toml","report.txt"),readdir(root))
        @test_throws ArgumentError storage_atomic_write(joinpath(root,"..","escape"),io->write(io,"bad");budget=budget)
        outside=joinpath(root,"outside")
        write(outside,"preserved")
        symlink(outside,joinpath(root,"link"))
        @test_throws ArgumentError storage_atomic_write(joinpath(root,"link"),io->write(io,"bad");budget=budget)
        @test read(outside,String)=="preserved"
        observation=storage_snapshot!(budget)
        @test observation["peak_observed_bytes"]>=observation["retained_bytes"]
        @test observation["retained_bytes"]<=observation["cap_bytes"]
        fill=joinpath(root,"fill")
        write(fill,zeros(UInt8,budget.cap-budget.checkpoint-budget.headroom))
        @test_throws StorageBudgetExceeded storage_dispatch_ready!(budget)
        @test_throws StorageBudgetExceeded storage_atomic_write(path,io->write(io,"new");budget=budget)
        @test read(path,String)=="original"
        rm(fill)
    end
    @test_throws ArgumentError storage_policy(1000)
end

@testset "Storage stops preserve a small durable checkpoint" begin
    mktempdir() do root
        budget=storage_fixture(root)
        original=Dict{String,Any}("status"=>"running","jobs"=>[Dict("id"=>"job-000001","status"=>"running")])
        save_atlas_state(root,original)
        proposed=merge(original,Dict("large_proposals"=>repeat("x",budget.checkpoint+1)))
        @test_throws StorageBudgetExceeded save_atlas_state(root,proposed)
        retained=TOML.parsefile(joinpath(root,"state.toml"))
        @test retained["status"]=="storage_budget"
        @test retained["jobs"][1]["status"]=="interrupted"
        @test !haskey(retained,"large_proposals")
    end
    mktempdir() do root
        budget=storage_fixture(root)
        baseline=max(0,budget.cap-budget.checkpoint-budget.report-2budget.result-2budget.headroom-65536)
        write(joinpath(root,"baseline"),zeros(UInt8,baseline))
        storage_dispatch_ready!(budget)
        for index in 1:4
            write(joinpath(root,"slot-$index"),zeros(UInt8,budget.file_limit))
        end
        @test_throws StorageBudgetExceeded storage_dispatch_ready!(budget)
        atomic_toml(joinpath(root,"state.toml"),Dict("status"=>"storage_budget","jobs"=>Any[]))
        @test TOML.parsefile(joinpath(root,"state.toml"))["status"]=="storage_budget"
        @test storage_snapshot!(budget)["retained_bytes"]<=budget.cap
    end
    mktempdir() do root
        budget=storage_fixture(root)
        config=Dict("mode"=>"pilot","storage_bytes"=>budget.cap)
        atomic_toml(joinpath(root,"config.toml"),config)
        atomic_toml(joinpath(root,"state.toml"),Dict("mode"=>"pilot","storage_policy_hash"=>toml_digest(storage_policy(budget.cap)),"config_hash"=>toml_digest(config)))
        @test storage_budget(root).cap==budget.cap
        open(joinpath(root,".storage.toml"),"w") do io
            TOML.print(io,storage_policy(2budget.cap))
        end
        @test_throws ArgumentError storage_budget(root)
    end
end

@testset "Strict worker has only fixed bounded writable transport" begin
    mktempdir() do root
        budget=storage_fixture(root;cap=16*1024^2)
        mkpath(joinpath(root,"workers"));mkpath(joinpath(root,"results"))
        script=joinpath(root,"mock_worker.jl")
        storage_source=abspath(joinpath(@__DIR__,"..","scripts","atlas","storage.jl"))
        engine_source=abspath(joinpath(@__DIR__,"..","scripts","atlas","engine.jl"))
        open(script,"w") do io
            println(io,"module MockStorageWorker; using TOML,SHA,LinearAlgebra")
            println(io,"include(",repr(storage_source),");include(",repr(engine_source),")")
            println(io,"function execute_job(job,config)")
            println(io,"if get(job,\"busy\",false);println(stderr,\"BUSY_ENTERED\");flush(stderr);while true;end;end")
            println(io,"blocked=try;write(joinpath(ENV[\"TMPDIR\"],\"forbidden\"),\"bad\");false;catch;true;end")
            println(io,"get(job,\"flood\",false) && write(stderr,repeat(\"x\",10^7))")
            println(io,"get(job,\"large\",false) && return Dict(\"status\"=>\"completed\",\"evidence\"=>Dict(\"large\"=>repeat(\"x\",10^6)),\"followups\"=>Any[])")
            println(io,"Dict(\"status\"=>\"completed\",\"evidence\"=>Dict(\"blocked\"=>blocked,\"pid\"=>getpid(),\"parent_pid\"=>ccall(:getppid,Cint,())),\"followups\"=>Any[])")
            println(io,"end;end;MockStorageWorker.atlas_worker_main(ARGS[2])")
        end
        worker=start_atlas_worker(root,60.0;script=script,storage=budget)
        try
            @test worker.strict
            result,status,seconds,message=execute_atlas_worker!(worker,Dict("kind"=>"test"),atlas_config(),30.0;storage=budget)
            @test status=="completed"
            @test result["evidence"]["blocked"]
            @test result["evidence"]["pid"]<=3
            @test !isfile(joinpath(root,"runtime","tmp","forbidden"))
            @test_throws ArgumentError execute_atlas_worker!(worker,Dict("kind"=>"test"),atlas_config(),30.0;storage=budget)
            job=Dict{String,Any}("id"=>"job-000001","result_file"=>joinpath("results","job-000001.toml"))
            atomic_toml(joinpath(root,job["result_file"]),result)
            job["result_hash"]=bytes2hex(sha256(read(joinpath(root,job["result_file"]))))
            atomic_toml(joinpath(root,"state.toml"),Dict("jobs"=>[job]))
            @test cleanup_transport_duplicate!(root,job,worker)
            @test isfile(joinpath(worker.directory,"response.toml"))
            result,status,seconds,message=execute_atlas_worker!(worker,Dict("kind"=>"large","large"=>true),atlas_config(),30.0;storage=budget)
            @test status=="storage_budget"
            @test occursin("reserved",message)
            @test filesize(joinpath(worker.directory,"response.toml"))<=budget.result
            observation=storage_snapshot!(budget)
            @test observation["retained_bytes"]<=budget.cap
            @test observation["peak_observed_bytes"]<=budget.cap
        finally
            stop_atlas_worker!(worker)
        end
        worker=start_atlas_worker(root,60.0;script=script,storage=budget)
        try
            result,status,seconds,message=execute_atlas_worker!(worker,Dict("kind"=>"flood","flood"=>true),atlas_config(),30.0;storage=budget)
            @test status=="storage_budget"
            @test result===nothing
            @test filesize(joinpath(worker.directory,"worker.log"))<=budget.file_limit
            @test storage_snapshot!(budget)["retained_bytes"]<=budget.cap
        finally
            stop_atlas_worker!(worker)
        end
        coordinator=joinpath(root,"coordinator-death.jl")
        open(coordinator,"w") do io
            println(io,"module Coordinator;using TOML,SHA,LinearAlgebra")
            println(io,"include(",repr(storage_source),");include(",repr(engine_source),");end")
            println(io,"budget=Coordinator.storage_budget(ARGS[1])")
            println(io,"worker=Coordinator.start_atlas_worker(ARGS[1],60.0;script=ARGS[2],storage=budget)")
            println(io,"Coordinator.atomic_toml(joinpath(ARGS[1],\"coordinator-worker.toml\"),Dict(\"wrapper_pid\"=>getpid(worker.process),\"worker_directory\"=>worker.directory))")
            println(io,"Coordinator.execute_atlas_worker!(worker,Dict(\"kind\"=>\"busy\",\"busy\"=>true),Coordinator.atlas_config(),120.0;storage=budget)")
        end
        command=`$(Base.julia_cmd()) --startup-file=no $coordinator $root $script`
        coordinator_process=run(pipeline(command;stdout=devnull,stderr=devnull);wait=false)
        descendants=Int[]
        identities=Dict{Int,Union{Nothing,String}}()
        function children(pid)
            path="/proc/$pid/task/$pid/children"
            direct=try parse.(Int,split(read(path,String))) catch; Int[] end
            vcat(direct,[nested for child in direct for nested in children(child)])
        end
        function running(pid)
            token=get(identities,pid,nothing)
            token===nothing && return false
            try
                process_start_token(pid)==token && !occursin(r"\) Z ",read("/proc/$pid/stat",String))
            catch
                false
            end
        end
        try
            info_path=joinpath(root,"coordinator-worker.toml")
            deadline=time()+60.0
            while !isfile(info_path) && time()<deadline && !process_exited(coordinator_process)
                sleep(0.02)
            end
            @test isfile(info_path)
            if isfile(info_path)
                info=TOML.parsefile(info_path)
                log_path=joinpath(info["worker_directory"],"worker.log")
                while !occursin("BUSY_ENTERED",read(log_path,String)) && time()<deadline && !process_exited(coordinator_process)
                    sleep(0.02)
                end
                @test occursin("BUSY_ENTERED",read(log_path,String))
                descendants=vcat([info["wrapper_pid"]],children(info["wrapper_pid"]))
                for pid in descendants
                    identities[pid]=process_start_token(pid)
                end
                @test length(descendants)>=3
                kill(coordinator_process,Base.SIGKILL);wait(coordinator_process)
                deadline=time()+5.0
                while any(running,descendants) && time()<deadline
                    sleep(0.02)
                end
                @test !any(running,descendants)
            end
        finally
            process_exited(coordinator_process) || kill(coordinator_process,Base.SIGKILL)
            wait(coordinator_process)
            for pid in descendants
                running(pid) && ccall(:kill,Cint,(Cint,Cint),pid,9)
            end
        end
    end
end

@testset "Only durable matching transport duplicates can be reused" begin
    mktempdir() do root
        storage_fixture(root)
        mkpath(joinpath(root,"workers","fixture"));mkpath(joinpath(root,"results"))
        process=run(`$(Base.julia_cmd()) --startup-file=no -e 'exit(0)'`;wait=false);wait(process)
        worker=AtlasWorker(process,joinpath(root,"workers","fixture"),1,true,false)
        job=Dict{String,Any}("id"=>"job-000001","result_file"=>joinpath("results","job-000001.toml"))
        result=Dict("status"=>"unresolved","evidence"=>Dict("failure"=>"retained"))
        atomic_toml(joinpath(root,job["result_file"]),result)
        job["result_hash"]=bytes2hex(sha256(read(joinpath(root,job["result_file"]))))
        atomic_toml(joinpath(root,"state.toml"),Dict("jobs"=>[job]))
        response=joinpath(worker.directory,"response.toml")
        atomic_toml(response,result)
        @test cleanup_transport_duplicate!(root,job,worker)
        @test worker.reusable
        @test isfile(response)
        worker.reusable=false
        write(response,"changed")
        @test !cleanup_transport_duplicate!(root,job,worker)
        @test !worker.reusable
        @test read(joinpath(root,job["result_file"]),String)!="changed"
    end
end
end

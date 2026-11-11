const ATLAS_SCHEMA_VERSION = 1
const ATLAS_QUEUES = ("discovery", "events", "cycles", "certification")
const ATLAS_BENCHMARK_KINDS = ("discovery", "nexus", "continuation", "section", "periodic", "certification")
const ATLAS_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"Write a complete TOML document atomically; an interrupted write preserves its predecessor."
function atomic_toml(path, value)
    mkpath(dirname(path))
    islink(path) && throw(ArgumentError("refusing to replace symbolic-link evidence"))
    temporary, stream = mktemp(dirname(path))
    try
        TOML.print(stream, value; sorted=true)
        flush(stream)
        close(stream)
        ccall(:rename, Cint, (Cstring, Cstring), temporary, path) == 0 ||
            Base.systemerror("atomic atlas checkpoint rename", true)
    finally
        isopen(stream) && close(stream)
        ispath(temporary) && rm(temporary)
    end
    return path
end

function toml_digest(value)
    stream = IOBuffer()
    TOML.print(stream, value; sorted=true)
    return bytes2hex(sha256(take!(stream)))
end

function atlas_config(path=nothing)
    config = TOML.parsefile(joinpath(@__DIR__, "defaults.toml"))
    if path !== nothing
        custom = TOML.parsefile(path)
        for (key, value) in custom
            if value isa AbstractDict && get(config, key, nothing) isa AbstractDict
                merge!(config[key], value)
            else
                config[key] = value
            end
        end
    end
    validate_atlas_config(config)
    return config
end

function validate_atlas_config(config)
    get(config, "schema_version", 0) == ATLAS_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported atlas configuration schema"))
    for key in ("warmup_seconds", "benchmark_seconds", "job_timeout_seconds")
        value = config[key]
        value isa Real && isfinite(value) && value > 0 ||
            throw(ArgumentError("$key must be finite and positive"))
    end
    for key in ("max_jobs", "max_followups_per_job")
        value = config[key]
        value isa Integer && value > 0 || throw(ArgumentError("$key must be a positive integer"))
    end
    allocations = config["allocation"]
    Set(keys(allocations)) == Set(ATLAS_QUEUES) || throw(ArgumentError("allocation must name the four atlas queues"))
    all(value -> value isa Real && isfinite(value) && value > 0, values(allocations)) ||
        throw(ArgumentError("queue allocations must be finite and positive"))
    isapprox(sum(values(allocations)), 1.0; atol=1e-12, rtol=0) ||
        throw(ArgumentError("queue allocations must sum to one"))
    science = get(config, "science", nothing)
    science isa AbstractDict || throw(ArgumentError("science configuration must be a table"))
    for key in ("seed_grid", "max_corrector_iters", "max_evaluations", "continuation_steps",
                "certification_boxes", "certification_evaluations")
        value = get(science, key, nothing)
        value isa Integer && !(value isa Bool) && value > 0 ||
            throw(ArgumentError("science.$key must be a positive integer"))
    end
    for key in ("certification_seconds", "balance_atol", "event_atol")
        value = get(science, key, nothing)
        value isa Real && isfinite(value) && value > 0 ||
            throw(ArgumentError("science.$key must be finite and positive"))
    end
    2 <= science["seed_grid"] <= 31 || throw(ArgumentError("science.seed_grid must be in 2:31"))
    return config
end

"Hash all executable/model inputs used by a resumable run."
function atlas_source_hashes(root=ATLAS_REPO_ROOT)
    paths = String["Project.toml", "Manifest.toml", "scripts/zero_drive_atlas.jl"]
    for subdir in ("src", "scripts/atlas")
        for (directory, _, files) in walkdir(joinpath(root, subdir))
            append!(paths, [relpath(joinpath(directory, file), root) for file in files
                            if endswith(file, ".jl") || endswith(file, ".toml")])
        end
    end
    return Dict(path => bytes2hex(sha256(read(joinpath(root, path)))) for path in sort!(unique(paths)))
end

function source_revision(root=ATLAS_REPO_ROOT)
    try
        return strip(read(Cmd(["git", "-C", root, "rev-parse", "HEAD"]), String))
    catch
        return "unavailable"
    end
end

function snapshot_atlas_source(output, hashes; root=ATLAS_REPO_ROOT)
    for path in keys(hashes)
        destination = joinpath(output, "source", path)
        mkpath(dirname(destination))
        cp(joinpath(root, path), destination; force=false)
        bytes2hex(sha256(read(destination))) == hashes[path] ||
            error("source changed while snapshotting $path; start a new run")
    end
end

function safe_atlas_path(output, relative)
    relative isa AbstractString && !isempty(relative) && !isabspath(relative) && normpath(relative) == relative ||
        throw(ArgumentError("invalid atlas-relative path"))
    any(part -> part in ("..", "."), splitpath(relative)) && throw(ArgumentError("atlas path escapes output"))
    islink(output) && throw(ArgumentError("atlas output must not be a symbolic link"))
    current = output
    for part in splitpath(relative)
        current = joinpath(current, part)
        islink(current) && throw(ArgumentError("symbolic links are not accepted in atlas evidence paths"))
    end
    return current
end

function process_start_token(pid)
    path = "/proc/$pid/stat"
    isfile(path) || return nothing
    try
        line = read(path, String)
        # The parenthesized command name can itself contain spaces or ')'.
        fields = split(line[nextind(line, findlast(')', line)):end])
        return fields[20] # field 22: Linux process start time, measured in clock ticks
    catch
        return nothing
    end
end

function acquire_atlas_lock(output)
    Sys.islinux() || throw(ArgumentError("bounded atlas workers currently require Linux parent-death signalling"))
    path = safe_atlas_path(output, ".coordinator.lock")
    stream = open(path, "a+")
    if ccall(:fcntl, Cint, (Cint, Cint, Cint), fd(stream), 2, 1) != 0 # F_SETFD, FD_CLOEXEC
        close(stream)
        error("cannot prevent worker inheritance of coordinator lock")
    end
    # Kernel-owned lock: automatically released on death, without PID-reuse or stale-lock races.
    if ccall(:flock, Cint, (Cint, Cint), fd(stream), 2 | 4) != 0 # LOCK_EX | LOCK_NB
        close(stream)
        throw(ArgumentError("another atlas coordinator acquired this output"))
    end
    token = process_start_token(getpid())
    seekstart(stream)
    truncate(stream, 0)
    TOML.print(stream, Dict("pid" => getpid(), "start_token" => something(token, "unavailable")))
    flush(stream)
    return stream
end

function release_atlas_lock(stream)
    stream === nothing && return
    isopen(stream) || return
    ccall(:flock, Cint, (Cint, Cint), fd(stream), 8) # LOCK_UN
    close(stream)
end

function save_atlas_state(output, state)
    atomic_toml(joinpath(output, "state.toml"), state)
end

function validate_atlas_resume(output; root=ATLAS_REPO_ROOT)
    for path in ("state.toml", "config.toml", "results", "source", "workers")
        safe_atlas_path(output, path)
    end
    state = TOML.parsefile(safe_atlas_path(output, "state.toml"))
    get(state, "schema_version", 0) == ATLAS_SCHEMA_VERSION || throw(ArgumentError("unsupported checkpoint schema"))
    config = TOML.parsefile(joinpath(output, "config.toml"))
    validate_atlas_config(config)
    toml_digest(config) == state["config_hash"] || throw(ArgumentError("configuration changed since checkpoint"))
    atlas_source_hashes(root) == state["source_hashes"] || throw(ArgumentError("source changed since checkpoint; start a new run"))
    for (path, digest) in state["source_hashes"]
        bytes2hex(sha256(read(safe_atlas_path(output, joinpath("source", path))))) == digest ||
            throw(ArgumentError("source snapshot changed: $path"))
    end
    seen_ids = Set{String}()
    for (index, job) in enumerate(state["jobs"])
        id = get(job, "id", "")
        occursin(r"^job-[0-9]{6,}$", id) || throw(ArgumentError("invalid checkpoint job identifier"))
        id == "job-" * lpad(string(index), 6, '0') || throw(ArgumentError("checkpoint job identifiers are not sequential"))
        id in seen_ids && throw(ArgumentError("duplicate checkpoint job identifier"))
        push!(seen_ids, id)
        job["payload_hash"] == toml_digest(job["payload"]) || throw(ArgumentError("checkpoint job payload changed"))
        job["queue"] in ATLAS_QUEUES && job["queue"] == get(job["payload"], "queue", "events") &&
            job["kind"] == job["payload"]["kind"] || throw(ArgumentError("inconsistent checkpoint job metadata"))
        job["status"] in ("pending", "running", "completed", "unresolved", "failed", "timed_out", "interrupted") ||
            throw(ArgumentError("unknown checkpoint job status"))
        job["result_file"] == joinpath("results", id * ".toml") || throw(ArgumentError("invalid checkpoint result path"))
        result_path = safe_atlas_path(output, job["result_file"])
        job["status"] == "pending" && isfile(result_path) && throw(ArgumentError("pending job would overwrite existing evidence"))
        job["status"] == "completed" && !haskey(job, "result_hash") && throw(ArgumentError("completed job lacks bound evidence"))
        if get(job, "status", "") == "running"
            job["status"] = "interrupted"
            job["error"] = "previous coordinator ended during this job; result is unresolved"
        end
        if haskey(job, "result_hash")
            bytes2hex(sha256(read(result_path))) == job["result_hash"] ||
                throw(ArgumentError("saved result changed: $(job["id"])"))
        end
    end
    return state, config
end

function register_atlas_job!(state, payload)
    queue = get(payload, "queue", "events")
    queue in ATLAS_QUEUES || throw(ArgumentError("unknown job queue $queue"))
    haskey(payload, "kind") || throw(ArgumentError("job has no kind"))
    digest = toml_digest(payload)
    any(job -> get(job, "payload_hash", "") == digest, state["jobs"]) && return nothing
    id = "job-" * lpad(string(length(state["jobs"]) + 1), 6, '0')
    job = Dict{String,Any}("id" => id, "kind" => payload["kind"], "queue" => queue,
        "status" => "pending", "elapsed_seconds" => 0.0, "payload_hash" => digest,
        "payload" => payload, "result_file" => joinpath("results", id * ".toml"), "error" => "")
    push!(state["jobs"], job)
    return job
end

function next_discovery!(state)
    index = state["discovery_index"] + 1
    state["discovery_index"] = index
    # Alternate the chart order every four jobs so boundary jobs reach both charts.
    chart = isodd(index + div(index - 1, 4)) ? "parameter" : "equilibrium"
    boundary = index % 4 == 0
    # Separate chart counters prevent the binary shell schedule from starving either chart.
    key = "$(chart)_$(boundary ? "boundary" : "interior")_index"
    ordinal = get(state, key, 0) + 1
    state[key] = ordinal
    # Realize potentially expensive shell rejection sampling inside the timed worker.
    payload = Dict{String,Any}("kind" => "discovery", "queue" => "discovery", "index" => ordinal,
        "chart" => chart, "boundary" => boundary, "generate" => true)
    return register_atlas_job!(state, payload)
end

function select_atlas_job!(state, config)
    pending = [job for job in state["jobs"] if job["status"] == "pending"]
    if !any(job -> job["queue"] == "discovery", pending)
        job = next_discovery!(state)
        job === nothing || push!(pending, job)
    end
    isempty(pending) && return nothing
    if get(state, "mode", "run") == "benchmark"
        # Calibration gets one attempt at each available job class before wall-
        # weighted scheduling. Cold compilation must not starve the remaining
        # classes behind hundreds of cheap discovery jobs. Failures count as
        # attempts; this policy does not retry until a class succeeds.
        attempted = Set(job["kind"] for job in state["jobs"] if job["status"] != "pending")
        for job in pending
            job["kind"] in ATLAS_BENCHMARK_KINDS && !(job["kind"] in attempted) && return job
        end
    end
    queue = first(sort!(unique([job["queue"] for job in pending]);
        by=q -> (state["queue_seconds"][q] / config["allocation"][q], findfirst(==(q), ATLAS_QUEUES))))
    return first(job for job in pending if job["queue"] == queue)
end

mutable struct AtlasWorker
    process::Base.Process
    directory::String
    sequence::Int
end

function stop_atlas_worker!(worker)
    worker === nothing && return
    process_exited(worker.process) || kill(worker.process, Base.SIGKILL)
    wait(worker.process)
end

function start_atlas_worker(output, timeout; script=joinpath(output, "source", "scripts", "zero_drive_atlas.jl"))
    Sys.islinux() || throw(ArgumentError("bounded atlas workers currently require Linux parent-death signalling"))
    directory = mktempdir(safe_atlas_path(output, "workers"); prefix="worker-", cleanup=false)
    logfile = open(joinpath(directory, "worker.log"), "w")
    # Install before loading the package, so even setup dies with its coordinator.
    launcher = """
    ccall(:prctl, Cint, (Cint, Culong, Culong, Culong, Culong), 1, 9, 0, 0, 0) == 0 || error("cannot install parent-death signal")
    ccall(:getppid, Cint, ()) == $(getpid()) || exit(2)
    include($(repr(abspath(script))))
    Main.ZeroDriveAtlas.atlas_main(ARGS)
    """
    project = normpath(joinpath(dirname(script), ".."))
    command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$project -e $launcher -- __worker $directory`
    command = addenv(command, "JULIA_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1", "OMP_NUM_THREADS" => "1")
    process = try
        run(pipeline(command; stdout=logfile, stderr=logfile); wait=false)
    finally
        close(logfile)
    end
    worker = AtlasWorker(process, directory, 0)
    started = time_ns()
    while !isfile(joinpath(directory, "ready.toml"))
        if process_exited(process)
            stop_atlas_worker!(worker)
            error("atlas worker failed during setup; inspect $(joinpath(directory, "worker.log"))")
        end
        if (time_ns() - started) / 1e9 >= timeout
            stop_atlas_worker!(worker)
            error("atlas worker setup timeout; inspect $(joinpath(directory, "worker.log"))")
        end
        sleep(0.02)
    end
    return worker
end

function execute_atlas_worker!(worker, payload, config, timeout)
    worker.sequence += 1
    prefix = joinpath(worker.directory, string(worker.sequence))
    atomic_toml(prefix * ".request.toml", Dict("job" => payload, "config" => config))
    started = time_ns()
    while !isfile(prefix * ".result.toml")
        elapsed = (time_ns() - started) / 1e9
        if process_exited(worker.process)
            return nothing, "failed", elapsed, "worker exited; inspect worker.log"
        elseif elapsed >= timeout
            stop_atlas_worker!(worker)
            return nothing, "timed_out", elapsed, "hard job timeout; numerical result unresolved"
        end
        sleep(min(0.02, max(0.001, timeout - elapsed)))
    end
    result = try
        TOML.parsefile(prefix * ".result.toml")
    catch exception
        return nothing, "failed", (time_ns() - started) / 1e9,
            "unreadable worker result: " * sprint(showerror, exception)
    end
    status = get(result, "status", "failed")
    status in ("completed", "unresolved", "failed") ||
        return nothing, "failed", (time_ns() - started) / 1e9, "invalid worker result status"
    return result, status, (time_ns() - started) / 1e9, get(result, "error", "")
end

"Persistent isolated worker; killed by the coordinator when a numerical job exceeds its cap."
function atlas_worker_main(directory)
    BLAS.set_num_threads(1)
    Base.cumulative_compile_timing(true)
    atomic_toml(joinpath(directory, "ready.toml"), Dict("julia_version" => string(VERSION)))
    sequence = 1
    while true
        prefix = joinpath(directory, string(sequence))
        request_file = prefix * ".request.toml"
        if !isfile(request_file)
            sleep(0.02)
            continue
        end
        request = TOML.parsefile(request_file)
        started = time_ns()
        compilation_before = Base.cumulative_compile_time_ns()
        executed_job = request["job"]
        result = try
            handler_config = merge(request["config"], get(request["config"], "science", Dict{String,Any}()))
            if get(executed_job, "kind", "") == "discovery" && get(executed_job, "generate", false) === true
                executed_job = merge(discovery_job(executed_job["index"], executed_job["chart"], executed_job["boundary"]),
                    Dict("id" => executed_job["id"]))
            end
            value = execute_job(executed_job, handler_config)
            value isa AbstractDict || throw(ArgumentError("execute_job must return a dictionary"))
            Dict{String,Any}(value)
        catch exception
            Dict{String,Any}("status" => "failed", "error" => sprint(showerror, exception, catch_backtrace()),
                "evidence" => Dict("completeness" => "CompletenessNotCertified"), "followups" => Any[])
        end
        compilation_after = Base.cumulative_compile_time_ns()
        result["executed_job"] = executed_job
        result["worker_timing"] = Dict("handler_seconds" => (time_ns() - started) / 1e9,
            "compilation_seconds" => Float64(compilation_after[1] - compilation_before[1]) / 1e9,
            "recompilation_seconds" => Float64(compilation_after[2] - compilation_before[2]) / 1e9,
            "scope" => "Julia compiler counters during handler; setup and process startup reported separately")
        try
            atomic_toml(prefix * ".result.toml", result)
        catch exception
            # A handler's evidence is a serialization contract, not permission to
            # coerce scientific values or terminate the persistent worker.
            message = "scientific result serialization failed: " * sprint(showerror, exception, catch_backtrace())
            println(stderr, message)
            flush(stderr)
            failure = Dict{String,Any}("status" => "failed", "error" => message,
                "evidence" => Dict("failure_stage" => "result_serialization",
                    "job_kind" => string(get(request["job"], "kind", "unknown")),
                    "completeness" => "CompletenessNotCertified"),
                "followups" => Any[], "executed_job" => request["job"],
                "worker_timing" => result["worker_timing"])
            atomic_toml(prefix * ".result.toml", failure)
        end
        sequence += 1
    end
end

function initialize_atlas(output, mode, config, budget)
    ispath(output) && throw(ArgumentError("output already exists; use resume or a new directory"))
    mkpath(dirname(output))
    mkdir(output)
    mkpath(joinpath(output, "workers"))
    mkpath(joinpath(output, "results"))
    hashes = atlas_source_hashes()
    snapshot_atlas_source(output, hashes)
    atomic_toml(joinpath(output, "config.toml"), config)
    state = Dict{String,Any}("schema_version" => ATLAS_SCHEMA_VERSION, "mode" => mode,
        "config_hash" => toml_digest(config), "source_hashes" => hashes,
        "source_revision" => source_revision(), "julia_version" => string(VERSION),
        "scheduling_policy" => mode == "benchmark" ? "first_attempt_per_available_kind_then_weighted_wall_time" : "weighted_wall_time",
        "budget_seconds" => Float64(budget), "elapsed_seconds" => 0.0,
        "setup_seconds" => 0.0, "setup_failures" => String[], "stop_reason" => "initialized",
        "status" => "running", "discovery_index" => 0, "jobs" => Dict{String,Any}[],
        "queue_seconds" => Dict(queue => 0.0 for queue in ATLAS_QUEUES),
        "completeness" => "CompletenessNotCertified")
    for payload in initial_jobs(merge(config, get(config, "science", Dict{String,Any}())))
        register_atlas_job!(state, payload)
    end
    save_atlas_state(output, state)
    return state
end

function drive_atlas!(output, state, config; additional_seconds,
                      worker_script=joinpath(output, "source", "scripts", "zero_drive_atlas.jl"), lock_held=false)
    isfinite(additional_seconds) && additional_seconds > 0 || throw(ArgumentError("budget must be finite and positive"))
    lock = lock_held ? nothing : acquire_atlas_lock(output)
    worker = nothing
    measured_start = UInt64(0)
    previous_elapsed = 0.0
    try
        previous_elapsed = Float64(state["elapsed_seconds"])
        state["budget_seconds"] = previous_elapsed + additional_seconds
        state["status"] = "running"
        state["stop_reason"] = "running"
        save_atlas_state(output, state)
        setup_start = time_ns()
        try
            worker = start_atlas_worker(output, config["warmup_seconds"]; script=worker_script)
        finally
            state["setup_seconds"] += (time_ns() - setup_start) / 1e9
        end
        measured_start = time_ns()
        while true
            elapsed = (time_ns() - measured_start) / 1e9
            state["elapsed_seconds"] = previous_elapsed + elapsed
            if elapsed >= additional_seconds
                state["stop_reason"] = "budget_exhausted"
                break
            end
            count(job -> job["status"] != "pending", state["jobs"]) >= config["max_jobs"] && begin
                state["stop_reason"] = "job_limit"
                break
            end
            job = select_atlas_job!(state, config)
            job === nothing && begin
                state["stop_reason"] = "queue_empty"
                break
            end
            save_atlas_state(output, state)
            if worker === nothing || process_exited(worker.process)
                restart_start = time_ns()
                remaining = additional_seconds - (restart_start - measured_start) / 1e9
                remaining <= 0 && continue
                try
                    worker = start_atlas_worker(output, min(remaining, config["job_timeout_seconds"]); script=worker_script)
                catch exception
                    push!(state["setup_failures"], sprint(showerror, exception))
                    state["stop_reason"] = "worker_restart_failed"
                    break
                end
            end
            remaining = additional_seconds - (time_ns() - measured_start) / 1e9
            remaining <= 0 && continue
            job["status"] = "running"
            save_atlas_state(output, state)
            worker_payload = merge(job["payload"], Dict("id" => job["id"]))
            result, status, seconds, error_message = execute_atlas_worker!(worker, worker_payload, config,
                min(config["job_timeout_seconds"], remaining))
            job["elapsed_seconds"] = seconds
            job["error"] = error_message
            state["queue_seconds"][job["queue"]] += seconds
            if result !== nothing
                result_path = safe_atlas_path(output, job["result_file"])
                atomic_toml(result_path, result)
                job["result_hash"] = bytes2hex(sha256(read(result_path)))
                followups = get(result, "followups", Any[])
                followups isa AbstractVector || throw(ArgumentError("worker followups must be an array"))
                # Validate the whole batch before changing queue state.
                for payload in followups
                    payload isa AbstractDict && haskey(payload, "kind") && get(payload, "queue", "events") in ATLAS_QUEUES ||
                        throw(ArgumentError("malformed worker followup"))
                    toml_digest(payload)
                end
                if length(followups) > config["max_followups_per_job"]
                    status = "unresolved"
                    job["error"] = "followup limit exceeded; complete unqueued proposals retained in result"
                else
                    for payload in followups
                        register_atlas_job!(state, payload)
                    end
                end
            end
            job["status"] = status
            state["elapsed_seconds"] = previous_elapsed + (time_ns() - measured_start) / 1e9
            save_atlas_state(output, state)
        end
        state["status"] = state["stop_reason"]
    catch exception
        for job in state["jobs"]
            if job["status"] == "running"
                job["status"] = "unresolved"
                job["error"] = "coordinator could not validate/persist complete job evidence: " * sprint(showerror, exception)
            end
        end
        state["status"] = "interrupted"
        state["stop_reason"] = sprint(showerror, exception)
        push!(state["setup_failures"], sprint(showerror, exception))
        rethrow()
    finally
        measured_start == 0 || (state["elapsed_seconds"] = previous_elapsed + (time_ns() - measured_start) / 1e9)
        try
            stop_atlas_worker!(worker)
            save_atlas_state(output, state)
        finally
            release_atlas_lock(lock)
        end
    end
    return state
end

function run_atlas(mode, output, config; budget_seconds=nothing)
    mode in ("benchmark", "run") || throw(ArgumentError("mode must be benchmark or run"))
    config = merge(config, Dict("mode" => mode))
    validate_atlas_config(config)
    budget = mode == "benchmark" ? get(config, "benchmark_seconds", 1200.0) : budget_seconds
    budget === nothing && throw(ArgumentError("pilot run requires an explicit --budget-seconds chosen after benchmarking"))
    budget isa Real && isfinite(budget) && budget > 0 || throw(ArgumentError("budget must be finite and positive"))
    state = initialize_atlas(output, mode, config, budget)
    lock = acquire_atlas_lock(output)
    try
        drive_atlas!(output, state, config; additional_seconds=budget, lock_held=true)
        write_report(output)
        return state
    finally
        release_atlas_lock(lock)
    end
end

function report_atlas(output)
    lock = acquire_atlas_lock(output)
    try
        return write_report(output)
    finally
        release_atlas_lock(lock)
    end
end

function resume_atlas(output; budget_seconds=nothing)
    budget_seconds === nothing && throw(ArgumentError("resume requires an explicit additional --budget-seconds"))
    lock = acquire_atlas_lock(output)
    try
        state, config = validate_atlas_resume(output)
        # The job cap is cumulative, so resuming cannot silently reset the agreed bound.
        drive_atlas!(output, state, config; additional_seconds=budget_seconds, lock_held=true)
        write_report(output)
        return state
    finally
        release_atlas_lock(lock)
    end
end

function atlas_main(arguments)
    isempty(arguments) && throw(ArgumentError("usage: zero_drive_atlas.jl benchmark|run|resume|report OUTPUT [--config FILE] [--budget-seconds N]"))
    if first(arguments) == "__worker"
        length(arguments) == 2 || throw(ArgumentError("invalid internal worker arguments"))
        return atlas_worker_main(arguments[2])
    end
    length(arguments) >= 2 || throw(ArgumentError("an output directory is required"))
    mode, output = arguments[1:2]
    options = Dict{String,String}()
    remainder = arguments[3:end]
    iseven(length(remainder)) || throw(ArgumentError("options require values"))
    for index in 1:2:length(remainder)
        key = remainder[index]
        key in ("--config", "--budget-seconds") || throw(ArgumentError("unknown option $key"))
        haskey(options, key) && throw(ArgumentError("duplicate option $key"))
        options[key] = remainder[index + 1]
    end
    budget = haskey(options, "--budget-seconds") ? parse(Float64, options["--budget-seconds"]) : nothing
    if mode == "report"
        isempty(options) || throw(ArgumentError("report accepts no run options"))
        return report_atlas(output)
    elseif mode == "resume"
        haskey(options, "--config") && throw(ArgumentError("resume uses its immutable saved configuration"))
        return resume_atlas(output; budget_seconds=budget)
    elseif mode == "benchmark"
        budget === nothing || throw(ArgumentError("set benchmark_seconds in the configuration for a shorter benchmark"))
    elseif mode != "run"
        throw(ArgumentError("unknown atlas command $mode"))
    end
    return run_atlas(mode, output, atlas_config(get(options, "--config", nothing)); budget_seconds=budget)
end

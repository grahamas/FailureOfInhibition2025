# Storage guards cover all run-owned files, including worker caches and scratch.
# Controlled writes reserve space before writing. The pilot worker has a read-only
# filesystem except four fixed transport/log inodes, each OS file-size limited.
# Monitoring adds early stops; it is not the mechanism establishing the ceiling.
struct StorageBudgetExceeded <: Exception
    message::String
end
is_storage_error(exception) = exception isa StorageBudgetExceeded
Base.showerror(io::IO, error::StorageBudgetExceeded) = print(io, error.message)

mutable struct StorageBudget
    root::String
    cap::Int
    checkpoint::Int
    report::Int
    result::Int
    headroom::Int
    file_limit::Int
    peak::Int
end

function storage_policy(cap::Integer)
    cap >= 1024^2 || throw(ArgumentError("storage cap must be at least one MiB for initialization reserves"))
    cap <= typemax(Int) || throw(ArgumentError("storage cap is too large"))
    cap = Int(cap)
    checkpoint = min(16 * 1024^2, max(256, cap ÷ 100))
    report = min(64 * 1024^2, max(256, cap ÷ 50))
    result = min(32 * 1024^2, max(256, cap ÷ 50))
    file_limit = 4096 * div(min(64 * 1024^2, cap ÷ 20),4096)
    metadata_padding=min(16*1024^2,max(128*1024,cap÷100))
    headroom = max(cap ÷ 20, 4file_limit+metadata_padding)
    checkpoint + report + 2result + 2headroom < cap ||
        throw(ArgumentError("storage cap is too small for checkpoint/report/in-flight reserves"))
    Dict{String,Any}("schema_version"=>1,"cap_bytes"=>cap,"checkpoint_reserve_bytes"=>checkpoint,
        "report_reserve_bytes"=>report,"result_reserve_bytes"=>result,
        "monitor_headroom_bytes"=>headroom,"file_limit_bytes"=>file_limit,
        "enforcement"=>"reserved_controlled_writes_readonly_worker_filesystem_fixed_capped_transport_inodes")
end

function storage_budget(output)
    policy_path = joinpath(output, ".storage.toml")
    islink(policy_path) && throw(ArgumentError("storage policy must not be a symlink"))
    isfile(policy_path) || return nothing
    policy = TOML.parsefile(policy_path)
    expected = storage_policy(policy["cap_bytes"])
    policy == expected || throw(ArgumentError("storage policy changed or is unsupported"))
    state_path=joinpath(output,"state.toml")
    if isfile(state_path)
        islink(state_path) && throw(ArgumentError("storage checkpoint must not be a symlink"))
        state=TOML.parsefile(state_path)
        if haskey(state,"storage_policy_hash") || get(state,"mode","")=="pilot"
            get(state,"storage_policy_hash","")==toml_digest(policy) || throw(ArgumentError("authorized storage policy hash changed"))
            config_path=joinpath(output,"config.toml")
            islink(config_path) && throw(ArgumentError("storage configuration must not be a symlink"))
            config=TOML.parsefile(config_path)
            config["storage_bytes"]==policy["cap_bytes"] && state["config_hash"]==toml_digest(config) ||
                throw(ArgumentError("authorized pilot storage configuration changed"))
        end
    end
    StorageBudget(abspath(output), policy["cap_bytes"],policy["checkpoint_reserve_bytes"],
        policy["report_reserve_bytes"],policy["result_reserve_bytes"],policy["monitor_headroom_bytes"],
        policy["file_limit_bytes"],0)
end

function storage_budget_for(path)
    directory = abspath(dirname(path))
    while true
        isfile(joinpath(directory, ".storage.toml")) && return storage_budget(directory)
        parent = dirname(directory)
        parent == directory && return nothing
        directory = parent
    end
end

function storage_owned_bytes(root)
    islink(root) && throw(ArgumentError("storage root must not be a symlink"))
    total = 0
    for (directory, dirs, files) in walkdir(root; follow_symlinks=false)
        for path in vcat([directory], [joinpath(directory,name) for name in files],
                         [joinpath(directory,name) for name in dirs if islink(joinpath(directory,name))])
            info = try lstat(path) catch exception
                ispath(path) && rethrow(exception)
                continue # concurrently renamed atomic temporary file
            end
            total = Base.checked_add(total, max(Int(info.size), Int(info.blocks)*512))
        end
    end
    total
end

function storage_snapshot!(budget::StorageBudget)
    retained = storage_owned_bytes(budget.root)
    budget.peak = max(budget.peak,retained)
    Dict{String,Any}("cap_bytes"=>budget.cap,"retained_bytes"=>retained,
        "peak_observed_bytes"=>budget.peak,"checkpoint_reserve_bytes"=>budget.checkpoint,
        "report_reserve_bytes"=>budget.report,"result_reserve_bytes"=>budget.result,
        "monitor_headroom_bytes"=>budget.headroom,"file_limit_bytes"=>budget.file_limit,
        "measurement"=>"sum_of_max_logical_and_allocated_entry_bytes_in_owned_tree",
        "enforcement"=>"reserved_controlled_writes_readonly_worker_filesystem_fixed_capped_transport_inodes",
        "worker_writable_inode_allowance_bytes"=>4budget.file_limit,
        "overshoot_bound_bytes"=>0)
end
storage_snapshot!(::Nothing) = Dict{String,Any}()

function storage_preflight!(budget::StorageBudget, additional; category="metadata")
    observation = storage_snapshot!(budget)
    reserve = category == "checkpoint" ? 0 : budget.checkpoint
    category in ("checkpoint","report") || (reserve += budget.report)
    reserve += budget.headroom # reserve all possible fixed-slot growth, including during checkpoints
    observation["retained_bytes"] + additional + reserve <= budget.cap ||
        throw(StorageBudgetExceeded("storage_budget: controlled $category write exceeds reserved capacity"))
    return observation
end

function storage_dispatch_ready!(budget::StorageBudget)
    observation = storage_snapshot!(budget)
    observation["retained_bytes"] + budget.checkpoint + budget.report + 2budget.result + 2budget.headroom <= budget.cap ||
        throw(StorageBudgetExceeded("storage_budget: insufficient capacity reserved for the next job and final reports"))
    return observation
end
storage_dispatch_ready!(::Nothing) = nothing

function storage_monitor!(budget::StorageBudget)
    observation = storage_snapshot!(budget)
    observation["retained_bytes"] + budget.checkpoint + budget.report + budget.headroom <= budget.cap ||
        throw(StorageBudgetExceeded("storage_budget: worker runtime/cache writes reached the reserved stop threshold"))
    return observation
end
storage_monitor!(::Nothing) = nothing

mutable struct StorageBuffer <: IO
    buffer::IOBuffer
    limit::Int
    count::Int
end
Base.isopen(io::StorageBuffer) = isopen(io.buffer)
Base.flush(io::StorageBuffer) = flush(io.buffer)
function Base.unsafe_write(io::StorageBuffer, pointer::Ptr{UInt8}, count::UInt)
    count <= io.limit-io.count || throw(StorageBudgetExceeded("storage_budget: serialized file exceeds its reserved per-file allowance"))
    wrote = Base.unsafe_write(io.buffer,pointer,count)
    io.count += Int(wrote)
    wrote
end
function Base.write(io::StorageBuffer, value::UInt8)
    io.count < io.limit || throw(StorageBudgetExceeded("storage_budget: serialized file exceeds its reserved per-file allowance"))
    io.count += 1
    write(io.buffer,value)
end

function storage_atomic_write(path, writer::Function; budget=nothing, category="report")
    if budget === nothing
        temporary,stream=mktemp(dirname(path))
        try
            writer(stream);flush(stream);close(stream)
            ccall(:rename,Cint,(Cstring,Cstring),temporary,path)==0 || Base.systemerror("atomic storage rename",true)
        finally
            isopen(stream) && close(stream)
            ispath(temporary) && rm(temporary)
        end
        return path
    end
    relative=relpath(abspath(path),budget.root)
    safe_atlas_path(budget.root,relative)
    limit=category=="checkpoint" ? budget.checkpoint : category=="report" ? budget.report : budget.result
    bounded=StorageBuffer(IOBuffer(),limit,0)
    writer(bounded)
    bytes=take!(bounded.buffer)
    # Round up for file allocation and allow one newly created directory entry.
    storage_preflight!(budget,cld(length(bytes),4096)*4096+8192;category=category)
    mkpath(dirname(path))
    temporary,stream=mktemp(dirname(path))
    try
        write(stream,bytes);flush(stream)
        ccall(:fsync,Cint,(Cint,),fd(stream))==0 || Base.systemerror("fsync storage evidence",true)
        close(stream)
        storage_snapshot!(budget)
        ccall(:rename,Cint,(Cstring,Cstring),temporary,path)==0 || Base.systemerror("atomic storage rename",true)
        directory_fd=ccall(:open,Cint,(Cstring,Cint),dirname(path),65536)
        directory_fd>=0 || Base.systemerror("open storage directory",true)
        try
            ccall(:fsync,Cint,(Cint,),directory_fd)==0 || Base.systemerror("fsync storage directory",true)
        finally
            ccall(:close,Cint,(Cint,),directory_fd)
        end
    finally
        isopen(stream) && close(stream)
        ispath(temporary) && rm(temporary)
    end
    return path
end
storage_atomic_write(writer::Function,path;kwargs...) = storage_atomic_write(path,writer;kwargs...)

function cleanup_transport_duplicate!(output,job,worker)
    transport=worker.strict ? joinpath(worker.directory,"response.toml") : joinpath(worker.directory,string(worker.sequence)*".result.toml")
    isfile(transport) && !islink(transport) || return false
    committed=safe_atlas_path(output,job["result_file"])
    checkpoint=TOML.parsefile(safe_atlas_path(output,"state.toml"))
    recorded=only(filter(row->row["id"]==job["id"],checkpoint["jobs"]))
    haskey(recorded,"result_hash") || return false
    digest=recorded["result_hash"]
    bytes2hex(sha256(read(committed)))==digest==bytes2hex(sha256(read(transport))) || return false
    if worker.strict
        worker.reusable=true # reuse only after the durable committed result and checkpoint agree
    else
        rm(transport)
    end
    return true
end

function storage_fixed_toml(path,value)
    basename(path) in ("ready.toml","response.toml","done.toml") || throw(ArgumentError("unapproved worker transport path"))
    root=get(ENV,"ATLAS_STORAGE_ROOT","")
    budget=storage_budget(root)
    budget===nothing && error("fixed worker transport requires its immutable storage policy")
    limit=basename(path)=="response.toml" ? budget.result : min(4096,budget.file_limit)
    bounded=StorageBuffer(IOBuffer(),limit,0)
    TOML.print(bounded,value;sorted=true)
    bytes=take!(bounded.buffer)
    # Existing inode is deliberately retained: bwrap grants precisely this inode write access.
    open(path,"w") do stream
        write(stream,bytes);flush(stream)
        ccall(:fsync,Cint,(Cint,),fd(stream))==0 || Base.systemerror("fsync fixed worker response",true)
    end
    path
end

function storage_sync_initialization!(root)
    directories=String[]
    for (directory,_,files) in walkdir(root;follow_symlinks=false)
        push!(directories,directory)
        for name in files
            path=joinpath(directory,name)
            islink(path) && throw(ArgumentError("initial pilot snapshot contains a symlink"))
            open(path,"r") do io
                ccall(:fsync,Cint,(Cint,),fd(io))==0 || Base.systemerror("fsync initial pilot snapshot",true)
            end
        end
    end
    for directory in vcat(reverse(directories),[dirname(root)])
        descriptor=ccall(:open,Cint,(Cstring,Cint),directory,65536)
        descriptor>=0 || Base.systemerror("open initial pilot directory",true)
        try
            ccall(:fsync,Cint,(Cint,),descriptor)==0 || Base.systemerror("fsync initial pilot directory",true)
        finally
            ccall(:close,Cint,(Cint,),descriptor)
        end
    end
end

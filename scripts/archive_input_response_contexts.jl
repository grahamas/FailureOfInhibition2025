"""Losslessly pack completed two-input geometry contexts without changing observations."""
module InputResponseArchives

using SHA, TOML, Tar

digest(path) = bytes2hex(open(SHA.sha256, path))

function relative_parts(name)
    name isa AbstractString && !isempty(name) && !isabspath(name) ||
        throw(ArgumentError("invalid checkpoint path: $name"))
    (occursin('\0', name) || occursin('\n', name)) &&
        throw(ArgumentError("invalid checkpoint path: $name"))
    parts = split(name, '/')
    all(part -> !isempty(part) && part != "." && part != "..", parts) ||
        throw(ArgumentError("invalid checkpoint path: $name"))
    parts
end

function checked_path(folder, name)
    path = folder
    for part in relative_parts(name)
        path = joinpath(path, part)
        islink(path) && throw(ArgumentError("checkpoint symlink is not supported: $name"))
    end
    path
end

function checkpoint_files(bytes)
    parsed = TOML.parse(String(copy(bytes)))
    files = parsed["files"]
    files isa AbstractDict || throw(ArgumentError("checkpoint has no file map"))
    for (name, hash) in files
        relative_parts(name)
        hash isa AbstractString && occursin(r"^[0-9a-f]{64}$", hash) ||
            throw(ArgumentError("invalid checkpoint hash: $name"))
    end
    Dict{String,String}(files)
end

function verify_checkpoint(folder)
    files = checkpoint_files(read(joinpath(folder, "done.toml")))
    for (name, expected) in files
        path = checked_path(folder, name)
        isfile(path) && digest(path) == expected ||
            throw(ArgumentError("checkpoint mismatch: $path"))
    end
    files
end

function context_files(files)
    Dict(name => hash for (name, hash) in files if first(relative_parts(name)) == "contexts")
end

"""Verify old Python and new Julia tar.gz archives without extracting their contents."""
function verify_archive(path)
    isfile(path) || throw(ArgumentError("missing archive: $path"))
    headers = Tar.list(`gzip -dc $path`)
    names = [header.path for header in headers]
    length(names) == length(unique(names)) && all(header -> header.type == :file, headers) ||
        throw(ArgumentError("archive must contain unique regular files"))
    !isempty(names) && first(names) == "_checkpoint/done.toml" ||
        throw(ArgumentError("archive lacks its original checkpoint"))
    open(`tar -xOzf $path`, "r") do stream
        original = read(stream, first(headers).size)
        length(original) == first(headers).size || throw(ArgumentError("truncated archive checkpoint"))
        expected = context_files(checkpoint_files(original))
        Set(names) == union(Set(keys(expected)), Set(["_checkpoint/done.toml"])) ||
            throw(ArgumentError("archive contents do not match original checkpoint"))
        for header in headers[2:end]
            bytes = read(stream, header.size)
            length(bytes) == header.size && bytes2hex(SHA.sha256(bytes)) == expected[header.path] ||
                throw(ArgumentError("archive mismatch: $(header.path)"))
        end
        eof(stream) || throw(ArgumentError("archive has unexpected content"))
        return original, expected
    end
end

function verify_raw(folder, expected)
    root = joinpath(folder, "contexts")
    isdir(root) && !islink(root) || throw(ArgumentError("raw contexts are missing"))
    actual = Set{String}()
    for (directory, subdirectories, files) in walkdir(root)
        any(islink(joinpath(directory, name)) for name in vcat(subdirectories, files)) &&
            throw(ArgumentError("context symlinks are not supported"))
        for name in files
            path = joinpath(directory, name)
            isfile(path) || throw(ArgumentError("nonregular context file: $path"))
            push!(actual, relpath(path, folder))
        end
    end
    actual == Set(keys(expected)) || throw(ArgumentError("context files differ from checkpoint"))
    for (name, hash) in expected
        digest(checked_path(folder, name)) == hash ||
            throw(ArgumentError("raw context mismatch: $name"))
    end
    nothing
end

function write_atomic(path, value)
    temporary = path * ".tmp"
    try
        open(temporary, "w") do stream
            write(stream, value)
        end
        mv(temporary, path; force=true)
    finally
        rm(temporary; force=true)
    end
end

function study_hashes(study)
    hashes = Dict{String,String}()
    for (directory, subdirectories, files) in walkdir(study)
        any(islink(joinpath(directory, name)) for name in vcat(subdirectories, files)) &&
            throw(ArgumentError("study symlinks are not supported"))
        for name in files
            path = joinpath(directory, name)
            relative = relpath(path, study)
            relative == "checksums.toml" && continue
            relative_parts(relative)
            isfile(path) || throw(ArgumentError("nonregular study file: $path"))
            hashes[relative] = digest(path)
        end
    end
    hashes
end

function verify_study_manifest(study)
    path = joinpath(study, "checksums.toml")
    isfile(path) && !islink(path) || throw(ArgumentError("missing regular study manifest"))
    data = TOML.parsefile(path)
    get(data, "schema_version", nothing) == 1 && get(data, "algorithm", nothing) == "SHA-256" ||
        throw(ArgumentError("unsupported study manifest"))
    files = get(data, "files", nothing)
    files isa AbstractDict || throw(ArgumentError("study manifest has no file map"))
    for (name, hash) in files
        relative_parts(name)
        hash isa AbstractString && occursin(r"^[0-9a-f]{64}$", hash) ||
            throw(ArgumentError("invalid study manifest hash: $name"))
    end
    study_hashes(study) == files || throw(ArgumentError("study manifest does not match output"))
    nothing
end

function refresh_study_manifest(study)
    files = study_hashes(study)
    data = Dict("schema_version" => 1, "algorithm" => "SHA-256", "files" => files)
    write_atomic(joinpath(study, "checksums.toml"),
        sprint(stream -> TOML.print(stream, data; sorted=true)))
    verify_study_manifest(study)
end

function create_archive(folder, original, expected, archive_path)
    marker_dir = joinpath(folder, "_checkpoint")
    marker = joinpath(marker_dir, "done.toml")
    ispath(marker_dir) && (!isdir(marker_dir) || islink(marker_dir) ||
        Set(readdir(marker_dir)) != Set(["done.toml"])) &&
        throw(ArgumentError("unexpected temporary checkpoint directory"))
    if isfile(marker)
        read(marker) == original || throw(ArgumentError("temporary checkpoint differs from original"))
    else
        mkpath(marker_dir)
        write(marker, original)
    end
    list_path, list_stream = mktemp(folder)
    temporary = archive_path * ".tmp"
    try
        for name in vcat(["_checkpoint/done.toml"], sort(collect(keys(expected))))
            write(list_stream, name)
            write(list_stream, UInt8(0))
        end
        close(list_stream)
        run(pipeline(pipeline(`tar -cf - -C $folder --no-recursion --null -T $list_path`,
            `gzip -6 -c`), stdout=temporary))
        archived_marker, archived_files = verify_archive(temporary)
        archived_marker == original && archived_files == expected ||
            throw(ArgumentError("new archive differs from original checkpoint"))
        mv(temporary, archive_path; force=true)
    finally
        isopen(list_stream) && close(list_stream)
        rm(list_path; force=true)
        rm(temporary; force=true)
        rm(marker; force=true)
        isdir(marker_dir) && isempty(readdir(marker_dir)) && rm(marker_dir)
    end
end

function archive_record(original, archive_path, expected, raw_bytes)
    # All string fields below are fixed ASCII; repr supplies JSON-compatible quotes.
    fields = [
        "  \"format\": 1",
        "  \"scope\": " * repr("geometry/contexts only; numerical data unchanged"),
        "  \"original_checkpoint_sha256\": " * repr(bytes2hex(SHA.sha256(original))),
        "  \"archive_sha256\": " * repr(digest(archive_path)),
        "  \"archived_files\": $(length(expected))",
        "  \"unpacked_bytes\": $raw_bytes",
        "  \"archive_bytes\": $(stat(archive_path).size)",
        "  \"original_checkpoint_member\": " * repr("_checkpoint/done.toml"),
        "  \"restore\": " * repr("Extract contexts/ members into this geometry directory."),
        "  \"script_sha256\": " * repr(digest(@__FILE__)),
    ]
    "{\n" * join(fields, ",\n") * "\n}\n"
end

function archive_geometry(folder)
    summary = TOML.parsefile(joinpath(folder, "summary.toml"))
    if !get(summary, "screen", false)
        confirmation = joinpath(dirname(folder), "geometry_confirmation")
        isfile(joinpath(confirmation, "done.toml")) || return false
        verify_checkpoint(confirmation)
    end
    marker = joinpath(folder, "done.toml")
    original = read(marker)
    files = verify_checkpoint(folder)
    contexts = joinpath(folder, "contexts")
    archive_path = joinpath(folder, "contexts.tar.gz")
    record_path = joinpath(folder, "context_archive.json")
    if haskey(files, "contexts.tar.gz")
        archived_marker, expected = verify_archive(archive_path)
        haskey(files, "context_archive.json") ||
            throw(ArgumentError("archived checkpoint has no provenance record"))
        if isdir(contexts)
            verify_raw(folder, expected)
            rm(contexts; recursive=true)
        end
        return false
    end
    isdir(contexts) || throw(ArgumentError("raw contexts are missing from an unpacked checkpoint"))
    expected = context_files(files)
    verify_raw(folder, expected)
    if isfile(archive_path)
        archived_marker, archived_files = verify_archive(archive_path)
        archived_marker == original && archived_files == expected ||
            throw(ArgumentError("existing archive belongs to a different checkpoint"))
    else
        create_archive(folder, original, expected, archive_path)
    end
    raw_bytes = sum(stat(checked_path(folder, name)).size for name in keys(expected))
    write_atomic(record_path, archive_record(original, archive_path, expected, raw_bytes))
    remaining = Dict(name => hash for (name, hash) in files if !haskey(expected, name))
    remaining["contexts.tar.gz"] = digest(archive_path)
    remaining["context_archive.json"] = digest(record_path)
    new_marker = "[files]\n" * join((repr(name) * " = " * repr(remaining[name]) * "\n"
        for name in sort(collect(keys(remaining)))))
    write_atomic(marker, new_marker)
    verify_checkpoint(folder)
    verify_raw(folder, expected)
    rm(contexts; recursive=true)
    true
end

function archive_study(study)
    had_manifest = isfile(joinpath(study, "checksums.toml"))
    had_manifest && verify_study_manifest(study)
    count = 0
    try
        for category in ("anchors", "expansion", "representatives")
            parent = joinpath(study, category)
            isdir(parent) || continue
            for case_name in sort(readdir(parent))
                folder = joinpath(parent, case_name, "geometry")
                isfile(joinpath(folder, "done.toml")) || continue
                files = checkpoint_files(read(joinpath(folder, "done.toml")))
                haskey(files, "contexts.tar.gz") && !isdir(joinpath(folder, "contexts")) && continue
                count += archive_geometry(folder)
            end
        end
    finally
        had_manifest && refresh_study_manifest(study)
    end
    count
end

function main(args)
    length(args) == 1 || throw(ArgumentError("usage: julia --project=. scripts/archive_input_response_contexts.jl STUDY"))
    count = archive_study(args[1])
    println("{\"newly_archived_geometries\": $count}")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    InputResponseArchives.main(ARGS)
end

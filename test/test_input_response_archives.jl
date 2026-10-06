using Test
include(joinpath(@__DIR__, "..", "scripts", "archive_input_response_contexts.jl"))
const Archive = InputResponseArchives

function archive_fixture(root; screen=true)
    geometry = joinpath(root, "geometry")
    contexts = joinpath(geometry, "contexts", "0.0_0.0")
    mkpath(contexts)
    write(joinpath(geometry, "summary.toml"), "screen = $screen\n")
    write(joinpath(contexts, "input.toml"), "B_E = 0.0\nB_I = 0.0\n")
    write(joinpath(contexts, "context.toml"), "raw = \"preserve every byte\"\n")
    write_archive_marker(geometry)
    geometry
end

function write_archive_marker(geometry)
    files = Dict{String,String}()
    for (directory, _, names) in walkdir(geometry), name in names
        path = joinpath(directory, name)
        name == "done.toml" && continue
        files[relpath(path, geometry)] = Archive.digest(path)
    end
    marker = "[files]\n" * join((repr(name) * " = " * repr(files[name]) * "\n"
        for name in sort(collect(keys(files)))))
    write(joinpath(geometry, "done.toml"), marker)
end

function write_study_manifest(root)
    files = Dict{String,String}()
    for (directory, _, names) in walkdir(root), name in names
        path = joinpath(directory, name)
        relative = relpath(path, root)
        relative == "checksums.toml" && continue
        files[relative] = Archive.digest(path)
    end
    open(joinpath(root, "checksums.toml"), "w") do stream
        Archive.TOML.print(stream, Dict("schema_version" => 1, "algorithm" => "SHA-256",
            "files" => files); sorted=true)
    end
end

@testset "Lossless Julia geometry archive" begin
    mktempdir() do root
        geometry = archive_fixture(root)
        original = read(joinpath(geometry, "done.toml"))
        @test Archive.archive_geometry(geometry)
        @test !isdir(joinpath(geometry, "contexts"))
        marker, files = Archive.verify_archive(joinpath(geometry, "contexts.tar.gz"))
        @test marker == original
        @test length(files) == 2
        @test haskey(Archive.verify_checkpoint(geometry), "contexts.tar.gz")
        @test !Archive.archive_geometry(geometry)
    end
    mktempdir() do root
        geometry = archive_fixture(root)
        path = joinpath(geometry, "contexts", "0.0_0.0", "context.toml")
        write(path, "changed = true\n")
        @test_throws ArgumentError Archive.archive_geometry(geometry)
        @test isfile(path)
        @test !isfile(joinpath(geometry, "contexts.tar.gz"))
    end
    mktempdir() do root
        geometry = archive_fixture(root)
        extra = joinpath(geometry, "contexts", "0.0_0.0", "extra.txt")
        write(extra, "untracked data")
        @test_throws ArgumentError Archive.archive_geometry(geometry)
        @test isfile(extra)
    end
    mktempdir() do root
        geometry = archive_fixture(root)
        path = joinpath(geometry, "contexts", "0.0_0.0", "context.toml")
        target = joinpath(root, "outside.toml")
        cp(path, target)
        rm(path)
        symlink(target, path)
        @test_throws ArgumentError Archive.archive_geometry(geometry)
        @test isfile(target)
    end
    mktempdir() do root
        geometry = archive_fixture(root; screen=false)
        @test !Archive.archive_geometry(geometry)
        @test isdir(joinpath(geometry, "contexts"))
        confirmation = joinpath(root, "geometry_confirmation")
        mkpath(confirmation)
        write(joinpath(confirmation, "done.toml"), "[files]\n")
        @test Archive.archive_geometry(geometry)
    end
    mktempdir() do root
        geometry = archive_fixture(root)
        @test Archive.archive_geometry(geometry)
        open(joinpath(geometry, "contexts.tar.gz"), "a") do stream
            write(stream, "changed")
        end
        @test_throws ArgumentError Archive.archive_geometry(geometry)
    end
    mktempdir() do root
        geometry = archive_fixture(root)
        original = read(joinpath(geometry, "done.toml"))
        backup = joinpath(root, "backup")
        cp(joinpath(geometry, "contexts"), backup)
        @test Archive.archive_geometry(geometry)
        cp(backup, joinpath(geometry, "contexts"))
        write(joinpath(geometry, "done.toml"), original)
        @test Archive.archive_geometry(geometry)
        Archive.verify_checkpoint(geometry)
    end
    mktempdir() do root
        geometry = archive_fixture(root)
        backup = joinpath(root, "backup")
        cp(joinpath(geometry, "contexts"), backup)
        @test Archive.archive_geometry(geometry)
        cp(backup, joinpath(geometry, "contexts"))
        @test !Archive.archive_geometry(geometry)
        @test !isdir(joinpath(geometry, "contexts"))
        Archive.verify_checkpoint(geometry)
    end
    mktempdir() do root
        geometry = archive_fixture(joinpath(root, "expansion", "case"))
        write_study_manifest(root)
        previous = read(joinpath(root, "checksums.toml"))
        @test Archive.archive_study(root) == 1
        @test !isdir(joinpath(geometry, "contexts"))
        @test read(joinpath(root, "checksums.toml")) != previous
        @test isnothing(Archive.verify_study_manifest(root))
        files = Archive.TOML.parsefile(joinpath(root, "checksums.toml"))["files"]
        @test haskey(files, "expansion/case/geometry/contexts.tar.gz")
        @test haskey(files, "expansion/case/geometry/context_archive.json")
        @test !haskey(files, "expansion/case/geometry/contexts/0.0_0.0/context.toml")
        @test Archive.archive_study(root) == 0
        @test isnothing(Archive.verify_study_manifest(root))
    end
    mktempdir() do root
        geometry = archive_fixture(joinpath(root, "expansion", "case"))
        write_study_manifest(root)
        manifest = read(joinpath(root, "checksums.toml"))
        path = joinpath(geometry, "contexts", "0.0_0.0", "context.toml")
        write(path, "changed = true\n")
        @test_throws ArgumentError Archive.archive_study(root)
        @test read(joinpath(root, "checksums.toml")) == manifest
        @test isdir(joinpath(geometry, "contexts"))
        @test !isfile(joinpath(geometry, "contexts.tar.gz"))
    end
    mktempdir() do root
        first_geometry = archive_fixture(joinpath(root, "expansion", "a"))
        later_geometry = archive_fixture(joinpath(root, "expansion", "b"))
        write(joinpath(later_geometry, "summary.toml"), "screen = \"invalid\"\n")
        write_archive_marker(later_geometry)
        write_study_manifest(root)
        @test_throws MethodError Archive.archive_study(root)
        @test !isdir(joinpath(first_geometry, "contexts"))
        @test isdir(joinpath(later_geometry, "contexts"))
        @test isnothing(Archive.verify_study_manifest(root))
        # A retry reaches the same bad case, not a stale top-level manifest.
        @test_throws MethodError Archive.archive_study(root)
        @test isnothing(Archive.verify_study_manifest(root))
    end
end

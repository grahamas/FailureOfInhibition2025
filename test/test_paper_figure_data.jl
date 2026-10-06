import TOML
include(joinpath(@__DIR__, "..", "scripts", "build_paper_figure_data.jl"))

@testset "Paper figure response archive provenance" begin
    anchor = TOML.parsefile(joinpath(@__DIR__, "..", "reproducibility",
        "selective_anchor_joint_20261005", "reference_summary.toml"))["baseline_parameters"]
    mktempdir() do archive
        reference = TOML.parsefile(PaperFigureData.RESPONSE_REFERENCE)["files"]
        for relative in keys(reference)
            path = joinpath(archive, relative)
            mkpath(dirname(path))
            write(path, "unrelated archive\n")
        end
        digest = PaperFigureData.hashfile(joinpath(archive, "metadata.toml"))
        open(joinpath(archive, "checksums.toml"), "w") do io
            TOML.print(io, Dict("algorithm" => "SHA-256",
                "files" => Dict(relative => digest for relative in keys(reference))))
        end
        baseline = joinpath(archive, "anchors", "selective_withdrawal",
            "responses", "baseline_1")
        @test_throws ErrorException PaperFigureData.check_response_archive(baseline, anchor)
    end
end

@testset "Paper figure numerical environment" begin
    root = joinpath(@__DIR__, "..")
    joint = TOML.parsefile(joinpath(root, "reproducibility",
        "selective_anchor_joint_20261005", "reference_summary.toml"))
    tonic = TOML.parsefile(joinpath(root, "reproducibility",
        "selective_tonic_e_release_20261005", "reference_summary.toml"))
    environment = TOML.parsefile(PaperFigureData.SOURCE_ARTIFACTS)["environment"]
    @test PaperFigureData.check_numerical_environment(joint, tonic, environment)
    changed = deepcopy(environment)
    changed["julia_version"] = "1.10.0"
    @test_throws ErrorException PaperFigureData.check_numerical_environment(
        joint, tonic, changed)
    changed = deepcopy(environment)
    changed["source_sha256"]["Manifest.toml"] = repeat("0", 64)
    @test_throws ErrorException PaperFigureData.check_numerical_environment(
        joint, tonic, changed)
    @test PaperFigureData.portable_path(raw"traces\tonic_held.csv") ==
        "traces/tonic_held.csv"
end

@testset "Paper figure source artifacts" begin
    mktempdir() do directory
        write(joinpath(directory, "first.csv"), "initial,final\n0,1\n")
        write(joinpath(directory, "second.toml"), "destination = \"active\"\n")
        files = ["first.csv", "second.toml"]
        reference = Dict("files" => 2,
            "sha256" => PaperFigureData.artifact_digest(directory, files))
        @test PaperFigureData.check_artifacts(directory, files, reference, "fixture")
        @test_throws ErrorException PaperFigureData.artifact_digest(
            directory, [raw"sub\first.csv", "sub/first.csv"])
        write(joinpath(directory, "second.toml"), "destination = \"seizure\"\n")
        @test_throws ErrorException PaperFigureData.check_artifacts(
            directory, files, reference, "fixture")
    end
end

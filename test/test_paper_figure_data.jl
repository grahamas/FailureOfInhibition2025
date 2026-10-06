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

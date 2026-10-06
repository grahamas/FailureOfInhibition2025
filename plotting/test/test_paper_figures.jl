using Test
import CSV
import TOML
include(joinpath(@__DIR__, "..", "..", "scripts", "render_paper_figures.jl"))
const PaperFigures = PaperFigureRenderer
const PAPER_BUNDLE = joinpath(@__DIR__, "..", "..", "reproducibility",
    "paper_figures_20261005", "data")

if VERSION == v"1.10.12"
@testset "Selective paper figure evidence" begin
    data, samples = PaperFigures.checked_data(PAPER_BUNDLE)
    @test data["julia_version"] == string(VERSION) == "1.10.12"
    @test all(haskey(data["source_sha256"], path) for path in
        ("Project.toml", "Manifest.toml", "plotting/Project.toml",
         "plotting/Manifest.toml", "src/responses.jl", "src/drives.jl",
         "src/stability.jl", "src/diagnostics.jl",
         "scripts/run_input_release_study.jl",
         "scripts/render_paper_figures.jl"))
    @test PaperFigures.portable_path(raw"traces\tonic_held.csv") ==
        "traces/tonic_held.csv"
    @test haskey(data["reference_sha256"], "response")
    @test haskey(data["reference_sha256"], "source_artifacts")
    @test length(samples) == 2058
    @test length(data["baseline_roots"]) == 7
    @test length(data["threshold_8p75_roots"]) == 5
    @test Set(row["role"] for row in data["baseline_roots"] if
        row["stability"] == "Attracting") == Set(("rest", "active", "herald", "seizure"))
    @test Set(row["role"] for row in data["threshold_8p75_roots"] if
        row["stability"] == "Attracting") == Set(("rest", "active", "herald"))

    for source in ("rest", "active")
        row = only(filter(row -> row.theta_off == 8.0 &&
            row.B_E_on == 1.2328125 && row.source == source, samples))
        @test row.induction_destination == "herald"
        @test row.control_destination == "herald"
        @test row.release_0p35_destination == "herald"
        @test row.release_0p17_destination == "active"
        @test row.release_0p0_destination == "active"
    end
    seizure = data["selected_tonic"]["seizure"]
    @test seizure["control"]["destination"] == "seizure"
    @test all(seizure["release"][key]["destination"] == "seizure" for key in
        ("B_0p35", "B_0p17", "B_0p0"))

    joint = TOML.parsefile(joinpath(@__DIR__, "..", "..", "reproducibility",
        "selective_anchor_joint_20261005", "reference_summary.toml"))
    for trial in joint["induction_trials"]
        name = "induce_$(trial["source"])_to_$(trial["destination"]).csv"
        endpoint = last(CSV.File(joinpath(PAPER_BUNDLE, "traces", name)))
        @test maximum(abs.([endpoint.E, endpoint.I] .- trial["final"])) < 1e-7
    end
    changed = only(filter(row -> row["theta_off"] == 8.75,
        joint["threshold_interventions"]))
    endpoint = last(CSV.File(joinpath(PAPER_BUNDLE, "traces",
        "theta_off_8.75_from_seizure.csv")))
    @test maximum(abs.([endpoint.E, endpoint.I] .-
        changed["seizure_source_final"])) < 1e-7

    reference = TOML.parsefile(joinpath(@__DIR__, "..", "..", "reproducibility",
        "selective_tonic_e_release_20261005", "reference_summary.toml"))
    for threshold in reference["thresholds"], entry in threshold["thresholds"]
        suffix = Dict(0.35 => "0p35", 0.17 => "0p17", 0.0 => "0p0")[entry["return_B_E"]]
        relevant = filter(row -> row.theta_off == threshold["theta_off"] &&
            row.source == entry["source"], samples)
        qualifies(row) = row.induction_status == "compatible" &&
            row.induction_destination == "herald" &&
            row.control_status == "compatible" &&
            row.control_destination == "herald" &&
            getproperty(row, Symbol("release_$(suffix)_status")) == "compatible" &&
            getproperty(row, Symbol("release_$(suffix)_destination")) == "active"
        @test count(qualifies, relevant) == entry["qualified_samples"]
        if !isempty(entry["brackets"])
            first_success = minimum(row.B_E_on for row in relevant if qualifies(row))
            @test first_success == first(entry["brackets"])["first_success"]
        end
    end

    mktempdir() do temporary
        destination = joinpath(temporary, "tampered")
        cp(PAPER_BUNDLE, destination)
        open(joinpath(destination, "tonic_samples.csv"), "a") do io
            println(io, "tampered")
        end
        @test_throws ErrorException PaperFigures.checked_data(destination)
    end
    mktempdir() do temporary
        destination = joinpath(temporary, "missing checksum")
        cp(PAPER_BUNDLE, destination)
        checksum_path = joinpath(destination, "checksums.toml")
        manifest = TOML.parsefile(checksum_path)
        delete!(manifest["files"], "traces/induce_rest_to_herald.csv")
        open(checksum_path, "w") do io
            TOML.print(io, manifest)
        end
        open(joinpath(destination, "traces", "induce_rest_to_herald.csv"), "a") do io
            println(io, "tampered")
        end
        @test_throws ErrorException PaperFigures.checked_data(destination)
    end
    if !Sys.iswindows()
        mktempdir() do temporary
            destination = joinpath(temporary, "duplicate path")
            cp(PAPER_BUNDLE, destination)
            write(joinpath(destination, raw"traces\induce_rest_to_herald.csv"), "extra")
            @test_throws ErrorException PaperFigures.checked_data(destination)
        end
    end
    mktempdir() do temporary
        destination = joinpath(temporary, "missing source")
        cp(PAPER_BUNDLE, destination)
        data_path = joinpath(destination, "data.toml")
        record = TOML.parsefile(data_path)
        delete!(record["source_sha256"], "src/diagnostics.jl")
        open(data_path, "w") do io
            TOML.print(io, record)
        end
        checksum_path = joinpath(destination, "checksums.toml")
        manifest = TOML.parsefile(checksum_path)
        manifest["files"]["data.toml"] = PaperFigures.hashfile(data_path)
        open(checksum_path, "w") do io
            TOML.print(io, manifest)
        end
        @test_throws ErrorException PaperFigures.checked_data(destination)
    end
    @test_throws ErrorException PaperFigures.render(PAPER_BUNDLE, PAPER_BUNDLE)
end
else
@testset "Selective paper figure runtime gate" begin
    @test_throws ErrorException PaperFigures.checked_data(PAPER_BUNDLE)
end
end

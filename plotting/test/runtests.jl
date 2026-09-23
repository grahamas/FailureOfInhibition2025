import CSV
using FailureOfInhibition2025
import SHA
using Test
import TOML

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "scripts", "plot_manuscript_evidence.jl"))
using .ManuscriptEvidenceFigures

struct ShiftedVector{T} <: AbstractVector{T}
    values::Vector{T}
    offset::Int
end

Base.size(vector::ShiftedVector) = size(vector.values)
Base.axes(vector::ShiftedVector) = (firstindex(vector.values) + vector.offset:
    lastindex(vector.values) + vector.offset,)
Base.getindex(vector::ShiftedVector, index::Int) = vector.values[index - vector.offset]

function fixture_equilibria()
    control = [
        ("Attracting", 0.00058037894167215, 2.1779885959755705e-9),
        ("Saddle", 0.05563596231976791, 4.0694307246910245e-7),
        ("Repelling", 0.3110145895276539, 0.42513816192176007),
        ("Saddle", 0.3508374078843997, 0.4924231309608498),
        ("Attracting", 0.4999990682649182, 0.49999999372194726),
    ]
    failure = [
        ("Attracting", 0.0005803789416721498, 2.177988591486394e-9),
        ("Saddle", 0.05563596231976733, 4.069430716303147e-7),
        ("Repelling", 0.3110145904135569, 0.4251381634276635),
        ("Saddle", 0.3508373937158447, 0.49242310720629645),
        ("Attracting", 0.4999999113740313, 0.447721024060851),
        ("Saddle", 0.49999993913883795, 0.43936930466260365),
        ("Attracting", 0.49999999999999983, 0.0005586739686350245),
    ]
    rows = NamedTuple[]
    for (condition, values) in (("control", control), ("failure_of_inhibition", failure))
        for (stability, E, I) in values
            push!(rows, (; anchor="figure3", e_to_i=19.0, failure_threshold=8.0,
                condition, stability, E, I))
        end
    end
    return rows
end

function fixture_map()
    rows = NamedTuple[]
    for condition in ("control", "failure_of_inhibition"),
        e_to_i in (18.0, 19.0), failure_threshold in (8.0, 9.0)
        attracting_equilibria = condition == "failure_of_inhibition" &&
            e_to_i == 19.0 && failure_threshold == 8.0 ? 3 : 2
        push!(rows, (; anchor="figure3", condition, e_to_i, failure_threshold,
            status="completed", attracting_equilibria))
    end
    return rows
end

function write_fixture(directory)
    mkpath(directory)
    cp(joinpath(ROOT, "experiments", "coexistence.toml"),
        joinpath(directory, "config.toml"))
    CSV.write(joinpath(directory, "equilibria.csv"), fixture_equilibria())
    CSV.write(joinpath(directory, "map.csv"), fixture_map())
    files = Dict(name => ManuscriptEvidenceFigures.sha256_file(joinpath(directory, name))
        for name in ("config.toml", "equilibria.csv", "map.csv"))
    ManuscriptEvidenceFigures.write_toml(joinpath(directory, "checksums.toml"),
        Dict("algorithm" => "SHA-256", "schema_version" => 1, "files" => files))
    return directory
end

function write_report_fixture(path)
    names = ("response_curve", "figure3_phase_portraits", "figure3_coexistence")
    open(path, "w") do stream
        for name in names
            println(stream,
                "<img data-figure=\"$name\" src=\"data:image/png;base64,AA==\" " *
                "data-sha256=\"$(repeat("0", 64))\" alt=\"claim evidence figure\">")
        end
    end
    return path
end

@testset "Manuscript evidence figures" begin
    @testset "Julia-only active source" begin
        active_python = String[]
        for (directory, subdirectories, files) in walkdir(ROOT)
            filter!(name -> name ∉ (".git", ".jj", "output"), subdirectories)
            append!(active_python,
                joinpath(directory, file) for file in files if endswith(file, ".py"))
        end
        @test isempty(active_python)
    end

    @testset "model-backed values" begin
        figure_models = ManuscriptEvidenceFigures.load_figure_models(
            joinpath(ROOT, "experiments", "coexistence.toml"))
        response_model = figure_models.models.failure_of_inhibition.inhibitory.response
        midpoint = (response_model.onset_threshold + response_model.failure_threshold) / 2
        @test response(response_model, midpoint - 0.75) ≈ response(response_model, midpoint + 0.75)
        @test response_derivative(response_model, midpoint - 0.75) > 0
        @test response_derivative(response_model, midpoint + 0.75) < 0

        e_values = [-1.0e8, 1.0e8]
        i_values = [0.0]
        for condition in ManuscriptEvidenceFigures.CONDITIONS
            model = getproperty(figure_models.models, Symbol(condition.key))
            balances = ManuscriptEvidenceFigures.balance_grid(model, e_values, i_values)
            @test all(isfinite, first(balances))
            @test all(isfinite, last(balances))

            shifted_e = ShiftedVector([-1.0e8, 1.0e8], -2)
            shifted_i = ShiftedVector([0.0], 3)
            shifted_balances = ManuscriptEvidenceFigures.balance_grid(
                model, shifted_e, shifted_i)
            @test size.(shifted_balances) == ((2, 1), (2, 1))
            @test all(isfinite, first(shifted_balances))
            @test all(isfinite, last(shifted_balances))
        end
    end

    @testset "renderer and provenance" begin
        mktempdir() do temporary
            source = write_fixture(joinpath(temporary, "source"))
            output = joinpath(temporary, "figures")
            mkpath(output)
            report = joinpath(temporary, "report.html")
            write_report_fixture(report)
            package = ManuscriptEvidenceFigures.render_figures(source, output; report)
            @test length(package.figures) == 3
            @test Set(readdir(output)) == Set([
                "response_curve.png", "response_curve.toml",
                "figure3_phase_portraits.png", "figure3_phase_portraits.toml",
                "figure3_coexistence.png", "figure3_coexistence.toml", "manifest.toml",
            ])
            manifest = TOML.parsefile(joinpath(output, "manifest.toml"))
            @test getindex.(manifest["figures"], "figure") == [
                "response_curve", "figure3_phase_portraits", "figure3_coexistence",
            ]
            for item in manifest["figures"]
                provenance = TOML.parsefile(joinpath(output, item["provenance"]))
                @test provenance["output"]["sha256"] ==
                    ManuscriptEvidenceFigures.sha256_file(joinpath(output, item["png"]))
                provenance_sources = Dict(
                    source["path"] => source["sha256"] for source in provenance["sources"])
                for model_source in ManuscriptEvidenceFigures.MODEL_SOURCE_PATHS
                    @test provenance_sources[model_source] ==
                        ManuscriptEvidenceFigures.sha256_file(model_source)
                end
            end
            embedded = read(report, String)
            @test length(collect(eachmatch(r"data-figure=", embedded))) == 3
            @test occursin("data:image/png;base64,", embedded)

            duplicate_report = joinpath(temporary, "duplicate-report.html")
            write_report_fixture(duplicate_report)
            open(duplicate_report, "a") do stream
                println(stream,
                    "<img data-figure=\"response_curve\" " *
                    "src=\"data:image/png;base64,AA==\" " *
                    "data-sha256=\"$(repeat("0", 64))\" alt=\"duplicate\">")
            end
            @test_throws ArgumentError ManuscriptEvidenceFigures.embed_figures!(
                duplicate_report, output)

            control_only_map = joinpath(temporary, "control-only-tristability.csv")
            control_only_rows = [
                merge(row, (; attracting_equilibria=
                    row.condition == "control" ? 3 : 2)) for row in fixture_map()
            ]
            CSV.write(control_only_map, control_only_rows)
            control_only_package = ManuscriptEvidenceFigures.FigurePackage(
                joinpath(temporary, "control-only-figures"))
            @test_throws ArgumentError ManuscriptEvidenceFigures.coexistence_figure(
                control_only_package,
                joinpath(source, "config.toml"),
                control_only_map,
            )

            @test_throws ArgumentError ManuscriptEvidenceFigures.render_figures(source, output)

            open(joinpath(source, "map.csv"), "a") do stream
                print(stream, "changed")
            end
            @test_throws ArgumentError ManuscriptEvidenceFigures.verify_inputs(
                source, ("map.csv",))
        end
    end

end

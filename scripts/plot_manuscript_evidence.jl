"""Render and optionally embed the three claim-changing manuscript figures."""
module ManuscriptEvidenceFigures

using Base64
using CairoMakie
import CSV
using FailureOfInhibition2025
import SHA
import TOML

const FIGURE_ANCHOR = "figure3"
const FIGURE_E_TO_I = 19.0
const FIGURE_FAILURE_THRESHOLD = 8.0
const RENDERER_PATH = abspath(@__FILE__)
const MODEL_SOURCE_PATHS = sort!(filter(
    path -> endswith(path, ".jl"),
    readdir(dirname(pathof(FailureOfInhibition2025)); join=true),
))
const CONDITIONS = (
    (key="control", label="Monotone control"),
    (key="failure_of_inhibition", label="Failure of inhibition"),
)
const BLUE = RGBf(33 / 255, 102 / 255, 172 / 255)
const RED = RGBf(178 / 255, 24 / 255, 43 / 255)
const PURPLE = RGBf(124 / 255, 58 / 255, 237 / 255)
const INK = RGBf(17 / 255, 24 / 255, 39 / 255)
const MUTED = RGBf(71 / 255, 85 / 255, 105 / 255)

function required(table, key::AbstractString, context::AbstractString)
    haskey(table, key) || throw(ArgumentError("$context is missing $key"))
    return table[key]
end

function finite_float(value, context::AbstractString)
    value isa Real || throw(ArgumentError("$context must be numeric"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$context must be finite"))
    return result
end

function load_figure_models(path::AbstractString)
    raw = TOML.parsefile(path)
    model = required(raw, "model", "configuration")
    excitatory_table = required(model, "excitatory", "model")
    inhibitory_table = required(model, "inhibitory", "model")
    population(table, context) = PopulationParameters(
        timescale=finite_float(required(table, "timescale", context), "$context.timescale"),
        response=LogisticResponse(
            slope=finite_float(required(table, "slope", context), "$context.slope"),
            threshold=finite_float(required(table, "threshold", context), "$context.threshold"),
        ),
    )
    excitatory = population(excitatory_table, "model.excitatory")
    inhibitory = population(inhibitory_table, "model.inhibitory")
    anchors = required(raw, "anchors", "configuration")
    index = findfirst(anchor -> get(anchor, "name", nothing) == FIGURE_ANCHOR, anchors)
    isnothing(index) && throw(ArgumentError("configuration has no $FIGURE_ANCHOR anchor"))
    anchor = anchors[index]
    coupling = PointCoupling(
        e_to_e=finite_float(required(anchor, "e_to_e", FIGURE_ANCHOR),
            "$FIGURE_ANCHOR.e_to_e"),
        i_to_e=finite_float(required(anchor, "i_to_e", FIGURE_ANCHOR),
            "$FIGURE_ANCHOR.i_to_e"),
        e_to_i=FIGURE_E_TO_I,
        i_to_i=finite_float(required(anchor, "i_to_i", FIGURE_ANCHOR),
            "$FIGURE_ANCHOR.i_to_i"),
    )
    models = matched_point_models(
        excitatory=excitatory,
        inhibitory_control=inhibitory,
        failure_threshold=FIGURE_FAILURE_THRESHOLD,
        coupling=coupling,
    )
    return (; raw, models)
end

sha256_file(path::AbstractString) = open(path, "r") do stream
    bytes2hex(SHA.sha256(stream))
end

function write_toml(path::AbstractString, value)
    open(path, "w") do stream
        TOML.print(stream, value; sorted=true)
        println(stream)
    end
    return path
end

function verify_inputs(directory::AbstractString, required)
    manifest_path = joinpath(directory, "checksums.toml")
    isfile(manifest_path) || throw(ArgumentError("missing checksum manifest: $manifest_path"))
    manifest = TOML.parsefile(manifest_path)
    haskey(manifest, "files") || throw(ArgumentError("checksum manifest has no files table"))
    checksums = manifest["files"]
    paths = Dict{String,String}()
    for relative in required
        haskey(checksums, relative) ||
            throw(ArgumentError("consumed input is not checksummed: $relative"))
        path = joinpath(directory, relative)
        isfile(path) || throw(ArgumentError("consumed input is missing: $relative"))
        sha256_file(path) == checksums[relative] ||
            throw(ArgumentError("consumed input has changed: $relative"))
        paths[relative] = path
    end
    return paths
end

mutable struct FigurePackage
    output::String
    figures::Vector{Dict{String,Any}}
    environment::Dict{String,Any}
end

function FigurePackage(output::AbstractString)
    if ispath(output)
        isdir(output) || throw(ArgumentError("output is not a directory: $output"))
        isempty(readdir(output)) || throw(ArgumentError("output is not empty: $output"))
    else
        mkpath(output)
    end
    environment = Dict{String,Any}(
        "julia" => string(VERSION),
        "cairo_makie" => string(pkgversion(CairoMakie)),
        "failure_of_inhibition_2025" => string(pkgversion(FailureOfInhibition2025)),
    )
    return FigurePackage(abspath(output), Dict{String,Any}[], environment)
end

function save_figure!(package::FigurePackage, figure, name::AbstractString, sources;
    claim::AbstractString, limitation::AbstractString)
    target = joinpath(package.output, "$name.png")
    save(target, figure; px_per_unit=2)
    figure_sources = collect(sources)
    push!(figure_sources, RENDERER_PATH)
    append!(figure_sources, MODEL_SOURCE_PATHS)
    sort!(unique!(figure_sources))
    provenance = Dict{String,Any}(
        "figure" => name,
        "claim" => claim,
        "limitation" => limitation,
        "environment" => package.environment,
        "sources" => [
            Dict("path" => abspath(path), "sha256" => sha256_file(path))
            for path in figure_sources
        ],
        "output" => Dict("path" => basename(target), "sha256" => sha256_file(target)),
    )
    provenance_path = joinpath(package.output, "$name.toml")
    write_toml(provenance_path, provenance)
    push!(package.figures, Dict{String,Any}(
        "figure" => name,
        "png" => basename(target),
        "provenance" => basename(provenance_path),
        "claim" => claim,
        "limitation" => limitation,
    ))
    return target
end

function finish!(package::FigurePackage)
    manifest = Dict{String,Any}(
        "figures" => package.figures,
        "environment" => package.environment,
        "interpretation" =>
            "Claim-specific numerical evidence; no completeness, biological-state, or clinical inference.",
    )
    write_toml(joinpath(package.output, "manifest.toml"), manifest)
    return package
end

function response_figure(package::FigurePackage, figure_models, config_path)
    model = figure_models.models.failure_of_inhibition
    inhibitory_response = model.inhibitory.response
    onset = inhibitory_response.onset_threshold
    failure = inhibitory_response.failure_threshold
    midpoint = (onset + failure) / 2
    inputs = range(onset - 3, failure + 3; length=500)
    values = response.(Ref(inhibitory_response), inputs)
    mirror_offset = (failure - onset) / 4
    mirror_inputs = [midpoint - mirror_offset, midpoint + mirror_offset]
    mirror_values = response.(Ref(inhibitory_response), mirror_inputs)

    figure = Figure(size=(920, 480), backgroundcolor=:white)
    axis = Axis(figure[1, 1];
        xlabel="Effective inhibitory input",
        ylabel="Response",
        title="Implemented equal-slope failure response",
        xgridcolor=(:gray, 0.18),
        ygridcolor=(:gray, 0.18),
    )
    lines!(axis, inputs, values; color=PURPLE, linewidth=3)
    vlines!(axis, [midpoint]; color=MUTED, linestyle=:dash, linewidth=1.5)
    scatter!(axis, mirror_inputs, mirror_values; color=INK, markersize=11)
    xlims!(axis, first(inputs), last(inputs))
    ylims!(axis, 0, 1.02)
    Label(figure[2, 1], "Midpoint = $(midpoint); paired markers have equal response",
        color=MUTED, fontsize=15, tellwidth=false)
    return save_figure!(package, figure, "response_curve", [config_path];
        claim="The implemented equal-slope response rises and then falls symmetrically.",
        limitation="This does not validate independently adjustable slopes or historical fits.")
end

function balance_grid(model, e_values, i_values)
    e_balance = Matrix{Float64}(undef, length(e_values), length(i_values))
    i_balance = similar(e_balance)
    residual = zeros(2)
    state = zeros(2)
    for (e_index, excitatory) in enumerate(e_values),
        (i_index, inhibitory) in enumerate(i_values)
        state[1] = excitatory
        state[2] = inhibitory
        point_balance!(residual, state, model, 0.0)
        e_balance[e_index, i_index] = residual[1]
        i_balance[e_index, i_index] = residual[2]
    end
    return e_balance, i_balance
end

function selected_equilibria(path::AbstractString)
    rows = collect(CSV.File(path))
    selected = filter(rows) do row
        string(row.anchor) == FIGURE_ANCHOR &&
            isapprox(Float64(row.e_to_i), FIGURE_E_TO_I; atol=1e-12, rtol=0) &&
            isapprox(Float64(row.failure_threshold), FIGURE_FAILURE_THRESHOLD;
                atol=1e-12, rtol=0)
    end
    by_condition = Dict(condition.key => filter(row -> string(row.condition) == condition.key,
        selected) for condition in CONDITIONS)
    length(by_condition["control"]) == 5 ||
        throw(ArgumentError("Figure 3 evidence must contain five control equilibria"))
    length(by_condition["failure_of_inhibition"]) == 7 ||
        throw(ArgumentError("Figure 3 evidence must contain seven FoI equilibria"))
    any(by_condition["failure_of_inhibition"]) do row
        string(row.stability) == "Attracting" && Float64(row.E) > 0.49 && Float64(row.I) < 0.01
    end || throw(ArgumentError("Figure 3 evidence lacks the attracting high-E/low-I root"))
    return by_condition
end

function phase_portrait_figure(package::FigurePackage, figure_models, config_path,
    equilibrium_path)
    models = figure_models.models
    evidence = selected_equilibria(equilibrium_path)
    e_values = range(0.0, 0.5; length=201)
    i_values = range(0.0, 0.5; length=201)

    figure = Figure(size=(1260, 610), backgroundcolor=:white)
    Label(figure[0, 1:2],
        "Figure 3 anchor · E-to-I = 19 · failure threshold = 8\n" *
        "blue: E nullcline · red: I nullcline · circle: attracting · cross: other",
        fontsize=19, tellwidth=false)
    for (column, condition) in enumerate(CONDITIONS)
        axis = Axis(figure[1, column];
            xlabel="E active fraction",
            ylabel="I active fraction",
            title=condition.label,
            aspect=DataAspect(),
            xgridcolor=(:gray, 0.12),
            ygridcolor=(:gray, 0.12),
        )
        model = getproperty(models, Symbol(condition.key))
        e_balance, i_balance = balance_grid(model, e_values, i_values)
        contour!(axis, e_values, i_values, e_balance;
            levels=[0.0], color=BLUE, linewidth=2.5)
        contour!(axis, e_values, i_values, i_balance;
            levels=[0.0], color=RED, linewidth=2.5)
        rows = evidence[condition.key]
        attracting = filter(row -> string(row.stability) == "Attracting", rows)
        other = filter(row -> string(row.stability) != "Attracting", rows)
        scatter!(axis, Float64.(getproperty.(attracting, :E)),
            Float64.(getproperty.(attracting, :I)); color=INK, markersize=13)
        scatter!(axis, Float64.(getproperty.(other, :E)), Float64.(getproperty.(other, :I));
            color=INK, marker=:x, markersize=15)
        limits!(axis, -0.01, 0.51, -0.01, 0.51)
    end
    return save_figure!(package, figure, "figure3_phase_portraits",
        [config_path, equilibrium_path];
        claim="At the supplied anchor, the FoI model has a discovered attracting high-E/low-I root.",
        limitation="The roots and local spectra are parameter-specific and do not certify completeness.")
end

function coexistence_matrix(rows, condition::AbstractString)
    selected = filter(row -> string(row.anchor) == FIGURE_ANCHOR &&
        string(row.condition) == condition, rows)
    isempty(selected) && throw(ArgumentError("coexistence map has no $condition rows"))
    all(row -> string(row.status) == "completed", selected) ||
        throw(ArgumentError("coexistence map contains incomplete $condition rows"))
    e_values = sort!(unique!(Float64.(getproperty.(selected, :e_to_i))))
    thresholds = sort!(unique!(Float64.(getproperty.(selected, :failure_threshold))))
    length(selected) == length(e_values) * length(thresholds) ||
        throw(ArgumentError("coexistence map is not rectangular for $condition"))
    counts = Matrix{Float64}(undef, length(e_values), length(thresholds))
    seen = Set{Tuple{Float64,Float64}}()
    for row in selected
        coordinate = (Float64(row.e_to_i), Float64(row.failure_threshold))
        coordinate in seen && throw(ArgumentError("duplicate map coordinate: $coordinate"))
        push!(seen, coordinate)
        e_index = findfirst(==(coordinate[1]), e_values)
        threshold_index = findfirst(==(coordinate[2]), thresholds)
        counts[e_index, threshold_index] = Float64(row.attracting_equilibria)
    end
    return e_values, thresholds, counts
end

function coexistence_figure(package::FigurePackage, config_path, map_path)
    rows = collect(CSV.File(map_path))
    completed_counts = Int[
        row.attracting_equilibria for row in rows
        if string(row.anchor) == FIGURE_ANCHOR &&
            string(row.condition) == "failure_of_inhibition" &&
            string(row.status) == "completed"
    ]
    maximum(completed_counts) == 3 ||
        throw(ArgumentError("claim figure requires sampled maximum of three attracting roots"))
    4 in completed_counts &&
        throw(ArgumentError("claim figure cannot state that no tetrastability was sampled"))

    figure = Figure(size=(1260, 570), backgroundcolor=:white)
    Label(figure[0, 1:2],
        "Figure 3 sampled coexistence plane\nsampled maximum = 3; no sampled cell = 4",
        fontsize=19, tellwidth=false)
    heatmap_plot = nothing
    for (column, condition) in enumerate(CONDITIONS)
        e_values, thresholds, counts = coexistence_matrix(rows, condition.key)
        axis = Axis(figure[1, column];
            xlabel="E-to-I coupling",
            ylabel="Failure threshold",
            title=condition.label,
        )
        current = heatmap!(axis, e_values, thresholds, counts;
            colormap=:viridis, colorrange=(0, 4))
        heatmap_plot = isnothing(heatmap_plot) ? current : heatmap_plot
    end
    Colorbar(figure[1, 3], heatmap_plot;
        ticks=0:4, label="Discovered locally attracting equilibria")
    return save_figure!(package, figure, "figure3_coexistence", [config_path, map_path];
        claim="The sampled FoI plane contains cells with three discovered attracting roots.",
        limitation="No sampled cell has four; the finite search is not an exhaustive attractor count.")
end

function embed_figures!(report_path::AbstractString, figure_directory::AbstractString)
    html = read(report_path, String)
    for name in ("response_curve", "figure3_phase_portraits", "figure3_coexistence")
        image_path = joinpath(figure_directory, "$name.png")
        isfile(image_path) || throw(ArgumentError("missing rendered figure: $image_path"))
        encoded = base64encode(read(image_path))
        checksum = sha256_file(image_path)
        pattern = Regex(
            "(<img\\s+data-figure=\\\"$name\\\"\\s+src=\\\")" *
            "data:image/png;base64,[^\\\"]+" *
            "(\\\"\\s+data-sha256=\\\")[^\\\"]+(\\\")",
            "s",
        )
        matches = collect(eachmatch(pattern, html))
        length(matches) == 1 ||
            throw(ArgumentError("report must contain exactly one $name embedding slot"))
        matched = only(matches)
        replacement = string(matched.captures[1], "data:image/png;base64,", encoded,
            matched.captures[2], checksum, matched.captures[3])
        html = replace(html, matched.match => replacement; count=1)
    end
    open(report_path, "w") do stream
        write(stream, html)
    end
    return report_path
end

function render_figures(coexistence::AbstractString, output::AbstractString;
    report::Union{Nothing,AbstractString}=nothing)
    sources = verify_inputs(coexistence, ("config.toml", "equilibria.csv", "map.csv"))
    config_path = sources["config.toml"]
    equilibrium_path = sources["equilibria.csv"]
    map_path = sources["map.csv"]
    config = load_figure_models(config_path)
    package = FigurePackage(output)
    response_figure(package, config, config_path)
    phase_portrait_figure(package, config, config_path, equilibrium_path)
    coexistence_figure(package, config_path, map_path)
    finish!(package)
    isnothing(report) || embed_figures!(report, package.output)
    return package
end

function parse_arguments(args)
    coexistence = "output/coexistence_study"
    output = nothing
    report = nothing
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option in ("--coexistence", "--output", "--report") ||
            throw(ArgumentError("unknown option: $option"))
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        index < length(args) || throw(ArgumentError("$option requires a value"))
        value = args[index + 1]
        startswith(value, "--") && throw(ArgumentError("$option requires a value"))
        option == "--coexistence" && (coexistence = value)
        option == "--output" && (output = value)
        option == "--report" && (report = value)
        index += 2
    end
    isnothing(output) && throw(ArgumentError("--output DIRECTORY is required"))
    return (; coexistence, output, report)
end

function main(args=ARGS)
    arguments = parse_arguments(args)
    package = render_figures(arguments.coexistence, arguments.output;
        report=arguments.report)
    println("Wrote $(length(package.figures)) claim figures to $(package.output)")
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(ManuscriptEvidenceFigures.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end

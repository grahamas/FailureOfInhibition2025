"""Render the selective-anchor paper figures from a portable Julia evidence bundle."""
module PaperFigureRenderer

using CairoMakie, FailureOfInhibition2025
import CSV, TOML, SHA
include("narrative_models.jl")
const ROOT = normpath(joinpath(@__DIR__, ".."))
const E_COLOR = "#17658a"
const I_COLOR = "#bd6a25"
const NEUTRAL = "#657383"
const ROLES = Dict("rest"=>"#64748b", "active"=>"#17805e",
    "herald"=>"#8059b0", "seizure"=>"#c74655", "unassigned"=>"#8c98a5")
const EXTRA_SOURCE_FILES = Set((
    "plotting/Project.toml", "plotting/Manifest.toml",
    "reproducibility/selective_tonic_e_release_20261005/replay.jl",
    "scripts/build_paper_figure_data.jl", "scripts/render_paper_figures.jl"))
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
portable_path(path) = replace(path, '\\' => '/')

function expected_bundle_files()
    files = Set(("data.toml", "tonic_samples.csv", "herald_sustained.csv",
        "herald_withdrawal_pulses.csv", "seizure_sustained.csv",
        "seizure_withdrawal_pulses.csv"))
    for setting in ("baseline_switching", "theta_off_8.75_switching"),
            cycle in 1:2, phase in ("rest_to_active", "active_to_rest"),
            suffix in ("_pulse", "")
        push!(files, portable_path(joinpath("traces", setting, "cycle$(cycle)_$(phase)$(suffix).csv")))
    end
    for name in ("induce_rest_to_herald_pulse", "induce_rest_to_herald",
            "induce_active_to_seizure_pulse", "induce_active_to_seizure",
            "theta_off_8.75_from_seizure", "preon_baseline_rest",
            "preon_baseline_active", "preon_herald_on")
        push!(files, portable_path(joinpath("traces", name * ".csv")))
    end
    for name in ("tonic_active_induction", "tonic_rest_induction",
            "tonic_herald_held", "tonic_seizure_held")
        for extension in ("csv", "toml")
            push!(files, portable_path(joinpath("traces", "$name.$extension")))
        end
    end
    for state in ("herald", "seizure"), level in ("0p0", "0p17", "0p35"),
            extension in ("csv", "toml")
        push!(files, portable_path(joinpath("traces", "tonic_$(state)_release_B_$(level).$(extension)")))
    end
    return files
end

function checked_data(directory)
    manifest = TOML.parsefile(joinpath(directory, "checksums.toml"))["files"]
    expected = expected_bundle_files()
    Set(keys(manifest)) == expected || error("incomplete figure-data checksum manifest")
    actual_files = [portable_path(relpath(joinpath(dir, file), directory))
        for (dir, _, filenames) in walkdir(directory) for file in filenames]
    actual = Set(actual_files)
    length(actual) == length(actual_files) || error("duplicate figure-data path")
    actual == union(expected, Set(["checksums.toml"])) ||
        error("figure-data bundle file set differs")
    for (relative, digest) in manifest
        (isabspath(relative) || ".." in splitpath(relative)) &&
            error("unsafe figure-data path: $relative")
        path = joinpath(directory, relative)
        isfile(path) && hashfile(path) == digest || error("figure-data hash mismatch: $relative")
    end
    data = TOML.parsefile(joinpath(directory, "data.toml"))
    environment = TOML.parsefile(joinpath(ROOT, "reproducibility",
        "paper_figures_20261005", "source_artifacts.toml"))["environment"]
    data["julia_version"] == environment["julia_version"] == string(VERSION) ||
        error("figure-data Julia version differs")
    joint_sources = TOML.parsefile(joinpath(ROOT, "reproducibility",
        "selective_anchor_joint_20261005", "reference_summary.toml"))["source_sha256"]
    tonic_sources = TOML.parsefile(joinpath(ROOT, "reproducibility",
        "selective_tonic_e_release_20261005", "reference_summary.toml"))["metadata"]["source_sha256"]
    joint_sources == tonic_sources || error("joint/tonic source hashes differ")
    Set(keys(data["source_sha256"])) == union(Set(keys(joint_sources)), EXTRA_SOURCE_FILES) ||
        error("incomplete figure-data source list")
    for (relative, digest) in joint_sources
        data["source_sha256"][relative] == digest ||
            error("figure-data reviewed source differs: $relative")
    end
    for (relative, digest) in environment["source_sha256"]
        data["source_sha256"][relative] == digest ||
            error("figure-data numerical environment differs: $relative")
    end
    Set(keys(data["reference_sha256"])) ==
        Set(("joint", "tonic", "response", "source_artifacts")) ||
        error("incomplete figure-data reference list")
    for (relative, expected) in data["source_sha256"]
        hashfile(joinpath(ROOT, relative)) == expected || error("model/protocol source changed: $relative")
    end
    for (name, relative, filename) in
            (("joint", "selective_anchor_joint_20261005", "reference_summary.toml"),
             ("tonic", "selective_tonic_e_release_20261005", "reference_summary.toml"),
             ("response", "paper_figures_20261005", "response_reference.toml"),
             ("source_artifacts", "paper_figures_20261005", "source_artifacts.toml"))
        path = joinpath(ROOT, "reproducibility", relative, filename)
        hashfile(path) == data["reference_sha256"][name] ||
            error("figure-data reference changed: $name")
    end
    samples = collect(CSV.File(joinpath(directory, "tonic_samples.csv")))
    length(samples) == data["sample_rows"] == 2058 || error("incomplete tonic sample table")
    for theta in (8.0, 8.25, 8.5, 8.75, 9.0, 10.0, 12.0), source in ("rest", "active")
        count(row -> row.theta_off == theta && row.source == source, samples) == 147 ||
            error("incomplete tonic row: $theta $source")
    end
    return data, samples
end

trace(bundle, relative) = collect(CSV.File(joinpath(bundle, "traces", relative)))
function save_figure(fig, output, name)
    save(joinpath(output, name * ".pdf"), fig)
    save(joinpath(output, name * ".svg"), fig)
    save(joinpath(output, name * ".png"), fig; px_per_unit=1.5)
end
function trace_lines!(axis, rows; offset=0.0, limit=300.0, alpha=1.0, label=true)
    short = filter(row -> row.time <= limit, rows)
    for (field, color) in ((:E, E_COLOR), (:I, I_COLOR))
        lines!(axis, [row.time + offset for row in short],
            [getproperty(row, field) for row in short];
            color=(color, alpha), linewidth=2.8, label=label ? string(field) : nothing)
    end
end
function role_points!(axis, rows; labels=true, limits=true)
    for row in rows
        role = row["role"]
        stable = row["stability"] == "Attracting"
        scatter!(axis, [row["E"]], [row["I"]];
            marker=stable ? :circle : :xcross,
            color=ROLES[role], markersize=stable ? 16 : 10)
        if labels && role != "unassigned"
            offset = role == "seizure" ? (-58, 12) : role == "herald" ? (-58, -18) : (8, 9)
            text!(axis, row["E"], row["I"]; text=role, offset, fontsize=13,
                color=ROLES[role])
        end
    end
    if limits
        xlims!(axis, -0.015, 0.53)
        ylims!(axis, -0.015, 0.53)
    end
end
function nullclines!(axis, parameters)
    model = NarrativeModels.models(parameters).failure_of_inhibition
    grid = collect(range(0.0, 0.505; length=180))
    d_e = zeros(length(grid), length(grid))
    d_i = similar(d_e)
    balance = zeros(2)
    for (j, i) in enumerate(grid), (k, e) in enumerate(grid)
        point_rhs!(balance, [e, i], model, 0.0)
        d_e[k, j], d_i[k, j] = balance
    end
    contour!(axis, grid, grid, d_e; levels=[0.0], color=E_COLOR, linewidth=2)
    contour!(axis, grid, grid, d_i; levels=[0.0], color=I_COLOR,
        linewidth=2, linestyle=:dash)
end
function endpoint_series(bundle, setting)
    suffix = setting == "baseline" ? "baseline_switching" : "theta_off_8.75_switching"
    files = ["cycle$(cycle)_$(phase).csv" for cycle in 1:2
        for phase in ("rest_to_active", "active_to_rest")]
    [last(trace(bundle, joinpath(suffix, file))) for file in files]
end
function endpoint_plot!(axis, bundle, setting, rest)
    ends = endpoint_series(bundle, setting)
    states = vcat([(E=rest["E"], I=rest["I"])], ends)
    x = collect(0:4)
    scatter!(axis, x, [row.E for row in states]; color=E_COLOR, markersize=12, label="E")
    scatter!(axis, x, [row.I for row in states]; color=I_COLOR, markersize=12, label="I")
    axis.xticks = (x, ["start\nrest", "1: E +0.5\nactive", "2: I +3\nrest",
        "3: E +0.5\nactive", "4: I +3\nrest"])
    xlims!(axis, -0.2, 4.2)
    ylims!(axis, -0.02, 0.52)
    axislegend(axis; position=:rt, orientation=:horizontal)
end

function step_input!(axis, before, during, after; onset=0.0, offset=100.0,
        left=-50.0, right=300.0, color=E_COLOR, label=nothing)
    lines!(axis, [left, onset, onset, offset, offset, right],
        [before, before, during, during, after, after];
        color, linewidth=2.7, label)
    vlines!(axis, [onset, offset]; color=(NEUTRAL, 0.5), linestyle=:dot)
    xlims!(axis, left, right)
end

function preon_lines!(axis, bundle, name; field=:E, color=E_COLOR,
        linestyle=:solid, label=nothing)
    rows = trace(bundle, "preon_$(name).csv")
    lines!(axis, [row.time for row in rows],
        [getproperty(row, field) for row in rows];
        color, linestyle, linewidth=2.4, label)
end

function figure1(bundle, output, data)
    fig = Figure(size=(1600, 760), fontsize=18)
    Label(fig[1, 1:3], "Failure response permits a second high-activity ordering";
        fontsize=25, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1], title="A. Ordered inhibitory recruitment and failure",
        xlabel="Effective input to I", ylabel="Population response")
    pair = NarrativeModels.models(data["anchor"])
    input = collect(range(0.0, 12.0; length=600))
    lines!(ax, input, response.(Ref(pair.failure_of_inhibition.inhibitory.response), input);
        color=I_COLOR, linewidth=3, label="Fire-then-fail response")
    lines!(ax, input, response.(Ref(pair.control.inhibitory.response), input);
        color=NEUTRAL, linewidth=2.5, linestyle=:dash,
        label="Monotone control response")
    vlines!(ax, [4.0, 8.0]; color=(NEUTRAL, 0.5), linestyle=:dot)
    text!(ax, 4.0, 0.12; text="onset", rotation=pi/2, fontsize=12)
    text!(ax, 8.0, 0.12; text="failure", rotation=pi/2, fontsize=12)
    Legend(fig[3, 1], ax; orientation=:horizontal, framevisible=false,
        tellwidth=false)
    ax_full = Axis(fig[2, 2], title="B. Physical occupancy domain",
        xlabel="E occupancy", ylabel="I occupancy", aspect=DataAspect())
    nullclines!(ax_full, data["anchor"])
    role_points!(ax_full, data["baseline_roots"]; labels=false, limits=false)
    lines!(ax_full, [0.0, 0.53, 0.53, 0.0, 0.0],
        [0.0, 0.0, 0.53, 0.53, 0.0]; color=NEUTRAL, linestyle=:dot)
    xlims!(ax_full, 0, 1); ylims!(ax_full, 0, 1)
    ax_zoom = Axis(fig[2, 3],
        title=rich("C. Equilibrium zoom at B", subscript("E"), " = 0.35"),
        xlabel="E occupancy", ylabel="I occupancy", aspect=DataAspect())
    nullclines!(ax_zoom, data["anchor"])
    role_points!(ax_zoom, data["baseline_roots"])
    Label(fig[4, 1:3], "Dashed gray and orange responses overlap before failure. Nullclines and equilibria lie below 0.5; admissible transients can start above it.\nBlue/orange: E/I nullclines. Circles: attracting roots; crosses: other roots. Role names are provisional.";
        fontsize=17, tellwidth=false)
    save_figure(fig, output, "01_mechanism")
end

function figure2(bundle, output, data)
    fig = Figure(size=(1450, 1090), fontsize=18)
    Label(fig[1, 1:2], "Ordinary switching and separate access to high-activity states";
        fontsize=24, font=:bold, tellwidth=false)
    rest = only(filter(row -> row["role"] == "rest", data["baseline_roots"]))
    ax = Axis(fig[2, 1], title="A. Two reviewed switching cycles",
        xlabel="Pulse event and 5,000 ms destination", ylabel="Occupancy")
    endpoint_plot!(ax, bundle, "baseline", rest)
    ax = Axis(fig[2, 2], title="B. Reviewed pulse regimes (schematic height)",
        xlabel="Time relative to pulse onset (ms)", ylabel="Protocol")
    protocols = ((4, "rest → active; E +0.5", ROLES["active"]),
        (3, "active → rest; I +3", ROLES["rest"]),
        (2, "rest → herald; E +1", ROLES["herald"]),
        (1, "active → seizure; I +4.5", ROLES["seizure"]))
    for (row, label, color) in protocols
        lines!(ax, [-50, 0, 0, 100, 100, 300],
            [row, row, row + 0.35, row + 0.35, row, row]; color, linewidth=2.8)
        text!(ax, 120, row + 0.2; text=label, color, fontsize=15)
    end
    ax.yticks = (1:4, ["4", "3", "2", "1"])
    xlims!(ax, -50, 300); ylims!(ax, 0.7, 4.65)
    for (column, name, source, title, target, amplitude) in
            ((1, "induce_rest_to_herald", "rest", "C. Rest → herald", "E", 1.0),
             (2, "induce_active_to_seizure", "active", "D. Active → seizure", "I", 4.5))
        ax = Axis(fig[3, column]; title, ylabel="Occupancy")
        for (field, color) in ((:E, E_COLOR), (:I, I_COLOR))
            preon_lines!(ax, bundle, "baseline_$(source)"; field, color,
                linestyle=:dash)
        end
        pulse = trace(bundle, name * "_pulse.csv")
        trace_lines!(ax, pulse; limit=100.0)
        trace_lines!(ax, trace(bundle, name * ".csv"); offset=100.0, limit=200.0,
            label=false)
        vlines!(ax, [100.0]; color=NEUTRAL, linestyle=:dash)
        xlims!(ax, -50, 300)
        ylims!(ax, -0.02, 0.52)
        column == 1 && axislegend(ax; position=:rb, orientation=:horizontal)
        strip = Axis(fig[4, column], xlabel="Time relative to pulse onset (ms)",
            ylabel=rich("Δ", target, " input"))
        step_input!(strip, 0.0, amplitude, 0.0; color=target == "E" ? E_COLOR : I_COLOR)
        strip.yticks = ([0.0, amplitude], ["0", string(amplitude)])
        ylims!(strip, -0.12 * amplitude, 1.18 * amplitude)
    end
    rowsize!(fig.layout, 4, 115)
    Label(fig[5, 1:2], rich("Dashed pre-onset segments are simulated at baseline B", subscript("E"),
        " = 0.35; input returns to baseline after 100 ms. Traces show early dynamics.\n",
        "Regime colors mark destinations; classifications use the full 5,000 ms follow-up. High-state examples have different sources and input directions.");
        fontsize=17, tellwidth=false)
    save_figure(fig, output, "02_switching_and_access")
end

function outcome_grid!(axis, selected)
    axis.xticks = (1:4, ["held at 1.2328", "return 0.35", "return 0.17", "return 0"])
    axis.yticks = ([1, 2], ["separate seizure source", "induced from rest / active"])
    axis.xgridvisible = false
    axis.ygridvisible = false
    outcomes(source) = [selected[source]["control"]["destination"],
        (selected[source]["release"][key]["destination"] for key in
            ("B_0p35", "B_0p17", "B_0p0"))...]
    outcomes("rest") == outcomes("active") ||
        error("rest and active tonic outcomes differ; outcome grid needs separate rows")
    for (y, roles) in ((2, outcomes("rest")), (1, outcomes("seizure")))
        for (x, role) in enumerate(roles)
            text!(axis, x, y; text=role, align=(:center, :center),
                color=ROLES[role], fontsize=19, font=:bold)
        end
    end
    for x in 1.5:1.0:3.5
        vlines!(axis, [x]; color=(NEUTRAL, 0.3))
    end
    hlines!(axis, [1.5]; color=(NEUTRAL, 0.3))
    xlims!(axis, 0.45, 4.55)
    ylims!(axis, 0.45, 2.55)
end

function figure3(bundle, output, data)
    fig = Figure(size=(1450, 990), fontsize=18)
    Label(fig[1, 1:2], "Lower tonic E returns induced herald to active, while seizure persists";
        fontsize=23, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1],
        title=rich("A. Increase B", subscript("E"), " to 1.2328125"),
        ylabel="E occupancy")
    for (source, style) in (("rest", :solid), ("active", :dash))
        preon_lines!(ax, bundle, "baseline_$(source)"; linestyle=style,
            color=(E_COLOR, 0.65))
        rows = filter(row -> row.time <= 300,
            trace(bundle, "tonic_$(source)_induction.csv"))
        lines!(ax, [row.time for row in rows], [row.E for row in rows];
            color=E_COLOR, linestyle=style, linewidth=2.8, label="from $source")
    end
    vlines!(ax, [0.0]; color=(NEUTRAL, 0.5), linestyle=:dot)
    xlims!(ax, -50, 300); ylims!(ax, -0.02, 0.52)
    axislegend(ax; position=:rb)
    ax = Axis(fig[2, 2], title="B. Release from the actual herald endpoint",
        ylabel="E occupancy")
    preon_lines!(ax, bundle, "herald_on"; color=NEUTRAL,
        linestyle=:dash)
    for (suffix, label, color, style) in (("held", "held on", ROLES["herald"], :dash),
            ("release_B_0p35", "return 0.35", ROLES["herald"], :solid),
            ("release_B_0p17", "return 0.17", ROLES["active"], :solid),
            ("release_B_0p0", "return 0", E_COLOR, :dot))
        rows = filter(row -> row.time <= 400,
            trace(bundle, "tonic_herald_$(suffix).csv"))
        lines!(ax, [row.time for row in rows], [row.E for row in rows];
            color, linestyle=style, linewidth=2.8, label)
    end
    vlines!(ax, [0.0]; color=(NEUTRAL, 0.5), linestyle=:dot)
    xlims!(ax, -50, 400); ylims!(ax, 0.25, 0.52)
    ax_input = Axis(fig[3, 1], xlabel="Time relative to increase (ms)",
        ylabel=rich("B", subscript("E")))
    lines!(ax_input, [-50.0, 0.0, 0.0, 300.0],
        [0.35, 0.35, 1.2328125, 1.2328125];
        color=E_COLOR, linewidth=2.7)
    vlines!(ax_input, [0.0]; color=(NEUTRAL, 0.5), linestyle=:dot)
    xlims!(ax_input, -50, 300)
    ax_input.yticks = ([0.35, 1.2328125], ["0.35", "1.2328"])
    ylims!(ax_input, 0.22, 1.36)
    ax_input = Axis(fig[3, 2], xlabel="Time relative to release (ms)",
        ylabel=rich("B", subscript("E")))
    for (value, color, style) in ((1.2328125, ROLES["herald"], :dash),
            (0.35, ROLES["herald"], :solid), (0.17, ROLES["active"], :solid),
            (0.0, E_COLOR, :dot))
        lines!(ax_input, [-50, 0, 0, 400],
            [1.2328125, 1.2328125, value, value];
            color, linestyle=style, linewidth=2.3)
    end
    vlines!(ax_input, [0.0]; color=(NEUTRAL, 0.5), linestyle=:dot)
    xlims!(ax_input, -50, 400); ylims!(ax_input, -0.12, 1.36)
    ax_input.yticks = ([0.0, 0.35, 1.2328125], ["0", "0.35", "1.2328"])
    rowsize!(fig.layout, 3, 135)
    Legend(fig[4, 1:2], ax; orientation=:horizontal, framevisible=false,
        tellwidth=false)
    ax = Axis(fig[5, 1:2], title="C. Destination after 5,000 ms under held or released input",
        xlabel=rich("Applied B", subscript("E")))
    outcome_grid!(ax, data["selected_tonic"])
    rowsize!(fig.layout, 5, 170)
    Label(fig[6, 1:2], rich("At θ", subscript("off"),
        " = 8, the seizure source was found at the held-on input; it was not induced from baseline.\n",
        "Pre-onset traces are simulated at fixed input and meet the reviewed t = 0 states; post-onset traces are the reviewed runs.");
        fontsize=17, tellwidth=false)
    save_figure(fig, output, "03_tonic_control")
end

function figure4(bundle, output, data)
    fig = Figure(size=(1450, 900), fontsize=18)
    Label(fig[1, 1:2], rich("At θ", subscript("off"),
        " = 8.75, the original seizure source reaches herald and switching persists");
        fontsize=23, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1], title="A. Changed-threshold phase portrait",
        xlabel="E occupancy", ylabel="I occupancy")
    role_points!(ax, data["threshold_8p75_roots"])
    path = filter(row -> row.time <= 500,
        trace(bundle, "theta_off_8.75_from_seizure.csv"))
    lines!(ax, [row.E for row in path], [row.I for row in path];
        color=ROLES["seizure"], linewidth=3, label="trajectory from original seizure source")
    scatter!(ax, [first(path).E], [first(path).I]; marker=:diamond,
        color=ROLES["seizure"], markersize=15, label="original source at θ = 8")
    text!(ax, 0.37, 0.055; text="original seizure source", color=ROLES["seizure"],
        fontsize=13)
    text!(ax, 0.35, 0.405; text="trajectory → herald", color=ROLES["seizure"],
        fontsize=13)
    ax = Axis(fig[2, 2], title="B. Outcomes at sampled failure thresholds",
        xlabel=rich("Sampled θ", subscript("off")))
    joint = TOML.parsefile(joinpath(ROOT, "reproducibility",
        "selective_anchor_joint_20261005", "reference_summary.toml"))
    trials = joint["threshold_interventions"]
    for (x, row) in enumerate(trials)
        role = row["seizure_source_destination"]
        text!(ax, x, 2; text=role, align=(:center, :center),
            color=ROLES[role], fontsize=17, font=:bold)
        text!(ax, x, 1; text="$(row["root_count"]) / $(row["sink_count"])",
            align=(:center, :center), fontsize=16)
    end
    ax.xticks = (1:7, [string(row["theta_off"]) for row in trials])
    ax.yticks = ([1, 2], ["roots / sinks", "source destination"])
    ax.xgridvisible = false; ax.ygridvisible = false
    xlims!(ax, 0.5, 7.5); ylims!(ax, 0.5, 2.5)
    ax = Axis(fig[3, 1:2], title="C. Same four switching pulses at both thresholds",
        xlabel="Pulse event and validated destination")
    ax.xticks = (0:4, ["start", "1: E +0.5", "2: I +3", "3: E +0.5", "4: I +3"])
    ax.yticks = ([1, 2], ["θ = 8.75", "θ = 8"])
    ax.xgridvisible = false; ax.ygridvisible = false
    for (y, setting) in ((2, "baseline"), (1, "changed"))
        ends = endpoint_series(bundle, setting)
        roles = ("rest", "active", "rest", "active", "rest")
        for (x, role) in enumerate(roles)
            endpoint = x == 1 ? nothing : ends[x-1]
            if endpoint !== nothing
                roots = setting == "baseline" ? data["baseline_roots"] :
                    data["threshold_8p75_roots"]
                root = only(filter(row -> row["role"] == role, roots))
                maximum(abs.([endpoint.E - root["E"], endpoint.I - root["I"]])) < 1e-6 ||
                    error("switching endpoint differs from displayed destination")
            end
            text!(ax, x-1, y; text=role, align=(:center, :center),
                color=ROLES[role], fontsize=18, font=:bold)
        end
    end
    xlims!(ax, -0.45, 4.45); ylims!(ax, 0.5, 2.5)
    rowsize!(fig.layout, 3, 200)
    Label(fig[4, 1:2], rich("Thresholds are sampled points; no bifurcation is located. Counts are discovered roots/sinks.\n",
        "At θ", subscript("off"), " = 8.75, the high-state destination is herald, not active or rest.");
        fontsize=17, tellwidth=false)
    save_figure(fig, output, "04_threshold_intervention")
end

function supplement1(output, samples)
    qualifies(row) = row.induction_status == "compatible" &&
        row.induction_destination == "herald" &&
        row.control_status == "compatible" &&
        row.control_destination == "herald" &&
        row.release_0p17_status == "compatible" &&
        row.release_0p17_destination == "active"
    groups = [(theta, source) for theta in (8.0, 8.25, 8.5, 8.75, 9.0, 10.0, 12.0)
        for source in ("rest", "active")]
    brackets = Tuple{Float64,Float64}[]
    counts = Tuple{Int,Int}[]
    for (theta, source) in groups
        rows = sort(filter(row -> row.theta_off == theta && row.source == source,
            samples); by=row -> row.B_E_on)
        passed = qualifies.(rows)
        first_pass = findfirst(identity, passed)
        first_pass !== nothing && first_pass > 1 &&
            all(!, passed[1:first_pass-1]) && all(passed[first_pass:end]) ||
            error("sampled onset is not a single shared transition")
        push!(brackets, (rows[first_pass-1].B_E_on, rows[first_pass].B_E_on))
        push!(counts, (first_pass-1, length(rows)-first_pass+1))
    end
    length(unique(brackets)) == 1 && length(unique(counts)) == 1 ||
        error("sampled thresholds or source states differ")
    lower, upper = only(unique(brackets))
    failures, successes = only(unique(counts))
    fig = Figure(size=(1150, 510), fontsize=19)
    Label(fig[1, 1], "The same sampled tonic-input bracket across 14 source–threshold groups";
        fontsize=24, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1], title="Last nonqualifying and first qualifying samples",
        xlabel=rich("Absolute held B", subscript("E"), " at onset"))
    lines!(ax, [lower, upper], [1.0, 1.0]; color=NEUTRAL,
        linestyle=:dash, linewidth=2)
    scatter!(ax, [lower], [1.0]; color=NEUTRAL, markersize=21)
    scatter!(ax, [upper], [1.0]; color=ROLES["active"], markersize=21)
    text!(ax, lower, 1.17; text="last nonqualifying\n$(lower)",
        align=(:center, :bottom), color=NEUTRAL, fontsize=17)
    text!(ax, upper, 0.83; text="first qualifying\n$(upper)",
        align=(:center, :top), color=ROLES["active"], fontsize=17)
    ax.yticks = (Float64[], String[])
    ax.ygridvisible = false
    xlims!(ax, 1.219, 1.243); ylims!(ax, 0.48, 1.52)
    Label(fig[3, 1], rich("Seven sampled θ", subscript("off"),
        " values (8, 8.25, 8.5, 8.75, 9, 10, 12) × rest/active sources.\n",
        "Each group: $(failures) other outcomes and $(successes) qualifying samples among 147 inputs over 0.35–16.\n",
        "Qualification: held-on herald followed by active after return to B", subscript("E"),
        " = 0.17. This bracket is sampled, not an exact threshold.");
        fontsize=16, tellwidth=false)
    save_figure(fig, output, "S1_sampled_input_threshold")
end

function supplement2(bundle, output)
    fig = Figure(size=(1370, 910), fontsize=19)
    Label(fig[1, 1:2], "Historical protocol: finite E withdrawal from established high states";
        fontsize=24, font=:bold, tellwidth=false)
    for (column, source) in enumerate(("herald", "seizure"))
        ax = Axis(fig[2, column],
            title=rich("Starting from established $source at B", subscript("E"), " = 0.35"),
            xlabel=rich("Pulse decrement ΔB", subscript("E")),
            ylabel="Pulse duration (ms)",
            yscale=log10)
        rows = collect(CSV.File(joinpath(bundle, "$(source)_withdrawal_pulses.csv")))
        for role in ("active", "herald", "seizure", "rest")
            subset = filter(row -> row.status == "compatible" && row.destination == role, rows)
            isempty(subset) && continue
            scatter!(ax, [row.amplitude for row in subset], [row.duration for row in subset];
                color=ROLES[role], marker=Dict("active"=>:circle,
                    "herald"=>:diamond, "seizure"=>:rect, "rest"=>:utriangle)[role],
                markersize=9, label=role)
        end
        unresolved = filter(row -> row.status != "compatible", rows)
        isempty(unresolved) || scatter!(ax, [row.amplitude for row in unresolved],
            [row.duration for row in unresolved]; color=NEUTRAL, marker=:xcross,
            markersize=8, label="unresolved")
        xlims!(ax, -0.01, 0.36); ylims!(ax, 0.9, 250)
        ax.yticks = ([1.0, 3.0, 10.0, 30.0, 100.0, 200.0],
            ["1", "3", "10", "30", "100", "200"])
        axislegend(ax; position=:rt, labelsize=16)
    end
    ax = Axis(fig[3, 1:2], title="Input protocol for each sampled point (time schematic)",
        xlabel="Before pulse              During pulse: duration d              After pulse",
        ylabel=rich("B", subscript("E")))
    lines!(ax, [0, 1, 1, 2, 2, 3], [0.35, 0.35, 0.12, 0.12, 0.35, 0.35];
        color=E_COLOR, linewidth=3)
    text!(ax, 0.5, 0.39; text="0.35", align=(:center, :bottom), fontsize=16)
    text!(ax, 1.5, 0.075; text=rich("0.35 − ΔB", subscript("E")),
        align=(:center, :top), fontsize=16)
    text!(ax, 2.5, 0.39; text="restored to 0.35",
        align=(:center, :bottom), fontsize=16)
    ax.xticks = (Float64[], String[])
    ax.yticks = (Float64[], String[])
    ax.xgridvisible = false; ax.ygridvisible = false
    xlims!(ax, 0, 3); ylims!(ax, 0, 0.48)
    rowsize!(fig.layout, 3, 175)
    Label(fig[4, 1:2], rich("The diagram shows timing, not a fixed decrement or duration; pulse end restores B",
        subscript("E"), " = 0.35. Destinations follow that restoration.\n",
        "Frozen two-input batch before region-selection and phase-handoff corrections; current-source replay remains pending.");
        fontsize=17, tellwidth=false)
    save_figure(fig, output, "S2_established_state_withdrawal")
end

function render(bundle, output)
    data, samples = checked_data(bundle)
    ispath(output) && error("output path already exists: $output")
    mkpath(output)
    figure1(bundle, output, data)
    figure2(bundle, output, data)
    figure3(bundle, output, data)
    figure4(bundle, output, data)
    supplement1(output, samples)
    supplement2(bundle, output)
    figure_files = filter(name -> any(ext -> endswith(name, ext),
        (".pdf", ".svg", ".png")), readdir(output))
    length(figure_files) == 18 || error("incomplete figure export")
    open(joinpath(output, "render.toml"), "w") do io
        TOML.print(io, Dict("bundle_sha256"=>hashfile(joinpath(bundle, "checksums.toml")),
            "julia_version"=>string(VERSION), "figure_count"=>6,
            "renderer_sha256"=>hashfile(@__FILE__),
            "plotting_manifest_sha256"=>hashfile(joinpath(ROOT, "plotting", "Manifest.toml")),
            "files"=>Dict(name=>hashfile(joinpath(output, name)) for name in figure_files),
            "limits"=>data["limits"]))
    end
end

function main(args)
    length(args) == 2 || error("usage: julia --project=plotting scripts/render_paper_figures.jl DATA_BUNDLE NEW_OUTPUT")
    render(abspath(args[1]), abspath(args[2]))
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    PaperFigureRenderer.main(ARGS)
end

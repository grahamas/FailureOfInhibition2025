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
hashfile(path) = bytes2hex(SHA.sha256(read(path)))

function checked_data(directory)
    manifest = TOML.parsefile(joinpath(directory, "checksums.toml"))["files"]
    for (relative, expected) in manifest
        (isabspath(relative) || ".." in splitpath(relative)) &&
            error("unsafe figure-data path: $relative")
        path = joinpath(directory, relative)
        isfile(path) && hashfile(path) == expected || error("figure-data hash mismatch: $relative")
    end
    data = TOML.parsefile(joinpath(directory, "data.toml"))
    for (relative, expected) in data["source_sha256"]
        hashfile(joinpath(ROOT, relative)) == expected || error("model/protocol source changed: $relative")
    end
    for (name, relative) in (("joint", "selective_anchor_joint_20261005"),
            ("tonic", "selective_tonic_e_release_20261005"),
            ("response", "paper_figures_20261005"))
        filename = name == "response" ? "response_reference.toml" : "reference_summary.toml"
        path = joinpath(ROOT, "reproducibility", relative, filename)
        hashfile(path) == data["reference_sha256"][name] ||
            error("reference summary changed: $name")
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
function role_points!(axis, rows; labels=true)
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
    xlims!(axis, -0.015, 0.53)
    ylims!(axis, -0.015, 0.53)
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
    lines!(axis, x, [row.E for row in states]; color=E_COLOR, linewidth=2.5)
    scatter!(axis, x, [row.E for row in states]; color=E_COLOR, markersize=12, label="E")
    lines!(axis, x, [row.I for row in states]; color=I_COLOR, linewidth=2.5)
    scatter!(axis, x, [row.I for row in states]; color=I_COLOR, markersize=12, label="I")
    axis.xticks = (x, ["rest", "active", "rest", "active", "rest"])
    xlims!(axis, -0.2, 4.2)
    ylims!(axis, -0.02, 0.52)
    axislegend(axis; position=:rt, orientation=:horizontal)
end

function figure1(bundle, output, data)
    fig = Figure(size=(1300, 680), fontsize=17)
    Label(fig[1, 1:2], "Failure response permits a second high-activity ordering";
        fontsize=25, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1], title="A. Ordered inhibitory recruitment and failure",
        xlabel="Effective input to I", ylabel="Population response")
    pair = NarrativeModels.models(data["anchor"])
    input = collect(range(0.0, 12.0; length=600))
    lines!(ax, input, response.(Ref(pair.control.inhibitory.response), input);
        color=NEUTRAL, linewidth=2.5, label="Monotone control response")
    lines!(ax, input, response.(Ref(pair.failure_of_inhibition.inhibitory.response), input);
        color=I_COLOR, linewidth=3, label="Fire-then-fail response")
    vlines!(ax, [4.0, 8.0]; color=(NEUTRAL, 0.5), linestyle=:dot)
    text!(ax, 4.0, 0.12; text="onset", rotation=pi/2, fontsize=12)
    text!(ax, 8.0, 0.12; text="failure", rotation=pi/2, fontsize=12)
    axislegend(ax; position=:lt)
    ax = Axis(fig[2, 2], title="B. Discovered equilibria at B_E = 0.35",
        xlabel="E occupancy", ylabel="I occupancy")
    nullclines!(ax, data["anchor"])
    role_points!(ax, data["baseline_roots"])
    Label(fig[3, 1:2], "For two equilibria at common parameters and input, a monotone inhibitory response forbids higher E with lower I.  Blue/orange: E/I nullclines.";
        fontsize=14, tellwidth=false)
    save_figure(fig, output, "01_mechanism")
end

function figure2(bundle, output, data)
    fig = Figure(size=(1320, 900), fontsize=16)
    Label(fig[1, 1:2], "Ordinary switching and separate access to high-activity states";
        fontsize=24, font=:bold, tellwidth=false)
    rest = only(filter(row -> row["role"] == "rest", data["baseline_roots"]))
    ax = Axis(fig[2, 1], title="A. Two cycles from actual previous endpoints",
        xlabel="Validated destination after each 100 ms pulse", ylabel="Occupancy")
    endpoint_plot!(ax, bundle, "baseline", rest)
    ax = Axis(fig[2, 2], title="B. Baseline state geometry",
        xlabel="E occupancy", ylabel="I occupancy")
    role_points!(ax, data["baseline_roots"])
    for (column, name, title) in ((1, "induce_rest_to_herald", "C. Rest → herald; E +1"),
            (2, "induce_active_to_seizure", "D. Active → seizure; I +4.5"))
        ax = Axis(fig[3, column], title=title, xlabel="Time within pulse and release (ms)",
            ylabel="Occupancy")
        pulse = trace(bundle, name * "_pulse.csv")
        trace_lines!(ax, pulse; limit=100.0)
        trace_lines!(ax, trace(bundle, name * ".csv"); offset=100.0, limit=200.0,
            label=false)
        vlines!(ax, [100.0]; color=NEUTRAL, linestyle=:dash)
        xlims!(ax, 0, 300)
        ylims!(ax, -0.02, 0.52)
        column == 1 && axislegend(ax; position=:rb, orientation=:horizontal)
    end
    Label(fig[4, 1:2], "Tonic input returns to B_E = 0.35 after each pulse.  Traces show early dynamics; destinations use the full 5,000 ms follow-up.";
        fontsize=14, tellwidth=false)
    save_figure(fig, output, "02_switching_and_access")
end

function outcome_grid!(axis, selected)
    axis.xticks = (1:4, ["held on", "0.35", "0.17", "0"])
    axis.yticks = ([1, 2], ["on-input seizure source", "induced herald"])
    outcomes(source) = [selected[source]["control"]["destination"],
        (selected[source]["release"][key]["destination"] for key in
            ("B_0p35", "B_0p17", "B_0p0"))...]
    outcomes("rest") == outcomes("active") ||
        error("rest and active tonic outcomes differ; outcome grid needs separate rows")
    for (y, roles) in ((2, outcomes("rest")), (1, outcomes("seizure")))
        for (x, role) in enumerate(roles)
            scatter!(axis, [x], [y]; marker=:rect, color=ROLES[role], markersize=65)
            text!(axis, x, y; text=role, align=(:center, :center), color=:white,
                fontsize=13)
        end
    end
    xlims!(axis, 0.45, 4.55)
    ylims!(axis, 0.45, 2.55)
end

function figure3(bundle, output, data)
    fig = Figure(size=(1380, 900), fontsize=16)
    Label(fig[1, 1:2], "Lower tonic E returns induced herald to active, while seizure persists";
        fontsize=23, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1], title="A. Hold B_E = 1.2328125: baseline states → herald",
        xlabel="Time after input increase (ms)", ylabel="E occupancy")
    for (source, style) in (("rest", :solid), ("active", :dash))
        rows = filter(row -> row.time <= 300,
            trace(bundle, "tonic_$(source)_induction.csv"))
        lines!(ax, [row.time for row in rows], [row.E for row in rows];
            color=E_COLOR, linestyle=style, linewidth=2.8, label="from $source")
    end
    xlims!(ax, 0, 300); ylims!(ax, -0.02, 0.52)
    axislegend(ax; position=:rb)
    ax = Axis(fig[2, 2], title="B. Release from the actual herald endpoint",
        xlabel="Time after release (ms)", ylabel="E occupancy")
    for (suffix, label, color, style) in (("held", "held on", ROLES["herald"], :dash),
            ("release_B_0p35", "return 0.35", ROLES["herald"], :solid),
            ("release_B_0p17", "return 0.17", ROLES["active"], :solid),
            ("release_B_0p0", "return 0", E_COLOR, :dot))
        rows = filter(row -> row.time <= 400,
            trace(bundle, "tonic_herald_$(suffix).csv"))
        lines!(ax, [row.time for row in rows], [row.E for row in rows];
            color, linestyle=style, linewidth=2.8, label)
    end
    xlims!(ax, 0, 400); ylims!(ax, 0.25, 0.52)
    axislegend(ax; position=:rb)
    ax = Axis(fig[3, 1], title="C. Same release inputs from a separate seizure source",
        xlabel="Time after release (ms)", ylabel="I occupancy")
    for (suffix, label, style) in (("held", "held on", :dash),
            ("release_B_0p35", "return 0.35", :solid),
            ("release_B_0p17", "return 0.17", :dot),
            ("release_B_0p0", "return 0", :dashdot))
        rows = filter(row -> row.time <= 400,
            trace(bundle, "tonic_seizure_$(suffix).csv"))
        lines!(ax, [row.time for row in rows], [row.I for row in rows];
            color=ROLES["seizure"], linestyle=style, linewidth=2.8, label)
    end
    xlims!(ax, 0, 400); ylims!(ax, -0.005, 0.04)
    axislegend(ax; position=:rt, labelsize=12)
    ax = Axis(fig[3, 2], title="D. Finite-window destination by applied input",
        xlabel="Held or release B_E", ylabel="Source at B_E,on")
    outcome_grid!(ax, data["selected_tonic"])
    Label(fig[4, 1:2], "At θ_off = 8.  The seizure source is discovered at B_E,on; it was not induced from baseline.  At θ_off ≥ 8.75 no seizure source was identified for this cessation trial.";
        fontsize=14, tellwidth=false)
    save_figure(fig, output, "03_tonic_control")
end

function figure4(bundle, output, data)
    fig = Figure(size=(1370, 930), fontsize=16)
    Label(fig[1, 1:2], "Increasing the failure threshold redirects the seizure source and preserves switching";
        fontsize=23, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1], title="A. θ_off = 8; four discovered sinks",
        xlabel="E occupancy", ylabel="I occupancy")
    role_points!(ax, data["baseline_roots"])
    ax = Axis(fig[2, 2], title="B. θ_off = 8.75; three sinks; seizure source → herald",
        xlabel="E occupancy", ylabel="I occupancy")
    role_points!(ax, data["threshold_8p75_roots"])
    path = filter(row -> row.time <= 500,
        trace(bundle, "theta_off_8.75_from_seizure.csv"))
    lines!(ax, [row.E for row in path], [row.I for row in path];
        color=ROLES["seizure"], linewidth=3)
    scatter!(ax, [first(path).E], [first(path).I]; marker=:diamond,
        color=ROLES["seizure"], markersize=15)
    ax = Axis(fig[3, 1], title="C. Sampled source destinations",
        xlabel="Failure threshold θ_off", ylabel="Observed destination")
    joint = TOML.parsefile(joinpath(ROOT, "reproducibility",
        "selective_anchor_joint_20261005", "reference_summary.toml"))
    trials = joint["threshold_interventions"]
    for row in trials
        role = row["seizure_source_destination"]
        y = role == "seizure" ? 2.0 : 1.0
        scatter!(ax, [row["theta_off"]], [y]; color=ROLES[role], markersize=17)
    end
    text!(ax, 8.0, 2.28; text="7 roots / 4 sinks at sampled lower values", fontsize=11)
    text!(ax, 8.75, 1.28; text="5 roots / 3 sinks at sampled higher values", fontsize=11)
    ax.yticks = ([1, 2], ["herald", "seizure"])
    xlims!(ax, 7.8, 12.2); ylims!(ax, 0.65, 2.4)
    rest = only(filter(row -> row["role"] == "rest", data["threshold_8p75_roots"]))
    ax = Axis(fig[3, 2], title="D. Two switching cycles at θ_off = 8.75",
        xlabel="Validated destination after each 100 ms pulse", ylabel="Occupancy")
    endpoint_plot!(ax, bundle, "changed", rest)
    Label(fig[4, 1:2], "The threshold strip shows sampled points, not a located bifurcation.  Root/sink counts are discovered counts; the altered high-state destination remains herald.";
        fontsize=14, tellwidth=false)
    save_figure(fig, output, "04_threshold_intervention")
end

function supplement1(output, samples)
    fig = Figure(size=(1400, 770), fontsize=16)
    Label(fig[1, 1:2], "Sampled tonic-on input versus failure threshold";
        fontsize=24, font=:bold, tellwidth=false)
    for (column, low, high, title) in ((1, 1.15, 1.30, "A. Sampled onset bracket"),
            (2, 0.35, 16.0, "B. Full sampled range"))
        ax = Axis(fig[2, column], title=title, xlabel="Absolute held B_E,on",
            ylabel="Failure threshold θ_off")
        for row in samples
            low <= row.B_E_on <= high || continue
            qualifies = row.induction_status == "compatible" &&
                row.induction_destination == "herald" &&
                row.control_status == "compatible" &&
                row.control_destination == "herald" &&
                row.release_0p17_status == "compatible" &&
                row.release_0p17_destination == "active"
            unresolved = row.induction_status != "compatible" ||
                (row.induction_destination == "herald" &&
                 (row.control_status != "compatible" || row.release_0p17_status != "compatible"))
            color = qualifies ? ROLES["active"] : unresolved ? ROLES["seizure"] : NEUTRAL
            offset = row.source == "rest" ? -0.045 : 0.045
            scatter!(ax, [row.B_E_on], [row.theta_off + offset]; color,
                markersize=column == 1 ? 6 : 4)
        end
        xlims!(ax, low, high); ylims!(ax, 7.8, 12.2)
        column == 1 && vlines!(ax, [1.22890625, 1.2328125];
            color=(NEUTRAL, 0.6), linestyle=:dash)
        ax.yticks = ([8.0, 8.25, 8.5, 8.75, 9.0, 10.0, 12.0],
            ["8", "8.25", "8.5", "8.75", "9", "10", "12"])
    end
    Label(fig[3, 1:2], "Green: held-on herald followed by active at B_E = 0.17; gray: another resolved outcome; red: unresolved.  Lower/upper dot: rest/active source; no interpolation.";
        fontsize=14, tellwidth=false)
    save_figure(fig, output, "S1_sampled_input_threshold")
end

function supplement2(bundle, output)
    fig = Figure(size=(1330, 690), fontsize=16)
    Label(fig[1, 1:2], "Historical protocol: finite E withdrawal from established high states";
        fontsize=24, font=:bold, tellwidth=false)
    for (column, source) in enumerate(("herald", "seizure"))
        ax = Axis(fig[2, column], title="Starting from established $source at B_E = 0.35",
            xlabel="E input withdrawn during pulse", ylabel="Pulse duration (ms)",
            yscale=log10)
        rows = collect(CSV.File(joinpath(bundle, "$(source)_withdrawal_pulses.csv")))
        for role in ("active", "herald", "seizure", "rest")
            subset = filter(row -> row.status == "compatible" && row.destination == role, rows)
            isempty(subset) && continue
            scatter!(ax, [row.amplitude for row in subset], [row.duration for row in subset];
                color=ROLES[role], markersize=7, label=role)
        end
        unresolved = filter(row -> row.status != "compatible", rows)
        isempty(unresolved) || scatter!(ax, [row.amplitude for row in unresolved],
            [row.duration for row in unresolved]; color=NEUTRAL, marker=:xcross,
            markersize=8, label="unresolved")
        xlims!(ax, -0.01, 0.36); ylims!(ax, 0.9, 250)
        axislegend(ax; position=:rt, labelsize=12)
    end
    Label(fig[3, 1:2], "Frozen two-input batch, before region-selection and phase-handoff corrections.  Pulses restore B_E = 0.35 afterward; current-source response replay remains pending.";
        fontsize=14, tellwidth=false)
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

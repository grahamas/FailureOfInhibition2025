"""Render an archived permanent-input-release example with CairoMakie."""
module InputReleaseRenderer
using CairoMakie
import CSV
import TOML
import SHA

function checked_inputs(input)
    checksums = TOML.parsefile(joinpath(input, "checksums.toml"))["files"]
    for (relative, expected) in checksums
        isabspath(relative) || ".." in splitpath(relative) ?
            throw(ArgumentError("invalid checksum path")) : nothing
        path = joinpath(input, relative)
        isfile(path) && bytes2hex(SHA.sha256(read(path))) == expected ||
            throw(ArgumentError("artifact checksum mismatch: $relative"))
    end
    selected = TOML.parsefile(joinpath(input, "selected.toml"))
    selected["success"] === true || throw(ArgumentError("example is not confirmed"))
    config = TOML.parsefile(joinpath(input, "config.toml"))
    catalogue = TOML.parsefile(joinpath(input, config["exemplars"]))
    case = only(filter(x -> x["name"] == selected["case"], catalogue["cases"]))
    directory = joinpath(input, selected["confirmation_directory"], "grid_$(selected["final_grid"])")
    data = Dict(name => CSV.File(joinpath(directory, "$name.csv")) for name in ("induction", "recovery", "control"))
    contexts = Dict(name => TOML.parsefile(joinpath(directory, "$(name)_context.toml")) for name in ("on", "off"))
    return (; selected, case, data, contexts)
end

function render(input_dir, output_dir; preview=false)
    input, output = abspath(input_dir), abspath(output_dir)
    inputs = checked_inputs(input)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or empty"))
    mkpath(output)
    selected, case, data, contexts = inputs.selected, inputs.case, inputs.data, inputs.contexts
    e_color, i_color = "#087e9c", "#ce702b"
    fig = Figure(size=(1280, 800), backgroundcolor="#f5f7f8", fontsize=17)
    Label(fig[1, 1:2], "Input on: sustained high E / low I. Input off: recovery.",
        fontsize=27, font=:bold, tellwidth=false)
    Label(fig[2, 1:2], "$(selected["case"]) · e_to_e = $(selected["e_to_e"]) · all other parameters matched",
        color="#506375", tellwidth=false)
    title = Observable("B_E = 8 · induction")
    Label(fig[3, 1:2], title, fontsize=20, color="#694b89", tellwidth=false)
    on_axis = Axis(fig[4, 1], title="After input turns on", xlabel="Time since input on (ms)", ylabel="Activity")
    off_axis = Axis(fig[4, 2], title="After input turns off", xlabel="Time since input off (ms)", ylabel="Activity")
    for ax in (on_axis, off_axis)
        xlims!(ax, 0, 400); ylims!(ax, -0.02, 0.53)
    end
    control = data["control"]
    control_indices = findall(<=(400), control.time)
    lines!(off_axis, control.time[control_indices], control.E[control_indices],
        color=(e_color, 0.5), linestyle=:dash, label="E with input held on")
    lines!(off_axis, control.time[control_indices], control.I[control_indices],
        color=(i_color, 0.5), linestyle=:dash)
    traces = Dict{Tuple{String,Symbol},Observable{Vector{Point2f}}}()
    for (name, ax) in (("induction", on_axis), ("recovery", off_axis)), (key, color) in ((:E, e_color), (:I, i_color))
        obs = Observable(Point2f[])
        traces[(name, key)] = obs
        lines!(ax, obs; color, linewidth=3, label=string(key))
    end
    axislegend(on_axis; position=:rt, orientation=:horizontal)
    axislegend(off_axis; position=:rt)
    phase = Axis(fig[5, 1], title="State space and current-input equilibria", xlabel="E", ylabel="I")
    xlims!(phase, -0.015, 0.52); ylims!(phase, -0.02, 0.52)
    trail = Observable(Point2f[])
    marker = Observable([Point2f(selected["initial_state"]...)])
    roots = Observable(Point2f[])
    other_roots = Observable(Point2f[])
    scatter!(phase, roots; marker=:star5, markersize=18, color="#c14d60", label="Attracting equilibrium")
    scatter!(phase, other_roots; marker=:xcross, markersize=13, color="#6d7884", label="Other equilibrium")
    lines!(phase, trail; linewidth=2, color="#2b927f")
    scatter!(phase, marker; markersize=15, color="#e6ba42", strokecolor="#203448", strokewidth=1.5)
    axislegend(phase; position=:lt, labelsize=12)
    info = Observable("")
    Label(fig[5, 2], info; justification=:left, halign=:left, fontsize=18, tellwidth=false)
    Label(fig[6, 1:2], "Numerical coordinate roles; finite-window recovery. Stars are equilibria of the currently applied input.",
        fontsize=14, color="#526575", tellwidth=false)
    Label(fig[7, 1:2], "Plots show the first 400 ms of each phase; the clock advances through the full validated follow-up.",
        fontsize=13, color="#526575", tellwidth=false)
    function update_frame(stage, time)
        current_input = stage == "induction" ? "on" : "off"
        entries = contexts[current_input]["equilibria"]
        roots[] = [Point2f(r["state"]...) for r in entries if r["stability"]["classification"] == "Attracting"]
        other_roots[] = [Point2f(r["state"]...) for r in entries if r["stability"]["classification"] != "Attracting"]
        for name in ("induction", "recovery")
            limit = name == stage ? time : stage == "recovery" ? selected["on_duration"] : -1.0
            table = data[name]
            indices = findall(t -> t <= min(limit, 400.0), table.time)
            for key in (:E, :I)
                traces[(name, key)][] = [Point2f(table.time[i], getproperty(table, key)[i]) for i in indices]
            end
        end
        table = data[stage]
        last_index = clamp(searchsortedlast(table.time, time), 1, length(table.time))
        trail[] = [Point2f(table.E[i], table.I[i]) for i in 1:last_index]
        marker[] = [last(trail[])]
        drive = current_input == "on" ? selected["on_input"] : 0.0
        title[] = "B_E = $drive · $(stage == "induction" ? "input on" : "input remains off") · $(round(Int,time)) ms into phase"
        info[] = "Current E = $(round(table.E[last_index]; sigdigits=6))\nCurrent I = $(round(table.I[last_index]; sigdigits=6))\n\n" *
            "$(length(roots[])) discovered attracting $(length(roots[]) == 1 ? "equilibrium" : "equilibria")\nat the current input\n\n" *
            "Validated input-on duration: $(round(Int,selected["on_duration"])) ms\nValidated input-off follow-up: $(round(Int,selected["off_duration"])) ms\n\n" *
            "Held-on control remains at high E / low I.\nRelease returns to the starting low-activity state."
    end
    frames = Tuple{String,Float64}[]
    for (stage, horizon) in (("induction", selected["on_duration"]), ("recovery", selected["off_duration"]))
        for time in vcat(collect(range(0.0, min(400.0,horizon); length=preview ? 6 : 90)),
            collect(range(min(400.0,horizon), horizon; length=preview ? 2 : 15)), fill(horizon, preview ? 1 : 12))
            push!(frames, (stage, time))
        end
    end
    record(fig, joinpath(output, "input_release.mp4"), frames; framerate=24) do frame
        update_frame(frame...)
    end
    update_frame("recovery", selected["off_duration"])
    save(joinpath(output, "recovery.png"), fig; px_per_unit=1.5)
    cp(joinpath(input, "source"), joinpath(output, "source"))
    for relative in ("scripts/render_input_release.jl", "plotting/Project.toml", "plotting/Manifest.toml")
        destination = joinpath(output, "source", relative)
        mkpath(dirname(destination))
        cp(joinpath(@__DIR__, "..", relative), destination; force=true)
    end
    metadata = Dict("input_artifact" => input,
        "input_checksums_sha256" => bytes2hex(SHA.sha256(read(joinpath(input,"checksums.toml")))),
        "preview" => preview, "frames" => length(frames), "framerate" => 24,
        "replay" => "julia --project=source/plotting source/scripts/render_input_release.jl $input replay",
        "renderer_sha256" => bytes2hex(SHA.sha256(read(@__FILE__))))
    open(joinpath(output,"metadata.toml"), "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    hashes = Dict(relpath(joinpath(dir,file), output) => bytes2hex(SHA.sha256(read(joinpath(dir,file))))
        for (dir,_,files) in walkdir(output) for file in files)
    open(joinpath(output,"checksums.toml"), "w") do io
        TOML.print(io, Dict("files" => hashes); sorted=true)
    end
    return output
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (2,3) || error("usage: render_input_release.jl INPUT OUTPUT [--preview]")
    length(ARGS) == 2 || ARGS[3] == "--preview" || error("unknown argument")
    InputReleaseRenderer.render(ARGS[1], ARGS[2]; preview=length(ARGS)==3)
end

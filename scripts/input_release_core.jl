"""Reusable finite input-release numerical phases."""
module InputReleaseCore
using FailureOfInhibition2025
include("study_support.jl")
const Evidence = StudySupport

function search_context(model, grid; tight=false)
    upper = FailureOfInhibition2025._equilibrium_upper_bounds(model, Float64)
    seeds = vcat([[e, i] for e in range(0, upper[1]; length=grid)
        for i in range(0, upper[2]; length=grid)], default_equilibrium_seeds(model))
    defaults = EquilibriumOptions()
    options = tight ? EquilibriumOptions(;
        (name => (name == :maxiters ? getfield(defaults, name) : getfield(defaults, name)/10)
         for name in fieldnames(typeof(defaults)))...) : defaults
    stability_options = tight ? StabilityOptions(spectral_atol=1e-11, spectral_rtol=1e-9) : StabilityOptions()
    return find_equilibria(model; seeds, options, stability_options)
end

sink_indices(search) = [i for (i, root) in enumerate(search.equilibria)
    if root.stability.classification == Attracting && !root.near_singular]

"""Track a driven source by coordinates; no zero-input search participates."""
function match_source(search, reference, tolerance)
    reference === nothing && return nothing
    candidates = [(i, maximum(abs.(search.equilibria[i].state .- reference))) for i in sink_indices(search)]
    sort!(candidates; by=last)
    isempty(candidates) && return nothing
    first(candidates)[2] <= tolerance || return nothing
    length(candidates) > 1 && candidates[2][2] - candidates[1][2] <= 1e-6 && return nothing
    return first(candidates)[1]
end

"""Integrate one fixed-input phase, retaining every attempted horizon."""
function observe_phase(model, search, initial, config; retain=false, tight=false,
    stop_at_saved_times=false, handoff=false)
    attempts = NamedTuple[]
    for horizon in config.horizons
        window = config.diagnostics.window_duration
        times = sort!(unique(vcat([0.0],
            collect(range(horizon-2window, horizon-window; length=config.diagnostics.min_samples)),
            collect(range(horizon-window, horizon; length=config.diagnostics.min_samples)),
            retain ? collect(0.0:1.0:horizon) : Float64[])))
        try
            solution = solve_point_model(initial, (0.0, horizon), model;
                saveat=times, save_everystep=false, dense=false,
                tstops=(stop_at_saved_times || handoff) ? times : (),
                abstol=config.abstol/(tight ? 10 : 1), reltol=config.reltol/(tight ? 10 : 1),
                domain_atol=handoff ? 0.0 : config.domain_atol, maxiters=config.maxiters)
            diagnostic = diagnose_trajectory(solution, model; equilibria=search,
                options=config.diagnostics)
            destination = diagnostic.matched_equilibrium
            status = !diagnostic.integration_success ? "integration_failed" :
                diagnostic.classification == EquilibriumCompatible &&
                destination in sink_indices(search) ? "compatible" : "unresolved"
            push!(attempts, (; horizon, status, destination=destination === nothing ? 0 : destination,
                solver_status=string(solution.retcode), diagnostics=diagnostic, error=""))
            if status != "unresolved" || horizon == last(config.horizons)
                return (; status, destination=status == "compatible" ? destination : nothing,
                    final=copy(last(solution.u)), horizon, solution, attempts)
            end
        catch error
            error isa InterruptException && rethrow()
            push!(attempts, (; horizon, status="integration_failed", destination=0,
                solver_status="exception", diagnostics=nothing, error=sprint(showerror, error)))
            return (; status="integration_failed", destination=nothing, final=copy(initial),
                horizon, solution=nothing, attempts)
        end
    end
    error("empty phase horizons")
end

phase_record(phase) = (; phase.status, phase.destination, phase.final, phase.horizon, phase.attempts)

"""Recovery requires actual induction followed by return to the initial sink."""
function qualifies_trial(source, start, induction, recovery, control)
    return induction.status == "compatible" && induction.destination == source &&
        recovery.status == "compatible" && recovery.destination == start &&
        control.status == "compatible" && control.destination == source
end

function full_trial(models, searches, source, start, config; retain=false, tight=false)
    initial = searches.off.equilibria[start].state
    induction = observe_phase(models.on, searches.on, initial, config; retain, tight, handoff=true)
    # Both continuations start from the actual end of induction, never a snapped root.
    recovery = observe_phase(models.off, searches.off, induction.final, config; retain, tight)
    control = observe_phase(models.on, searches.on, induction.final, config; retain, tight)
    success = qualifies_trial(source, start, induction, recovery, control)
    return (; success, initial, source, start, induction, recovery, control)
end

function save_phase(output, name, phase)
    Evidence.write_toml(joinpath(output, name * ".toml"), phase_record(phase))
    phase.solution === nothing || write_trajectory_csv(joinpath(output, name * ".csv"), phase.solution)
end

function save_trial(output, trial)
    mkpath(output)
    for phase in (:induction, :recovery, :control)
        save_phase(output, string(phase), getproperty(trial, phase))
    end
    Evidence.write_toml(joinpath(output, "trial.toml"),
        (; trial.success, trial.initial, trial.source, trial.start,
            switch_state=trial.induction.final, on_duration=trial.induction.horizon,
            off_duration=trial.recovery.horizon))
end

end

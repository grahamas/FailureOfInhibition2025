"""
    PulseExperimentOptions(; amplitudes=0:0.25:8,
        durations=[1,2,5,10,20,50,100,200], targets=(:E,:I,:equal,:negative_E),
        followup_times=[5000,10000,20000], diagnostic_options, refinement_levels=2,
        duration_refinement_levels=0, retain_trajectories=false,
        trajectory_saveat=1, abstol=1e-10, reltol=1e-10,
        domain_atol=1e-8, maxiters=1000000)

Rectangular pulse exploration policy. Durations and follow-up times are in ms;
amplitudes are nonnegative effective-input magnitudes. `:negative_E` applies
the negative magnitude as an `AbstractIntervention`; the other targets use
positive E, positive I, or the same positive magnitude on both populations.
Each unresolved follow-up is independently reintegrated from the original
initial state and pulse, retaining every attempt. This avoids stitching
different diagnostic sampling windows across integration segments.

By default only pulse endpoints and two exactly sampled terminal diagnostic
windows are saved. `retain_trajectories=true` additionally saves the full
interval on `trajectory_saveat` spacing. Boundary refinement samples every
adjacent pair of differing resolved outcomes, without a monotonicity
assumption. Unresolved brackets remain explicit and are not treated as
successful or unsuccessful interventions. Neither finite sampling nor
refinement certifies all transitions between amplitudes. Duration refinement
compares resolved-destination sets, ordered amplitude-boundary topology,
unresolved presence, and integration-failure presence at adjacent durations.
It inserts arithmetic midpoints where those signatures differ and retains
finite duration brackets; it does not estimate an exact minimum duration.
"""
struct PulseExperimentOptions{T<:AbstractFloat,D<:DiagnosticOptions}
    amplitudes::Vector{T}
    durations::Vector{T}
    targets::Vector{Symbol}
    followup_times::Vector{T}
    diagnostic_options::D
    refinement_levels::Int
    duration_refinement_levels::Int
    retain_trajectories::Bool
    trajectory_saveat::T
    abstol::T
    reltol::T
    domain_atol::T
    maxiters::Int
end

function _pulse_numeric_sequence(values, name; positive=false)
    applicable(iterate, values) || throw(ArgumentError("$name must be iterable"))
    sequence = collect(values)
    !isempty(sequence) && all(x -> x isa Real && !(x isa Bool) && isfinite(x) &&
        (positive ? x > zero(x) : x >= zero(x)), sequence) ||
        throw(ArgumentError("$name must contain finite $(positive ? "positive" : "nonnegative") real values"))
    all(pair -> first(pair) < last(pair), zip(sequence, Iterators.drop(sequence, 1))) ||
        throw(ArgumentError("$name must be strictly increasing"))
    return sequence
end

function PulseExperimentOptions(;
    amplitudes=0:0.25:8, durations=[1, 2, 5, 10, 20, 50, 100, 200],
    targets=(:E, :I, :equal, :negative_E), followup_times=[5000, 10000, 20000],
    diagnostic_options=DiagnosticOptions(window_duration=100, min_samples=21),
    refinement_levels=2, duration_refinement_levels=0,
    retain_trajectories=false, trajectory_saveat=1,
    abstol=1e-10, reltol=1e-10, domain_atol=1e-8, maxiters=1000000,
)
    amplitudes = _pulse_numeric_sequence(amplitudes, "amplitudes")
    durations = _pulse_numeric_sequence(durations, "durations"; positive=true)
    followup_times = _pulse_numeric_sequence(followup_times, "followup_times"; positive=true)
    diagnostic_options isa DiagnosticOptions ||
        throw(ArgumentError("diagnostic_options must be DiagnosticOptions"))
    first(followup_times) >= 2diagnostic_options.window_duration ||
        throw(ArgumentError("every follow-up must cover both diagnostic windows after withdrawal"))
    applicable(iterate, targets) || throw(ArgumentError("targets must be iterable"))
    targets = collect(targets)
    !isempty(targets) && all(x -> x in (:E, :I, :equal, :negative_E), targets) &&
        length(unique(targets)) == length(targets) ||
        throw(ArgumentError("targets must be unique members of (:E,:I,:equal,:negative_E)"))
    for (name, value) in (("refinement_levels", refinement_levels),
                          ("duration_refinement_levels", duration_refinement_levels))
        value isa Integer && !(value isa Bool) && 0 <= value <= typemax(Int) ||
            throw(ArgumentError("$name must be a nonnegative integer"))
    end
    retain_trajectories isa Bool || throw(ArgumentError("retain_trajectories must be Boolean"))
    maxiters isa Integer && !(maxiters isa Bool) && 0 < maxiters <= typemax(Int) ||
        throw(ArgumentError("maxiters must be a positive integer"))
    for (name, value) in (("trajectory_saveat", trajectory_saveat), ("abstol", abstol),
                          ("reltol", reltol))
        value isa Real && !(value isa Bool) && isfinite(value) && value > zero(value) ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    _validate_domain_atol(domain_atol)
    values = promote(float.((amplitudes..., durations..., followup_times...,
        trajectory_saveat, abstol, reltol, domain_atol))...)
    T = eltype(values)
    return PulseExperimentOptions(T.(amplitudes), T.(durations), Symbol.(targets),
        T.(followup_times), diagnostic_options, Int(refinement_levels),
        Int(duration_refinement_levels), retain_trajectories, T(trajectory_saveat),
        T(abstol), T(reltol), T(domain_atol), Int(maxiters))
end

"""
    PulseTrialResult

A single pulse with all follow-up attempts retained. `outcome_equilibrium` is
a unique finite-window compatible equilibrium whose local stability is
`Attracting`; it remains `nothing` otherwise. It is not a proof of the
asymptotic destination. `status` is `:compatible`, `:unresolved`, or
`:integration_failed`. `integrated_E` and `integrated_I` are signed pulse
increments integrated over time. `absolute_input_cost` is their summed
absolute value, in effective-input units times ms, not biological energy.
Each attempt contains diagnostics, two tail summaries of E, I, u_I and F_I',
the solver status, retained sampled states, and an explicit error if present.
"""
struct PulseTrialResult{T<:AbstractFloat,O<:PulseExperimentOptions}
    initial_id::String
    initial_state::Vector{T}
    initial_provenance::String
    target::Symbol
    amplitude::T
    duration::T
    integrated_E::T
    integrated_I::T
    absolute_input_cost::T
    status::Symbol
    outcome_equilibrium::Union{Nothing,Int}
    attempts::Vector{NamedTuple}
    options::O
end

"""Pulse trials and retained amplitude and duration protocol-change brackets."""
struct PulseExperimentResult{M,E,O}
    model::M
    equilibria::E
    options::O
    initial_states::Vector{NamedTuple}
    trials::Vector{PulseTrialResult}
    boundaries::Vector{NamedTuple}
    duration_refinements::Vector{NamedTuple}
    duration_brackets::Vector{NamedTuple}
end

function _pulse_context(model, equilibria)
    equilibria isa EquilibriumSearchResult ||
        throw(ArgumentError("equilibria must be an EquilibriumSearchResult"))
    context = _frozen_point_context(model, nothing)
    _same_diagnostic_parameters(model, equilibria.model) &&
        _same_diagnostic_parameters(model, equilibria.frozen_model) &&
        context.frozen_drive == equilibria.frozen_drive ||
        throw(ArgumentError("pulse model must match the autonomous equilibrium context"))
    return context.frozen_drive
end

function _pulse_increment(target, amplitude)
    target == :E && return (amplitude, zero(amplitude))
    target == :I && return (zero(amplitude), amplitude)
    target == :equal && return (amplitude, amplitude)
    target == :negative_E && return (-amplitude, zero(amplitude))
    throw(ArgumentError("unknown pulse target: $target"))
end

function _pulse_save_times(duration, followup, options)
    stop = duration + followup
    middle = stop - options.diagnostic_options.window_duration
    start = middle - options.diagnostic_options.window_duration
    samples = options.diagnostic_options.min_samples
    times = vcat([zero(stop), duration],
        collect(range(start, middle; length=samples)),
        collect(range(middle, stop; length=samples)))
    if options.retain_trajectories
        append!(times, first(_save_times(options.trajectory_saveat, (zero(stop), stop))))
    end
    return _sorted_unique_times(times)
end

function _pulse_scalar_summary(values)
    isempty(values) && return (mean=NaN, minimum=NaN, maximum=NaN)
    all(isfinite, values) || return (mean=NaN, minimum=NaN, maximum=NaN)
    return (mean=sum(x / length(values) for x in values),
        minimum=minimum(values), maximum=maximum(values))
end

function _pulse_tail_summaries(solution, model, diagnostics)
    return ntuple(2) do window
        lower, upper = diagnostics.window_bounds[window]
        indices = findall(t -> lower <= t <= upper, solution.t)
        e = [solution.u[i][1] for i in indices]
        inh = [solution.u[i][2] for i in indices]
        inputs = [_point_inputs(solution.u[i], model, solution.t[i])[4] for i in indices]
        slopes = [response_derivative(model.inhibitory.response, input) for input in inputs]
        (bounds=(lower, upper), samples=length(indices), E=_pulse_scalar_summary(e),
            I=_pulse_scalar_summary(inh), u_I=_pulse_scalar_summary(inputs),
            F_I_prime=_pulse_scalar_summary(slopes))
    end
end

"""
    run_pulse_trial(model, equilibria, initial_state;
        target, amplitude, duration, options=PulseExperimentOptions(),
        initial_id="supplied", initial_provenance="caller-supplied state")

Apply one rectangular pulse on `[0,duration)` to an autonomous model. Extend
an unresolved observation through the configured follow-ups; retain numerical
failures and every finite-window diagnostic. Input validation errors throw.
Pulse magnitudes and durations need not belong to the batch grid. A supplied
initial state is not automatically an equilibrium or a validated cycle phase.
No periodic destination or biological role is inferred.
"""
function run_pulse_trial(model::PointModelParameters, equilibria, initial_state;
    target, amplitude, duration, options=PulseExperimentOptions(),
    initial_id="supplied", initial_provenance="caller-supplied state",
)
    options isa PulseExperimentOptions || throw(ArgumentError("options must be PulseExperimentOptions"))
    baseline = _pulse_context(model, equilibria)
    _validate_initial_state(initial_state)
    for (name, value, positive) in (("amplitude", amplitude, false), ("duration", duration, true))
        value isa Real && !(value isa Bool) && isfinite(value) &&
            (positive ? value > zero(value) : value >= zero(value)) ||
            throw(ArgumentError("$name must be finite and $(positive ? "positive" : "nonnegative")"))
    end
    initial_id isa AbstractString && !isempty(initial_id) ||
        throw(ArgumentError("initial_id must be a nonempty string"))
    initial_provenance isa AbstractString ||
        throw(ArgumentError("initial_provenance must be a string"))
    T = promote_type(eltype(options.amplitudes), eltype(float.(initial_state)),
        typeof(float(amplitude)), typeof(float(duration)))
    amplitude, duration = T(amplitude), T(duration)
    increment = _pulse_increment(target, amplitude)
    # Signed background drive also requires the abstract interpretation.
    interpretation = target == :negative_E || any(x -> x < zero(x), baseline) ?
        AbstractIntervention : AfferentExcitation
    drive = PiecewiseConstantDrive(baseline=baseline,
        pulses=(DrivePulse(onset=zero(T), offset=duration, increment=increment),),
        interpretation=interpretation)
    driven = PointModelParameters(excitatory=model.excitatory, inhibitory=model.inhibitory,
        coupling=model.coupling, drive=drive)
    attempts = NamedTuple[]
    status, outcome = :unresolved, nothing
    for followup in options.followup_times
        stop = duration + T(followup)
        times = _pulse_save_times(duration, T(followup), options)
        try
            solution = solve_point_model(initial_state, (zero(T), stop), driven;
                saveat=times, save_everystep=false, dense=false,
                abstol=options.abstol, reltol=options.reltol,
                domain_atol=options.domain_atol, maxiters=options.maxiters)
            diagnostics = diagnose_trajectory(solution, driven;
                equilibria=equilibria, options=options.diagnostic_options)
            reasons = copy(diagnostics.reasons)
            matched = diagnostics.matched_equilibrium
            attracting = matched !== nothing &&
                equilibria.equilibria[matched].stability.classification == Attracting
            matched !== nothing && !attracting && push!(reasons, :matched_equilibrium_not_attracting)
            resolved = diagnostics.classification == EquilibriumCompatible && attracting
            status = !diagnostics.integration_success ? :integration_failed :
                resolved ? :compatible : :unresolved
            outcome = resolved ? matched : nothing
            push!(attempts, (followup_time=T(followup), status=status,
                outcome_equilibrium=outcome, reasons=reasons,
                solver_status=Symbol(string(solution.retcode)), diagnostics=diagnostics,
                tail=_pulse_tail_summaries(solution, driven, diagnostics),
                times=collect(solution.t), states=[copy(state) for state in solution.u],
                error=nothing))
            status != :unresolved && break
        catch error
            error isa InterruptException && rethrow()
            status = :integration_failed
            push!(attempts, (followup_time=T(followup), status=status,
                outcome_equilibrium=nothing, reasons=[:integration_exception],
                solver_status=:exception, diagnostics=nothing, tail=nothing,
                times=T[], states=Vector{T}[],
                error=(type=string(typeof(error)), message=sprint(showerror, error))))
            break
        end
    end
    costs = duration .* increment
    return PulseTrialResult(String(initial_id), T.(collect(initial_state)),
        String(initial_provenance), target, amplitude, duration, costs...,
        sum(abs, costs), status, outcome, attempts, options)
end

function _pulse_initial_states(equilibria, initial_states)
    if initial_states === nothing
        return NamedTuple[(id="equilibrium_$index", state=copy(equilibrium.state),
            provenance=_pulse_equilibrium_provenance(equilibria, index, equilibrium))
            for (index, equilibrium) in enumerate(equilibria.equilibria)
            if equilibrium.stability.classification == Attracting]
    end
    applicable(iterate, initial_states) || throw(ArgumentError("initial_states must be iterable"))
    states = NamedTuple[]
    for initial in initial_states
        all(key -> hasproperty(initial, key), (:id, :state, :provenance)) ||
            throw(ArgumentError("each initial state requires id, state, and provenance"))
        initial.id isa AbstractString && !isempty(initial.id) &&
            initial.provenance isa AbstractString ||
            throw(ArgumentError("initial state IDs and provenance must be strings; IDs must be nonempty"))
        _validate_initial_state(initial.state)
        push!(states, (id=String(initial.id), state=collect(float.(initial.state)),
            provenance=String(initial.provenance)))
    end
    length(unique(initial.id for initial in states)) == length(states) ||
        throw(ArgumentError("initial state IDs must be unique"))
    return states
end

function _pulse_equilibrium_provenance(search, index, equilibrium)
    issues = String[]
    equilibrium.near_singular && push!(issues, "near-singular balance Jacobian")
    any(component -> any(attempt -> attempt in component, equilibrium.member_attempts),
        search.unresolved_nearby) && push!(issues, "unresolved nearby root candidates")
    base = "admissible equilibrium $index; local spectral classification Attracting"
    return isempty(issues) ? base : base * "; source uncertainty: " * join(issues, "; ")
end

function _pulse_boundary_kind(left, right)
    if left.status == :integration_failed || right.status == :integration_failed
        return :integration_failed
    end
    if left.outcome_equilibrium === nothing || right.outcome_equilibrium === nothing
        return :unresolved
    end
    return left.outcome_equilibrium == right.outcome_equilibrium ? nothing : :outcome_change
end

function _refine_pulse_amplitudes!(trials, indices, evaluate, levels)
    for _ in 1:levels
        sort!(indices; by=i -> trials[i].amplitude)
        brackets = [(left, right) for (left, right) in zip(indices, Iterators.drop(indices, 1))
            if _pulse_boundary_kind(trials[left], trials[right]) == :outcome_change]
        isempty(brackets) && break
        for (left, right) in brackets
            amplitude = (trials[left].amplitude + trials[right].amplitude) / 2
            trials[left].amplitude < amplitude < trials[right].amplitude || continue
            push!(trials, evaluate(amplitude))
            push!(indices, length(trials))
        end
    end
    sort!(indices; by=i -> trials[i].amplitude)
    return indices
end

function _pulse_duration_signature(trials, indices)
    ordered = [trials[index] for index in indices]
    destinations = sort!(unique(trial.outcome_equilibrium for trial in ordered
        if trial.outcome_equilibrium !== nothing))
    boundary_sequence = NamedTuple[]
    for (left, right) in zip(ordered, Iterators.drop(ordered, 1))
        kind = _pulse_boundary_kind(left, right)
        kind === nothing && continue
        push!(boundary_sequence, (kind=kind, lower_outcome=left.outcome_equilibrium,
            upper_outcome=right.outcome_equilibrium, lower_status=left.status,
            upper_status=right.status))
    end
    return (destinations=Tuple(destinations),
        boundary_sequence=Tuple(boundary_sequence),
        unresolved_present=any(trial -> trial.status == :unresolved, ordered),
        integration_failure_present=any(trial -> trial.status == :integration_failed, ordered))
end

function _pulse_duration_difference_reasons(lower, upper)
    reasons = Symbol[]
    lower.destinations != upper.destinations && push!(reasons, :destination_set)
    lower_present = any(item -> item.kind == :outcome_change, lower.boundary_sequence)
    upper_present = any(item -> item.kind == :outcome_change, upper.boundary_sequence)
    lower_present != upper_present && push!(reasons, :transition_presence)
    length(lower.boundary_sequence) != length(upper.boundary_sequence) &&
        push!(reasons, :boundary_count)
    length(lower.boundary_sequence) == length(upper.boundary_sequence) &&
        lower.boundary_sequence != upper.boundary_sequence && push!(reasons, :boundary_order)
    lower.unresolved_present != upper.unresolved_present &&
        push!(reasons, :unresolved_presence)
    lower.integration_failure_present != upper.integration_failure_present &&
        push!(reasons, :integration_failure_presence)
    return reasons
end

function _refine_pulse_durations!(durations, signatures, evaluate, levels)
    refinements = NamedTuple[]
    for level in 1:levels
        sort!(durations)
        marked = NamedTuple[]
        for (lower, upper) in zip(durations, Iterators.drop(durations, 1))
            reasons = _pulse_duration_difference_reasons(signatures[lower], signatures[upper])
            isempty(reasons) && continue
            midpoint = (lower + upper) / 2
            midpoint in durations && continue
            push!(marked, (; level, lower_duration=lower, inserted_duration=midpoint,
                upper_duration=upper, reasons=Tuple(reasons)))
        end
        isempty(marked) && break
        for refinement in marked
            midpoint = refinement.inserted_duration
            signatures[midpoint] = evaluate(midpoint)
            push!(durations, midpoint)
            push!(refinements, refinement)
        end
    end
    sort!(durations)
    return refinements
end

function _pulse_duration_brackets(durations, signatures)
    brackets = NamedTuple[]
    for (lower, upper) in zip(durations, Iterators.drop(durations, 1))
        reasons = _pulse_duration_difference_reasons(signatures[lower], signatures[upper])
        isempty(reasons) && continue
        push!(brackets, (lower_duration=lower, upper_duration=upper,
            reasons=Tuple(reasons), lower_signature=signatures[lower],
            upper_signature=signatures[upper]))
    end
    return brackets
end

"""
    run_pulse_experiments(model; equilibria, options=PulseExperimentOptions(),
        initial_states=nothing)

Run every target, duration and amplitude from each admissible locally
attracting discovered equilibrium, then refine adjacent differing outcomes.
Optional explicit initial states are `(id, state, provenance)` records; this
allows externally validated cycle phases while preserving their provenance.
The caller must verify any periodic state and its model context separately.
The destination classifier currently recognizes only finite-window
compatibility with locally attracting equilibria. An empty discovered set is
retained as an empty result, without an attractor-completeness claim.
"""
function run_pulse_experiments(model::PointModelParameters; equilibria,
    options=PulseExperimentOptions(), initial_states=nothing,
)
    options isa PulseExperimentOptions || throw(ArgumentError("options must be PulseExperimentOptions"))
    _pulse_context(model, equilibria)
    initials = _pulse_initial_states(equilibria, initial_states)
    trials = PulseTrialResult[]
    boundaries, duration_refinements, duration_brackets = NamedTuple[], NamedTuple[], NamedTuple[]
    for initial in initials, target in options.targets
        indices_by_duration = Dict{eltype(options.durations),Vector{Int}}()
        signatures = Dict{eltype(options.durations),NamedTuple}()
        function evaluate_duration(duration)
            evaluate = amplitude -> run_pulse_trial(model, equilibria, initial.state;
                initial_id=initial.id, initial_provenance=initial.provenance,
                target=target, amplitude=amplitude, duration=duration, options=options)
            indices = Int[]
            for amplitude in options.amplitudes
                push!(trials, evaluate(amplitude))
                push!(indices, length(trials))
            end
            _refine_pulse_amplitudes!(trials, indices, evaluate, options.refinement_levels)
            indices_by_duration[duration] = indices
            return _pulse_duration_signature(trials, indices)
        end
        sampled_durations = copy(options.durations)
        for duration in sampled_durations
            signatures[duration] = evaluate_duration(duration)
        end
        refinements = _refine_pulse_durations!(sampled_durations, signatures,
            evaluate_duration, options.duration_refinement_levels)
        append!(duration_refinements, [merge((initial_id=initial.id, target=target), item)
            for item in refinements])
        for duration in sampled_durations
            indices = indices_by_duration[duration]
            for (left, right) in zip(indices, Iterators.drop(indices, 1))
                kind = _pulse_boundary_kind(trials[left], trials[right])
                kind === nothing && continue
                push!(boundaries, (initial_id=initial.id, target=target, duration=duration,
                    lower_amplitude=trials[left].amplitude,
                    upper_amplitude=trials[right].amplitude,
                    lower_trial=left, upper_trial=right,
                    lower_outcome=trials[left].outcome_equilibrium,
                    upper_outcome=trials[right].outcome_equilibrium,
                    lower_status=trials[left].status, upper_status=trials[right].status,
                    kind=kind))
            end
        end
        append!(duration_brackets,
            [merge((initial_id=initial.id, target=target), bracket)
             for bracket in _pulse_duration_brackets(sampled_durations, signatures)])
    end
    return PulseExperimentResult(model, equilibria, options, initials, trials, boundaries,
        duration_refinements, duration_brackets)
end

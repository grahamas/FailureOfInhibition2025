using LinearAlgebra: eigvals, norm

"""Numerical periodic-orbit validation; neither value is an existence certificate."""
@enum PeriodicOrbitValidation begin
    NumericallyValidatedPeriodicOrbit
    PeriodicOrbitUnresolved
end

"""Orbital stability from the nontrivial Floquet multiplier of a planar orbit."""
@enum PeriodicOrbitStability begin
    PeriodicOrbitAttracting
    PeriodicOrbitRepelling
    PeriodicOrbitStabilityUnresolved
end

"""
    PeriodicOrbitOptions(; kwargs...)

Policies for Float64, phase-conditioned shooting. `shooting_atol` bounds the
closure and phase equations. `validation_atol` bounds closure, the sampled
normalized-time ODE defect, and discrepancies between two tolerance levels
(period discrepancies are relative to `max(1, period)`). `amplitude_atol`
excludes constant solutions. `floquet_atol` bounds the trivial-multiplier
error and separates the transverse multiplier from the unit circle.
`phase_speed_atol` excludes a degenerate phase reference. Integration is
repeated with tolerances multiplied by `refinement_factor`.
"""
struct PeriodicOrbitOptions
    ode_abstol::Float64
    ode_reltol::Float64
    shooting_atol::Float64
    validation_atol::Float64
    amplitude_atol::Float64
    phase_speed_atol::Float64
    floquet_atol::Float64
    refinement_factor::Float64
    min_period::Float64
    max_period::Float64
    maxiters::Int
    ode_maxiters::Int
    samples::Int

    function PeriodicOrbitOptions(;
        ode_abstol=1e-10, ode_reltol=1e-10, shooting_atol=1e-9,
        validation_atol=1e-6, amplitude_atol=1e-5,
        phase_speed_atol=1e-10, floquet_atol=1e-4,
        refinement_factor=0.1, min_period=1e-6, max_period=1e6,
        maxiters=20, ode_maxiters=1_000_000, samples=257,
    )
        values = (ode_abstol, ode_reltol, shooting_atol, validation_atol,
                  amplitude_atol, phase_speed_atol, floquet_atol,
                  refinement_factor, min_period, max_period)
        for value in values
            _periodic_float64(value, "periodic-orbit tolerance or bound")
            value > 0 || throw(ArgumentError("periodic-orbit tolerances and bounds must be positive"))
        end
        refinement_factor < 1 || throw(ArgumentError("refinement_factor must be below one"))
        min_period < max_period || throw(ArgumentError("min_period must precede max_period"))
        for (value, name) in ((maxiters, "maxiters"), (ode_maxiters, "ode_maxiters"),
                              (samples, "samples"))
            value isa Integer && value > 0 || throw(ArgumentError("$name must be a positive integer"))
        end
        samples >= 9 || throw(ArgumentError("samples must be at least nine"))
        return new(Float64.(values)..., Int(maxiters), Int(ode_maxiters), Int(samples))
    end
end

"""
    PeriodicOrbitResult

Final shooting evidence, including unresolved outcomes. The caller should
retain the original seed and period guess to replay the phase condition;
intermediate Newton and line-search iterates are not retained. `model` retains
the PointModelParameters context (or generic callback parameters), and
`solution` retains the refined normalized-time integration for phase sampling.
`closure_residual`, `phase_residual`, and `equation_residual` are infinity
norms; the equation defect compares the dense interpolant derivative against
`period * rhs` at interior sample times. `refinement_difference` reports
waveform, relative period, and monodromy discrepancies. `times` use original
time units and `states` are ordered `[E, I]` for the point-model wrapper.

Validation is numerical evidence for a nonconstant periodic solution, not
a rigorous existence proof, a completeness claim, or certification that the
reported period is primitive. Near-neutral orbital stability stays unresolved.
"""
struct PeriodicOrbitResult{M,S}
    model::M
    options::PeriodicOrbitOptions
    initial_state::Vector{Float64}
    period::Float64
    times::Vector{Float64}
    states::Vector{Vector{Float64}}
    amplitudes::Vector{Float64}
    closure_residual::Float64
    phase_residual::Float64
    equation_residual::Float64
    monodromy::Matrix{Float64}
    floquet_multipliers::Vector{ComplexF64}
    transverse_multiplier::ComplexF64
    refinement_difference::NamedTuple{(:waveform, :period, :monodromy),Tuple{Float64,Float64,Float64}}
    integration_success::Bool
    validation::PeriodicOrbitValidation
    stability::PeriodicOrbitStability
    reasons::Vector{Symbol}
    solution::S
end

function _periodic_float64(value, name)
    value isa Union{Integer,Rational,Float32,Float64} || throw(
        ArgumentError("$name must use integers, rationals, Float32 or Float64; periodic shooting uses Float64"),
    )
    isfinite(value) && isfinite(Float64(value)) || throw(ArgumentError("$name must be finite"))
    # Large exact inputs must not silently lose precision at this boundary.
    value isa Union{Integer,Rational} && Float64(value) != value && throw(
        ArgumentError("$name is not exactly representable by the Float64 shooting implementation"),
    )
    return Float64(value)
end

function _periodic_seed(initial_state)
    _require_point_state(initial_state, "initial_state")
    return [_periodic_float64(value, "initial_state coordinate") for value in initial_state]
end

_periodic_parameter_check(::Nothing) = nothing
_periodic_parameter_check(value::Real) = _periodic_float64(value, "callback parameter")
function _periodic_parameter_check(values::Union{Tuple,NamedTuple})
    foreach(_periodic_parameter_check, values)
    return nothing
end
function _periodic_parameter_check(value)
    throw(ArgumentError("generic callback parameters must be nothing, real numbers, tuples or named tuples"))
end

function _periodic_integrate(rhs!, jacobian!, parameters, state, period, options, factor, domain)
    augmented = [state[1], state[2], 1.0, 0.0, 0.0, 1.0]
    derivative = zeros(2)
    jacobian = zeros(2, 2)
    function augmented_rhs!(output, input, _, time)
        x = @view input[1:2]
        rhs!(derivative, x, parameters)
        jacobian!(jacobian, x, parameters)
        output[1] = period * derivative[1]
        output[2] = period * derivative[2]
        for column in 0:1
            first = 3 + 2column
            output[first] = period * (jacobian[1, 1] * input[first] + jacobian[1, 2] * input[first + 1])
            output[first + 1] = period * (jacobian[2, 1] * input[first] + jacobian[2, 2] * input[first + 1])
        end
        return nothing
    end
    outside = (u, _, _) -> !all(isfinite, u) ||
        (domain !== nothing && any(value -> !(domain[1] <= value <= domain[2]), @view u[1:2]))
    problem = ODEProblem(augmented_rhs!, augmented, (0.0, 1.0); isoutofdomain=outside)
    solution = solve(problem, Tsit5(); abstol=options.ode_abstol * factor,
                     reltol=options.ode_reltol * factor, maxiters=options.ode_maxiters,
                     dense=true, save_everystep=true, save_start=true, save_end=true,
                     verbose=false)
    complete = successful_retcode(solution) && !isempty(solution.t) &&
               first(solution.t) == 0.0 && last(solution.t) == 1.0 &&
               all(u -> !outside(u, nothing, nothing), solution.u)
    return solution, complete
end

function _periodic_shoot(rhs!, jacobian!, parameters, seed, period, reference,
                         tangent, options, factor, domain)
    state = copy(seed)
    end_rhs = zeros(2)
    solution = nothing
    complete = false
    for iteration in 0:options.maxiters
        solution, complete = _periodic_integrate(rhs!, jacobian!, parameters,
                                                 state, period, options, factor, domain)
        complete || return (; state, period, solution, complete, reason=:incomplete_integration)
        endpoint = solution.u[end]
        residual = [endpoint[1] - state[1], endpoint[2] - state[2],
                    sum((state .- reference) .* tangent)]
        maximum(abs, residual) <= options.shooting_atol &&
            return (; state, period, solution, complete, reason=nothing)
        iteration == options.maxiters && break
        monodromy = reshape(endpoint[3:6], 2, 2)
        rhs!(end_rhs, @view(endpoint[1:2]), parameters)
        bordered = [monodromy[1, 1]-1 monodromy[1, 2] period*end_rhs[1];
                    monodromy[2, 1] monodromy[2, 2]-1 period*end_rhs[2];
                    tangent[1] tangent[2] 0.0]
        singular_values = svdvals(bordered)
        minimum(singular_values) <= 1e-12 * maximum(singular_values) &&
            return (; state, period, solution, complete, reason=:singular_shooting_jacobian)
        correction = -(bordered \ residual)
        all(isfinite, correction) ||
            return (; state, period, solution, complete, reason=:nonfinite_newton_step)
        accepted = false
        for backtrack in 0:12
            scale = 0.5^backtrack
            trial_state = state .+ scale .* correction[1:2]
            trial_period = period * exp(scale * correction[3])
            options.min_period <= trial_period <= options.max_period || continue
            domain !== nothing && any(value -> !(domain[1] <= value <= domain[2]), trial_state) && continue
            trial_solution, trial_complete = _periodic_integrate(
                rhs!, jacobian!, parameters, trial_state, trial_period, options, factor, domain,
            )
            trial_complete || continue
            trial_end = trial_solution.u[end]
            trial_residual = [trial_end[1] - trial_state[1], trial_end[2] - trial_state[2],
                              sum((trial_state .- reference) .* tangent)]
            if maximum(abs, trial_residual) < maximum(abs, residual)
                state, period = trial_state, trial_period
                accepted = true
                break
            end
        end
        accepted || return (; state, period, solution, complete, reason=:shooting_line_search_failed)
    end
    return (; state, period, solution, complete, reason=:shooting_iteration_limit)
end

function _periodic_unresolved(parameters, options, state, period, reason; solution=nothing, complete=false)
    return PeriodicOrbitResult(
        parameters, options, state, period, Float64[], Vector{Float64}[], fill(NaN, 2),
        NaN, NaN, NaN, fill(NaN, 2, 2), fill(ComplexF64(NaN), 2), ComplexF64(NaN),
        (waveform=NaN, period=NaN, monodromy=NaN), complete,
        PeriodicOrbitUnresolved, PeriodicOrbitStabilityUnresolved, [reason], solution,
    )
end

"""
    solve_periodic_orbit(model, initial_state, period_guess; options=PeriodicOrbitOptions())
    solve_periodic_orbit(rhs!, jacobian!, parameters, initial_state, period_guess; options=PeriodicOrbitOptions())

Correct a proposed nonconstant periodic solution with a phase condition,
repeat at tighter integration tolerance, check the ODE defect, and integrate
the variational equations for Floquet stability. The point-model wrapper
requires a globally constant drive; even pulses outside the proposed period
are rejected. Times and period use the model's time units.

The general planar overload requires autonomous callbacks with signatures
`rhs!(du, u, parameters)` and `jacobian!(J, u, parameters)`. Callers must supply
the exact Jacobian of the same autonomous vector field and callbacks compatible
with Float64 arithmetic. Point-model trajectories must remain in `[0, 1]^2`.
Generic callback parameters must be `nothing`, real numbers, or nested tuples
and named tuples of those values, so their precision can be checked explicitly.
Float32 inputs are promoted explicitly to Float64; BigFloat inputs/models are
unsupported and rejected. The seed and period are hypotheses: failure to
correct them does not establish absence of a periodic orbit.
"""
function solve_periodic_orbit(model::PointModelParameters, initial_state, period_guess;
                              options=PeriodicOrbitOptions())
    options isa PeriodicOrbitOptions || throw(ArgumentError("options must be PeriodicOrbitOptions"))
    model.drive isa NoDrive || (model.drive isa PiecewiseConstantDrive && isempty(model.drive.pulses)) ||
        throw(ArgumentError("periodic-orbit shooting requires a globally constant drive"))
    for value in (_model_numeric_values(model)..., drive_value(model.drive, 0.0)...)
        _periodic_float64(value, "model parameter")
    end
    _validate_initial_state(initial_state)
    rhs! = (du, u, parameters) -> point_rhs!(du, u, parameters, 0.0)
    jacobian! = (J, u, parameters) -> point_jacobian!(J, u, parameters, 0.0)
    return _solve_periodic_orbit(rhs!, jacobian!, model, initial_state, period_guess,
                                 options, (0.0, 1.0))
end

function solve_periodic_orbit(rhs!, jacobian!, parameters, initial_state, period_guess;
                              options=PeriodicOrbitOptions())
    options isa PeriodicOrbitOptions || throw(ArgumentError("options must be PeriodicOrbitOptions"))
    _periodic_parameter_check(parameters)
    return _solve_periodic_orbit(rhs!, jacobian!, parameters, initial_state, period_guess,
                                 options, nothing)
end

function _solve_periodic_orbit(rhs!, jacobian!, parameters, initial_state, period_guess,
                               options, domain)
    seed = _periodic_seed(initial_state)
    period = _periodic_float64(period_guess, "period_guess")
    options.min_period <= period <= options.max_period ||
        throw(ArgumentError("period_guess must lie within the configured positive period bounds"))
    tangent = zeros(2)
    rhs!(tangent, seed, parameters)
    if !all(isfinite, tangent) || norm(tangent) <= options.phase_speed_atol
        return _periodic_unresolved(parameters, options, seed, period, :degenerate_phase_reference)
    end
    tangent ./= norm(tangent)
    coarse = _periodic_shoot(rhs!, jacobian!, parameters, seed, period, seed, tangent,
                             options, 1.0, domain)
    coarse.reason === nothing || return _periodic_unresolved(
        parameters, options, coarse.state, coarse.period, coarse.reason;
        solution=coarse.solution, complete=coarse.complete,
    )
    refined = _periodic_shoot(rhs!, jacobian!, parameters, coarse.state, coarse.period,
                              seed, tangent, options, options.refinement_factor, domain)
    refined.reason === nothing || return _periodic_unresolved(
        parameters, options, refined.state, refined.period, refined.reason;
        solution=refined.solution, complete=refined.complete,
    )
    phases = collect(range(0.0, 1.0; length=options.samples))
    states = [refined.solution(phase)[1:2] for phase in phases]
    coarse_states = [coarse.solution(phase)[1:2] for phase in phases]
    amplitudes = [maximum(state[i] for state in states) - minimum(state[i] for state in states) for i in 1:2]
    monodromy = reshape(refined.solution.u[end][3:6], 2, 2)
    coarse_monodromy = reshape(coarse.solution.u[end][3:6], 2, 2)
    closure = maximum(abs, last(states) .- refined.state)
    phase_residual = abs(sum((refined.state .- seed) .* tangent))
    difference = (
        waveform=maximum(maximum(abs, state .- coarse_state) for (state, coarse_state) in zip(states, coarse_states)),
        period=abs(refined.period - coarse.period) / max(1.0, abs(refined.period)),
        monodromy=maximum(abs, monodromy .- coarse_monodromy),
    )
    derivative = zeros(2)
    equation_residual = 0.0
    for phase in @view phases[2:end-1]
        state = refined.solution(phase)[1:2]
        rhs!(derivative, state, parameters)
        interpolant_derivative = refined.solution(phase, Val{1})[1:2]
        equation_residual = max(equation_residual, maximum(abs, interpolant_derivative .- refined.period .* derivative))
    end
    multipliers = ComplexF64.(eigvals(monodromy))
    trivial_index = argmin(abs.(multipliers .- 1))
    transverse = multipliers[3 - trivial_index]
    reasons = Symbol[]
    maximum(amplitudes) > options.amplitude_atol || push!(reasons, :constant_or_small_amplitude_solution)
    closure <= options.validation_atol || push!(reasons, :large_closure_residual)
    phase_residual <= options.validation_atol || push!(reasons, :large_phase_residual)
    equation_residual <= options.validation_atol || push!(reasons, :large_equation_residual)
    difference.waveform <= options.validation_atol && difference.period <= options.validation_atol &&
        difference.monodromy <= options.floquet_atol || push!(reasons, :tolerance_refinement_disagreement)
    abs(multipliers[trivial_index] - 1) <= options.floquet_atol || push!(reasons, :missing_trivial_floquet_multiplier)
    abs(imag(transverse)) <= options.floquet_atol && real(transverse) >= -options.floquet_atol ||
        push!(reasons, :invalid_planar_floquet_multiplier)
    domain !== nothing && any(state -> any(value -> !(domain[1] <= value <= domain[2]), state), states) &&
        push!(reasons, :outside_physical_domain)
    validation = isempty(reasons) ? NumericallyValidatedPeriodicOrbit : PeriodicOrbitUnresolved
    stability = PeriodicOrbitStabilityUnresolved
    if validation == NumericallyValidatedPeriodicOrbit && abs(imag(transverse)) <= options.floquet_atol
        abs(transverse) < 1 - options.floquet_atol && (stability = PeriodicOrbitAttracting)
        abs(transverse) > 1 + options.floquet_atol && (stability = PeriodicOrbitRepelling)
    end
    return PeriodicOrbitResult(
        parameters, options, refined.state, refined.period, phases .* refined.period, states,
        amplitudes, closure, phase_residual, equation_residual, monodromy, multipliers,
        transverse, difference, true, validation, stability, reasons, refined.solution,
    )
end

"""
    periodic_orbit_phases(result, phases)

Sample copied states at fractional phases in `[0, 1)` from a numerically
validated periodic result. Validation does not imply attraction; check
`result.stability` separately before treating these states as an attractor.
"""
function periodic_orbit_phases(result::PeriodicOrbitResult, phases)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("phase sampling requires a numerically validated periodic orbit"))
    normalized_phases = [_periodic_float64(phase, "phase") for phase in phases]
    all(phase -> 0 <= phase < 1, normalized_phases) ||
        throw(ArgumentError("fractional phases must lie in [0, 1)"))
    return [result.solution(phase)[1:2] for phase in normalized_phases]
end

using ForwardDiff
using LinearAlgebra: I, det, dot, eigen, eigvals, norm

const FIGURE5B_REFINEMENT_GRID_POINTS = (11, 21, 41)

function _positive_float64_values(values, label)
    converted = [_hopf_float64(value, label) for value in values]
    all(value -> isfinite(value) && value > 0, converted) ||
        throw(ArgumentError("$label must remain finite and positive in Float64"))
    return converted
end

function _half_quotient(numerator::Float64, denominator::Float64)
    half_numerator = numerator / 2
    return half_numerator == 0 && numerator != 0 ?
        (numerator / denominator) / 2 : half_numerator / denominator
end

function _rounded_power_of_two_scale(mantissa::Float64, exponent::Int)
    if -1074 <= exponent <= -1022
        return mantissa * ldexp(1.0, exponent)
    elseif exponent < -1074
        return 0.0
    end
    return ldexp(mantissa, exponent)
end

function _scaled_hopf_frequency(determinant::Float64, ratio::Float64,
    tau_e::Float64)
    determinant_mantissa, determinant_exponent = frexp(determinant)
    ratio_mantissa, ratio_exponent = frexp(ratio)
    tau_mantissa, tau_exponent = frexp(tau_e)
    exponent_difference = determinant_exponent - ratio_exponent
    root_exponent = fld(exponent_difference, 2)
    root_argument = determinant_mantissa / ratio_mantissa
    isodd(exponent_difference) && (root_argument *= 2)
    mantissa = sqrt(root_argument) / tau_mantissa
    normalized_mantissa, normalization_exponent = frexp(mantissa)
    exponent = root_exponent - tau_exponent + normalization_exponent
    return _rounded_power_of_two_scale(normalized_mantissa, exponent)
end

function _scaled_mean(values)
    scale = maximum(abs, values)
    scale == 0 && return 0.0
    return scale * (sum(value / scale for value in values) / length(values))
end

function _finite_difference_plateau(values, atol, rtol)
    scale = maximum(abs, values)
    scale == 0 && return true
    normalized = [value / scale for value in values]
    normalized_mean = sum(normalized) / length(normalized)
    normalized_span = maximum(normalized) - minimum(normalized)
    return normalized_span <= atol / scale + rtol * abs(normalized_mean)
end

"""
    Figure5bTopologyOptions(; kwargs...)

Numerical gates for recognizing the seven-equilibrium Figure-5b topology in
three independently searched refinements. Coordinate matching is deliberately
stricter than equilibrium discovery deduplication. This finite classifier does
not certify search completeness or an attractor count. Tolerances use the
same explicit `Float64` input contract as `hopf_diagnostics` and must remain
finite and positive after conversion.
"""
struct Figure5bTopologyOptions
    coordinate_match_atol::Float64
    minimum_root_separation::Float64
    residual_atol::Float64
    jacobian_atol::Float64
    slope_atol::Float64
    spectral_margin::Float64
end

function Figure5bTopologyOptions(;
    coordinate_match_atol=1e-6,
    minimum_root_separation=1e-5,
    residual_atol=1e-9,
    jacobian_atol=1e-9,
    slope_atol=1e-8,
    spectral_margin=1e-8,
)
    raw = (coordinate_match_atol, minimum_root_separation, residual_atol,
        jacobian_atol, slope_atol, spectral_margin)
    converted = _positive_float64_values(raw, "Figure-5b topology tolerances")
    converted[1] < converted[2] / 2 || throw(ArgumentError(
        "coordinate_match_atol must be less than half minimum_root_separation"))
    return Figure5bTopologyOptions(converted...)
end

"""Finite result of matching the Figure-5b seven-root topology across refinements."""
struct Figure5bTopologyResult
    qualified::Bool
    reasons::Vector{Symbol}
    root_tracks::Vector{NTuple{3,Vector{Float64}}}
    central_track::Union{Nothing,Int}
    central_state::Union{Nothing,Vector{Float64}}
    nullcline_slopes::Union{Nothing,NTuple{2,Float64}}
    minimum_root_separation::Float64
    refinement_grid_points::NTuple{3,Int}
end

function _same_search_context(left, right)
    return typeof(left.frozen_model.excitatory.response) ===
               typeof(right.frozen_model.excitatory.response) &&
        typeof(left.frozen_model.inhibitory.response) ===
               typeof(right.frozen_model.inhibitory.response) &&
        isequal(_model_numeric_values(left.frozen_model),
            _model_numeric_values(right.frozen_model)) &&
        isequal(Tuple(left.frozen_drive), Tuple(right.frozen_drive)) &&
        isequal(left.source_time, right.source_time)
end

function _root_separation(equilibria)
    length(equilibria) < 2 && return Inf
    separation = Inf
    for left in 1:(length(equilibria) - 1)
        for right in (left + 1):length(equilibria)
            separation = min(separation,
                norm(equilibria[left].state .- equilibria[right].state))
        end
    end
    return separation
end

function _unique_root_mapping(reference, candidate, tolerance)
    distances = [norm(left.state .- right.state) for left in reference, right in candidate]
    mapping = Int[]
    for row in axes(distances, 1)
        matches = findall(value -> value <= tolerance, @view distances[row, :])
        length(matches) == 1 || return nothing
        push!(mapping, only(matches))
    end
    length(unique(mapping)) == length(candidate) || return nothing
    for column in axes(distances, 2)
        count(value -> value <= tolerance, @view distances[:, column]) == 1 || return nothing
    end
    return mapping
end

function _matches_refinement_schedule(search, grid_points)
    expected = default_equilibrium_seeds(search.frozen_model)
    upper_e, upper_i = last(expected)
    append!(expected, [[e, i] for e in range(0.0, upper_e; length=grid_points)
        for i in range(0.0, upper_i; length=grid_points)])
    unique!(expected)
    actual = [attempt.seed for attempt in search.attempts]
    length(actual) == length(expected) || return false
    return Set(Tuple.(actual)) == Set(Tuple.(expected))
end

function _safe_spectral_classification(jacobian, classification, margin)
    real_parts = sort(real.(eigvals(jacobian)))
    return if classification == Attracting
        all(value -> value < -margin, real_parts)
    elseif classification == Repelling
        all(value -> value > margin, real_parts)
    elseif classification == Saddle
        first(real_parts) < -margin && last(real_parts) > margin
    else
        false
    end
end

function _recomputed_equilibrium_quality(search, equilibrium,
    options::Figure5bTopologyOptions)
    state = equilibrium.state
    residual = zeros(Float64, 2)
    jacobian = zeros(Float64, 2, 2)
    point_balance!(residual, state, search.frozen_model, zero(eltype(state)))
    point_balance_jacobian!(jacobian, state, search.frozen_model, zero(eltype(state)))
    ode_jacobian = zeros(Float64, 2, 2)
    point_jacobian!(ode_jacobian, state, search.frozen_model, zero(eltype(state)))
    residual_ok = maximum(abs, residual) <= options.residual_atol
    jacobian_ok = maximum(abs, jacobian .- equilibrium.balance_jacobian) <=
        options.jacobian_atol
    spectral_ok = _safe_spectral_classification(ode_jacobian,
        equilibrium.stability.classification, options.spectral_margin)
    return residual_ok && jacobian_ok && spectral_ok && !equilibrium.near_singular
end

"""
    classify_figure5b_topology(searches; options=Figure5bTopologyOptions())

Recognize exactly seven matched roots with three attracting equilibria, three
saddles, and one repeller across three equilibrium-search refinements. The
unique repeller must lie on two locally rising nullcline arms. Nonqualifying
scientific results are returned with reasons rather than raised as errors.
"""
function classify_figure5b_topology(searches;
    options=Figure5bTopologyOptions())
    options isa Figure5bTopologyOptions ||
        throw(ArgumentError("options must be Figure5bTopologyOptions"))
    searches isa Tuple && length(searches) == 3 ||
        throw(ArgumentError("searches must be a tuple of three refinements"))
    reasons = Symbol[]
    all(search -> search isa EquilibriumSearchResult, searches) ||
        throw(ArgumentError("each refinement must be an EquilibriumSearchResult"))
    all(search -> _same_search_context(first(searches), search), searches) ||
        push!(reasons, :model_context_mismatch)
    all(_matches_refinement_schedule(search, grid_points)
        for (search, grid_points) in
            zip(searches, FIGURE5B_REFINEMENT_GRID_POINTS)) ||
        push!(reasons, :refinement_schedule_mismatch)
    all(search -> isempty(search.unresolved_nearby), searches) ||
        push!(reasons, :unresolved_nearby_roots)
    all(search -> length(search.equilibria) == 7, searches) ||
        push!(reasons, :root_count_mismatch)
    minimum_separation = minimum(_root_separation(search.equilibria) for search in searches)
    minimum_separation >= options.minimum_root_separation ||
        push!(reasons, :insufficient_root_separation)

    for search in searches, equilibrium in search.equilibria
        _recomputed_equilibrium_quality(search, equilibrium, options) ||
            push!(reasons, :root_quality_failure)
    end

    expected = Dict(Attracting => 3, Saddle => 3, Repelling => 1,
        StabilityUnresolved => 0)
    for search in searches
        counts = Dict(label => count(equilibrium ->
            equilibrium.stability.classification == label, search.equilibria)
            for label in keys(expected))
        counts == expected || push!(reasons, :stability_pattern_mismatch)
    end

    tracks = NTuple{3,Vector{Float64}}[]
    central_track = nothing
    central_state = nothing
    slopes = nothing
    if all(search -> length(search.equilibria) == 7, searches)
        reference = first(searches).equilibria
        second_mapping = _unique_root_mapping(reference, searches[2].equilibria,
            options.coordinate_match_atol)
        third_mapping = _unique_root_mapping(reference, searches[3].equilibria,
            options.coordinate_match_atol)
        if second_mapping === nothing || third_mapping === nothing
            push!(reasons, :ambiguous_root_matching)
        else
            for index in eachindex(reference)
                push!(tracks, (Float64.(reference[index].state),
                    Float64.(searches[2].equilibria[second_mapping[index]].state),
                    Float64.(searches[3].equilibria[third_mapping[index]].state)))
            end
            for index in eachindex(reference)
                classifications = (
                    reference[index].stability.classification,
                    searches[2].equilibria[second_mapping[index]].stability.classification,
                    searches[3].equilibria[third_mapping[index]].stability.classification,
                )
                all(==(first(classifications)), classifications) ||
                    push!(reasons, :classification_flip)
            end
            repellers = findall(equilibrium ->
                equilibrium.stability.classification == Repelling, reference)
            if length(repellers) == 1
                central_track = only(repellers)
                central_state = tracks[central_track][3]
                jacobian = zeros(Float64, 2, 2)
                point_balance_jacobian!(jacobian, central_state,
                    searches[3].frozen_model, zero(eltype(central_state)))
                if abs(jacobian[1, 2]) <= options.slope_atol ||
                        abs(jacobian[2, 2]) <= options.slope_atol
                    push!(reasons, :unresolved_nullcline_slope)
                else
                    candidate_slopes = (-jacobian[1, 1] / jacobian[1, 2],
                        -jacobian[2, 1] / jacobian[2, 2])
                    if all(value -> isfinite(value) && value > options.slope_atol,
                            candidate_slopes)
                        slopes = candidate_slopes
                    else
                        push!(reasons, :nonrising_nullcline_arm)
                    end
                end
            else
                push!(reasons, :central_repeller_not_unique)
            end
        end
    end
    unique!(reasons)
    return Figure5bTopologyResult(isempty(reasons), reasons, tracks,
        central_track, central_state, slopes, minimum_separation,
        FIGURE5B_REFINEMENT_GRID_POINTS)
end

"""
    HopfDiagnosticOptions(; kwargs...)

Fail-closed tolerances for local planar Hopf diagnostics. The automatic-
differentiation Lyapunov coefficient is cross-checked using an independently
finite-differenced Guckenheimer-Holmes planar formula over `fd_steps`.
Tolerances and steps use the same explicit `Float64` input contract as
`hopf_diagnostics` and must remain finite and positive after conversion.
"""
struct HopfDiagnosticOptions
    balance_residual_atol::Float64
    diagonal_atol::Float64
    determinant_atol::Float64
    transversality_atol::Float64
    lyapunov_atol::Float64
    lyapunov_agreement_atol::Float64
    lyapunov_agreement_rtol::Float64
    fd_plateau_atol::Float64
    fd_plateau_rtol::Float64
    fd_steps::Tuple{Vararg{Float64}}
end

function HopfDiagnosticOptions(;
    balance_residual_atol=1e-10,
    diagonal_atol=1e-10,
    determinant_atol=1e-10,
    transversality_atol=1e-8,
    lyapunov_atol=1e-7,
    lyapunov_agreement_atol=1e-4,
    lyapunov_agreement_rtol=1e-4,
    fd_plateau_atol=1e-4,
    fd_plateau_rtol=1e-4,
    fd_steps=(5e-4, 7.5e-4, 1e-3),
)
    scalars = (balance_residual_atol, diagonal_atol, determinant_atol,
        transversality_atol, lyapunov_atol, lyapunov_agreement_atol,
        lyapunov_agreement_rtol, fd_plateau_atol, fd_plateau_rtol)
    converted_scalars = _positive_float64_values(
        scalars, "Hopf diagnostic tolerances")
    fd_steps isa Tuple && length(fd_steps) >= 3 ||
        throw(ArgumentError("fd_steps must be a tuple containing at least three steps"))
    converted_steps = _positive_float64_values(
        fd_steps, "finite-difference steps")
    length(unique(converted_steps)) == length(converted_steps) ||
        throw(ArgumentError("finite-difference steps must be unique"))
    return HopfDiagnosticOptions(converted_scalars...,
        Tuple(sort(collect(converted_steps))))
end

"""
Local trace-zero and first-Lyapunov diagnostics. `classification` is one of
`:supercritical_candidate`, `:subcritical_candidate`, or `:unresolved` and is
not a certification that a periodic orbit exists.
"""
struct HopfDiagnosticResult
    resolved::Bool
    reasons::Vector{Symbol}
    critical_ratio::Union{Nothing,Float64}
    critical_tau_i::Union{Nothing,Float64}
    determinant::Union{Nothing,Float64}
    frequency::Union{Nothing,Float64}
    linear_period::Union{Nothing,Float64}
    transversality::Union{Nothing,Float64}
    nullcline_slopes::Union{Nothing,NTuple{2,Float64}}
    eigenvalues::Vector{ComplexF64}
    g21::Union{Nothing,ComplexF64}
    c1::Union{Nothing,ComplexF64}
    lyapunov_ad::Union{Nothing,Float64}
    lyapunov_fd::Union{Nothing,Float64}
    lyapunov_fd_values::Vector{Float64}
    modal_radius_squared_slope::Union{Nothing,Float64}
    classification::Symbol
end

function _derivative_tensors(vector_field, state)
    x = Float64.(state)
    A = ForwardDiff.jacobian(vector_field, x)
    hessians = [ForwardDiff.hessian(y -> vector_field(y)[component], x)
        for component in 1:2]
    thirds = zeros(Float64, 2, 2, 2, 2)
    for component in 1:2
        raw = ForwardDiff.jacobian(x0 -> vec(ForwardDiff.hessian(
            y -> vector_field(y)[component], x0)), x)
        for i in 1:2, j in 1:2, k in 1:2
            thirds[component, i, j, k] = raw[i + 2(j - 1), k]
        end
    end
    return A, hessians, thirds
end

function _kuznetsov_l1(vector_field, state, omega)
    A, hessians, thirds = _derivative_tensors(vector_field, state)
    decomposition = eigen(complex.(A))
    q_index = argmin(abs.(decomposition.values .- im * omega))
    q = decomposition.vectors[:, q_index]
    q ./= norm(q)
    adjoint_decomposition = eigen(adjoint(complex.(A)))
    p_index = argmin(abs.(adjoint_decomposition.values .+ im * omega))
    p0 = adjoint_decomposition.vectors[:, p_index]
    p = p0 / conj(dot(p0, q))
    B(u, v) = ComplexF64[sum(hessians[k][i, j] * u[i] * v[j]
        for i in 1:2, j in 1:2) for k in 1:2]
    C(u, v, w) = ComplexF64[sum(thirds[k, i, j, l] * u[i] * v[j] * w[l]
        for i in 1:2, j in 1:2, l in 1:2) for k in 1:2]
    h20 = (2im * omega * Matrix{ComplexF64}(I, 2, 2) - A) \ B(q, q)
    h11 = A \ B(q, conj.(q))
    g21 = dot(p, C(q, q, conj.(q)) + B(conj.(q), h20) - 2B(q, h11))
    return ComplexF64(g21), q
end

function _fd_weights(order)
    order == 0 && return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    nodes = Float64.(-3:3)
    vandermonde = [nodes[column]^(row - 1) for row in 1:7, column in 1:7]
    target = zeros(7)
    target[order + 1] = factorial(order)
    return vandermonde \ target
end

function _fd_partial(function_value, component, dx, dy, step)
    wx = _fd_weights(dx)
    wy = _fd_weights(dy)
    total = 0.0
    for (ix, nx) in enumerate(-3:3), (iy, ny) in enumerate(-3:3)
        total += wx[ix] * wy[iy] * function_value([step * nx, step * ny])[component]
    end
    return total / step^(dx + dy)
end

function _gh_l1(vector_field, state, step)
    x = Float64.(state)
    # This Jacobian and all nonlinear derivatives are finite-differenced and
    # do not reuse the ForwardDiff tensors or Kuznetsov resolvent contraction.
    A = [_fd_partial(z -> vector_field(x .+ z), component, dx, dy, step)
        for component in 1:2, (dx, dy) in ((1, 0), (0, 1))]
    decomposition = eigen(complex.(A))
    positive = findall(value -> imag(value) > 0, decomposition.values)
    length(positive) == 1 || return NaN
    q = decomposition.vectors[:, only(positive)]
    q ./= norm(q)
    omega = imag(decomposition.values[only(positive)])
    P = hcat(real.(q), -imag.(q))
    abs(det(P)) > eps(Float64) || return NaN
    transformed(z) = P \ vector_field(x .+ P * z)
    Fxx = _fd_partial(transformed, 1, 2, 0, step)
    Fyy = _fd_partial(transformed, 1, 0, 2, step)
    Fxy = _fd_partial(transformed, 1, 1, 1, step)
    Gxx = _fd_partial(transformed, 2, 2, 0, step)
    Gyy = _fd_partial(transformed, 2, 0, 2, step)
    Gxy = _fd_partial(transformed, 2, 1, 1, step)
    Fxxx = _fd_partial(transformed, 1, 3, 0, step)
    Fxyy = _fd_partial(transformed, 1, 1, 2, step)
    Gxxy = _fd_partial(transformed, 2, 2, 1, step)
    Gyyy = _fd_partial(transformed, 2, 0, 3, step)
    coefficient = (Fxxx + Fxyy + Gxxy + Gyyy) / 16 +
        (Fxy * (Fxx + Fyy) - Gxy * (Gxx + Gyy) - Fxx * Gxx + Fyy * Gyy) /
        (16 * omega)
    return 4 * coefficient / omega
end

function _unresolved_hopf(reasons; critical_ratio=nothing, critical_tau_i=nothing,
    determinant=nothing, frequency=nothing, linear_period=nothing,
    transversality=nothing, slopes=nothing, eigenvalues=ComplexF64[], g21=nothing,
    c1=nothing, lyapunov_ad=nothing, lyapunov_fd=nothing,
    lyapunov_fd_values=Float64[], radius_slope=nothing)
    unique!(reasons)
    return HopfDiagnosticResult(false, reasons, critical_ratio, critical_tau_i,
        determinant, frequency, linear_period, transversality, slopes,
        eigenvalues, g21, c1, lyapunov_ad, lyapunov_fd,
        lyapunov_fd_values, radius_slope, :unresolved)
end

function _hopf_float64(value, label)
    value isa Union{Integer,Rational,Float32,Float64} && !(value isa Bool) ||
        throw(ArgumentError("$label must use integers, rationals, Float32 or Float64; Hopf diagnostics use Float64"))
    isfinite(value) && isfinite(Float64(value)) ||
        throw(ArgumentError("$label must be finite"))
    value isa Union{Integer,Rational} && Float64(value) != value &&
        throw(ArgumentError("$label is not exactly representable by the Float64 Hopf implementation"))
    return Float64(value)
end

"""
    hopf_diagnostics(balance, state, tau_e; options=HopfDiagnosticOptions())
    hopf_diagnostics(model, state; time=0.0, options=HopfDiagnosticOptions())

Compute the trace-zero timescale ratio and local planar Hopf diagnostics while
holding the balance equations fixed. A resolved Lyapunov sign describes a
local Hopf candidate only; it neither discovers nor validates a periodic
orbit. Diagnostics use `Float64` arithmetic. State coordinates and `tau_e`
may be `Integer`, `Rational`, `Float32`, or `Float64`; integer and rational
values must be exactly representable as `Float64`. Higher-precision inputs,
including `BigFloat`, are rejected rather than silently narrowed.
"""
function hopf_diagnostics(balance, state, tau_e;
    options=HopfDiagnosticOptions())
    options isa HopfDiagnosticOptions ||
        throw(ArgumentError("options must be HopfDiagnosticOptions"))
    state isa AbstractVector && length(state) == 2 ||
        throw(ArgumentError("state must contain two coordinates"))
    state64 = [_hopf_float64(value, "state coordinate") for value in state]
    tau_e64 = _hopf_float64(tau_e, "tau_e")
    tau_e64 > 0 || throw(ArgumentError("tau_e must be positive"))
    sample = balance(state64)
    sample isa AbstractVector && length(sample) == 2 &&
        all(value -> value isa Real && isfinite(value), sample) ||
        throw(ArgumentError("balance must return two finite real values"))
    reasons = Symbol[]
    maximum(abs, sample) <= options.balance_residual_atol ||
        push!(reasons, :balance_residual_too_large)
    balance_jacobian = ForwardDiff.jacobian(balance, state64)
    all(isfinite, balance_jacobian) || push!(reasons, :nonfinite_balance_jacobian)
    b11, b22 = balance_jacobian[1, 1], balance_jacobian[2, 2]
    abs(b11) > options.diagonal_atol && abs(b22) > options.diagonal_atol ||
        push!(reasons, :unresolved_diagonal)
    critical_ratio = b11 == 0 ? nothing : -b22 / b11
    critical_ratio !== nothing && isfinite(critical_ratio) && critical_ratio > 0 ||
        push!(reasons, :invalid_critical_ratio)
    determinant = det(balance_jacobian)
    isfinite(determinant) && determinant > options.determinant_atol ||
        push!(reasons, :nonpositive_determinant)
    if !isempty(reasons)
        return _unresolved_hopf(reasons; critical_ratio,
            critical_tau_i=critical_ratio === nothing ? nothing : critical_ratio * tau_e64,
            determinant)
    end
    ratio = Float64(critical_ratio)
    critical_tau_i = ratio * tau_e64
    frequency = _scaled_hopf_frequency(determinant, ratio, tau_e64)
    period = 2pi / frequency
    # At trace zero, -b22 / ratio == b11. Use the critical inhibitory
    # timescale and divide b11 before the timescale to avoid an intermediate
    # overflow that subsequent ratio scaling would otherwise undo.
    transversality = _half_quotient(b11, critical_tau_i)
    isfinite(critical_tau_i) && critical_tau_i > 0 ||
        push!(reasons, :nonfinite_critical_tau_i)
    isfinite(frequency) && frequency > 0 || push!(reasons, :nonfinite_frequency)
    isfinite(period) && period > 0 || push!(reasons, :nonfinite_linear_period)
    isfinite(transversality) || push!(reasons, :nonfinite_transversality)
    if !isempty(reasons)
        return _unresolved_hopf(reasons; critical_ratio=ratio,
            critical_tau_i, determinant, frequency,
            linear_period=period, transversality)
    end
    abs(transversality) > options.transversality_atol ||
        push!(reasons, :unresolved_transversality)
    if abs(balance_jacobian[1, 2]) <= options.diagonal_atol ||
            abs(balance_jacobian[2, 2]) <= options.diagonal_atol
        slopes = nothing
        push!(reasons, :unresolved_nullcline_slope)
    else
        slopes = (-balance_jacobian[1, 1] / balance_jacobian[1, 2],
            -balance_jacobian[2, 1] / balance_jacobian[2, 2])
        all(isfinite, slopes) || push!(reasons, :nonfinite_nullcline_slope)
    end
    vector_field = x -> begin
        values = balance(x)
        [values[1] / tau_e64, values[2] / (ratio * tau_e64)]
    end
    A = ForwardDiff.jacobian(vector_field, state64)
    if !all(isfinite, A)
        return _unresolved_hopf(vcat(reasons, :nonfinite_ode_jacobian);
            critical_ratio=ratio, critical_tau_i, determinant, frequency,
            linear_period=period, transversality, slopes)
    end
    eigenvalues = ComplexF64.(eigen(complex.(A)).values)
    g21 = nothing
    c1 = nothing
    lyapunov_ad = nothing
    q = nothing
    try
        g21, q = _kuznetsov_l1(vector_field, state64, frequency)
        c1 = g21 / 2
        lyapunov_ad = real(c1) / frequency
        isfinite(g21) && isfinite(c1) && isfinite(lyapunov_ad) ||
            push!(reasons, :nonfinite_automatic_differentiation_lyapunov)
    catch error
        error isa InterruptException && rethrow()
        push!(reasons, :automatic_differentiation_failure)
    end
    fd_values = Float64[]
    for step in options.fd_steps
        value = try
            _gh_l1(vector_field, state64, step)
        catch error
            error isa InterruptException && rethrow()
            NaN
        end
        push!(fd_values, value)
    end
    all(isfinite, fd_values) || push!(reasons, :finite_difference_failure)
    lyapunov_fd = all(isfinite, fd_values) ? _scaled_mean(fd_values) : nothing
    if lyapunov_fd !== nothing
        if !isfinite(lyapunov_fd)
            push!(reasons, :nonfinite_finite_difference_lyapunov)
        end
        _finite_difference_plateau(fd_values,
            options.fd_plateau_atol, options.fd_plateau_rtol) ||
            push!(reasons, :finite_difference_no_plateau)
    end
    if lyapunov_ad === nothing || !isfinite(lyapunov_ad) ||
            lyapunov_fd === nothing || !isfinite(lyapunov_fd) ||
            abs(lyapunov_ad) <= options.lyapunov_atol ||
            abs(lyapunov_fd) <= options.lyapunov_atol
        push!(reasons, :lyapunov_near_zero)
    else
        sign(lyapunov_ad) == sign(lyapunov_fd) ||
            push!(reasons, :lyapunov_sign_disagreement)
        isapprox(lyapunov_ad, lyapunov_fd;
            atol=options.lyapunov_agreement_atol,
            rtol=options.lyapunov_agreement_rtol) ||
            push!(reasons, :lyapunov_magnitude_disagreement)
    end
    if !isempty(reasons)
        return _unresolved_hopf(reasons; critical_ratio=ratio,
            critical_tau_i, determinant, frequency,
            linear_period=period, transversality, slopes, eigenvalues,
            g21, c1, lyapunov_ad, lyapunov_fd, lyapunov_fd_values=fd_values)
    end
    radius_slope = -transversality / real(c1)
    radius_reason = if !isfinite(radius_slope)
        :nonfinite_modal_radius_squared_slope
    elseif iszero(radius_slope) && !iszero(transversality) && !iszero(real(c1))
        :underflowed_modal_radius_squared_slope
    else
        nothing
    end
    if radius_reason !== nothing
        return _unresolved_hopf([radius_reason];
            critical_ratio=ratio, critical_tau_i, determinant, frequency,
            linear_period=period, transversality, slopes, eigenvalues,
            g21, c1, lyapunov_ad, lyapunov_fd,
            lyapunov_fd_values=fd_values, radius_slope)
    end
    classification = lyapunov_ad < 0 ?
        :supercritical_candidate : :subcritical_candidate
    return HopfDiagnosticResult(true, Symbol[], ratio, critical_tau_i,
        determinant, frequency, period, transversality, slopes, eigenvalues,
        g21, c1, lyapunov_ad, lyapunov_fd, fd_values, radius_slope,
        classification)
end

function hopf_diagnostics(model::PointModelParameters, state;
    time=0.0, options=HopfDiagnosticOptions())
    time isa Real && !(time isa Bool) && isfinite(time) ||
        throw(ArgumentError("time must be finite and real"))
    balance = x -> begin
        values = similar(x, 2)
        point_balance!(values, x, model, time)
        values
    end
    return hopf_diagnostics(balance, state, model.excitatory.timescale; options)
end

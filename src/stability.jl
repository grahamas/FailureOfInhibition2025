using LinearAlgebra: det, eigvals, tr

"""Local linear stability labels for a two-dimensional equilibrium."""
@enum StabilityClassification begin
    Attracting
    Repelling
    Saddle
    StabilityUnresolved
end

"""Spectral geometry reported independently of local stability."""
@enum SpectralGeometry begin
    RealDistinctSpectrum
    RealRepeatedSpectrum
    ComplexConjugateSpectrum
end

"""
    StabilityOptions(; spectral_atol=1e-10, spectral_rtol=1e-8)

Tolerances for local spectral classification. `spectral_atol` has units of
inverse milliseconds and `spectral_rtol` is dimensionless. An eigenvalue real
part is resolved only when its magnitude exceeds
`spectral_atol + spectral_rtol * abs(eigenvalue)`. The largest resulting
threshold is also used for spectral geometry: imaginary parts no larger than
that threshold are treated as numerically real, after which the real parts
determine whether the spectrum is repeated or distinct.
"""
struct StabilityOptions{T<:AbstractFloat}
    spectral_atol::T
    spectral_rtol::T
end

function StabilityOptions(; spectral_atol=1.0e-10, spectral_rtol=1.0e-8)
    values = (spectral_atol, spectral_rtol)
    all(value -> value isa Real, values) ||
        throw(ArgumentError("spectral tolerances must be real"))
    all(isfinite, values) || throw(ArgumentError("spectral tolerances must be finite"))
    all(value -> value >= zero(value), values) ||
        throw(ArgumentError("spectral tolerances must be nonnegative"))
    promoted = promote(float.(values)...)
    return StabilityOptions(promoted...)
end

"""
    LocalStabilityResult

Raw local linear diagnostics for a two-dimensional Jacobian. `jacobian`,
`eigenvalues`, `trace`, and `spectral_abscissa` use inverse-millisecond units;
`determinant` uses inverse-square-millisecond units. `thresholds` contains the
per-eigenvalue resolution thresholds. `geometry` is reported separately from
`classification`.
"""
struct LocalStabilityResult{T<:AbstractFloat}
    jacobian::Matrix{T}
    eigenvalues::Vector{Complex{T}}
    trace::T
    determinant::T
    spectral_abscissa::T
    thresholds::Vector{T}
    classification::StabilityClassification
    geometry::SpectralGeometry
end

function _require_supported_analysis_type(::Type{T}) where {T}
    T === Float32 || T === Float64 || throw(
        ArgumentError(
            "equilibrium and stability analysis supports Float32 and Float64; " *
            "arbitrary-precision eigensolvers are not enabled",
        ),
    )
    return T
end

"""
    classify_local_stability(jacobian; options=StabilityOptions())

Classify a finite real `2x2` Jacobian as `Attracting`, `Repelling`, `Saddle`,
or `StabilityUnresolved`. Repeated eigenvalues do not by themselves make the
classification unresolved. The result does not establish global stability,
a center, a Hopf bifurcation, or a periodic orbit.
"""
function classify_local_stability(jacobian; options=StabilityOptions())
    options isa StabilityOptions ||
        throw(ArgumentError("options must be StabilityOptions"))
    jacobian isa AbstractMatrix || throw(ArgumentError("jacobian must be a 2x2 matrix"))
    size(jacobian) == (2, 2) || throw(ArgumentError("jacobian must be a 2x2 matrix"))
    all(value -> value isa Real && isfinite(value), jacobian) ||
        throw(ArgumentError("jacobian entries must be finite and real"))

    floating = float.(jacobian)
    T = _require_supported_analysis_type(
        promote_type((typeof(value) for value in floating)...),
    )
    matrix = Matrix{T}(floating)
    values = Complex{T}.(eigvals(matrix))
    sort!(values; by=value -> (real(value), imag(value)))
    thresholds = T[
        convert(T, options.spectral_atol) +
        convert(T, options.spectral_rtol) * abs(value) for value in values
    ]
    signs = map(zip(values, thresholds)) do (value, threshold)
        real(value) < -threshold ? -1 : real(value) > threshold ? 1 : 0
    end
    classification = if all(==(-1), signs)
        Attracting
    elseif all(==(1), signs)
        Repelling
    elseif sort(signs) == [-1, 1]
        Saddle
    else
        StabilityUnresolved
    end

    geometry_tolerance = maximum(thresholds)
    geometry = if maximum(abs ∘ imag, values) > geometry_tolerance
        ComplexConjugateSpectrum
    elseif abs(real(values[2]) - real(values[1])) <= geometry_tolerance
        RealRepeatedSpectrum
    else
        RealDistinctSpectrum
    end

    return LocalStabilityResult(
        matrix,
        values,
        tr(matrix),
        det(matrix),
        maximum(real, values),
        thresholds,
        classification,
        geometry,
    )
end

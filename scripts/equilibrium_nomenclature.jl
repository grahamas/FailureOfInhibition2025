"""Exploratory activity definitions. No defaults are biological classifications."""
module EquilibriumNomenclature

"""Explicit occupancy cutoffs; independent boundaries may overlap or leave gaps."""
struct Boundaries
    rest_E::Float64
    rest_I::Float64
    seizure_E::Float64
    seizure_I::Float64
    herald_E::Float64
    herald_I::Float64
    function Boundaries(values::Vararg{Real,6})
        all(x -> isfinite(x) && 0 <= x <= .5, values) || throw(ArgumentError("occupancy cutoffs must be finite and in [0, 0.5]"))
        new(Float64.(values)...)
    end
end

"""Return independent geometric/activity memberships, without stability or precedence.

Rest uses inclusive low boundaries; elevated E and partial-failure I use strict
lower boundaries. Herald includes its high boundaries. An inhibitory-response
peak is not a failure arm. Stability must be reported separately by consumers.
"""
function memberships(E, I, side, b::Boundaries)
    isfinite(E) && isfinite(I) && 0 <= E <= .5 && 0 <= I <= .5 || throw(ArgumentError("invalid equilibrium occupancy"))
    side in ("ascending", "descending", "peak") || throw(ArgumentError("unknown inhibitory response side"))
    (;rest=E <= b.rest_E && I <= b.rest_I,
      seizure=E > b.seizure_E && I <= b.seizure_I && side == "descending",
      herald=E >= b.herald_E && I >= b.herald_I,
      partial_failure=E >= b.herald_E && b.seizure_I < I < b.herald_I && side == "descending")
end

"""Observed-value empirical quantile (inverse ECDF, no interpolation)."""
function observed_quantile(sorted_values, q)
    isempty(sorted_values) && throw(ArgumentError("empty empirical distribution"))
    0 <= q <= 1 || throw(ArgumentError("invalid quantile"))
    sorted_values[max(1, ceil(Int, q * length(sorted_values)))]
end

"""Exact rectangular counts at sorted candidate boundaries in O(N log K + K²)."""
function lower_counts(E, I, xs, ys)
    length(E) == length(I) || throw(DimensionMismatch("coordinates differ"))
    issorted(xs) && issorted(ys) || throw(ArgumentError("unsorted boundaries"))
    h = zeros(Int, length(xs), length(ys))
    for (e, i) in zip(E, I)
        x, y = searchsortedfirst(xs, e), searchsortedfirst(ys, i)
        x <= length(xs) && y <= length(ys) && (h[x, y] += 1)
    end
    cumsum(cumsum(h; dims=1); dims=2)
end
end

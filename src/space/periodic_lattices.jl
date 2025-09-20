"""
    AbstractPeriodicLattice{T,N_ARR,N_CDT} <: AbstractLattice{T,N_ARR,N_CDT}

Abstract type for lattices with periodic (wrapping) boundary conditions.

Periodic lattices represent spatial domains where the boundaries wrap around,
creating a toroidal topology. This is useful for simulations where you want
to avoid edge effects or when modeling systems with natural periodicity.

# Type Parameters
- `T`: Numeric type for coordinates (e.g., Float64)
- `N_ARR`: Array dimensionality
- `N_CDT`: Coordinate dimensionality
"""
abstract type AbstractPeriodicLattice{T,N_ARR,N_CDT} <: AbstractLattice{T,N_ARR,N_CDT} end

"""
    PeriodicLattice{T,N_ARR} <: AbstractPeriodicLattice{T,N_ARR,N_ARR}

A concrete lattice type for spatial domains with periodic boundary conditions.

# Fields
- `arr::Array{NTuple{N_ARR,T},N_ARR}`: Array storing coordinate tuples at each grid point
- `extent::NTuple{N_ARR,T}`: The period/extent along each dimension

# Boundary Conditions
Uses periodic distance calculations where the shortest path between two points
may wrap around the boundary. For example, in a 1D periodic domain of length L,
the distance between points at positions 0.1 and 0.9 could be either 0.8 
(direct path) or 0.2 (wrapping around).

# Construction
```julia
# Create a 2D periodic lattice with extent (2π, 2π) and 32x32 points
lattice = PeriodicLattice(extent=(2π, 2π), n_points=(32, 32))
```

# Applications
- Fourier analysis (natural for FFTs)
- Fluid dynamics simulations
- Crystallography and solid state physics
- Any system where boundary effects should be minimized
"""
struct PeriodicLattice{T,N_ARR} <: AbstractPeriodicLattice{T,N_ARR,N_ARR}
    arr::Array{NTuple{N_ARR,T},N_ARR}
    extent::NTuple{N_ARR,T}
end
# Constructor methods for PeriodicLattice
"""
    PeriodicLattice(; extent, n_points)

Construct a periodic lattice with specified extent and number of points.

# Arguments
- `extent`: Period/total length along each dimension
- `n_points`: Number of discretization points along each dimension

# Note
The lattice points are placed such that the last point does not coincide with
the first point (since they are equivalent due to periodicity). The actual
spacing is extent/n_points, not extent/(n_points-1) as in compact lattices.
"""
(t::Type{<:PeriodicLattice})(; extent, n_points) =_periodiclattice(t, extent, n_points)

"""
    _periodiclattice(t, extent::Number, n_points::Number)

Internal helper for 1D periodic lattice construction.
"""
function _periodiclattice(t, extent::Number, n_points::Number)
    _periodiclattice(t, (extent,), (n_points,))
end

"""
    _periodiclattice(t, extent::Tuple, n_points::Tuple)

Internal helper for multi-dimensional periodic lattice construction.
Creates coordinates that span [-extent/2, extent/2) with the specified spacing.
"""
function _periodiclattice(t, extent::Tuple, n_points::Tuple)
    t(discrete_lattice(.-extent ./ 2, extent ./ 2 .- (extent ./ n_points), n_points), extent)
end

"""
    extent(p::PeriodicLattice)

Return the period/extent of the periodic lattice along each dimension.
"""
extent(p::PeriodicLattice) = p.extent

"""
    difference(p_lattice::PeriodicLattice, edge)

Compute distance between two points using periodic metric.
This automatically accounts for wrapping - returns the shortest distance
considering all possible paths through the periodic boundaries.
"""
difference(p_lattice::PeriodicLattice, edge) = abs_difference_periodic(edge, extent(p_lattice))

# Integration weights for periodic lattices
export simpson_weights

"""
    simpson_weights(lattice::PeriodicLattice{T}) where T

Compute integration weights for periodic lattices.

For periodic domains, all points are equivalent due to the periodic boundary
conditions, so uniform weighting is appropriate. Unlike compact lattices,
there are no special boundary conditions to consider.

# Returns
Array of uniform weights (all ones) with the same shape as the lattice.

# Note
TODO: Should be completely uniform in periodic case? 
Currently returns uniform weights, which is mathematically correct for
periodic boundary conditions.
"""
function simpson_weights(lattice::PeriodicLattice{T}) where T
    w = ones(T, size(lattice)...)
    return w
end
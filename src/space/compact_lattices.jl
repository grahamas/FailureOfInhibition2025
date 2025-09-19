"""
    AbstractCompactLattice{T,N_ARR,N_CDT} <: AbstractLattice{T,N_ARR,N_CDT}

Abstract type for lattices with compact (finite) boundary conditions.

Compact lattices represent finite spatial domains where the boundaries are "hard" - 
there is no wrapping or special boundary behavior. Distances are computed using 
standard Euclidean metrics without any topological considerations.

# Type Parameters
- `T`: Numeric type for coordinates (e.g., Float64)
- `N_ARR`: Array dimensionality 
- `N_CDT`: Coordinate dimensionality
"""
abstract type AbstractCompactLattice{T,N_ARR,N_CDT} <: AbstractLattice{T,N_ARR,N_CDT} end

"""
    CompactLattice{T,N_ARR} <: AbstractCompactLattice{T,N_ARR,N_ARR}

A concrete lattice type for finite spatial domains with regular grid spacing.

# Fields
- `arr::Array{NTuple{N_ARR,T},N_ARR}`: Array storing coordinate tuples at each grid point

# Construction
Can be constructed using the standard AbstractLattice constructors:
- `CompactLattice(discrete_lattice(start, stop, n_points))`
- `CompactLattice(extent=extent, n_points=n_points)`

# Boundary Conditions
Uses regular Euclidean distance calculations without any wrapping or special 
boundary treatment. Suitable for problems on finite domains.

# Example
```julia
# Create a 2D compact lattice from (-1,-1) to (1,1) with 11x11 points
lattice = CompactLattice(discrete_lattice((-1.0, -1.0), (1.0, 1.0), (11, 11)))

# Or equivalently, using extent notation:
lattice = CompactLattice(extent=(2.0, 2.0), n_points=(11, 11))
```
"""
struct CompactLattice{T,N_ARR} <: AbstractCompactLattice{T,N_ARR,N_ARR}
    arr::Array{NTuple{N_ARR,T},N_ARR}
end

"""
    difference(lattice::AbstractCompactLattice, edge)

Compute distance between two points using standard Euclidean metric.
For compact lattices, this simply calls `abs_difference` without any 
boundary considerations.
"""
difference(lattice::AbstractCompactLattice, edge) = abs_difference(edge)

# Integration weights for numerical methods
export simpson_weights

"""
    simpson_weights(lattice::CompactLattice{T,1}) where T

Compute Simpson's rule integration weights for 1D compact lattices.

Simpson's rule provides higher-order accuracy for numerical integration
compared to simple trapezoidal rule. For 1D, the weights alternate between
1, 4, 2, 4, 2, ..., 4, 1 with appropriate scaling.

# Requirements
- The lattice must have an odd number of points for proper Simpson's rule application

# Returns
Array of weights with the same shape as the lattice, where boundary points
get weight 0.5 and interior points get weight 1.0.

# Note
TODO: Should be completely uniform in periodic case?
"""
function simpson_weights(lattice::CompactLattice{T,1}) where T
    @assert all(size(lattice) .% 2 .== 1)
    w = ones(T, size(lattice)...)
    w[1] = 0.5
    w[end] = 0.5
    return w
end

"""
    simpson_weights(lattice::CompactLattice{T,2}) where T

Compute Simpson's rule integration weights for 2D compact lattices.

Extends Simpson's rule to 2D by applying the 1D weights along each dimension.
This follows the standard 2D Simpson's rule implementation.

# Requirements  
- The lattice must have an odd number of points along each dimension

# Returns
2D array of weights where the weights follow the 2D Simpson's pattern:
interior points get weights that are products of the 1D weights.

# Reference
http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html
"""
function simpson_weights(lattice::CompactLattice{T,2}) where T
    # http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html
    @assert all(size(lattice) .% 2 .== 1)
    w = ones(T, size(lattice)...)
    w[2:2:end-1,:] .*= 4.0
    w[3:2:end-2,:] .*= 2.0
    w[:,2:2:end-1] .*= 4.0
    w[:,3:2:end-2] .*= 2.0
    return w
end

"""
    getindex(lat::CompactLattice, dx)

Access coordinate at the given index in the lattice.
"""
Base.getindex(lat::CompactLattice, dx) = getindex(lat.arr, dx)
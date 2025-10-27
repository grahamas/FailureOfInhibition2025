"""
    AbstractPointLattice{T} <: AbstractLattice{T,0,0}

Abstract type for point lattices (zero-dimensional spaces).

A point lattice represents a single point with no spatial extent.
This is useful for non-spatial models where you want to use the same
functions (like wcm1973!) that work with spatial lattices, but without
any spatial structure.
"""
abstract type AbstractPointLattice{T} <: AbstractLattice{T,0,0} end

"""
    PointLattice{T} <: AbstractPointLattice{T}

A concrete lattice type representing a single point (zero-dimensional space).

This allows using spatial model functions like wcm1973! for non-spatial
models where there is no spatial structure, just population dynamics.

# Fields
- `arr::Array{Tuple{},0}`: A 0-dimensional array containing an empty tuple

# Construction
```julia
lattice = PointLattice()  # Default Float64
lattice = PointLattice{Float32}()  # Specify type
```

# Example
```julia
# Create a point lattice
lattice = PointLattice()

# Use with Wilson-Cowan model for non-spatial dynamics
params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (1.0, 0.8),
    connectivity = nothing,  # No spatial connectivity
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Activity state for 2 populations (no spatial dimension)
A = [0.3, 0.5]  # Shape: (2,) for 2 populations
dA = zeros(2)
wcm1973!(dA, A, params, 0.0)
```
"""
struct PointLattice{T} <: AbstractPointLattice{T}
    arr::Array{Tuple{},0}
    
    function PointLattice{T}() where {T}
        # Create a 0-dimensional array containing an empty tuple
        arr = fill((), ())
        new{T}(arr)
    end
end

# Default constructor uses Float64
PointLattice() = PointLattice{Float64}()

"""
    size(lattice::PointLattice)

Return the dimensions of the point lattice, which is an empty tuple ().
"""
Base.size(lattice::PointLattice) = ()

"""
    size(lattice::PointLattice, d::Int)

Return the size along dimension d. For a point lattice, throws an error
since there are no dimensions.
"""
Base.size(lattice::PointLattice, d::Int) = throw(BoundsError("PointLattice has no dimensions"))

"""
    zeros(lattice::PointLattice{T})

Create a zero scalar value (not an array) for a point lattice.
Since there's no spatial structure, this returns 0 of type T.
"""
Base.zeros(lattice::PointLattice{T}) where T = zero(T)

"""
    step(lattice::PointLattice)

Return the spacing between points. For a point lattice, this is an empty tuple
since there is no spatial extent.
"""
Base.step(lattice::PointLattice) = ()

"""
    start(lattice::PointLattice)

Return the starting coordinates of the point lattice (empty tuple).
"""
start(lattice::PointLattice) = ()

"""
    stop(lattice::PointLattice)

Return the ending coordinates of the point lattice (empty tuple).
"""
stop(lattice::PointLattice) = ()

"""
    extent(lattice::PointLattice)

Return the extent of the point lattice (empty tuple, as there is no spatial extent).
"""
extent(lattice::PointLattice) = ()

"""
    coordinates(lattice::PointLattice)

Return the coordinates of the point lattice.
For a point lattice, this is the 0-dimensional array containing an empty tuple.
"""
coordinates(lattice::PointLattice) = lattice.arr

"""
    coordinate_axes(lattice::PointLattice)

Return the coordinate axes. For a point lattice, this is an empty tuple.
"""
coordinate_axes(lattice::PointLattice) = ()

"""
    CartesianIndices(lattice::PointLattice)

Return CartesianIndices for the point lattice.
For a 0-dimensional array, this returns a single CartesianIndex().
"""
Base.CartesianIndices(lattice::PointLattice) = CartesianIndices(lattice.arr)

"""
    difference(lattice::AbstractPointLattice, edge)

Compute distance between two points in a point lattice.
Since there's only one point, the distance is always the empty tuple.
"""
difference(lattice::AbstractPointLattice, edge) = ()

"""
    getindex(lat::PointLattice, dx)

Access the coordinate at the given index.
For a point lattice, only the empty index () is valid.
"""
Base.getindex(lat::PointLattice, dx) = getindex(lat.arr, dx)

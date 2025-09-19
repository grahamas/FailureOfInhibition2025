"""
    AbstractLattice{T,N_ARR,N_CDT} <: AbstractSpace{T,N_ARR,N_CDT}

Abstract type for lattice-based spatial discretizations.

# Type Parameters
- `T`: The numeric type used for coordinates and distances (e.g., Float64)
- `N_ARR`: The dimensionality of the array storage (e.g., 2 for 2D grids)  
- `N_CDT`: The dimensionality of the coordinate tuples (e.g., 2 for (x,y) coordinates)

Lattices are spatial discretizations where points are arranged in a regular grid pattern.
Different lattice types can have different boundary conditions (compact vs periodic).
"""
abstract type AbstractLattice{T,N_ARR,N_CDT} <: AbstractSpace{T,N_ARR,N_CDT} end

"""
    (::Type{<:AbstractLattice})(start, stop, n_points)

Construct a lattice with explicit start and stop coordinates.

# Arguments
- `start`: Starting coordinates for each dimension
- `stop`: Ending coordinates for each dimension  
- `n_points`: Number of points along each dimension
"""
(t::Type{<:AbstractLattice})(start, stop, n_points) = t(discrete_lattice(start, stop, n_points))

"""
    (::Type{<:AbstractLattice})(; extent, n_points)

Construct a lattice centered at origin with specified extent.

# Arguments  
- `extent`: Total length along each dimension
- `n_points`: Number of points along each dimension

The lattice will be centered at the origin, spanning from -extent/2 to +extent/2.
"""
(t::Type{<:AbstractLattice})(; extent, n_points) = t(.-extent ./ 2, extent ./ 2, n_points)

# Base interface implementations for AbstractLattice
"""
    CartesianIndices(lattice::AbstractLattice)

Return CartesianIndices for indexing into the lattice array.
"""
Base.CartesianIndices(lattice::AbstractLattice) = CartesianIndices(lattice.arr)

"""
    step(space::AbstractLattice)

Return the spacing between adjacent points along each dimension.
"""
Base.step(space::AbstractLattice) = extent(space) ./ (size(space) .- 1)

"""
    size(lattice::AbstractLattice)

Return the dimensions of the lattice as a tuple.
"""
Base.size(lattice::AbstractLattice) = size(lattice.arr)

"""
    size(lattice::AbstractLattice, d::Int)

Return the size along dimension d.
"""
Base.size(lattice::AbstractLattice, d::Int) = size(lattice.arr, d)

"""
    zeros(lattice::AbstractLattice{T})

Create a zero array with the same dimensions and element type as the lattice.
"""
Base.zeros(lattice::AbstractLattice{T}) where T = zeros(T,size(lattice)...)

"""
    start(space::AbstractLattice)

Return the starting coordinates of the lattice (minimum corner).
"""
start(space::AbstractLattice) = space.arr[1]

"""
    stop(space::AbstractLattice)

Return the ending coordinates of the lattice (maximum corner).
"""
stop(space::AbstractLattice) = space.arr[end]

"""
    extent(space::AbstractLattice)

Return the total extent (size) of the lattice along each dimension.
"""
extent(space::AbstractLattice) = stop(space) .- start(space)


"""
    discrete_segment(start::T, stop::T, n_points::Int) where {T <: Number}

Create a discrete segment with `n_points` evenly spaced coordinates from `start` to `stop`.

# Arguments
- `start`: Starting coordinate
- `stop`: Ending coordinate  
- `n_points`: Number of points to include

# Returns
A LinRange object containing the discretized coordinates.

# Example
```jldoctest
julia> seg = discrete_segment(0.0, 5.0, 7);

julia> length(seg) == 7
true

julia> seg[end] - seg[1] ≈ 5.0
true
```
"""
function discrete_segment(start::T, stop::T, n_points::Int) where {T <: Number}
    LinRange{T}(start, stop, n_points)
end
# function discrete_segment(extent::T, n_points::Int) where {T <: Number}
#     discrete_segment(-extent/2,extent/2,n_points)
# end
"""
    discrete_lattice(start::Tup, stop::Tup, n_points::IntTup)

Create a discrete lattice (multidimensional grid) of coordinates.

# Arguments
- `start`: Starting coordinates for each dimension as a tuple
- `stop`: Ending coordinates for each dimension as a tuple
- `n_points`: Number of points along each dimension as a tuple

# Returns
An array where each element is a coordinate tuple, organized in a grid structure.
The array has the same dimensionality as specified by `n_points`.

# Example
```julia
# Create a 2D grid from (-1,-1) to (1,1) with 5x5 points
lattice = discrete_lattice((-1.0, -1.0), (1.0, 1.0), (5, 5))
```
"""
function discrete_lattice(start::Tup, stop::Tup, n_points::IntTup) where {Nminusone,T,Tup<:Tuple{T,Vararg{T,Nminusone}},IT<:Int,IntTup<:Tuple{IT,Vararg{IT,Nminusone}}}
    Iterators.product(discrete_segment.(start, stop, n_points)...) |> collect
end

"""
    discrete_lattice(start::T, stop::T, n_points::Int) where T

Create a 1D discrete lattice (convenience method for 1D case).
"""
discrete_lattice(start::T, stop::T, n_points::Int) where T = discrete_lattice((start,), (stop,), (n_points,))

"""
    coordinates(lattice::AbstractLattice)

Return the coordinate array of the lattice.
Each element in the returned array contains the spatial coordinates of that lattice point.
"""
coordinates(lattice::AbstractLattice) = lattice.arr

"""
    coordinate_axes(lattice::AbstractLattice)

Return the coordinate axes as separate 1D ranges for each dimension.
This is useful for plotting or when you need the axes separately rather than as a grid.
"""
coordinate_axes(lattice::AbstractLattice) = (discrete_segment.(start(lattice), stop(lattice), size(lattice))...,)

using Statistics
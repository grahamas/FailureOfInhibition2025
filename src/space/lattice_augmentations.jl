"""
    AbstractAugmentedLattice{T,N_ARR,N_CDT,L} <: AbstractLattice{T,N_ARR,N_CDT}

Abstract type for lattices that have been augmented with additional structure.

# Type Parameters
- `T`: Numeric type for coordinates  
- `N_ARR`: Array dimensionality of the base lattice
- `N_CDT`: Coordinate dimensionality (may be higher than N_ARR for embedded lattices)
- `L`: Type of the underlying base lattice

# Purpose
FIXME: The exact purpose and use cases of augmented lattices are unclear from the
original Simulation73 code. They appear to be a mechanism for extending lattices
with additional spatial dimensions or structure, but the specific applications
need clarification.

# Note  
This is adapted from Simulation73 but the original design intent is not fully
documented. Consider clarifying the intended use cases.
"""
abstract type AbstractAugmentedLattice{T,N_ARR,N_CDT,L} <: AbstractLattice{T,N_ARR,N_CDT} end

"""
    AbstractEmbeddedLattice{T,N_ARR,N_CDT,L} <: AbstractAugmentedLattice{T,N_ARR,N_CDT,L}

Abstract type for lattices embedded in higher-dimensional spaces.

An embedded lattice takes a base lattice and embeds it into a higher-dimensional
coordinate space. For example, you might embed a 2D lattice into 3D space by
adding z-coordinates to each (x,y) point.

# Type Parameters
- `T`: Numeric type for coordinates
- `N_ARR`: Array dimensionality of the base lattice  
- `N_CDT`: Total coordinate dimensionality after embedding (N_CDT > N_ARR)
- `L`: Type of the underlying base lattice

# Purpose
FIXME: The specific use cases for embedded lattices are not well documented.
Possible applications might include:
- Embedding lower-dimensional dynamics in higher-dimensional spaces
- Creating coordinate mappings for visualization  
- Handling systems with mixed dimensionality
More documentation needed about when and why to use this.
"""
abstract type AbstractEmbeddedLattice{T,N_ARR,N_CDT,L} <: AbstractAugmentedLattice{T,N_ARR,N_CDT,L} end

# Interface methods for embedded lattices
"""
    coordinates(lattice::AbstractEmbeddedLattice)

Return the embedded coordinates of the lattice.

For embedded lattices, this returns the higher-dimensional coordinates rather
than the base lattice coordinates. Each coordinate tuple has N_CDT elements
instead of N_ARR elements.
"""
function coordinates(lattice::AbstractEmbeddedLattice)
    lattice.coordinates
end

"""
    size(lattice::AbstractEmbeddedLattice)

Return the array dimensions of the underlying base lattice.

The embedded lattice maintains the same array structure as its base lattice,
only the coordinate dimensionality changes.
"""
function size(lattice::AbstractEmbeddedLattice)
    size(lattice.lattice)
end

"""
    difference(aug_lattice::AbstractEmbeddedLattice, edge::Tuple{PT,PT})

Compute distance between two points in an embedded lattice.

For embedded lattices, the distance calculation is split into two parts:
1. Distance along the base lattice dimensions (using base lattice metrics)
2. Distance along the embedded dimensions (using embedded lattice metrics)

# Arguments
- `aug_lattice`: The embedded lattice
- `edge`: Tuple of two coordinate tuples to measure distance between

# Returns
Combined distance tuple with components from both base and embedded spaces

# Implementation Note
FIXME: The coordinate splitting logic (edge_first_dims, edge_trailing_dims) 
assumes a specific embedding structure that may not be general. The relationship
between L_N_CDT and the actual coordinate structure needs clarification.
"""
function difference(aug_lattice::AbstractEmbeddedLattice{T,N_ARR,N_CDT,L},
                    edge::Tuple{PT,PT}) where {T,N_ARR,N_CDT,L_N_CDT,
                                               L<:AbstractLattice{T,N_ARR,L_N_CDT},
                                               PT<:NTuple{N_CDT,T}
                                               }
    edge_first_dims = (edge[1][1:L_N_CDT], edge[2][1:L_N_CDT])
    edge_trailing_dims = (edge[1][L_N_CDT+1:end], edge[2][L_N_CDT+1:end])
    return (difference(aug_lattice.lattice, edge_first_dims)...,
        difference(aug_lattice.embedded_lattice, edge_trailing_dims)...)
end

"""
    step(aug_lattice::AbstractEmbeddedLattice)

Return the step size for an embedded lattice.

Combines step sizes from both the base lattice and the embedded lattice dimensions.

# Returns
Tuple containing step sizes for all dimensions (base + embedded)
"""
function Base.step(aug_lattice::AbstractEmbeddedLattice)
    (step(aug_lattice.lattice)..., step(aug_lattice.embedded_lattice)...)
end

"""
    unembed_values(lattice::AbstractEmbeddedLattice, values::AbstractArray)

Extract values corresponding to the embedded lattice coordinates.

# Arguments
- `lattice`: The embedded lattice
- `values`: Array of values defined on the base lattice

# Returns
Array of values reorganized according to the embedded coordinate structure

# Purpose
FIXME: The purpose and correct usage of this function is unclear. It appears to
reorganize data from the base lattice according to the embedding structure, but:
1. The algorithm uses `findall` with approximate matching which is inefficient
2. The mapping between base and embedded coordinates is not well-defined
3. The intended use case is not documented

This function may be intended for data analysis or visualization, but needs
clarification and possibly a more efficient implementation.
"""
function unembed_values(lattice::AbstractEmbeddedLattice{T,N_ARR,N_CDT}, values::AbstractArray{T,N_ARR}) where {T,N_ARR,N_CDT}
    inner_coords = [coord[N_ARR+1:end] for coord in coordinates(lattice)]
    return [values[findall(map((x) -> all(isapprox.(embedded_coord, x)), inner_coords))] for embedded_coord in coordinates(lattice.embedded_lattice)]
end

"""
    linear_next(num::Int)

Simple utility function to increment an integer.

# Purpose
FIXME: This function appears to be a utility for some plotting or indexing
functionality that was commented out. Its current purpose is unclear and it
may be vestigial code from the original Simulation73 implementation.
"""
linear_next(num::Int) = num + 1

"""
    RandomlyEmbeddedLattice{T,N_ARR,N_CDT,L,E} <: AbstractEmbeddedLattice{T,N_ARR,N_CDT,L}

A concrete implementation of an embedded lattice where the embedding coordinates are random.

# Fields
- `lattice::L`: The base lattice  
- `embedded_lattice::E`: The lattice defining the embedding space
- `coordinates::Array{NTuple{N_CDT,T},N_ARR}`: The combined coordinates after embedding

# Purpose
FIXME: The purpose of random embedding is not well documented. Possible uses:
- Adding noise or randomness to spatial coordinates
- Creating irregular lattices for testing algorithms
- Modeling systems with spatial uncertainty
- Testing robustness of algorithms to coordinate perturbations

The original Simulation73 code does not clearly explain when and why to use this.
Consider adding examples and use cases.

# Example
```julia
base = CompactLattice(discrete_lattice((-1.0,), (1.0,), (5,)))
embedding = CompactLattice(discrete_lattice((-0.5,), (0.5,), (3,)))
embedded = RandomlyEmbeddedLattice(lattice=base, embedded_lattice=embedding)
```
"""
# FIXME what is this for?
struct RandomlyEmbeddedLattice{T,N_ARR,N_CDT,L<:AbstractLattice{T,N_ARR},E<:AbstractSpace{T}} <: AbstractEmbeddedLattice{T,N_ARR,N_CDT,L}
    lattice::L
    embedded_lattice::E
    coordinates::Array{NTuple{N_CDT,T},N_ARR}
end

"""
    RandomlyEmbeddedLattice(; lattice, embedded_lattice)

Construct a randomly embedded lattice.

# Arguments
- `lattice`: The base lattice to embed
- `embedded_lattice`: The lattice defining the embedding space structure

# Returns
A new RandomlyEmbeddedLattice where each base coordinate is augmented with
random coordinates sampled from the embedding space.

# Note
FIXME: The random seed is not controlled, so results are not reproducible.
Consider adding a seed parameter for reproducible random embeddings.
"""
function RandomlyEmbeddedLattice(; lattice, embedded_lattice)
    embedded_coordinates = embed_randomly(lattice, embedded_lattice)
    RandomlyEmbeddedLattice(lattice, embedded_lattice, embedded_coordinates)
end

"""
    embed_randomly(lattice, embedded_lattice)

Create random embedding coordinates by augmenting base lattice coordinates.

For each coordinate in the base lattice, this appends random coordinates
sampled from the embedding lattice's spatial domain.

# Arguments
- `lattice`: Base lattice providing the initial coordinates
- `embedded_lattice`: Lattice defining the embedding space

# Returns
Array of augmented coordinate tuples

# Implementation Note
FIXME: Uses `sample(embedded_lattice)` which samples uniformly from the embedding
space, but this may not always be the desired distribution. Consider supporting
different sampling strategies.
"""
function embed_randomly(lattice, embedded_lattice)
    [(lattice_coord..., sample(embedded_lattice)...) for lattice_coord in coordinates(lattice)]
end

"""
    sample(lattice::AbstractLattice)

Sample a random coordinate from the lattice's spatial domain.

# Arguments
- `lattice`: The lattice to sample from

# Returns
Random coordinate tuple sampled uniformly from the lattice's extent

# Implementation Note
FIXME: This samples from the continuous domain defined by the lattice extent,
not from the discrete lattice points themselves. The name `sample` is misleading
since it doesn't sample from the actual lattice coordinates.

Consider renaming to `sample_domain` or similar to clarify that this samples
from the continuous spatial domain, not the discrete lattice points.
"""
function sample(lattice::AbstractLattice)
    (rand(length(extent(lattice))...) .* extent(lattice)) .- (extent(lattice) ./ 2)
end
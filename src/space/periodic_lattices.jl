abstract type AbstractPeriodicLattice{T,N_ARR,N_CDT} <: AbstractLattice{T,N_ARR,N_CDT} end

@doc """
A Lattice of points with `extent` describing the length along each dimension and `n_points` describing the number of points representing each dimension.
"""
struct PeriodicLattice{T,N_ARR} <: AbstractPeriodicLattice{T,N_ARR,N_ARR}
    arr::Array{NTuple{N_ARR,T},N_ARR}
    extent::NTuple{N_ARR,T}
end
(t::Type{<:PeriodicLattice})(; extent, n_points) =_periodiclattice(t, extent, n_points)
function _periodiclattice(t, extent::Number, n_points::Number)
    _periodiclattice(t, (extent,), (n_points,))
end
function _periodiclattice(t, extent::Tuple, n_points::Tuple)
    t(discrete_lattice(.-extent ./ 2, extent ./ 2 .- (extent ./ n_points), n_points), extent)
end
extent(p::PeriodicLattice) = p.extent

difference(p_lattice::PeriodicLattice, edge) = abs_difference_periodic(edge, extent(p_lattice))

# TODO: Should be completely uniform in periodic case?
export simpson_weights
function simpson_weights(lattice::PeriodicLattice{T}) where T
    w = ones(T, size(lattice)...)
    return w
end
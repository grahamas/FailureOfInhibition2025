

"""
    distances(space)

Return the distances between every pair of points in `space`
"""
function differences(space::AbstractSpace{T}) where T
    edges = Iterators.product(coordinates(space), coordinates(space))
    return difference.(Ref(space), edges)
end
function differences(space::AbstractSpace{T,N_ARR,N_CDT},
                     reference_location::NTuple{N_CDT,T}) where {T,N_ARR,N_CDT}
    edges = ((coord, reference_location) for coord in coordinates(space))
    return difference.(Ref(space), edges)
end

"""
    abs_difference(edge::Tuple{T,T}) where T<:Number

Compute the absolute difference between two scalar values.

# Arguments
- `edge`: A tuple containing two scalar values (point1, point2)

# Returns
The absolute difference |point1 - point2|

# Example
```jldoctest
julia> abs_difference( (5,1) )
4
```
"""
abs_difference(edge::Tuple{T,T}) where T<:Number = abs(edge[1] - edge[2])

"""
    abs_difference(edge::Tuple{Tup,Tup}) where {Nminusone,T,Tup<:Tuple{T,Vararg{T,Nminusone}}}

Compute the absolute difference between two coordinate tuples (multidimensional case).

# Arguments  
- `edge`: A tuple containing two coordinate tuples ((x1,y1,...), (x2,y2,...))

# Returns
A tuple of absolute differences along each dimension

# Example
```jldoctest
julia> abs_difference( ((2,2), (5,-5)) )
(3, 7)
```

# Note
FIXME: should really be L2 norm - currently computes componentwise differences
rather than true Euclidean distance. This gives L∞ (max) norm behavior.
"""
# FIXME should really be L2 norm
abs_difference(edge::Tuple{Tup,Tup}) where {Nminusone,T,Tup<:Tuple{T,Vararg{T,Nminusone}}} = abs.(edge[1] .- edge[2])



"""
    abs_difference_periodic(edge::Tuple{T,T}, (period,)::Tuple{T}) where T<:Number

Compute the shortest distance between two scalar points on a periodic domain.

In a periodic domain, there are two possible paths between any two points:
the direct path and the path that wraps around the boundary. This function
returns the distance along the shorter path.

# Arguments
- `edge`: Tuple of two scalar coordinates (point1, point2) 
- `period`: Tuple containing the period length

# Returns
The shortest distance considering periodic wrapping

# Example
```jldoctest
julia> abs_difference_periodic( (5,1), 3 )
-1
```

# Note
A negative result indicates the wrapping path is shorter and goes in the
opposite direction.
"""
function abs_difference_periodic(edge::Tuple{T,T}, (period,)::Tuple{T}) where T<:Number
    diff = abs_difference(edge)
    if diff >= period / 2
        return period - diff
    else
        return diff
    end
end

"""
    abs_difference_periodic(edge::Tuple{Tup,Tup}, periods::Tup) where {Nminusone,T,Tup<:Tuple{T,Vararg{T,Nminusone}}}

Compute the shortest distance between two coordinate tuples on a periodic domain.

For multidimensional periodic domains, this applies the periodic distance
calculation independently along each dimension.

# Arguments
- `edge`: Tuple of two coordinate tuples ((x1,y1,...), (x2,y2,...))
- `periods`: Tuple of period lengths for each dimension

# Returns
Tuple of shortest distances along each dimension

# Example
```jldoctest
julia> abs_difference_periodic( ((2,2), (5,-5)), (3,4) )
(0, -3)
```

# Note
Each component can be negative, indicating the wrapping direction for that dimension.
"""
function abs_difference_periodic(edge::Tuple{Tup,Tup}, periods::Tup) where {Nminusone,T,Tup<:Tuple{T,Vararg{T,Nminusone}}}
    diffs = abs_difference(edge)
    diffs = map(zip(diffs, periods)) do (diff, period)
        if diff > period / 2
            return period - diff
        else
            return diff
        end
    end
    return Tup(diffs)
end
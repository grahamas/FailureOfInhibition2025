
"""
    population(array, index)

Extract population `index` from a multi-population array.
For simplicity, assumes populations are stored along the last dimension.
"""
function population(array, index)
    # For a 2D array, assume populations are along the second dimension
    # This is a simplified implementation - may need adjustment based on actual data structure
    if ndims(array) == 1
        return array  # Single population case
    elseif ndims(array) == 2
        return view(array, :, index)
    else
        # For higher dimensions, assume last dimension is populations
        indices = ntuple(i -> i == ndims(array) ? index : Colon(), ndims(array))
        return view(array, indices...)
    end
end

"""
    stimulate(dA, A, stimulus, t)

Generic stimulate function that dispatches to stimulate! methods.
"""
function stimulate(dA, A, stimulus, t)
    stimulate!(dA, A, stimulus, t)
end

function wcm1973!(dA, A, p, t)
    stimulate(dA, A, p.stimulus, t)
    propagate_activation(dA, A, p.connectivity, t)
    apply_nonlinearity(dA, A, p.nonlinearity, t)
    for i in 1:2 # number of populations
        dAi = population(dA, i); Ai = population(A, i)
        dAi .*= p.β[i] .* (1 .- Ai)
        dAi .+= -p.α[i] .* Ai
        dAi ./= p.τ[i]
    end
end

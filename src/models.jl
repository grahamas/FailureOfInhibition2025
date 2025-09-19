
function wcm1973!(dA, A, p, t)
    stimulate(dA, A, p.stimulus, t)
    propagate_activation(dA, A, p.connectivity, t)
    apply_nonlinearity(dA, A, p.nonlinearity, t)
    for 1 in 1:2 # number of populations
        dAi = population(dA, i); Ai = population(A, i)
        dAi .*= p.β[i] .* (1 .- Ai)
        dAi .+= -p.α[i] .* Ai
        dAi ./= p.τ[i]
    end
end

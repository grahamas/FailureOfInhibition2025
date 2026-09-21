using Test
using FailureOfInhibition2025

include("test_support.jl")

@testset "FailureOfInhibition2025" begin
    include("test_responses.jl")
    include("test_drives.jl")
    include("test_point_model.jl")
    include("test_jacobian.jl")
    include("test_stability.jl")
    include("test_equilibria.jl")
    include("test_configurations.jl")
    include("test_simulation.jl")
end

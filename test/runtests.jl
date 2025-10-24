#!/usr/bin/env julia

"""
Main test runner for FailureOfInhibition2025 package.
This file is executed when running `Pkg.test()` or `julia --project=. test/runtests.jl`
"""

using Test
using FailureOfInhibition2025

# Include all test files
println("Running FailureOfInhibition2025 test suite...")

@testset "FailureOfInhibition2025 Tests" begin
    
    # Run sigmoid tests
    @testset "Sigmoid Functionality" begin
        include("test_sigmoid.jl")
        
        # Run the main test function from test_sigmoid.jl
        test_sigmoid_functions()
        test_model_integration()
    end
    
    # Additional test groups can be added here as the package grows
    @testset "Basic Package Functionality" begin
        @test isdefined(FailureOfInhibition2025, :greet)
        @test isdefined(FailureOfInhibition2025, :SigmoidNonlinearity)
        @test isdefined(FailureOfInhibition2025, :RectifiedZeroedSigmoidNonlinearity)
        @test isdefined(FailureOfInhibition2025, :simple_sigmoid)
        @test isdefined(FailureOfInhibition2025, :apply_nonlinearity)
        @test isdefined(FailureOfInhibition2025, :wcm1973!)
        @test isdefined(FailureOfInhibition2025, :population)
        @test isdefined(FailureOfInhibition2025, :stimulate)
    end
    
    @testset "Space and Lattice Functions" begin
        @test isdefined(FailureOfInhibition2025, :AbstractSpace)
        @test isdefined(FailureOfInhibition2025, :AbstractLattice)
        @test isdefined(FailureOfInhibition2025, :CompactLattice)
        @test isdefined(FailureOfInhibition2025, :PeriodicLattice)
    end

    # Run comprehensive space/lattice/coordinates tests
    @testset "Space/Lattice/Coordinates Functionality" begin
        include("test_space.jl")
        
        # Run the comprehensive test function from test_space.jl
        run_all_space_tests()
    end

end

println("✅ All tests completed successfully!")
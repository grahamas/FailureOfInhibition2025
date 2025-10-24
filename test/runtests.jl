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
    
    # Run stimulation tests
    @testset "Stimulation Functionality" begin
        include("test_stimulate.jl")
        
        # Run the main test functions from test_stimulate.jl
        test_euclidean_distance()
        test_circle_stimulus_construction()
        test_stimulate_1d()
        test_stimulate_2d()
        test_time_windows()
        test_edge_cases()
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
        @test isdefined(FailureOfInhibition2025, :stimulate!)
        @test isdefined(FailureOfInhibition2025, :CircleStimulus)
    end
    
    @testset "Space and Lattice Functions" begin
        @test isdefined(FailureOfInhibition2025, :AbstractSpace)
        @test isdefined(FailureOfInhibition2025, :AbstractLattice)
        @test isdefined(FailureOfInhibition2025, :CompactLattice)
        @test isdefined(FailureOfInhibition2025, :PeriodicLattice)
    end

end

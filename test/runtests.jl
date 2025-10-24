#!/usr/bin/env julia

"""
Main test runner for FailureOfInhibition2025 package.
This file is executed when running `Pkg.test()` or `julia --project=. test/runtests.jl`
"""

using Test
using FailureOfInhibition2025

@testset "FailureOfInhibition2025 Tests" begin
    
    # Run comprehensive nonlinearity tests
    @testset "Nonlinearity Tests" begin
        include("test_nonlinearity.jl")
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
    
    # Run Wilson-Cowan model tests
    @testset "Wilson-Cowan Model" begin
        include("test_wilson_cowan.jl")
        
        # Run the main test functions from test_wilson_cowan.jl
        test_wilson_cowan_parameters()
        test_wilson_cowan_dynamics()
        test_implementation_documentation()
    end
    
    # Additional test groups can be added here as the package grows
    @testset "Basic Package Functionality" begin
        @test isdefined(FailureOfInhibition2025, :greet)
        @test isdefined(FailureOfInhibition2025, :SigmoidNonlinearity)
        @test isdefined(FailureOfInhibition2025, :RectifiedZeroedSigmoidNonlinearity)
        @test isdefined(FailureOfInhibition2025, :simple_sigmoid)
        @test isdefined(FailureOfInhibition2025, :apply_nonlinearity!)
        @test isdefined(FailureOfInhibition2025, :wcm1973!)
        @test isdefined(FailureOfInhibition2025, :WilsonCowanParameters)
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
    
    # Test Gaussian connectivity
    @testset "Gaussian Connectivity" begin
        include("test_gaussian_connectivity.jl")
        
        # Run the main test functions from test_gaussian_connectivity.jl
        test_gaussian_connectivity_parameter()
        test_apply_connectivity_unscaled()
        test_calculate_kernel()
        test_gaussian_connectivity_construction()
        test_propagate_activation()
        test_fftshift()
    end

    # Test ConnectivityMatrix
    @testset "ConnectivityMatrix" begin
        include("test_connectivity_matrix.jl")
        
        # Run the main test functions from test_connectivity_matrix.jl
        test_connectivity_matrix_construction()
        test_connectivity_matrix_indexing_convention()
        test_propagate_activation_with_connectivity_matrix()
        test_wilson_cowan_with_connectivity_matrix()
    end

    # Run comprehensive space/lattice/coordinates tests
    @testset "Space/Lattice/Coordinates Functionality" begin
        include("test_space.jl")
        
        # Run the comprehensive test function from test_space.jl
        run_all_space_tests()
    end

end

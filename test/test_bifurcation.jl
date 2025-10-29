#!/usr/bin/env julia

"""
Tests for bifurcation analysis functionality using BifurcationKit.
"""

using Test
using FailureOfInhibition2025
using BifurcationKit

# Load WCM 1973 parameter creation functions
include("test_wcm1973_validation.jl")

@testset "Bifurcation Analysis Tests (BifurcationKit)" begin
    
    @testset "wcm_rhs! Function" begin
        # Create a simple point model
        params = create_point_model_wcm1973(:active_transient)
        
        # Test activity state
        A = reshape([0.1, 0.1], 1, 2)
        dA = zeros(size(A))
        
        # Call wcm_rhs!
        result = wcm_rhs!(dA, A, params, 0.0)
        
        # Check that it returns dA
        @test result === dA
        
        # Check that derivatives were computed (not all zero)
        @test !all(dA .== 0.0)
        
        # Check that output has correct shape
        @test size(dA) == size(A)
        
        println("  ✓ wcm_rhs! function works correctly")
    end
    
    @testset "create_bifurcation_problem - Point Model" begin
        # Create point model parameters
        params = create_point_model_wcm1973(:active_transient)
        
        # Create a simple lens (placeholder - actual lens would come from BifurcationKit)
        # For testing purposes, we'll use a minimal setup
        param_lens = (@lens _)
        
        # Initial condition
        u0 = reshape([0.1, 0.1], 1, 2)
        
        # Create bifurcation problem
        prob = create_bifurcation_problem(params, param_lens, u0=u0)
        
        # Check that problem was created
        @test prob isa BifurcationProblem
        
        # Check initial condition
        @test size(prob.u0) == size(u0)
        
        println("  ✓ create_bifurcation_problem works for point models")
    end
    
    @testset "create_bifurcation_problem - Spatial Model" begin
        # Create spatial model parameters
        params = create_wcm1973_parameters(:active_transient)
        
        # Create a simple lens
        param_lens = (@lens _)
        
        # Let function auto-generate initial condition
        prob = create_bifurcation_problem(params, param_lens)
        
        # Check that problem was created
        @test prob isa BifurcationProblem
        
        # Check that initial condition has correct shape
        n_points = size(params.lattice)[1]
        P = length(params.pop_names)
        @test size(prob.u0) == (n_points, P)
        
        println("  ✓ create_bifurcation_problem works for spatial models")
    end
    
    @testset "create_bifurcation_problem - Auto Initial Condition" begin
        # Test auto-generation of initial condition
        params = create_point_model_wcm1973(:steady_state)
        param_lens = (@lens _)
        
        # Don't provide u0
        prob = create_bifurcation_problem(params, param_lens)
        
        # Check that initial condition was generated
        @test !isnothing(prob.u0)
        @test size(prob.u0) == (1, 2)  # Point model with 2 populations
        
        # Check that values are reasonable (should be small positive)
        @test all(prob.u0 .> 0.0)
        @test all(prob.u0 .< 0.2)
        
        println("  ✓ create_bifurcation_problem auto-generates initial conditions")
    end
    
end

println("\n" * "="^70)
println("✓ All BifurcationKit integration tests passed")
println("="^70)

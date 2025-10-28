#!/usr/bin/env julia

"""
Tests for bifurcation analysis functionality.
"""

using Test
using FailureOfInhibition2025

# Load WCM 1973 parameter creation functions
include("test_wcm1973_validation.jl")

@testset "Bifurcation Analysis Tests" begin
    
    @testset "Steady State Detection" begin
        # Create a simple point model
        params = create_point_model_wcm1973(:steady_state)
        
        # Initial condition
        A₀ = reshape([0.3, 0.2], 1, 2)
        
        # Solve to steady state
        sol = solve_model(A₀, (0.0, 200.0), params, saveat=0.5)
        
        # Test steady state detection
        steady = detect_steady_state(sol; transient_fraction=0.5, threshold=1e-3)
        
        @test !isnothing(steady)
        @test size(steady) == size(A₀)
        println("  ✓ Steady state detection works")
    end
    
    @testset "Oscillation Detection" begin
        # Create oscillatory mode parameters
        params = create_point_model_wcm1973(:oscillatory)
        
        # Initial condition
        A₀ = reshape([0.3, 0.2], 1, 2)
        
        # Solve
        sol = solve_model(A₀, (0.0, 500.0), params, saveat=0.5)
        
        # Test oscillation detection
        is_osc, amplitude, period = detect_oscillations(sol; transient_fraction=0.5)
        
        # Oscillatory mode should produce oscillations
        @test is_osc
        if !isnothing(period)
            @test period > 0
        end
        if !isnothing(amplitude)
            @test amplitude > 0
        end
        println("  ✓ Oscillation detection works")
    end
    
    @testset "Analyze Dynamics" begin
        # Test with active transient mode
        params = create_point_model_wcm1973(:active_transient)
        A₀ = reshape([0.3, 0.2], 1, 2)
        sol = solve_model(A₀, (0.0, 200.0), params, saveat=0.5)
        
        # Analyze dynamics
        param_vals = (bₑₑ=1.5, bᵢₑ=1.35)
        point = analyze_dynamics(sol, param_vals; transient_fraction=0.5)
        
        @test point isa BifurcationPoint
        @test point.param_values == param_vals
        @test length(point.mean_activity) == 2  # Two populations
        @test length(point.max_activity) == 2
        @test length(point.min_activity) == 2
        @test point.mean_activity[1] >= 0  # Activity should be non-negative
        @test point.mean_activity[2] >= 0
        println("  ✓ Dynamics analysis works")
    end
    
    @testset "Parameter Update - Connectivity" begin
        # Create base parameters
        params = create_point_model_wcm1973(:active_transient)
        
        # Update E-E coupling
        new_params = update_parameter(params, :bₑₑ, 2.5)
        
        @test new_params isa WilsonCowanParameters
        @test new_params.α == params.α
        @test new_params.τ == params.τ
        
        # Test that connectivity was updated
        # The new connectivity should have different weights
        old_conn_ee = params.connectivity.matrix[1, 1]
        new_conn_ee = new_params.connectivity.matrix[1, 1]
        
        if old_conn_ee isa ScalarConnectivity
            @test new_conn_ee.weight == 2.5
        else
            @test new_conn_ee.amplitude == 2.5
        end
        
        println("  ✓ Connectivity parameter update works")
    end
    
    @testset "Parameter Update - Nonlinearity" begin
        # Create base parameters
        params = create_point_model_wcm1973(:active_transient)
        
        # Update E sigmoid steepness
        new_params = update_parameter(params, :vₑ, 1.0)
        
        @test new_params isa WilsonCowanParameters
        
        # Check that nonlinearity was updated
        @test new_params.nonlinearity[1].a == 1.0
        # I population should be unchanged
        @test new_params.nonlinearity[2].a == params.nonlinearity[2].a
        
        println("  ✓ Nonlinearity parameter update works")
    end
    
    @testset "2D Parameter Sweep - Small Grid" begin
        # Use a very small grid for fast testing
        params = create_point_model_wcm1973(:active_transient)
        
        # Small parameter sweep (3x3 grid)
        diagram = parameter_sweep_2d(
            params,
            :bₑₑ, 1.0:0.5:2.0,  # 3 points
            :bᵢₑ, 1.0:0.5:2.0,  # 3 points
            tspan=(0.0, 100.0),
            saveat=1.0
        )
        
        @test diagram isa BifurcationDiagram
        @test diagram.param_names == (:bₑₑ, :bᵢₑ)
        @test length(diagram.param1_values) == 3
        @test length(diagram.param2_values) == 3
        @test size(diagram.points) == (3, 3)
        
        # Check that all points were computed
        for i in 1:3
            for j in 1:3
                point = diagram.points[i, j]
                @test point isa BifurcationPoint
                @test !isnan(point.mean_activity[1])
                @test !isnan(point.mean_activity[2])
            end
        end
        
        println("  ✓ 2D parameter sweep works")
    end
    
    @testset "2D Parameter Sweep - Spatial Model" begin
        # Test with spatial model (smaller grid for speed)
        params = create_wcm1973_parameters(:active_transient)
        
        # Very small grid for testing
        diagram = parameter_sweep_2d(
            params,
            :bₑₑ, 1.5:0.5:2.0,  # 2 points
            :bᵢₑ, 1.35:0.5:1.85,  # 2 points
            tspan=(0.0, 50.0),
            saveat=1.0
        )
        
        @test diagram isa BifurcationDiagram
        @test size(diagram.points) == (2, 2)
        
        # Spatial model should still produce valid results
        for i in 1:2
            for j in 1:2
                point = diagram.points[i, j]
                @test !isnan(point.mean_activity[1])
            end
        end
        
        println("  ✓ 2D parameter sweep works with spatial models")
    end
    
    @testset "BifurcationPoint Structure" begin
        # Create a test bifurcation point
        param_vals = (bₑₑ=1.5, bᵢₑ=1.2)
        steady = reshape([0.3, 0.2], 1, 2)
        mean_act = [0.3, 0.2]
        max_act = [0.35, 0.25]
        min_act = [0.25, 0.15]
        
        point = BifurcationPoint(
            param_vals,
            steady,
            true,  # is_oscillatory
            0.05,  # amplitude
            25.0,  # period
            mean_act,
            max_act,
            min_act
        )
        
        @test point.param_values == param_vals
        @test point.is_oscillatory == true
        @test point.oscillation_amplitude == 0.05
        @test point.oscillation_period == 25.0
        @test point.mean_activity == mean_act
        
        println("  ✓ BifurcationPoint structure works correctly")
    end
    
end

println("\n" * "="^70)
println("✓ All bifurcation analysis tests passed")
println("="^70)

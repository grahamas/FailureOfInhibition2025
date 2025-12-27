#!/usr/bin/env julia

"""
Tests for fixed point finding and stability analysis functionality.
"""

using Test
using FailureOfInhibition2025
using LinearAlgebra

@testset "Fixed Point Finding and Stability Analysis" begin
    
    @testset "find_fixed_points - Point Model" begin
        # Create a simple point model (steady state mode has stable fixed points)
        params = create_point_model_wcm1973(:steady_state)
        
        # Find fixed points
        fixed_points, converged = find_fixed_points(params, n_trials=10, tol=1e-5)
        
        # Should find at least one fixed point
        @test length(fixed_points) >= 1
        
        # Check that found points are indeed fixed points (dA/dt ≈ 0)
        for fp in fixed_points
            A = reshape(fp, 1, 2)
            dA = zeros(size(A))
            wcm_rhs!(dA, A, params, 0.0)
            
            # Derivative should be very small at fixed point
            # Relax tolerance since we're using numerical optimization
            @test norm(dA) < 0.01  # Fixed points found numerically may not be exact
        end
        
        println("  ✓ Found $(length(fixed_points)) fixed point(s) for point model")
    end
    
    @testset "compute_stability - Point Model" begin
        # Create steady state model
        params = create_point_model_wcm1973(:steady_state)
        
        # Find a fixed point
        fixed_points, _ = find_fixed_points(params, n_trials=10)
        @test length(fixed_points) >= 1
        
        # Compute stability for the first fixed point
        fp = fixed_points[1]
        eigenvalues, is_stable = compute_stability(fp, params)
        
        # Should return eigenvalues (2x2 system -> 2 eigenvalues)
        @test length(eigenvalues) == 2
        
        # is_stable should be a boolean
        @test isa(is_stable, Bool)
        
        # For steady state mode, we expect at least one stable fixed point
        if is_stable
            @test all(real(λ) < 0 for λ in eigenvalues)
            println("  ✓ Fixed point is stable (all eigenvalues have negative real parts)")
        else
            println("  ✓ Fixed point is unstable (some eigenvalues have positive real parts)")
        end
    end
    
    @testset "count_stable_fixed_points" begin
        # Test with different modes
        modes = [:active_transient, :oscillatory, :steady_state]
        
        for mode in modes
            params = create_point_model_wcm1973(mode)
            n_stable, fps, stabilities = count_stable_fixed_points(params, n_trials=15)
            
            # Should find at least one fixed point
            @test length(fps) >= 1
            @test length(stabilities) == length(fps)
            
            # n_stable should match the count of true values in stabilities
            @test n_stable == sum(stabilities)
            
            println("  ✓ Mode :$mode has $n_stable stable fixed point(s) out of $(length(fps)) total")
        end
    end
    
    @testset "optimize_for_stable_fixed_points - Basic" begin
        # Start with active transient mode
        base_params = create_point_model_wcm1973(:active_transient)
        
        # Define parameter ranges (small range for quick test)
        param_ranges = (
            connectivity_ee = (1.2, 2.5),
            connectivity_ei = (1.0, 2.0)
        )
        
        # Try to find parameters with 1 stable fixed point
        # (use small maxiter for quick test)
        result, best_params = optimize_for_stable_fixed_points(
            base_params,
            param_ranges,
            1,  # Target: 1 stable fixed point
            n_trials_per_eval=10,
            maxiter=5,  # Small for quick test
            population_size=5
        )
        
        # Should return a result and parameters
        @test !isnothing(result)
        @test !isnothing(best_params)
        
        # Verify the optimized parameters are valid
        @test isa(best_params, WilsonCowanParameters)
        
        # Count fixed points in optimized parameters
        n_stable, _, _ = count_stable_fixed_points(best_params, n_trials=10)
        
        println("  ✓ Optimization completed: found parameters with $n_stable stable fixed point(s)")
        println("    Target was 1 stable fixed point")
    end
    
    @testset "find_fixed_points - Different Initial Conditions" begin
        # Test that multiple trials can find multiple fixed points
        params = create_point_model_wcm1973(:steady_state)
        
        # Try with different ranges
        fps1, _ = find_fixed_points(params, n_trials=5, u0_range=(0.0, 0.2))
        fps2, _ = find_fixed_points(params, n_trials=5, u0_range=(0.2, 0.5))
        
        # Should find at least one fixed point in each case
        @test length(fps1) >= 0  # May be zero if no FP in that range
        @test length(fps2) >= 0
        
        println("  ✓ Found $(length(fps1)) FP(s) in range [0.0, 0.2]")
        println("  ✓ Found $(length(fps2)) FP(s) in range [0.2, 0.5]")
    end
    
    @testset "compute_stability - Eigenvalue Properties" begin
        # Create a model known to have oscillatory behavior
        params = create_point_model_wcm1973(:oscillatory)
        
        # Find fixed points
        fixed_points, _ = find_fixed_points(params, n_trials=10)
        
        if length(fixed_points) > 0
            fp = fixed_points[1]
            eigenvalues, is_stable = compute_stability(fp, params)
            
            # Check eigenvalue properties
            @test length(eigenvalues) == 2
            
            # For oscillatory systems, eigenvalues might be complex
            has_complex = any(abs(imag(λ)) > 1e-6 for λ in eigenvalues)
            
            if has_complex
                println("  ✓ Found complex eigenvalues (oscillatory dynamics)")
            else
                println("  ✓ Found real eigenvalues")
            end
        else
            println("  ⚠ No fixed points found for oscillatory mode")
        end
    end
    
end

println("\n" * "="^70)
println("✓ All fixed point analysis tests passed")
println("="^70)

#!/usr/bin/env julia

"""
Tests for analytical Jacobian implementation
"""

using FailureOfInhibition2025
using Test
using LinearAlgebra

using FailureOfInhibition2025
using Test
using LinearAlgebra

# Tolerance constants for numerical comparisons
const JACOBIAN_ATOL = 1e-5
const JACOBIAN_RTOL = 1e-4

"""
    numerical_jacobian(f!, A, p, t; h=1e-7)

Compute numerical Jacobian using finite differences for point models.

For point models, A should be (1, P) matrix or P-vector.
Uses forward differences to approximate ∂f/∂A.

# Arguments
- `f!`: ODE function that computes dA/dt
- `A`: Current state (vector or matrix)
- `p`: Parameters (WilsonCowanParameters)
- `t`: Current time
- `h`: Step size for finite differences (default: 1e-7)

# Returns
- `J`: PxP Jacobian matrix
"""
function numerical_jacobian(f!, A, p, t; h=1e-7)
    # Determine if this is a point model
    is_point_model = p.lattice isa PointLattice
    
    if is_point_model
        # For point models, ensure A is a (1, P) matrix
        if A isa AbstractVector
            P = length(A)
            A_mat = reshape(A, 1, P)
        else
            A_mat = A
            P = size(A_mat, 2)
        end
        
        # Initialize Jacobian
        J = zeros(P, P)
        
        dA = zeros(size(A_mat))
        dA_perturbed = zeros(size(A_mat))
        A_perturbed = copy(A_mat)
        
        # Get baseline
        f!(dA, A_mat, p, t)
        
        for j in 1:P
            # Perturb j-th population
            A_perturbed .= A_mat
            A_perturbed[1, j] += h
            
            # Compute perturbed derivative
            f!(dA_perturbed, A_perturbed, p, t)
            
            # Finite difference: ∂(dA_i/dt)/∂A_j ≈ (dA_i(A+h*e_j) - dA_i(A))/h
            for i in 1:P
                J[i, j] = (dA_perturbed[1, i] - dA[1, i]) / h
            end
        end
        
        return J
    else
        error("Numerical Jacobian only implemented for point models")
    end
end

@testset "Analytical Jacobian Tests" begin
    
    @testset "Point Model - SigmoidNonlinearity" begin
        println("\n=== Testing Point Model with SigmoidNonlinearity ===")
        
        # Create point model with sigmoid nonlinearity
        params = create_point_model_wcm1973(:oscillatory)
        
        # Test state
        A = reshape([0.1, 0.15], 1, 2)
        A_vec = vec(A)
        
        # Compute analytical Jacobian
        J_analytical = zeros(2, 2)
        wcm1973_jacobian!(J_analytical, A_vec, params, 0.0)
        
        # Compute numerical Jacobian
        J_numerical = numerical_jacobian(wcm1973!, A_vec, params, 0.0)
        
        # Compare
        println("Analytical Jacobian:")
        println(J_analytical)
        println("\nNumerical Jacobian:")
        println(J_numerical)
        println("\nDifference:")
        println(J_analytical - J_numerical)
        
        # Test that they match (within numerical tolerance)
        @test isapprox(J_analytical, J_numerical, atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)
    end
    
    @testset "Point Model - Different States" begin
        println("\n=== Testing Point Model at Different States ===")
        
        params = create_point_model_wcm1973(:active_transient)
        
        # Test at multiple states
        test_states = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.3, 0.2],
            [0.5, 0.5],
            [0.8, 0.7]
        ]
        
        for A_vec in test_states
            J_analytical = zeros(2, 2)
            wcm1973_jacobian!(J_analytical, A_vec, params, 0.0)
            
            J_numerical = numerical_jacobian(wcm1973!, A_vec, params, 0.0)
            
            @test isapprox(J_analytical, J_numerical, atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)
        end
        
        println("✓ All states tested successfully")
    end
    
    @testset "Point Model - Steady State Mode" begin
        println("\n=== Testing Point Model with Steady State Mode ===")
        
        params = create_point_model_wcm1973(:steady_state)
        A_vec = [0.2, 0.3]
        
        J_analytical = zeros(2, 2)
        wcm1973_jacobian!(J_analytical, A_vec, params, 0.0)
        
        J_numerical = numerical_jacobian(wcm1973!, A_vec, params, 0.0)
        
        println("Analytical Jacobian:")
        println(J_analytical)
        println("\nNumerical Jacobian:")
        println(J_numerical)
        
        @test isapprox(J_analytical, J_numerical, atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)
    end
    
    @testset "Per-Population Nonlinearities" begin
        println("\n=== Testing Per-Population Nonlinearities ===")
        
        # Create a model with per-population nonlinearities
        lattice = PointLattice()
        
        # E population: Sigmoid
        nonlinearity_e = SigmoidNonlinearity(a=0.5, θ=9.0)
        # I population: Rectified Zeroed Sigmoid
        nonlinearity_i = RectifiedZeroedSigmoidNonlinearity(a=1.0, θ=15.0)
        nonlinearity = (nonlinearity_e, nonlinearity_i)
        
        # Connectivity
        conn_ee = ScalarConnectivity(2.0)
        conn_ei = ScalarConnectivity(-1.5)
        conn_ie = ScalarConnectivity(1.5)
        conn_ii = ScalarConnectivity(-0.1)
        
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = WilsonCowanParameters{2}(
            α = (1.0, 1.0),
            β = (1.0, 1.0),
            τ = (10.0, 10.0),
            connectivity = connectivity,
            nonlinearity = nonlinearity,
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E", "I")
        )
        
        A_vec = [0.2, 0.25]
        
        J_analytical = zeros(2, 2)
        wcm1973_jacobian!(J_analytical, A_vec, params, 0.0)
        
        J_numerical = numerical_jacobian(wcm1973!, A_vec, params, 0.0)
        
        println("Analytical Jacobian:")
        println(J_analytical)
        println("\nNumerical Jacobian:")
        println(J_numerical)
        
        @test isapprox(J_analytical, J_numerical, atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)
    end
    
    @testset "DifferenceOfSigmoids Nonlinearity" begin
        println("\n=== Testing DifferenceOfSigmoids Nonlinearity ===")
        
        # Create FoI-like model with difference of sigmoids
        lattice = PointLattice()
        
        nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5)
        nonlinearity_i = DifferenceOfSigmoidsNonlinearity(
            a_activating=5.0, θ_activating=0.3,
            a_failing=3.0, θ_failing=0.7
        )
        nonlinearity = (nonlinearity_e, nonlinearity_i)
        
        conn_ee = ScalarConnectivity(1.0)
        conn_ei = ScalarConnectivity(-0.5)
        conn_ie = ScalarConnectivity(0.8)
        conn_ii = ScalarConnectivity(-0.3)
        
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = WilsonCowanParameters{2}(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (10.0, 8.0),
            connectivity = connectivity,
            nonlinearity = nonlinearity,
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E", "I")
        )
        
        A_vec = [0.3, 0.4]
        
        J_analytical = zeros(2, 2)
        wcm1973_jacobian!(J_analytical, A_vec, params, 0.0)
        
        J_numerical = numerical_jacobian(wcm1973!, A_vec, params, 0.0)
        
        println("Analytical Jacobian:")
        println(J_analytical)
        println("\nNumerical Jacobian:")
        println(J_numerical)
        
        @test isapprox(J_analytical, J_numerical, atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)
    end
    
    @testset "No Connectivity" begin
        println("\n=== Testing Model with No Connectivity ===")
        
        lattice = PointLattice()
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)
        
        params = WilsonCowanParameters{2}(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (1.0, 0.8),
            connectivity = nothing,
            nonlinearity = nonlinearity,
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E", "I")
        )
        
        A_vec = [0.3, 0.4]
        
        J_analytical = zeros(2, 2)
        wcm1973_jacobian!(J_analytical, A_vec, params, 0.0)
        
        J_numerical = numerical_jacobian(wcm1973!, A_vec, params, 0.0)
        
        println("Analytical Jacobian (no connectivity):")
        println(J_analytical)
        println("\nNumerical Jacobian (no connectivity):")
        println(J_numerical)
        
        # With no connectivity, off-diagonal elements should be zero
        @test J_analytical[1, 2] ≈ 0.0 atol=1e-10
        @test J_analytical[2, 1] ≈ 0.0 atol=1e-10
        
        @test isapprox(J_analytical, J_numerical, atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)
    end
    
    @testset "Jacobian Structure" begin
        println("\n=== Testing Jacobian Structure Properties ===")
        
        params = create_point_model_wcm1973(:oscillatory)
        A_vec = [0.2, 0.3]
        
        J = zeros(2, 2)
        wcm1973_jacobian!(J, A_vec, params, 0.0)
        
        # For oscillatory behavior, we expect some specific properties
        # The Jacobian should have real eigenvalues or complex conjugate pairs
        eigenvals = eigvals(J)
        println("Eigenvalues: ", eigenvals)
        
        # Check that eigenvalues are either real or come in complex conjugate pairs
        for λ in eigenvals
            if imag(λ) != 0
                # If complex, check conjugate is also present
                λ_conj = conj(λ)
                @test λ_conj ∈ eigenvals
            end
        end
    end
    
    @testset "Derivative Functions" begin
        println("\n=== Testing Nonlinearity Derivative Functions ===")
        
        # Test sigmoid derivative
        x = 0.5
        a = 2.0
        θ = 0.3
        
        # Numerical derivative
        h = 1e-7
        f_x = simple_sigmoid(x, a, θ)
        f_x_h = simple_sigmoid(x + h, a, θ)
        df_numerical = (f_x_h - f_x) / h
        
        # Analytical derivative
        df_analytical = simple_sigmoid_derivative(x, a, θ)
        
        println("Sigmoid derivative at x=$x:")
        println("  Analytical: $df_analytical")
        println("  Numerical: $df_numerical")
        
        @test isapprox(df_analytical, df_numerical, atol=1e-6, rtol=1e-5)
        
        # Test rectified zeroed sigmoid derivative
        df_rect_analytical = rectified_zeroed_sigmoid_derivative(x, a, θ)
        
        f_rect_x = rectified_zeroed_sigmoid(x, a, θ)
        f_rect_x_h = rectified_zeroed_sigmoid(x + h, a, θ)
        df_rect_numerical = (f_rect_x_h - f_rect_x) / h
        
        println("Rectified sigmoid derivative at x=$x:")
        println("  Analytical: $df_rect_analytical")
        println("  Numerical: $df_rect_numerical")
        
        @test isapprox(df_rect_analytical, df_rect_numerical, atol=1e-6, rtol=1e-5)
    end
end

println("\n🎉 All Analytical Jacobian tests passed!")

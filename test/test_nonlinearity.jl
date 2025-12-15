#!/usr/bin/env julia

"""
Comprehensive tests for nonlinearity functions in FailureOfInhibition2025.
Tests mathematical properties, edge cases, numerical stability, and model integration.
"""

using Test
using FailureOfInhibition2025

@testset "Simple Sigmoid Tests" begin
    @testset "Basic Properties" begin
        # Test sigmoid at threshold
        @test abs(simple_sigmoid(0.0, 1.0, 0.0) - 0.5) < 1e-10
        @test abs(simple_sigmoid(1.0, 2.0, 1.0) - 0.5) < 1e-10
        @test abs(simple_sigmoid(5.0, 3.0, 5.0) - 0.5) < 1e-10
        
        # Test sigmoid bounds (should be in (0, 1] or [0, 1))
        @test 0.0 < simple_sigmoid(0.0, 1.0, 0.0) < 1.0
        @test 0.0 < simple_sigmoid(100.0, 1.0, 0.0) <= 1.0
        @test 0.0 <= simple_sigmoid(-100.0, 1.0, 0.0) < 1.0
        
        # Test positive and negative inputs
        @test simple_sigmoid(1.0, 2.0, 0.0) > 0.5
        @test simple_sigmoid(-1.0, 2.0, 0.0) < 0.5
    end
    
    @testset "Monotonicity" begin
        # Sigmoid should be monotonically increasing for positive slope
        a, θ = 2.0, 0.5
        x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
        y_vals = [simple_sigmoid(x, a, θ) for x in x_vals]
        
        for i in 1:length(y_vals)-1
            @test y_vals[i] < y_vals[i+1]
        end
    end
    
    @testset "Asymptotic Behavior" begin
        a, θ = 2.0, 0.0
        
        # As x → ∞, sigmoid → 1
        @test simple_sigmoid(10.0, a, θ) > 0.99
        @test simple_sigmoid(20.0, a, θ) > 0.999
        
        # As x → -∞, sigmoid → 0
        @test simple_sigmoid(-10.0, a, θ) < 0.01
        @test simple_sigmoid(-20.0, a, θ) < 0.001
    end
    
    @testset "Slope Parameter Effects" begin
        x, θ = 1.0, 0.0
        
        # Higher slope (a) should make sigmoid steeper
        # Test gradient around threshold
        steep = simple_sigmoid(x, 10.0, θ) - simple_sigmoid(0.0, 10.0, θ)
        shallow = simple_sigmoid(x, 1.0, θ) - simple_sigmoid(0.0, 1.0, θ)
        
        @test steep > shallow
    end
    
    @testset "Threshold Parameter Effects" begin
        x, a = 1.0, 2.0
        
        # Test that threshold shifts the function
        sig_theta0 = simple_sigmoid(x, a, 0.0)
        sig_theta1 = simple_sigmoid(x, a, 1.0)
        
        # At x=1, sigmoid with θ=1 should be at 0.5
        @test abs(sig_theta1 - 0.5) < 1e-10
        # At x=1, sigmoid with θ=0 should be > 0.5
        @test sig_theta0 > 0.5
    end
    
    @testset "Numerical Stability" begin
        # Test extreme values that could cause overflow/underflow
        a = 100.0
        
        # Very large positive input
        @test !isnan(simple_sigmoid(1000.0, a, 0.0))
        @test !isinf(simple_sigmoid(1000.0, a, 0.0))
        @test simple_sigmoid(1000.0, a, 0.0) ≈ 1.0
        
        # Very large negative input
        @test !isnan(simple_sigmoid(-1000.0, a, 0.0))
        @test !isinf(simple_sigmoid(-1000.0, a, 0.0))
        @test simple_sigmoid(-1000.0, a, 0.0) ≈ 0.0
    end
end

@testset "Rectified Zeroed Sigmoid Tests" begin
    @testset "Basic Properties" begin
        # Test that rectified zeroed sigmoid is zero at x=0
        a, θ = 2.0, 1.0
        @test abs(rectified_zeroed_sigmoid(0.0, a, θ)) < 1e-10
        
        # Test non-negativity
        x_vals = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
        for x in x_vals
            @test rectified_zeroed_sigmoid(x, a, θ) >= 0.0
        end
        
        # Test that it works correctly
        @test rectified_zeroed_sigmoid(1.0, 2.0, 1.0) >= 0.0
    end
    
    @testset "Rectification Behavior" begin
        # When simple_sigmoid(x) < simple_sigmoid(0), result should be 0
        a, θ = 2.0, 1.0
        
        # For negative x far from threshold, sigmoid(x) < sigmoid(0)
        @test rectified_zeroed_sigmoid(-5.0, a, θ) == 0.0
        
        # For positive x, rectified should be > 0
        @test rectified_zeroed_sigmoid(2.0, a, θ) > 0.0
    end
    
    @testset "Comparison with Simple Sigmoid" begin
        a, θ = 2.0, 0.5
        
        # Rectified should equal simple - simple(0)
        for x in [0.5, 1.0, 1.5, 2.0]
            expected = simple_sigmoid(x, a, θ) - simple_sigmoid(0.0, a, θ)
            expected = max(0.0, expected)
            @test abs(rectified_zeroed_sigmoid(x, a, θ) - expected) < 1e-10
        end
    end
    
    @testset "Monotonicity" begin
        # Should be monotonically increasing where not rectified
        a, θ = 2.0, 0.0
        x_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
        y_vals = [rectified_zeroed_sigmoid(x, a, θ) for x in x_vals]
        
        for i in 1:length(y_vals)-1
            @test y_vals[i] <= y_vals[i+1]
        end
    end
end

@testset "Difference of Simple Sigmoids Tests" begin
    @testset "Basic Properties" begin
        # Test that function returns a number
        result = difference_of_simple_sigmoids(1.0, 2.0, 0.5, 1.0, 1.5)
        @test isa(result, Float64)
        @test !isnan(result)
        @test !isinf(result)
    end
    
    @testset "Bump-like Behavior" begin
        # With appropriate parameters, should create a bump
        a_up, θ_up = 5.0, 0.5
        a_down, θ_down = 5.0, 1.5
        
        # Far left: both sigmoids ≈ 0, difference ≈ 0
        left = difference_of_simple_sigmoids(-2.0, a_up, θ_up, a_down, θ_down)
        @test abs(left) < 0.01
        
        # Far right: both sigmoids ≈ 1, difference ≈ 0
        right = difference_of_simple_sigmoids(3.0, a_up, θ_up, a_down, θ_down)
        @test abs(right) < 0.01
        
        # Middle: up sigmoid activated, down sigmoid not yet, difference > 0
        middle = difference_of_simple_sigmoids(1.0, a_up, θ_up, a_down, θ_down)
        @test middle > 0.0
        @test middle > left
        @test middle > right
    end
    
    @testset "Parameter Effects" begin
        # Test that swapping parameters reverses sign
        a1, θ1 = 2.0, 0.5
        a2, θ2 = 1.0, 1.5
        x = 1.0
        
        diff1 = difference_of_simple_sigmoids(x, a1, θ1, a2, θ2)
        diff2 = difference_of_simple_sigmoids(x, a2, θ2, a1, θ1)
        
        @test abs(diff1 + diff2) < 1e-10  # Should be negatives of each other
    end
end

@testset "Difference of Rectified Zeroed Sigmoids Tests" begin
    @testset "Basic Properties" begin
        # Test that function returns a number
        result = difference_of_rectified_zeroed_sigmoids(1.0, 2.0, 0.5, 1.0, 1.5)
        @test isa(result, Float64)
        @test !isnan(result)
        @test !isinf(result)
        
        # Test with parameters from test_sigmoid
        result2 = difference_of_rectified_zeroed_sigmoids(0.5, 2.0, 0.3, 1.0, 0.7)
        @test isa(result2, Float64)
    end
    
    @testset "Bump-like Behavior with Rectification" begin
        # With appropriate parameters, should create a bump
        a_up, θ_up = 5.0, 0.5
        a_down, θ_down = 5.0, 1.5
        
        # Far left: both rectified sigmoids ≈ 0, difference ≈ 0
        left = difference_of_rectified_zeroed_sigmoids(-2.0, a_up, θ_up, a_down, θ_down)
        @test abs(left) < 0.01
        
        # Far right: rectified down sigmoid larger, difference <= 0
        right = difference_of_rectified_zeroed_sigmoids(3.0, a_up, θ_up, a_down, θ_down)
        @test right <= 0.1
        
        # Middle: up sigmoid activated, down sigmoid not yet, difference > 0
        middle = difference_of_rectified_zeroed_sigmoids(1.0, a_up, θ_up, a_down, θ_down)
        @test middle > 0.0
        @test middle > left
        @test middle > right
        
        # Test regions where function should be zero or minimal
        before_up = difference_of_rectified_zeroed_sigmoids(0.0, a_up, θ_up, a_down, θ_down)
        @test abs(before_up) < 0.1
        
        after_down = difference_of_rectified_zeroed_sigmoids(2.0, a_up, θ_up, a_down, θ_down)
        @test after_down <= 0.1
        
        # Test maximal region
        max_region = difference_of_rectified_zeroed_sigmoids(1.0, a_up, θ_up, a_down, θ_down)
        mid_left = difference_of_rectified_zeroed_sigmoids(0.7, a_up, θ_up, a_down, θ_down)
        mid_right = difference_of_rectified_zeroed_sigmoids(1.3, a_up, θ_up, a_down, θ_down)
        
        @test max_region > 0.0
        @test max_region > mid_left
        @test max_region > mid_right
    end
    
    @testset "Comparison with Non-Rectified" begin
        # Test that rectified version is computed correctly
        a_up, θ_up = 2.0, 0.5
        a_down, θ_down = 1.0, 1.5
        
        for x in [0.0, 0.5, 1.0, 1.5, 2.0]
            rect_result = difference_of_rectified_zeroed_sigmoids(x, a_up, θ_up, a_down, θ_down)
            manual_result = rectified_zeroed_sigmoid(x, a_up, θ_up) - rectified_zeroed_sigmoid(x, a_down, θ_down)
            
            @test abs(rect_result - manual_result) < 1e-10
        end
    end
end

@testset "Nonlinearity Types Tests" begin
    @testset "SigmoidNonlinearity" begin
        # Test construction with keyword arguments
        nl = SigmoidNonlinearity(a=2.0, θ=1.0)
        @test nl.a == 2.0
        @test nl.θ == 1.0
        
        # Test apply_nonlinearity! with input values
        input = [0.0, 1.0, 2.0]
        dA = copy(input)  # dA contains the accumulated input
        A = [0.5, 0.6, 0.7]  # Current activity (not used by nonlinearity)
        original_A = copy(A)
        
        apply_nonlinearity!(dA, A, nl, 0.0)
        
        # A should be unchanged
        @test A == original_A
        
        # dA should now contain f(input), where input was the original dA
        for i in 1:3
            expected = simple_sigmoid(input[i], nl.a, nl.θ)
            @test abs(dA[i] - expected) < 1e-10
        end
    end
    
    @testset "RectifiedZeroedSigmoidNonlinearity" begin
        # Test construction with keyword arguments
        nl = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=1.0)
        @test nl.a == 2.0
        @test nl.θ == 1.0
        
        # Test apply_nonlinearity! with input values
        input = [0.0, 1.0, 2.0]
        dA = copy(input)  # dA contains the accumulated input
        A = [0.5, 0.6, 0.7]  # Current activity (not used by nonlinearity)
        original_A = copy(A)
        
        apply_nonlinearity!(dA, A, nl, 0.0)
        
        # A should be unchanged
        @test A == original_A
        
        # dA should now contain f(input), where input was the original dA
        for i in 1:3
            expected = rectified_zeroed_sigmoid(input[i], nl.a, nl.θ)
            @test abs(dA[i] - expected) < 1e-10
        end
    end
    
    @testset "DifferenceOfSigmoidsNonlinearity" begin
        # Test construction with keyword arguments
        nl = DifferenceOfSigmoidsNonlinearity(a_activating=2.0, θ_activating=0.5, a_failing=1.0, θ_failing=1.5)
        @test nl.a_activating == 2.0
        @test nl.θ_activating == 0.5
        @test nl.a_failing == 1.0
        @test nl.θ_failing == 1.5
        
        # Test construction with different parameters
        nl2 = DifferenceOfSigmoidsNonlinearity(a_activating=2.0, θ_activating=0.3, a_failing=1.0, θ_failing=0.7)
        @test nl2.a_activating == 2.0
        @test nl2.θ_activating == 0.3
        @test nl2.a_failing == 1.0
        @test nl2.θ_failing == 0.7
        
        # Test apply_nonlinearity! with input values
        input = [0.0, 1.0, 2.0]
        dA = copy(input)  # dA contains the accumulated input
        A = [0.5, 0.6, 0.7]  # Current activity (not used by nonlinearity)
        original_A = copy(A)
        
        apply_nonlinearity!(dA, A, nl, 0.0)
        
        # A should be unchanged
        @test A == original_A
        
        # dA should now contain f(input), where input was the original dA
        for i in 1:3
            expected = difference_of_rectified_zeroed_sigmoids(input[i], nl.a_activating, nl.θ_activating, nl.a_failing, nl.θ_failing)
            @test abs(dA[i] - expected) < 1e-10
        end
        
        # Test with specific parameters from test_sigmoid
        a_activating, θ_activating = 5.0, 0.5
        a_failing, θ_failing = 5.0, 1.5
        diff_nl = DifferenceOfSigmoidsNonlinearity(a_activating=a_activating, θ_activating=θ_activating, a_failing=a_failing, θ_failing=θ_failing)
        input_test = [1.0]
        dA_test = copy(input_test)
        A_test = [0.5]
        apply_nonlinearity!(dA_test, A_test, diff_nl, 0.0)
        
        expected = difference_of_rectified_zeroed_sigmoids(input_test[1], a_activating, θ_activating, a_failing, θ_failing)
        @test abs(dA_test[1] - expected) < 1e-10
    end
end

@testset "Array Operations Tests" begin
    @testset "Broadcasting Behavior" begin
        # Test that apply_nonlinearity! works correctly with arrays
        nl = SigmoidNonlinearity(a=2.0, θ=0.5)
        
        # 1D array - dA contains accumulated input
        input_1d = [-1.0, -0.5, 0.0, 0.5, 1.0]
        dA = copy(input_1d)
        A = [0.3, 0.4, 0.5, 0.6, 0.7]  # Current activity
        apply_nonlinearity!(dA, A, nl, 0.0)
        
        for i in 1:5
            expected = simple_sigmoid(input_1d[i], nl.a, nl.θ)
            @test abs(dA[i] - expected) < 1e-10
        end
        
        # 2D array - dA contains accumulated input
        input_2d = [0.0 1.0; 0.5 1.5; 1.0 2.0]
        dA_2d = copy(input_2d)
        A_2d = [0.3 0.4; 0.5 0.6; 0.7 0.8]  # Current activity
        apply_nonlinearity!(dA_2d, A_2d, nl, 0.0)
        
        for i in 1:3, j in 1:2
            expected = simple_sigmoid(input_2d[i, j], nl.a, nl.θ)
            @test abs(dA_2d[i, j] - expected) < 1e-10
        end
    end
end

@testset "Edge Cases and Special Values" begin
    @testset "Zero Parameters" begin
        # Test with a=0 (flat sigmoid)
        @test abs(simple_sigmoid(1.0, 0.0, 0.0) - 0.5) < 1e-10
        @test abs(simple_sigmoid(100.0, 0.0, 0.0) - 0.5) < 1e-10
    end
    
    @testset "Very Large Parameters" begin
        # Test with very large slope
        a = 1000.0
        θ = 0.0
        
        # Should still produce valid results
        @test !isnan(simple_sigmoid(0.1, a, θ))
        @test !isinf(simple_sigmoid(0.1, a, θ))
        @test simple_sigmoid(0.1, a, θ) > 0.99  # Very steep, so 0.1 should be near 1
        
        @test !isnan(simple_sigmoid(-0.1, a, θ))
        @test !isinf(simple_sigmoid(-0.1, a, θ))
        @test simple_sigmoid(-0.1, a, θ) < 0.01  # Very steep, so -0.1 should be near 0
    end
    
    @testset "Negative Slope" begin
        # Test with negative slope (decreasing sigmoid)
        a = -2.0
        θ = 0.0
        
        # Should be decreasing
        @test simple_sigmoid(1.0, a, θ) < simple_sigmoid(0.0, a, θ)
        @test simple_sigmoid(0.0, a, θ) < simple_sigmoid(-1.0, a, θ)
    end
end

@testset "Model Integration Tests" begin
    # FIXME needs more integration testing
    @testset "Function Availability" begin
        # Test that all required functions exist
        @test isdefined(FailureOfInhibition2025, :wcm1973!)
        @test isdefined(FailureOfInhibition2025, :population)
        @test isdefined(FailureOfInhibition2025, :apply_nonlinearity!)
        @test isdefined(FailureOfInhibition2025, :SigmoidNonlinearity)
        @test isdefined(FailureOfInhibition2025, :RectifiedZeroedSigmoidNonlinearity)
        @test isdefined(FailureOfInhibition2025, :DifferenceOfSigmoidsNonlinearity)
        @test isdefined(FailureOfInhibition2025, :difference_of_simple_sigmoids)
        @test isdefined(FailureOfInhibition2025, :difference_of_rectified_zeroed_sigmoids)
    end
    
    @testset "Population Function" begin
        # Test 1D case (single population)
        array_1d = [1.0, 2.0, 3.0]
        pop1d = population(array_1d, 1)
        @test pop1d == array_1d
        
        # Test 2D case (multiple populations)
        array_2d = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3x2 array
        pop1 = population(array_2d, 1)
        pop2 = population(array_2d, 2)
        @test pop1 == [1.0, 3.0, 5.0]
        @test pop2 == [2.0, 4.0, 6.0]
    end
end

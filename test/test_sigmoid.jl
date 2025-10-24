#!/usr/bin/env julia

"""
Simple test for sigmoid nonlinearity implementation
"""

using FailureOfInhibition2025

function test_sigmoid_functions()
    println("=== Testing Sigmoid Functions ===")
    
    # Test simple_sigmoid
    println("\n1. Testing simple_sigmoid function:")
    @assert abs(simple_sigmoid(0.0, 1.0, 0.0) - 0.5) < 1e-10
    @assert simple_sigmoid(1.0, 2.0, 0.0) > 0.5  # Positive input with positive slope
    @assert simple_sigmoid(-1.0, 2.0, 0.0) < 0.5  # Negative input with positive slope
    println("   ✓ simple_sigmoid tests passed")
    
    # Test SigmoidNonlinearity type
    println("\n2. Testing SigmoidNonlinearity type:")
    sigmoid = SigmoidNonlinearity(a=2.0, θ=1.0)
    @assert sigmoid.a == 2.0
    @assert sigmoid.θ == 1.0
    println("   ✓ SigmoidNonlinearity construction passed")
    
    # Test RectifiedZeroedSigmoidNonlinearity type
    println("\n3. Testing RectifiedZeroedSigmoidNonlinearity type:")
    rect_sigmoid = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=1.0)
    @assert rect_sigmoid.a == 2.0
    @assert rect_sigmoid.θ == 1.0
    println("   ✓ RectifiedZeroedSigmoidNonlinearity construction passed")
    
    # Test DifferenceOfSigmoidsNonlinearity type
    println("\n4. Testing DifferenceOfSigmoidsNonlinearity type:")
    diff_sigmoid = DifferenceOfSigmoidsNonlinearity(a_up=2.0, θ_up=0.3, a_down=1.0, θ_down=0.7)
    @assert diff_sigmoid.a_up == 2.0
    @assert diff_sigmoid.θ_up == 0.3
    @assert diff_sigmoid.a_down == 1.0
    @assert diff_sigmoid.θ_down == 0.7
    println("   ✓ DifferenceOfSigmoidsNonlinearity construction passed")
    
    # Test applying sigmoid functions directly
    println("\n5. Testing sigmoid function application:")
    # Check that input at threshold gives 0.5
    @assert abs(simple_sigmoid(1.0, 2.0, 1.0) - 0.5) < 1e-10
    # Check that rectified zeroed sigmoid works correctly
    @assert rectified_zeroed_sigmoid(1.0, 2.0, 1.0) >= 0.0
    # Check that difference of simple sigmoids works correctly
    result = difference_of_simple_sigmoids(0.5, 2.0, 0.3, 1.0, 0.7)
    @assert isa(result, Float64)  # Should return a number
    # Check that difference of rectified zeroed sigmoids works correctly
    result_rect = difference_of_rectified_zeroed_sigmoids(0.5, 2.0, 0.3, 1.0, 0.7)
    @assert isa(result_rect, Float64)  # Should return a number
    println("   ✓ Sigmoid function application tests passed")
    
    # Test apply_nonlinearity! function
    println("\n6. Testing apply_nonlinearity! function:")
    dA1 = zeros(3)
    dA2 = zeros(3)
    dA3 = zeros(3)
    A = [0.0, 1.0, 2.0]
    original_A = copy(A)
    
    apply_nonlinearity!(dA1, A, sigmoid, 0.0)
    apply_nonlinearity!(dA2, A, rect_sigmoid, 0.0)
    apply_nonlinearity!(dA3, A, diff_sigmoid, 0.0)
    
    # A should be unchanged
    @assert A == original_A
    # dA should contain the nonlinearity contribution
    @assert !all(dA1 .== 0.0)  # dA1 should have been modified
    @assert !all(dA2 .== 0.0)  # dA2 should have been modified
    @assert !all(dA3 .== 0.0)  # dA3 should have been modified
    println("   ✓ apply_nonlinearity! tests passed")
    
    # Test zero and maximal regions of difference of rectified zeroed sigmoids
    println("\n7. Testing zero and maximal regions of difference of rectified zeroed sigmoids:")
    
    # Create a difference of sigmoids with well-defined parameters for testing
    # Up sigmoid: steep at θ_up=0.5, Down sigmoid: steep at θ_down=1.5
    a_up, θ_up = 5.0, 0.5
    a_down, θ_down = 5.0, 1.5
    
    # Test zero regions: far left and far right should be zero
    far_left = difference_of_rectified_zeroed_sigmoids(-2.0, a_up, θ_up, a_down, θ_down)
    far_right = difference_of_rectified_zeroed_sigmoids(3.0, a_up, θ_up, a_down, θ_down)
    @assert abs(far_left) < 1e-10  # Should be essentially zero
    @assert far_right <= 0.0       # Should be negative or zero (down sigmoid dominates)
    
    # Test regions where function should be zero or minimal
    # Before θ_up, both sigmoids should be near zero, so difference ≈ 0
    before_up = difference_of_rectified_zeroed_sigmoids(0.0, a_up, θ_up, a_down, θ_down)
    @assert abs(before_up) < 0.1   # Should be small
    
    # After θ_down, down sigmoid should dominate, making difference negative or zero
    after_down = difference_of_rectified_zeroed_sigmoids(2.0, a_up, θ_up, a_down, θ_down)
    @assert after_down <= 0.1      # Should be small or negative
    
    # Test maximal region: should occur between θ_up and θ_down
    max_region = difference_of_rectified_zeroed_sigmoids(1.0, a_up, θ_up, a_down, θ_down)
    mid_left = difference_of_rectified_zeroed_sigmoids(0.7, a_up, θ_up, a_down, θ_down)
    mid_right = difference_of_rectified_zeroed_sigmoids(1.3, a_up, θ_up, a_down, θ_down)
    
    # The maximum should be positive and larger than nearby points
    @assert max_region > 0.0
    @assert max_region > mid_left
    @assert max_region > mid_right
    
    # Test that DifferenceOfSigmoidsNonlinearity uses rectified zeroed sigmoids by default
    diff_nl = DifferenceOfSigmoidsNonlinearity(a_up=a_up, θ_up=θ_up, a_down=a_down, θ_down=θ_down)
    dA_test = zeros(1)
    A_test = [1.0]
    apply_nonlinearity!(dA_test, A_test, diff_nl, 0.0)
    
    # The result should match our direct function call
    expected = difference_of_rectified_zeroed_sigmoids(1.0, a_up, θ_up, a_down, θ_down) - 1.0
    @assert abs(dA_test[1] - expected) < 1e-10
    
    println("   ✓ Zero and maximal region tests passed")
    
    println("\n=== All Sigmoid Tests Passed! ===")
end

function test_model_integration()
    println("\n=== Testing Model Integration ===")
    
    # Test that all required functions exist
    println("\n1. Testing function availability:")
    @assert isdefined(FailureOfInhibition2025, :wcm1973!)
    @assert isdefined(FailureOfInhibition2025, :population)
    @assert isdefined(FailureOfInhibition2025, :stimulate)
    @assert isdefined(FailureOfInhibition2025, :apply_nonlinearity!)
    @assert isdefined(FailureOfInhibition2025, :SigmoidNonlinearity)
    @assert isdefined(FailureOfInhibition2025, :RectifiedZeroedSigmoidNonlinearity)
    @assert isdefined(FailureOfInhibition2025, :DifferenceOfSigmoidsNonlinearity)
    @assert isdefined(FailureOfInhibition2025, :difference_of_simple_sigmoids)
    @assert isdefined(FailureOfInhibition2025, :difference_of_rectified_zeroed_sigmoids)
    println("   ✓ All required functions are available")
    
    # Test population function
    println("\n2. Testing population function:")
    # Test 1D case (single population)
    array_1d = [1.0, 2.0, 3.0]
    pop1d = population(array_1d, 1)
    @assert pop1d == array_1d
    
    # Test 2D case (multiple populations)
    array_2d = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3x2 array
    pop1 = population(array_2d, 1)
    pop2 = population(array_2d, 2)
    @assert pop1 == [1.0, 3.0, 5.0]
    @assert pop2 == [2.0, 4.0, 6.0]
    println("   ✓ Population function tests passed")
    
    println("\n=== Model Integration Tests Passed! ===")
end

function main()
    println("Running sigmoid nonlinearity tests...")
    test_sigmoid_functions()
    test_model_integration()
    println("\n🎉 All tests completed successfully!")
    println("\nSigmoid nonlinearity implementation is ready for use in neural models.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

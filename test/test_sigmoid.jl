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
    
    # Test applying sigmoid to array
    println("\n3. Testing sigmoid application to arrays:")
    test_array = [0.0, 1.0, 2.0]
    original_array = copy(test_array)
    sigmoid(test_array)
    
    # Check that values are in sigmoid range (0, 1)
    @assert all(0 .< test_array .< 1)
    # Check that input at threshold gives 0.5
    @assert abs(simple_sigmoid(1.0, 2.0, 1.0) - 0.5) < 1e-10
    println("   ✓ Array application tests passed")
    
    # Test apply_nonlinearity function
    println("\n4. Testing apply_nonlinearity function:")
    dA = zeros(3)
    A = [0.0, 1.0, 2.0]
    original_A = copy(A)
    
    apply_nonlinearity(dA, A, sigmoid, 0.0)
    
    # A should be unchanged
    @assert A == original_A
    # dA should contain the nonlinearity contribution
    @assert !all(dA .== 0.0)  # dA should have been modified
    println("   ✓ apply_nonlinearity tests passed")
    
    println("\n=== All Sigmoid Tests Passed! ===")
end

function test_model_integration()
    println("\n=== Testing Model Integration ===")
    
    # Test that all required functions exist
    println("\n1. Testing function availability:")
    @assert isdefined(FailureOfInhibition2025, :wcm1973!)
    @assert isdefined(FailureOfInhibition2025, :population)
    @assert isdefined(FailureOfInhibition2025, :stimulate)
    @assert isdefined(FailureOfInhibition2025, :apply_nonlinearity)
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
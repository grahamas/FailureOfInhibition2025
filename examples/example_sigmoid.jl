#!/usr/bin/env julia

"""
Example usage of sigmoid nonlinearity in FailureOfInhibition2025

This demonstrates how to create and use sigmoid nonlinearities with the wcm1973! model.
"""

using FailureOfInhibition2025

function demo_sigmoid_usage()
    println("=== Sigmoid Nonlinearity Usage Demo ===")
    
    # Create a sigmoid nonlinearity with custom parameters
    println("\n1. Creating a sigmoid nonlinearity:")
    # a = steepness parameter (higher = steeper)
    # θ = threshold parameter (shifts the sigmoid left/right)
    sigmoid = SigmoidNonlinearity(a=2.0, θ=0.5)
    println("   Created: SigmoidNonlinearity(a=2.0, θ=0.5)")
    
    # Demonstrate the sigmoid function shape
    println("\n2. Sigmoid function values:")
    test_inputs = [-1.0, 0.0, 0.5, 1.0, 2.0]
    for x in test_inputs
        y = simple_sigmoid(x, sigmoid.a, sigmoid.θ)
        println("   f($x) = $(round(y, digits=4))")
    end
    
    # Show how to apply it to neural field data using apply_nonlinearity
    println("\n3. Using apply_nonlinearity with SigmoidNonlinearity:")
    # Simulate some neural field activity
    field_activity = [0.2, 0.8, -0.1, 1.5, 0.0]
    println("   Original activity: $field_activity")
    
    # Show how it would be used in a model differential equation
    dA = zeros(length(field_activity))
    A = copy(field_activity)
    
    println("   dA before: $dA")
    apply_nonlinearity(dA, A, sigmoid, 0.0)  # t=0.0 for this example
    println("   dA after:  $(round.(dA, digits=4))")
    println("   A unchanged: $A")
    
    # Show RectifiedZeroedSigmoidNonlinearity
    println("\n4. Using RectifiedZeroedSigmoidNonlinearity:")
    rect_sigmoid = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5)
    dA_rect = zeros(length(field_activity))
    A_rect = copy(field_activity)
    
    println("   dA_rect before: $dA_rect")
    apply_nonlinearity(dA_rect, A_rect, rect_sigmoid, 0.0)
    println("   dA_rect after:  $(round.(dA_rect, digits=4))")
    println("   A_rect unchanged: $A_rect")
    
    println("\n5. Different sigmoid shapes:")
    # Steeper sigmoid
    steep_sigmoid = SigmoidNonlinearity(a=5.0, θ=0.5)
    # Gentler sigmoid  
    gentle_sigmoid = SigmoidNonlinearity(a=0.5, θ=0.5)
    
    x = 0.7  # Test input
    steep_val = simple_sigmoid(x, steep_sigmoid.a, steep_sigmoid.θ)
    gentle_val = simple_sigmoid(x, gentle_sigmoid.a, gentle_sigmoid.θ)
    
    println("   Input x = $x:")
    println("   Steep sigmoid (a=5.0):   $(round(steep_val, digits=4))")
    println("   Gentle sigmoid (a=0.5):  $(round(gentle_val, digits=4))")
    
    println("\n=== Demo Complete ===")
    println("\nThe sigmoid nonlinearity is now ready to use in neural field models!")
    println("Simply pass a SigmoidNonlinearity or RectifiedZeroedSigmoidNonlinearity object as p.nonlinearity to wcm1973!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo_sigmoid_usage()
end
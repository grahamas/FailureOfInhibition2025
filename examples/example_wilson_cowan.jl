#!/usr/bin/env julia

"""
Example usage of the Wilson-Cowan model in FailureOfInhibition2025

This demonstrates how to set up and use the Wilson-Cowan model (Wilson & Cowan 1973)
with the new implementation that avoids callable objects.
"""

using FailureOfInhibition2025

function demo_wilson_cowan_model()
    println("=== Wilson-Cowan Model Usage Demo ===")
    println("\nThis demonstrates the Wilson-Cowan neural population model")
    println("reimplemented without callable objects.\n")
    
    # 1. Create a 2-population Wilson-Cowan model (E and I populations)
    println("1. Creating Wilson-Cowan model parameters:")
    println("   2 populations: Excitatory (E) and Inhibitory (I)")
    
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),          # Decay rates [E, I]
        β = (1.0, 1.0),          # Saturation coefficients [E, I]
        τ = (1.0, 0.8),          # Time constants [E, I]
        connectivity = nothing,   # No spatial connectivity for this simple example
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),  # Sigmoid activation
        stimulus = nothing,       # No external stimulus
        pop_names = ("E", "I")   # Population names
    )
    
    println("   α (decay rates): ", params.α)
    println("   β (saturation): ", params.β)
    println("   τ (time constants): ", params.τ)
    println("   Populations: ", params.pop_names)
    println("   Nonlinearity: SigmoidNonlinearity(a=2.0, θ=0.5)")
    
    # 2. Set up initial conditions
    println("\n2. Setting up initial conditions:")
    println("   3 spatial points, 2 populations")
    
    # Activity state: 3 spatial points × 2 populations
    A = [0.3 0.2;   # Point 1: [E=0.3, I=0.2]
         0.5 0.4;   # Point 2: [E=0.5, I=0.4]
         0.7 0.6]   # Point 3: [E=0.7, I=0.6]
    
    println("   Initial E population: ", A[:, 1])
    println("   Initial I population: ", A[:, 2])
    
    # Derivative array
    dA = zeros(size(A))
    
    # 3. Compute derivatives using Wilson-Cowan equations
    println("\n3. Computing derivatives using wcm1973!:")
    wcm1973!(dA, A, params, 0.0)
    
    println("   dE/dt: ", round.(dA[:, 1], digits=4))
    println("   dI/dt: ", round.(dA[:, 2], digits=4))
    
    # 4. Show what the function computes
    println("\n4. Understanding the Wilson-Cowan equations:")
    println("   τᵢ dAᵢ/dt = -αᵢ Aᵢ + βᵢ (1 - Aᵢ) f(Sᵢ + Iᵢ)")
    println("   where:")
    println("   - Aᵢ is the activity of population i")
    println("   - αᵢ is the decay rate")
    println("   - βᵢ is the saturation coefficient")
    println("   - τᵢ is the time constant")
    println("   - f is the nonlinearity (firing rate function)")
    println("   - Sᵢ is external stimulus")
    println("   - Iᵢ is recurrent input from connectivity")
    
    # 5. Different nonlinearity examples
    println("\n5. Using different nonlinearities:")
    
    # Rectified zeroed sigmoid
    println("\n   a) RectifiedZeroedSigmoidNonlinearity:")
    params_rect = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = nothing,
        nonlinearity = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        pop_names = ("E", "I")
    )
    
    dA_rect = zeros(size(A))
    wcm1973!(dA_rect, A, params_rect, 0.0)
    println("      dE/dt: ", round.(dA_rect[:, 1], digits=4))
    println("      dI/dt: ", round.(dA_rect[:, 2], digits=4))
    
    # Difference of sigmoids (creates bump-like activation)
    println("\n   b) DifferenceOfSigmoidsNonlinearity (bump function):")
    params_diff = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = nothing,
        nonlinearity = DifferenceOfSigmoidsNonlinearity(
            a_up=5.0, θ_up=0.3, a_down=3.0, θ_down=0.7
        ),
        stimulus = nothing,
        pop_names = ("E", "I")
    )
    
    dA_diff = zeros(size(A))
    wcm1973!(dA_diff, A, params_diff, 0.0)
    println("      dE/dt: ", round.(dA_diff[:, 1], digits=4))
    println("      dI/dt: ", round.(dA_diff[:, 2], digits=4))
    
    # 6. Single population example
    println("\n6. Single population Wilson-Cowan model:")
    params_single = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        pop_names = ("E",)
    )
    
    A_single = [0.3, 0.5, 0.7]  # 1D array for single population
    dA_single = zeros(size(A_single))
    
    wcm1973!(dA_single, A_single, params_single, 0.0)
    
    println("   Initial state: ", A_single)
    println("   Derivatives:   ", round.(dA_single, digits=4))
    
    # 7. Explain implementation differences
    println("\n7. Implementation Differences from WilsonCowanModel.jl:")
    println("   ✓ No callable objects: We use WilsonCowanParameters struct")
    println("     with separate wcm1973! function, not a callable model object")
    println("   ✓ Direct function dispatch: Call wcm1973!(dA, A, params, t)")
    println("     instead of model(space) creating an Action object")
    println("   ✓ Simpler structure: Single parameter struct instead of")
    println("     separate Parameter and Action types")
    println("   ✓ Functional style: Parameters are passed to functions")
    println("     rather than methods being defined on the model struct")
    
    # 8. Time evolution hint
    println("\n8. Time evolution (conceptual):")
    println("   To integrate over time, use an ODE solver like:")
    println("   ```julia")
    println("   using DifferentialEquations")
    println("   prob = ODEProblem(wcm1973!, A₀, tspan, params)")
    println("   sol = solve(prob, Tsit5())")
    println("   ```")
    
    println("\n=== Demo Complete ===")
    println("\nThe Wilson-Cowan model is ready for use in neural dynamics simulations!")
    println("See the implementation documentation in src/models.jl for more details.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo_wilson_cowan_model()
end

#!/usr/bin/env julia

"""
Example usage of the Wilson-Cowan model with PointLattice (non-spatial model)

This demonstrates how to use the same wcm1973! function for non-spatial models
(simple ODEs) by using a PointLattice instead of a spatial lattice.
"""

using FailureOfInhibition2025

function demo_point_model()
    println("=== Wilson-Cowan Point Model Demo ===")
    println("\nThis demonstrates using wcm1973! for non-spatial (point) models")
    println("where there is no spatial structure - just population dynamics.\n")
    
    # 1. Create a point lattice (zero-dimensional)
    println("1. Creating PointLattice (zero-dimensional space):")
    lattice = PointLattice()
    println("   Type: ", typeof(lattice))
    println("   Size: ", size(lattice))
    println("   Coordinates: ", coordinates(lattice))
    
    # 2. Create model components (same as spatial models)
    println("\n2. Creating model components:")
    
    # Sigmoid nonlinearity
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)
    println("   Nonlinearity: SigmoidNonlinearity(a=2.0, θ=0.5)")
    
    # No spatial connectivity for point model
    println("   Connectivity: nothing (no spatial structure)")
    
    # No stimulus for this simple demo
    println("   Stimulus: nothing")
    
    # 3. Create Wilson-Cowan parameters for a point model
    println("\n3. Creating Wilson-Cowan parameters for point model:")
    println("   2 populations: Excitatory (E) and Inhibitory (I)")
    
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),          # Decay rates [E, I]
        β = (1.0, 1.0),          # Saturation coefficients [E, I]
        τ = (1.0, 0.8),          # Time constants [E, I]
        connectivity = nothing,   # No spatial connectivity
        nonlinearity = nonlinearity,
        stimulus = nothing,       # No external stimulus
        lattice = lattice,        # PointLattice - no spatial structure
        pop_names = ("E", "I")   # Population names
    )
    
    println("   α (decay rates): ", params.α)
    println("   β (saturation): ", params.β)
    println("   τ (time constants): ", params.τ)
    println("   Populations: ", params.pop_names)
    
    # 4. Set up initial conditions
    println("\n4. Setting up initial conditions:")
    println("   No spatial dimension - just 2 population values")
    
    # Activity state: 2 populations (no spatial dimension)
    # For a point model, this is just a 1D array
    A = [0.3, 0.5]  # [E_activity, I_activity]
    
    println("   Initial E population: ", A[1])
    println("   Initial I population: ", A[2])
    
    # Derivative array
    dA = zeros(2)
    
    # 5. Compute derivatives using Wilson-Cowan equations
    println("\n5. Computing derivatives using wcm1973!:")
    wcm1973!(dA, A, params, 0.0)
    
    println("   dE/dt: ", round(dA[1], digits=4))
    println("   dI/dt: ", round(dA[2], digits=4))
    
    # 6. Demonstrate that the same function works for both point and spatial models
    println("\n6. Key Insight:")
    println("   ✓ Same wcm1973! function works for both:")
    println("     - Point models (PointLattice): Simple ODEs")
    println("     - Spatial models (CompactLattice, PeriodicLattice): PDEs")
    println("   ✓ Only difference is the lattice type and array shape:")
    println("     - Point: A has shape (P,) for P populations")
    println("     - Spatial: A has shape (N₁, N₂, ..., P) for spatial points")
    
    # 7. Show comparison with spatial model
    println("\n7. Comparison with spatial model:")
    println("\n   Point model (this example):")
    println("   - Lattice: PointLattice()")
    println("   - Activity shape: (2,) for 2 populations")
    println("   - Dynamics: dA/dt = f(A) (ODE)")
    
    println("\n   Spatial model (for comparison):")
    println("   - Lattice: CompactLattice(extent=(10.0,), n_points=(21,))")
    println("   - Activity shape: (21, 2) for 21 spatial points, 2 populations")
    println("   - Dynamics: dA/dt = f(A) + spatial_coupling(A) (PDE)")
    
    # 8. Simple time integration example (conceptual)
    println("\n8. Time evolution (conceptual - forward Euler for illustration):")
    println("   For actual simulations, use DifferentialEquations.jl")
    
    dt = 0.01
    n_steps = 5
    A_current = copy(A)
    
    println("\n   t=0.00: E=", round(A_current[1], digits=3), ", I=", round(A_current[2], digits=3))
    
    for step in 1:n_steps
        dA_step = zeros(2)
        wcm1973!(dA_step, A_current, params, (step-1)*dt)
        A_current .+= dt .* dA_step
        println("   t=", round(step*dt, digits=2), ": E=", round(A_current[1], digits=3), 
                ", I=", round(A_current[2], digits=3))
    end
    
    # 9. Use case scenarios
    println("\n9. When to use PointLattice:")
    println("   ✓ Testing model dynamics without spatial complications")
    println("   ✓ Parameter exploration for non-spatial models")
    println("   ✓ Mean-field approximations")
    println("   ✓ Classical Wilson-Cowan equations (original 1972/1973 formulation)")
    println("   ✓ Faster simulations when spatial structure is not needed")
    
    # 10. Single population example
    println("\n10. Bonus: Single population point model:")
    
    params_single = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    A_single = [0.6]
    dA_single = zeros(1)
    wcm1973!(dA_single, A_single, params_single, 0.0)
    
    println("    Initial activity: ", A_single[1])
    println("    Derivative: ", round(dA_single[1], digits=4))
    println("    (Single population, no spatial structure)")
    
    println("\n=== Demo Complete ===")
    println("\nThe PointLattice allows you to use wcm1973! for simple ODE models")
    println("without spatial structure, using the exact same interface as spatial models!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo_point_model()
end

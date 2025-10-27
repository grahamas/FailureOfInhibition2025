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
    
    # Scalar connectivity for point model (population-to-population weights)
    println("\n   Connectivity: ScalarConnectivity for population interactions")
    conn_ee = ScalarConnectivity(1.0)    # E → E (excitatory self-connection)
    conn_ei = ScalarConnectivity(-0.5)   # I → E (inhibitory to excitatory)
    conn_ie = ScalarConnectivity(0.8)    # E → I (excitatory to inhibitory)
    conn_ii = ScalarConnectivity(-0.3)   # I → I (inhibitory self-connection)
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    println("   - E → E: ", conn_ee.weight, " (excitatory self-connection)")
    println("   - I → E: ", conn_ei.weight, " (inhibitory to excitatory)")
    println("   - E → I: ", conn_ie.weight, " (excitatory to inhibitory)")
    println("   - I → I: ", conn_ii.weight, " (inhibitory self-connection)")
    
    # No stimulus for this simple demo
    println("\n   Stimulus: nothing")
    
    # 3. Create Wilson-Cowan parameters for a point model
    println("\n3. Creating Wilson-Cowan parameters for point model:")
    println("   2 populations: Excitatory (E) and Inhibitory (I)")
    
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),          # Decay rates [E, I]
        β = (1.0, 1.0),          # Saturation coefficients [E, I]
        τ = (1.0, 0.8),          # Time constants [E, I]
        connectivity = connectivity,  # Population-to-population connectivity
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
    println("   Point model with connectivity: use (1, P) shape for P populations")
    
    # Activity state: For point models with connectivity between populations,
    # use shape (1, P) to be consistent with spatial models (N_spatial, P)
    A = reshape([0.3, 0.5], 1, 2)  # Shape: (1, 2) for 1 point, 2 populations
    
    println("   Initial E population: ", A[1,1])
    println("   Initial I population: ", A[1,2])
    println("   Array shape: ", size(A))
    
    # Derivative array
    dA = zeros(1, 2)
    
    # 5. Compute derivatives using Wilson-Cowan equations
    println("\n5. Computing derivatives using wcm1973!:")
    wcm1973!(dA, A, params, 0.0)
    
    println("   dE/dt: ", round(dA[1,1], digits=4))
    println("   dI/dt: ", round(dA[1,2], digits=4))
    
    # 5b. Show effect of connectivity
    println("\n5b. Demonstrating effect of population connectivity:")
    
    # Create same model but without connectivity (can use 1D array)
    params_no_conn = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = nothing,  # No connectivity
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    # Without connectivity, can use simpler 1D array
    A_no_conn = [0.3, 0.5]
    dA_no_conn = zeros(2)
    wcm1973!(dA_no_conn, A_no_conn, params_no_conn, 0.0)
    
    println("   Without connectivity (1D array [E, I]):")
    println("     dE/dt: ", round(dA_no_conn[1], digits=4))
    println("     dI/dt: ", round(dA_no_conn[2], digits=4))
    println("   With connectivity (2D array (1, P)):")
    println("     dE/dt: ", round(dA[1,1], digits=4))
    println("     dI/dt: ", round(dA[1,2], digits=4))
    println("   → Connectivity enables population interactions")
    
    # 6. Demonstrate that the same function works for both point and spatial models
    println("\n6. Key Insight:")
    println("   ✓ Same wcm1973! function works for both:")
    println("     - Point models (PointLattice): Simple ODEs")
    println("     - Spatial models (CompactLattice, PeriodicLattice): PDEs")
    println("   ✓ Point models use ScalarConnectivity for population interactions")
    println("   ✓ Spatial models use GaussianConnectivity for spatial interactions")
    println("   ✓ Array shapes:")
    println("     - Point without connectivity: (P,) for P populations")
    println("     - Point with connectivity: (1, P) for 1 point, P populations")
    println("     - Spatial: (N₁, N₂, ..., P) for spatial points and P populations")
    
    # 7. Show comparison with spatial model
    println("\n7. Comparison with spatial model:")
    println("\n   Point model (this example):")
    println("   - Lattice: PointLattice()")
    println("   - Connectivity: ScalarConnectivity (population weights)")
    println("   - Activity shape: (1, 2) for 1 point, 2 populations")
    println("   - Dynamics: dA/dt = f(A) (ODE)")
    
    println("\n   Spatial model (for comparison):")
    println("   - Lattice: CompactLattice(extent=(10.0,), n_points=(21,))")
    println("   - Connectivity: GaussianConnectivity (spatial kernels)")
    println("   - Activity shape: (21, 2) for 21 spatial points, 2 populations")
    println("   - Dynamics: dA/dt = f(A) + spatial_coupling(A) (PDE)")
    
    # 8. Simple time integration example (conceptual)
    println("\n8. Time evolution (conceptual - forward Euler for illustration):")
    println("   For actual simulations, use DifferentialEquations.jl")
    
    dt = 0.01
    n_steps = 5
    A_current = copy(A)
    
    println("\n   t=0.00: E=", round(A_current[1,1], digits=3), ", I=", round(A_current[1,2], digits=3))
    
    for step in 1:n_steps
        dA_step = zeros(1, 2)
        wcm1973!(dA_step, A_current, params, (step-1)*dt)
        A_current .+= dt .* dA_step
        println("   t=", round(step*dt, digits=2), ": E=", round(A_current[1,1], digits=3), 
                ", I=", round(A_current[1,2], digits=3))
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

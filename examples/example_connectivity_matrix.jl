#!/usr/bin/env julia

"""
Example demonstrating per-population-pair connectivity in the Wilson-Cowan model.

This example shows how to use ConnectivityMatrix to specify different connectivity
kernels for each population pair (E→E, E→I, I→E, I→I).
"""

using FailureOfInhibition2025

function demo_connectivity_matrix()
    println("=== Per-Population-Pair Connectivity Demo ===")
    println("\nThis demonstrates using ConnectivityMatrix for a 2-population")
    println("Wilson-Cowan model with different connectivity for each population pair.\n")
    
    # 1. Create a spatial lattice
    println("1. Creating spatial lattice:")
    println("   1D lattice with 21 points from -5.0 to 5.0")
    
    lattice = CompactLattice(extent=(10.0,), n_points=(21,))
    println("   Lattice extent: ", extent(lattice))
    println("   Number of points: ", size(lattice.arr))
    
    # 2. Create per-population-pair connectivity
    println("\n2. Creating per-population-pair connectivity:")
    println("   For 2 populations (E=Excitatory, I=Inhibitory), we need a 2x2 matrix")
    println("   where connectivity[i,j] maps population j → population i")
    
    # Define connectivity for each pair
    # E → E: Strong excitatory self-connection
    conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
    println("   E → E (conn[1,1]): amplitude=1.0, spread=2.0 (strong excitation)")
    
    # I → E: Moderate inhibitory connection
    conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
    println("   I → E (conn[1,2]): amplitude=-0.5, spread=1.5 (moderate inhibition)")
    
    # E → I: Moderate excitatory connection
    conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
    println("   E → I (conn[2,1]): amplitude=0.8, spread=2.5 (moderate excitation)")
    
    # I → I: Weak inhibitory self-connection
    conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
    println("   I → I (conn[2,2]): amplitude=-0.3, spread=1.0 (weak inhibition)")
    
    # Create the connectivity matrix
    # Note: [i,j] entry represents connection from j to i
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;   # Row 1: inputs to E from [E, I]
        conn_ie conn_ii    # Row 2: inputs to I from [E, I]
    ])
    
    println("\n   Created 2×2 ConnectivityMatrix")
    println("   Matrix structure (A[i,j] maps j → i):")
    println("   [E→E  I→E]   [ 1.0  -0.5]")
    println("   [E→I  I→I] = [ 0.8  -0.3]")
    
    # 3. Create other model components
    println("\n3. Creating other model components:")
    
    # No stimulus for this example
    stimulus = nothing
    println("   Stimulus: none")
    
    # Rectified zeroed sigmoid nonlinearity (biologically realistic)
    nonlinearity = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5)
    println("   Nonlinearity: RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5)")
    
    # 4. Create Wilson-Cowan model parameters
    println("\n4. Creating Wilson-Cowan model parameters:")
    println("   2 populations: Excitatory (E) and Inhibitory (I)")
    
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),          # Decay rates [E, I]
        β = (1.0, 1.0),          # Saturation coefficients [E, I]
        τ = (1.0, 0.8),          # Time constants [E, I]
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = stimulus,
        lattice = lattice,
        pop_names = ("E", "I")   # Population names
    )
    
    println("   α (decay rates): ", params.α)
    println("   β (saturation): ", params.β)
    println("   τ (time constants): ", params.τ)
    println("   Populations: ", params.pop_names)
    
    # 5. Set up initial conditions
    println("\n5. Setting up initial conditions:")
    println("   21 spatial points, 2 populations")
    
    # Activity state: 21 spatial points × 2 populations
    A = zeros(21, 2)
    A[:, 1] .= 0.3  # E population baseline
    A[:, 2] .= 0.2  # I population baseline
    # Add a small perturbation in the middle
    A[11, 1] = 0.5
    
    println("   Initial E population: baseline=0.3, center=0.5")
    println("   Initial I population: baseline=0.2")
    
    # Derivative array
    dA = zeros(size(A))
    
    # 6. Compute derivatives using Wilson-Cowan equations
    println("\n6. Computing derivatives using wcm1973!:")
    
    wcm1973!(dA, A, params, 0.0)
    
    println("   ✓ Derivatives computed successfully")
    println("   E population derivative range: [", 
            round(minimum(dA[:, 1]), digits=4), ", ",
            round(maximum(dA[:, 1]), digits=4), "]")
    println("   I population derivative range: [",
            round(minimum(dA[:, 2]), digits=4), ", ",
            round(maximum(dA[:, 2]), digits=4), "]")
    
    # 7. Explain the connectivity matrix convention
    println("\n7. Understanding the connectivity matrix convention:")
    println("   Matrix indexing follows matrix multiplication rules:")
    println("   connectivity[i,j] describes how population j affects population i")
    println()
    println("   For a 2-population E-I model:")
    println("   ┌─────────┬─────────┐")
    println("   │  E → E  │  I → E  │  Row 1: Inputs to E")
    println("   ├─────────┼─────────┤")
    println("   │  E → I  │  I → I  │  Row 2: Inputs to I")
    println("   └─────────┴─────────┘")
    println("     Col 1       Col 2")
    println("   (from E)    (from I)")
    println()
    println("   This convention matches matrix-vector multiplication:")
    println("   When computing dA = C * A, we get:")
    println("   dA[i] = Σⱼ C[i,j] * A[j]")
    println("   meaning: change in population i = sum over source populations j")
    
    # 8. Demonstrate sparse connectivity
    println("\n8. Sparse connectivity example:")
    println("   You can set connections to 'nothing' to disable them:")
    
    sparse_connectivity = ConnectivityMatrix{2}([
        conn_ee   nothing;   # E → E only, no I → E
        conn_ie   conn_ii    # Both E → I and I → I
    ])
    
    println("   Sparse matrix with no I → E connection:")
    println("   [E→E  ---]")
    println("   [E→I  I→I]")
    println("   This allows flexible network architectures!")
    
    println("\n=== Demo Complete ===")
    println("\nKey takeaways:")
    println("• Use ConnectivityMatrix{P} for P populations")
    println("• Each population pair can have its own connectivity kernel")
    println("• Indexing follows matrix multiplication: [i,j] maps j → i")
    println("• Supports sparse connectivity by using 'nothing'")
    println("• Enables modeling complex E-I networks with realistic connectivity")
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo_connectivity_matrix()
end

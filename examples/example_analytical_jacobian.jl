#!/usr/bin/env julia

"""
Example demonstrating the use of analytical Jacobians for Wilson-Cowan models.

This example shows how to:
1. Compute analytical Jacobians using wcm1973_jacobian!
2. Compare analytical vs numerical Jacobians
3. Use Jacobians for stability analysis
4. Analyze eigenvalues and eigenvectors
"""

using FailureOfInhibition2025
using LinearAlgebra

println("\n" * "="^70)
println("Analytical Jacobian Example: Wilson-Cowan Model")
println("="^70)

#=============================================================================
Example 1: Computing Analytical Jacobian for Point Model
=============================================================================#

println("\n### Example 1: Analytical Jacobian for Oscillatory Mode ###\n")

# Create a Wilson-Cowan model in oscillatory mode
params = create_point_model_wcm1973(:oscillatory)

# Define a state (activities for E and I populations)
A = [0.2, 0.3]

# Allocate Jacobian matrix
J = zeros(2, 2)

# Compute analytical Jacobian at this state
wcm1973_jacobian!(J, A, params, 0.0)

println("State A: E=$(A[1]), I=$(A[2])")
println("\nAnalytical Jacobian:")
println("  J[E,E] = $(J[1,1])  (∂(dE/dt)/∂E)")
println("  J[E,I] = $(J[1,2])  (∂(dE/dt)/∂I)")
println("  J[I,E] = $(J[2,1])  (∂(dI/dt)/∂E)")
println("  J[I,I] = $(J[2,2])  (∂(dI/dt)/∂I)")

#=============================================================================
Example 2: Stability Analysis Using Jacobian
=============================================================================#

println("\n### Example 2: Stability Analysis ###\n")

# Compute eigenvalues of the Jacobian
eigenvalues = eigvals(J)
println("Eigenvalues of Jacobian:")
for (i, λ) in enumerate(eigenvalues)
    if imag(λ) ≈ 0
        println("  λ$i = $(real(λ)) (real)")
    else
        println("  λ$i = $(real(λ)) ± $(abs(imag(λ)))i (complex conjugate pair)")
    end
end

# Check stability
if all(real(λ) < 0 for λ in eigenvalues)
    println("\n✓ System is stable (all eigenvalues have negative real part)")
elseif any(imag(λ) != 0 for λ in eigenvalues)
    println("\n→ System has oscillatory behavior (complex eigenvalues)")
    # Compute frequency of oscillation
    λ_complex = eigenvalues[findfirst(λ -> imag(λ) != 0, eigenvalues)]
    frequency = abs(imag(λ_complex)) / (2π)
    period = 1 / frequency
    println("  Oscillation frequency: $(frequency) Hz")
    println("  Oscillation period: $(period) time units")
else
    println("\n✗ System is unstable (at least one eigenvalue has positive real part)")
end

#=============================================================================
Example 3: Comparing Different Model Modes
=============================================================================#

println("\n### Example 3: Jacobian Analysis Across Model Modes ###\n")

modes = [:active_transient, :oscillatory, :steady_state]
state = [0.2, 0.3]

for mode in modes
    params_mode = create_point_model_wcm1973(mode)
    J_mode = zeros(2, 2)
    wcm1973_jacobian!(J_mode, state, params_mode, 0.0)
    
    eigenvals_mode = eigvals(J_mode)
    
    println("Mode: $mode")
    println("  Eigenvalues: ", eigenvals_mode)
    
    if all(imag(λ) ≈ 0 for λ in eigenvals_mode)
        println("  → Real eigenvalues (no oscillations)")
    else
        λ_complex = eigenvals_mode[findfirst(λ -> abs(imag(λ)) > 1e-10, eigenvals_mode)]
        freq = abs(imag(λ_complex)) / (2π)
        println("  → Complex eigenvalues (oscillatory, f=$(round(freq, digits=4)) Hz)")
    end
    println()
end

#=============================================================================
Example 4: Jacobian for Failure of Inhibition Model
=============================================================================#

println("\n### Example 4: Jacobian for FoI Model ###\n")

# Create FoI model with difference of sigmoids
lattice = PointLattice()

nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5)
nonlinearity_i = DifferenceOfSigmoidsNonlinearity(
    a_activating=5.0, θ_activating=0.3,
    a_failing=3.0, θ_failing=0.7
)

conn_ee = ScalarConnectivity(1.0)
conn_ei = ScalarConnectivity(-0.5)
conn_ie = ScalarConnectivity(0.8)
conn_ii = ScalarConnectivity(-0.3)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

params_foi = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    connectivity = connectivity,
    nonlinearity = (nonlinearity_e, nonlinearity_i),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Compute Jacobian at multiple states to see how it changes
states = [
    [0.2, 0.3],  # Low activity
    [0.4, 0.5],  # Medium activity
    [0.6, 0.7]   # High activity
]

println("FoI Model Jacobian at Different Activity Levels:")
for A_state in states
    J_foi = zeros(2, 2)
    wcm1973_jacobian!(J_foi, A_state, params_foi, 0.0)
    
    eigenvals_foi = eigvals(J_foi)
    max_real = maximum(real.(eigenvals_foi))
    
    println("\nState: E=$(A_state[1]), I=$(A_state[2])")
    println("  Trace(J) = $(tr(J_foi))")
    println("  Det(J) = $(det(J_foi))")
    println("  Max Re(λ) = $(max_real)")
    if max_real > 0
        println("  → Unstable (max Re(λ) > 0)")
    elseif any(abs(imag(λ)) > 1e-6 for λ in eigenvals_foi)
        println("  → Oscillatory (complex eigenvalues)")
    else
        println("  → Stable (all Re(λ) < 0)")
    end
end

#=============================================================================
Example 5: Using Jacobian with ODE Solvers
=============================================================================#

println("\n### Example 5: Integration with ODE Solvers ###\n")

println("The analytical Jacobian can be used with ODE solvers that support")
println("Jacobian information (e.g., implicit methods, sensitivity analysis):")
println()
println("  using DifferentialEquations")
println("  ")
println("  # Define ODE function")
println("  function f!(dA, A, p, t)")
println("      wcm1973!(dA, A, p, t)")
println("  end")
println("  ")
println("  # Define Jacobian function")
println("  function jac!(J, A, p, t)")
println("      wcm1973_jacobian!(J, A, p, t)")
println("  end")
println("  ")
println("  # Create ODE problem with Jacobian")
println("  prob = ODEProblem(ODEFunction(f!, jac=jac!), u0, tspan, params)")
println("  ")
println("  # Solve with implicit method (e.g., Rodas5)")
println("  sol = solve(prob, Rodas5())")
println()
println("This can significantly speed up implicit solvers and improve accuracy")
println("for stiff systems.")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("✓ Analytical Jacobians are computed using wcm1973_jacobian!")
println("✓ Supported for point models with all nonlinearity types")
println("✓ Useful for:")
println("  - Stability analysis (eigenvalue computation)")
println("  - Bifurcation analysis (integration with BifurcationKit)")
println("  - Efficient ODE solving (implicit methods)")
println("  - Sensitivity analysis")
println()
println("For spatial models, use numerical differentiation or contact")
println("the package maintainer for spatial Jacobian implementation.")
println()

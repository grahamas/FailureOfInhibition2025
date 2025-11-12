#!/usr/bin/env julia

"""
Example demonstrating bifurcation analysis of Wilson-Cowan models using BifurcationKit.

This example shows how to use continuation methods to trace bifurcation curves and
detect bifurcation points (Hopf, fold, etc.) in the Wilson-Cowan model. Based on the
three dynamical modes from Wilson & Cowan 1973, we demonstrate how to analyze
transitions between different behaviors.

For more information on BifurcationKit, see:
https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/
"""

using FailureOfInhibition2025
using BifurcationKit

println("\n" * "="^70)
println("Bifurcation Analysis: Wilson-Cowan Model")
println("Using BifurcationKit Continuation Methods")
println("="^70)

#=============================================================================
Introduction to BifurcationKit with Wilson-Cowan Models
=============================================================================#

println("\n### BifurcationKit Continuation Analysis ###\n")
println("This example demonstrates how to use BifurcationKit to:")
println("  - Create a bifurcation problem from Wilson-Cowan parameters")
println("  - Set up continuation parameters")
println("  - Trace solution branches as parameters vary")
println("  - Detect bifurcation points (Hopf, fold, etc.)")
println()

#=============================================================================
Example 1: Setting up a Bifurcation Problem
=============================================================================#

println("\n### Example 1: Creating a BifurcationProblem ###\n")

# Create point model parameters for faster computation
base_params = create_point_model_wcm1973(:active_transient)

# Initial guess for steady state
u0 = reshape([0.1, 0.1], 1, 2)

println("Created Wilson-Cowan parameters:")
println("  - Mode: Active Transient")
println("  - Populations: $(base_params.pop_names)")
println("  - Lattice type: $(typeof(base_params.lattice))")
println()

# Note: To use continuation methods, you need to specify which parameter to vary
# using BifurcationKit's lens system. For example:
# 
# using Setfield
# param_lens = @lens _.α[1]  # Vary decay rate of E population
#
# Then create the bifurcation problem:
# prob = create_bifurcation_problem(base_params, param_lens, u0=u0)
#
# And run continuation:
# opts = ContinuationPar(
#     dsmax = 0.1,      # Maximum continuation step
#     dsmin = 1e-4,     # Minimum continuation step  
#     ds = -0.01,       # Initial step (negative = decrease parameter)
#     maxSteps = 100,   # Maximum number of steps
#     pMin = 0.0,       # Minimum parameter value
#     pMax = 5.0        # Maximum parameter value
# )
# br = continuation(prob, PALC(), opts)

println("To use continuation methods:")
println("  1. Define a parameter lens (e.g., @lens _.α[1])")
println("  2. Create bifurcation problem: create_bifurcation_problem(params, lens)")
println("  3. Set continuation options: ContinuationPar(...)")
println("  4. Run continuation: continuation(prob, PALC(), opts)")
println()

#=============================================================================
Example 2: Understanding the Continuation Interface
=============================================================================#

println("\n### Example 2: Continuation Method Overview ###\n")

println("BifurcationKit continuation methods provide:")
println("  ✓ Automatic detection of bifurcation points")
println("    - Fold (saddle-node) bifurcations")
println("    - Hopf bifurcations (birth of oscillations)")
println("    - Branch points")
println()
println("  ✓ Stability analysis via eigenvalue computation")
println("  ✓ Branch switching at bifurcation points")
println("  ✓ Periodic orbit continuation (for limit cycles)")
println()

#=============================================================================
Example 3: Working with wcm_rhs!
=============================================================================#

println("\n### Example 3: Using wcm_rhs! Function ###\n")

# The wcm_rhs! function adapts the Wilson-Cowan model for BifurcationKit
A = reshape([0.15, 0.12], 1, 2)
dA = zeros(size(A))

# Compute derivatives
wcm_rhs!(dA, A, base_params, 0.0)

println("Wilson-Cowan dynamics at A = [$(A[1,1]), $(A[1,2])]:")
println("  dE/dt = $(round(dA[1,1], digits=6))")
println("  dI/dt = $(round(dA[1,2], digits=6))")
println()
println("This function is used internally by create_bifurcation_problem")
println("to interface with BifurcationKit's continuation methods.")
println()

#=============================================================================
Example 4: Typical Workflow
=============================================================================#

println("\n### Example 4: Typical BifurcationKit Workflow ###\n")

println("```julia")
println("using FailureOfInhibition2025")
println("using BifurcationKit")
println("using Setfield")
println()
println("# 1. Create Wilson-Cowan parameters")
println("params = create_point_model_wcm1973(:active_transient)")
println()
println("# 2. Define which parameter to vary")
println("# Example: vary E-E coupling strength")
println("# Note: This requires modifying connectivity, which is complex")
println("# For simple parameters like α, τ, use:")
println("param_lens = @lens _.α[1]")
println()
println("# 3. Create bifurcation problem")
println("u0 = reshape([0.1, 0.1], 1, 2)")
println("prob = create_bifurcation_problem(params, param_lens, u0=u0)")
println()
println("# 4. Set continuation parameters")
println("opts = ContinuationPar(")
println("    dsmax = 0.1,")
println("    dsmin = 1e-4,")
println("    ds = -0.01,")
println("    maxSteps = 100,")
println("    pMin = 0.0,")
println("    pMax = 5.0")
println(")")
println()
println("# 5. Run continuation")
println("br = continuation(prob, PALC(), opts)")
println()
println("# 6. Analyze results")
println("# - br contains the solution branch")
println("# - Bifurcation points are marked in br")
println("# - Can plot with: plot(br)")
println("```")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ How to create BifurcationProblem objects for Wilson-Cowan models")
println("  ✓ The wcm_rhs! function for BifurcationKit compatibility")
println("  ✓ Overview of continuation method workflow")
println()
println("Key functions:")
println("  - create_bifurcation_problem(): Create BifurcationKit problem")
println("  - wcm_rhs!(): Right-hand side for continuation methods")
println()
println("For detailed BifurcationKit usage, see:")
println("  https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/")
println()
println("Note: Parameter variation requires using BifurcationKit's lens system")
println("to modify nested structures like connectivity matrices.")
println()
println("="^70)

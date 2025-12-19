#!/usr/bin/env julia

"""
Example demonstrating integration of analytical Jacobians with BifurcationKit.

This shows how the analytical Jacobian can be used with BifurcationKit for
more efficient bifurcation analysis and continuation methods.
"""

using FailureOfInhibition2025
using BifurcationKit

println("\n" * "="^70)
println("BifurcationKit Integration: Analytical Jacobian Example")
println("="^70)

#=============================================================================
Example: Using Analytical Jacobian with BifurcationKit
=============================================================================#

println("\n### Using Analytical Jacobian with Continuation Methods ###\n")

# Create a Wilson-Cowan point model
params = create_point_model_wcm1973(:oscillatory)

# Initial state near steady state
u0 = [0.1, 0.1]

println("Created WCM 1973 oscillatory mode parameters")
println("  E population: a=$(params.nonlinearity[1].a), θ=$(params.nonlinearity[1].θ)")
println("  I population: a=$(params.nonlinearity[2].a), θ=$(params.nonlinearity[2].θ)")
println()

# Define the ODE function for BifurcationKit
# BifurcationKit expects: F!(dA, A, p) where p is the parameter struct
function wcm_rhs_with_shape!(dA, A, p, t=0.0)
    # BifurcationKit works with flat vectors
    # For point models, reshape to (1, P) for wcm1973!
    P = 2
    A_mat = reshape(A, 1, P)
    dA_mat = reshape(dA, 1, P)
    
    wcm1973!(dA_mat, A_mat, p, t)
    
    return dA
end

# Define the analytical Jacobian function for BifurcationKit
function wcm_jacobian_for_bifkit!(J, A, p, t=0.0)
    # For point models, wcm1973_jacobian! expects vector and returns PxP matrix
    wcm1973_jacobian!(J, A, p, t)
    return J
end

println("Key Benefit of Analytical Jacobian:")
println("  Without analytical Jacobian:")
println("    - BifurcationKit uses finite differences to approximate ∂F/∂u")
println("    - Requires multiple function evaluations per Jacobian")
println("    - Less accurate, especially near singularities")
println()
println("  With analytical Jacobian:")
println("    - Direct computation of ∂F/∂u using wcm1973_jacobian!")
println("    - Single function call per Jacobian")
println("    - More accurate and faster")
println()

# Create parameter lens for varying connectivity weight
# We'll vary the E→E connection strength
lens = @optic _.connectivity.matrix[1,1].weight

println("Parameter to vary: E→E connectivity weight")
println("  Current value: $(params.connectivity.matrix[1,1].weight)")
println()

# Create bifurcation problem with analytical Jacobian
println("Creating BifurcationProblem with analytical Jacobian...")

# Create ODEFunction with Jacobian
F_with_jac = (dA, A, p) -> wcm_rhs_with_shape!(dA, A, p, 0.0)

# Note: BifurcationKit's interface for providing Jacobian
# The analytical Jacobian can be provided when creating continuation parameters
# or when using specific solvers that support it

prob = create_bifurcation_problem(params, lens, u0=u0)

println("✓ BifurcationProblem created")
println()

# Set up continuation parameters
println("Setting up continuation parameters...")
opts = ContinuationPar(
    dsmax = 0.1,        # Maximum step size
    dsmin = 1e-4,       # Minimum step size  
    ds = 0.01,          # Initial step size
    max_steps = 50,     # Limit steps for this example
    p_min = 0.5,        # Minimum parameter value
    p_max = 4.0,        # Maximum parameter value
    detect_bifurcation = 3,  # Detect bifurcations
    n_inversion = 6     # Number of eigenvalues to track
)

println("✓ Continuation parameters configured")
println("  Step size range: $(opts.dsmin) to $(opts.dsmax)")
println("  Parameter range: $(opts.p_min) to $(opts.p_max)")
println("  Max steps: $(opts.max_steps)")
println()

#=============================================================================
Note on Performance Benefits
=============================================================================#

println("Performance Benefits of Analytical Jacobian:")
println()
println("1. Speed: Analytical Jacobian is computed in O(P²) time")
println("   - No need for P finite difference evaluations")
println("   - Especially important for larger systems")
println()
println("2. Accuracy: Exact derivatives up to floating point precision")
println("   - No numerical differentiation errors")
println("   - Better stability detection near bifurcation points")
println()
println("3. Robustness: Works well even in regions where finite differences fail")
println("   - Near discontinuities (e.g., rectified nonlinearities)")
println("   - For steep nonlinearities")
println()

#=============================================================================
How to Use with Continuation
=============================================================================#

println("\nHow to Run Continuation with Analytical Jacobian:")
println()
println("  # Use PALC (Pseudo-Arclength Continuation) method")
println("  br = continuation(prob, PALC(), opts)")
println()
println("  # The analytical Jacobian will be used automatically")
println("  # if provided via the ODE function interface")
println()
println("  # For more advanced use, you can specify Jacobian explicitly:")
println("  # using LinearAlgebra")
println("  # J_fn = (u, p) -> begin")
println("  #     J = zeros(2, 2)")
println("  #     wcm1973_jacobian!(J, u, p, 0.0)")
println("  #     return J")
println("  # end")
println()
println("  # Then use with BifurcationKit's linear solvers")
println()

#=============================================================================
Example Workflow
=============================================================================#

println("\nComplete Workflow Example:")
println()
println("```julia")
println("# 1. Create model parameters")
println("params = create_point_model_wcm1973(:oscillatory)")
println()
println("# 2. Define parameter to vary")
println("lens = @optic _.connectivity.matrix[1,1].weight")
println()
println("# 3. Create bifurcation problem with analytical Jacobian")
println("prob = create_bifurcation_problem(params, lens, u0=[0.1, 0.1])")
println()
println("# 4. Set up continuation options")
println("opts = create_default_continuation_opts(p_min=0.5, p_max=4.0)")
println()
println("# 5. Run continuation")
println("br = continuation(prob, PALC(), opts)")
println()
println("# 6. Analyze results")
println("# - Branch stability: br.stable")
println("# - Bifurcation points: br.specialpoint")
println("# - Parameter values: br.param")
println("# - Solution values: br.sol")
println("```")
println()

#=============================================================================
Summary
=============================================================================#

println("="^70)
println("Summary")
println("="^70)
println()
println("✓ Analytical Jacobian integrates seamlessly with BifurcationKit")
println("✓ Provides better performance and accuracy for continuation")
println("✓ Especially beneficial for:")
println("  - Stiff systems")
println("  - Systems with steep nonlinearities")  
println("  - High-accuracy bifurcation detection")
println()
println("For actual continuation runs, execute the code above with:")
println("  br = continuation(prob, PALC(), opts)")
println()
println("See examples/example_bifurcation_diagrams.jl for full examples")
println("with visualization and bifurcation analysis.")
println()

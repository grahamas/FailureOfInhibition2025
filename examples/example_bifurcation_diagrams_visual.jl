#!/usr/bin/env julia

"""
Example demonstrating ergonomic bifurcation analysis of Wilson-Cowan models.

This example shows how to use the ergonomic interface to BifurcationKit for
analyzing Wilson-Cowan models. The interface provides helper functions that
make it easy to:
- Create parameter lenses for common parameters (connectivity, nonlinearity)
- Set up continuation with sensible defaults
- Generate bifurcation diagrams

For more advanced usage, see the BifurcationKit documentation:
https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/
"""

using FailureOfInhibition2025
using BifurcationKit
using Plots

println("\n" * "="^70)
println("Bifurcation Diagrams: Wilson-Cowan Model")
println("Using BifurcationKit Continuation Methods")
println("="^70)

#=============================================================================
Example 1: Using the Ergonomic Interface - Varying E-E Connectivity
=============================================================================#

println("\n### Example 1: Ergonomic Interface - E-E Connectivity ###\n")

# Create point model parameters
params1 = create_point_model_wcm1973(:active_transient)
u0_1 = reshape([0.05, 0.05], 1, 2)  # Low initial values work better

println("Using ergonomic helper functions:")
println("  - create_connectivity_lens(i, j) for parameter selection")
println("  - create_default_continuation_opts() for sensible defaults")
println()

# Create lens for E→E connectivity using helper function
lens1 = create_connectivity_lens(1, 1)  # E→E (row 1, col 1)
println("✓ Created connectivity lens for E→E connection")

# Create bifurcation problem
prob1 = create_bifurcation_problem(params1, lens1, u0=u0_1)
println("✓ Created bifurcation problem")

# Use default continuation options with custom range
opts1 = create_default_continuation_opts(p_min=0.5, p_max=3.0, max_steps=150)
println("✓ Created continuation options")
println("  Parameter range: [$(opts1.p_min), $(opts1.p_max)]")
println()

# Note: Continuation can be numerically challenging
# This example demonstrates the ergonomic interface setup
println("Ready for continuation analysis!")
println("  prob1 = bifurcation problem")
println("  opts1 = continuation parameters")
println()
println("To run continuation:")
println("  br = continuation(prob1, PALC(), opts1)")
println()

#=============================================================================
Example 2: Varying Nonlinearity Parameters
=============================================================================#

println("\n### Example 2: Ergonomic Interface - Nonlinearity Threshold ###\n")

# Create oscillatory mode parameters
params2 = create_point_model_wcm1973(:oscillatory)
u0_2 = reshape([0.05, 0.05], 1, 2)

println("Using nonlinearity lens helper:")
println("  - create_nonlinearity_lens(pop_index, :θ or :a)")
println()

# Create lens for E population threshold using helper
lens2 = create_nonlinearity_lens(1, :θ)  # Population 1 (E), threshold
println("✓ Created nonlinearity lens for E threshold")

# Create bifurcation problem  
prob2 = create_bifurcation_problem(params2, lens2, u0=u0_2)
println("✓ Created bifurcation problem")

# Use default options with custom range for threshold
opts2 = create_default_continuation_opts(
    p_min=5.0, 
    p_max=15.0, 
    max_steps=150,
    dsmax=0.1,
    ds=0.05
)
println("✓ Created continuation options")
println("  Threshold range: [$(opts2.p_min), $(opts2.p_max)]")
println()

println("Ready for continuation analysis!")
println("  prob2 = bifurcation problem")
println("  opts2 = continuation parameters")
println()
println("To run continuation:")
println("  br2 = continuation(prob2, PALC(), opts2)")
println()

#=============================================================================
Example 3: Multiple Connectivity Parameters
=============================================================================#

println("\n### Example 3: Working with Multiple Connectivity Parameters ###\n")

# Demonstrate creating lenses for different connectivity pairs
params3 = create_point_model_wcm1973(:steady_state)

println("Creating lenses for all connectivity pairs:")
lens_ee = create_connectivity_lens(1, 1)  # E→E
lens_ei = create_connectivity_lens(1, 2)  # I→E  
lens_ie = create_connectivity_lens(2, 1)  # E→I
lens_ii = create_connectivity_lens(2, 2)  # I→I

println("✓ Created lenses for:")
println("  - E→E (excitatory self-connection)")
println("  - I→E (inhibitory to excitatory)")
println("  - E→I (excitatory to inhibitory)")
println("  - I→I (inhibitory self-connection)")
println()

println("Each lens can be used to create a separate bifurcation problem")
println("to explore how that specific connection affects dynamics.")
println()

#=============================================================================
Example 4: Using Different WCM Modes
=============================================================================#

println("\n### Example 4: Setup for All Three WCM Modes ###\n")

println("Setting up bifurcation problems for all three modes...")

modes = [:active_transient, :oscillatory, :steady_state]
mode_names = ["Active Transient", "Oscillatory", "Steady-State"]

problems_and_opts = []

for (mode, mode_name) in zip(modes, mode_names)
    params = create_point_model_wcm1973(mode)
    u0 = reshape([0.05, 0.05], 1, 2)
    
    # Use helper functions for ergonomic setup
    lens = create_connectivity_lens(1, 1)  # E→E connectivity
    prob = create_bifurcation_problem(params, lens, u0=u0)
    opts = create_default_continuation_opts(p_min=0.5, p_max=3.0, max_steps=150)
    
    push!(problems_and_opts, (mode=mode_name, prob=prob, opts=opts))
    println("✓ Setup complete for $mode_name mode")
end

println()
println("All bifurcation problems are ready!")
println("To run continuation for each:")
println("  for item in problems_and_opts")
println("      br = continuation(item.prob, PALC(), item.opts)")
println("      # Process results...")
println("  end")
println()

#=============================================================================
Summary - Ergonomic Interface to BifurcationKit
=============================================================================#

println("\n" * "="^70)
println("Summary: Ergonomic Interface to BifurcationKit")
println("="^70)
println()
println("This example demonstrated the ergonomic interface which provides:")
println()
println("1. Helper Functions:")
println("   • create_connectivity_lens(i, j) - Easy parameter lens creation")
println("   • create_nonlinearity_lens(pop, :θ|:a) - Nonlinearity parameters")
println("   • create_default_continuation_opts() - Sensible defaults")
println()
println("2. Simplified Workflow:")
println("   params = create_point_model_wcm1973(:oscillatory)")
println("   lens = create_connectivity_lens(1, 1)  # E→E")
println("   prob = create_bifurcation_problem(params, lens)")
println("   opts = create_default_continuation_opts(p_min=0.5, p_max=3.0)")
println("   br = continuation(prob, PALC(), opts)")
println()
println("3. Advantages:")
println("   • No need to manually navigate nested structures")
println("   • Sensible default parameters for WCM models")
println("   • Clear, readable code")
println("   • Easy to explore different parameters")
println()
println("4. Common Parameters:")
println("   • Connectivity: E→E, I→E, E→I, I→I (use indices 1,2)")
println("   • Nonlinearity: θ (threshold), a (slope)")
println("   • All standard BifurcationKit features available")
println()
println("For advanced features:")
println("  - Branch switching and periodic orbits")
println("  - Two-parameter continuation")
println("  - Custom stability analysis")
println()
println("See BifurcationKit documentation:")
println("  https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/")
println()
println("="^70)

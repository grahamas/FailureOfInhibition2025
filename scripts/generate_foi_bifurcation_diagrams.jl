#!/usr/bin/env julia

"""
Generate bifurcation diagrams for Failure of Inhibition (FoI) models.

This script uses a simple parameter sweep approach to explore how
system dynamics change as parameters vary, creating bifurcation-style diagrams.

Usage:
    julia --project=. scripts/generate_foi_bifurcation_diagrams_simple.jl
"""

using FailureOfInhibition2025
using Plots
using DifferentialEquations

println("\n" * "="^70)
println("Bifurcation Analysis: Failure of Inhibition Model")
println("Parameter Sweep Approach")
println("="^70)

#=============================================================================
Setup: Create Base FoI Parameters
=============================================================================#

println("\n### Setting up FoI parameters ###\n")

# Use the oscillatory WCM1973 model as a base
base_params = create_point_model_wcm1973(:oscillatory)

println("✓ Created FoI model parameters (based on WCM1973 oscillatory mode)")
println("  E nonlinearity: $(base_params.nonlinearity[1])")
println("  I nonlinearity: $(base_params.nonlinearity[2])")
println()

#=============================================================================
Bifurcation Analysis 1: Varying E-E Connectivity
=============================================================================#

println("\n### Bifurcation 1: Varying E-E Connectivity ###\n")

# Parameter sweep for E-E connectivity
ee_values = range(1.0, 3.0, length=30)
equilibria_E = Float64[]
equilibria_I = Float64[]
is_stable = Bool[]

println("Running parameter sweep...")
println("  Parameter: E→E connectivity")
println("  Range: [$(first(ee_values)), $(last(ee_values))]")
println("  Points: $(length(ee_values))")

for (idx, ee_weight) in enumerate(ee_values)
    # Create modified parameters
    conn_ee = ScalarConnectivity(ee_weight)
    conn_ei = ScalarConnectivity(-1.5)
    conn_ie = ScalarConnectivity(1.5)
    conn_ii = ScalarConnectivity(-0.1)
    
    connectivity = ConnectivityMatrix{2}([conn_ee conn_ei; conn_ie conn_ii])
    
    params_sweep = WilsonCowanParameters{2}(
        α = base_params.α,
        β = base_params.β,
        τ = base_params.τ,
        connectivity = connectivity,
        nonlinearity = base_params.nonlinearity,
        stimulus = nothing,
        lattice = base_params.lattice,
        pop_names = base_params.pop_names
    )
    
    # Find equilibrium by running to steady state
    u0 = reshape([0.01, 0.01], 1, 2)
    tspan = (0.0, 500.0)
    
    prob = ODEProblem(wcm1973!, u0, tspan, params_sweep)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-8, reltol=1e-6)
    
    # Get final state as equilibrium
    eq_state = sol.u[end]
    push!(equilibria_E, eq_state[1,1])
    push!(equilibria_I, eq_state[1,2])
    
    # Check stability by perturbing and seeing if it returns
    u_perturb = eq_state .+ 0.001 .* randn(size(eq_state))
    prob_stab = ODEProblem(wcm1973!, u_perturb, (0.0, 50.0), params_sweep)
    sol_stab = solve(prob_stab, Tsit5(), abstol=1e-8, reltol=1e-6)
    
    # If final state is close to equilibrium, it's stable
    final_dist = sqrt(sum(abs2, sol_stab.u[end] .- eq_state))
    push!(is_stable, final_dist < 0.01)
    
    if idx % 5 == 0
        println("  Progress: $(idx)/$(length(ee_values))")
    end
end

println("✓ Parameter sweep complete")

# Plot results
println("\nGenerating bifurcation diagram...")

p1 = plot(ee_values, equilibria_E,
          linecolor=ifelse.(is_stable, :blue, :red),
          marker=:circle,
          markersize=3,
          xlabel="E→E Connectivity Strength",
          ylabel="Excitatory Activity (E)",
          title="FoI Bifurcation: E→E Connectivity",
          legend=false,
          linewidth=2)

plot!(p1, [], [], linecolor=:blue, linewidth=2, label="Stable")
plot!(p1, [], [], linecolor=:red, linewidth=2, label="Unstable")

savefig(p1, "foi_bifurcation_ee_connectivity.png")
println("✓ Saved: foi_bifurcation_ee_connectivity.png")

p2 = plot(ee_values, equilibria_I,
          linecolor=ifelse.(is_stable, :blue, :red),
          marker=:circle,
          markersize=3,
          xlabel="E→E Connectivity Strength",
          ylabel="Inhibitory Activity (I)",
          title="FoI Bifurcation: E→E Connectivity (I population)",
          legend=false,
          linewidth=2)

plot!(p2, [], [], linecolor=:blue, linewidth=2, label="Stable")
plot!(p2, [], [], linecolor=:red, linewidth=2, label="Unstable")

savefig(p2, "foi_bifurcation_ee_connectivity_inhibitory.png")
println("✓ Saved: foi_bifurcation_ee_connectivity_inhibitory.png")

p_combined = plot(p1, p2, layout=(2,1), size=(800, 800))
savefig(p_combined, "foi_bifurcation_ee_connectivity_combined.png")
println("✓ Saved: foi_bifurcation_ee_connectivity_combined.png")

#=============================================================================
Bifurcation Analysis 2: Varying Inhibitory Threshold
=============================================================================#

println("\n### Bifurcation 2: Varying Inhibitory Threshold ###\n")

# Parameter sweep for inhibitory threshold
theta_values = range(10.0, 20.0, length=25)
equilibria2_E = Float64[]
equilibria2_I = Float64[]
is_stable2 = Bool[]

println("Running parameter sweep...")
println("  Parameter: Inhibitory threshold (θ_I)")
println("  Range: [$(first(theta_values)), $(last(theta_values))]")
println("  Points: $(length(theta_values))")

for (idx, theta_i) in enumerate(theta_values)
    # Create modified nonlinearity
    nonlin_e = base_params.nonlinearity[1]
    nonlin_i = SigmoidNonlinearity(a=1.0, θ=theta_i)
    
    params_sweep = WilsonCowanParameters{2}(
        α = base_params.α,
        β = base_params.β,
        τ = base_params.τ,
        connectivity = base_params.connectivity,
        nonlinearity = (nonlin_e, nonlin_i),
        stimulus = nothing,
        lattice = base_params.lattice,
        pop_names = base_params.pop_names
    )
    
    # Find equilibrium by running to steady state
    u0 = reshape([0.01, 0.01], 1, 2)
    tspan = (0.0, 500.0)
    
    prob = ODEProblem(wcm1973!, u0, tspan, params_sweep)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-8, reltol=1e-6)
    
    # Get final state as equilibrium
    eq_state = sol.u[end]
    push!(equilibria2_E, eq_state[1,1])
    push!(equilibria2_I, eq_state[1,2])
    
    # Check stability
    u_perturb = eq_state .+ 0.001 .* randn(size(eq_state))
    prob_stab = ODEProblem(wcm1973!, u_perturb, (0.0, 50.0), params_sweep)
    sol_stab = solve(prob_stab, Tsit5(), abstol=1e-8, reltol=1e-6)
    
    final_dist = sqrt(sum(abs2, sol_stab.u[end] .- eq_state))
    push!(is_stable2, final_dist < 0.01)
    
    if idx % 5 == 0
        println("  Progress: $(idx)/$(length(theta_values))")
    end
end

println("✓ Parameter sweep complete")

# Plot results
println("\nGenerating bifurcation diagram...")

p3 = plot(theta_values, equilibria2_E,
          linecolor=ifelse.(is_stable2, :green, :orange),
          marker=:circle,
          markersize=3,
          xlabel="Inhibitory Threshold (θ_I)",
          ylabel="Excitatory Activity (E)",
          title="FoI Bifurcation: Inhibitory Threshold",
          legend=false,
          linewidth=2)

plot!(p3, [], [], linecolor=:green, linewidth=2, label="Stable")
plot!(p3, [], [], linecolor=:orange, linewidth=2, label="Unstable")

savefig(p3, "foi_bifurcation_inhibitory_threshold.png")
println("✓ Saved: foi_bifurcation_inhibitory_threshold.png")

p4 = plot(theta_values, equilibria2_I,
          linecolor=ifelse.(is_stable2, :green, :orange),
          marker=:circle,
          markersize=3,
          xlabel="Inhibitory Threshold (θ_I)",
          ylabel="Inhibitory Activity (I)",
          title="FoI Bifurcation: Inhibitory Threshold (I population)",
          legend=false,
          linewidth=2)

plot!(p4, [], [], linecolor=:green, linewidth=2, label="Stable")
plot!(p4, [], [], linecolor=:orange, linewidth=2, label="Unstable")

savefig(p4, "foi_bifurcation_inhibitory_threshold_inhibitory.png")
println("✓ Saved: foi_bifurcation_inhibitory_threshold_inhibitory.png")

p_combined2 = plot(p3, p4, layout=(2,1), size=(800, 800))
savefig(p_combined2, "foi_bifurcation_inhibitory_threshold_combined.png")
println("✓ Saved: foi_bifurcation_inhibitory_threshold_combined.png")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary: FoI Bifurcation Analysis Complete")
println("="^70)
println()
println("Generated bifurcation diagrams:")
println("  1. E→E Connectivity:")
println("     - foi_bifurcation_ee_connectivity.png")
println("     - foi_bifurcation_ee_connectivity_inhibitory.png")
println("     - foi_bifurcation_ee_connectivity_combined.png")
println()
println("  2. Inhibitory Threshold:")
println("     - foi_bifurcation_inhibitory_threshold.png")
println("     - foi_bifurcation_inhibitory_threshold_inhibitory.png")
println("     - foi_bifurcation_inhibitory_threshold_combined.png")
println()
println("These diagrams show how FoI dynamics emerge as parameters vary,")
println("revealing transitions between different dynamical regimes.")
println("="^70)

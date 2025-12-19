#!/usr/bin/env julia

"""
Generate bifurcation diagrams for full_dynamics_blocking and full_dynamics_monotonic models.

This script performs bifurcation analysis on the two full dynamics models, with particular
focus on the effect of stimulus strength. It generates bifurcation diagrams showing how
the steady states and their stability change as parameters vary.

The analysis includes:
1. Varying stimulus strength for both models
2. Varying E→E connectivity strength
3. Varying I→E connectivity strength  
4. Varying excitatory nonlinearity threshold

Results are saved as PNG files and CSV data files for further analysis.

Usage:
    julia --project=. scripts/generate_bifurcation_diagrams_full_dynamics.jl

Note: This script uses a wrapper approach to make bifurcation analysis work with
the Wilson-Cowan parameter structures. The wrapper allows BifurcationKit to vary
a single parameter while properly reconstructing the full parameter set.
"""

using FailureOfInhibition2025
using BifurcationKit
using Plots
using DataFrames
using CSV

println("\n" * "="^80)
println("Bifurcation Analysis: Full Dynamics Models")
println("="^80)

# Create output directory
output_dir = joinpath(@__DIR__, "..", "data", "bifurcation_diagrams")
mkpath(output_dir)
println("\nOutput directory: $output_dir")

#=============================================================================
Helper Functions
=============================================================================#

"""
Create point model version of full dynamics models.
Converts GaussianConnectivityParameter to ScalarConnectivity using amplitudes.
"""
function create_point_model_full_dynamics(model_name::Symbol)
    # Get default parameters for spatial model to extract values
    if model_name == :full_dynamics_monotonic
        spatial_params = create_full_dynamics_monotonic_parameters()
    elseif model_name == :full_dynamics_blocking
        spatial_params = create_full_dynamics_blocking_parameters()
    else
        error("Unknown model: $model_name")
    end
    
    # Extract connectivity amplitudes from Gaussian parameters
    conn_ee_amp = spatial_params.connectivity.matrix[1,1].amplitude
    conn_ei_amp = spatial_params.connectivity.matrix[1,2].amplitude
    conn_ie_amp = spatial_params.connectivity.matrix[2,1].amplitude
    conn_ii_amp = spatial_params.connectivity.matrix[2,2].amplitude
    
    # Create scalar connectivity for point model
    conn_ee = ScalarConnectivity(conn_ee_amp)
    conn_ei = ScalarConnectivity(conn_ei_amp)
    conn_ie = ScalarConnectivity(conn_ie_amp)
    conn_ii = ScalarConnectivity(conn_ii_amp)
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Create point lattice
    lattice = PointLattice()
    
    # Create point model parameters
    params = WilsonCowanParameters{2}(
        α = spatial_params.α,
        β = spatial_params.β,
        τ = spatial_params.τ,
        connectivity = connectivity,
        nonlinearity = spatial_params.nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = spatial_params.pop_names
    )
    
    return params
end

"""
Parameter wrapper that allows varying a single parameter for bifurcation analysis.
This stores the parameter value and a function to reconstruct the full WCM parameters.
"""
struct BifurcationParamWrapper{F}
    value::Float64
    params_builder::F  # Function that takes value and returns WilsonCowanParameters
end

"""
RHS function for bifurcation analysis that accepts parameter wrapper.
"""
function wcm_bifurcation_rhs!(dA_flat, A_flat, p_wrap::BifurcationParamWrapper, t=0.0)
    params = p_wrap.params_builder(p_wrap.value)
    
    # Reshape for point model: (1, 2) for 1 spatial point, 2 populations
    A = reshape(A_flat, 1, 2)
    dA = reshape(dA_flat, 1, 2)
    
    wcm1973!(dA, A, params, t)
    
    # Flatten back for BifurcationKit
    dA_flat .= vec(dA)
    return dA_flat
end

"""
Save bifurcation branch data to CSV file.
"""
function save_branch_to_csv(br, filename, param_name)
    # Extract parameter values and solution data
    param_vals = br.branch[:, 1]  # Parameter values
    
    # Get number of populations from solution dimension
    n_vars = size(br.branch, 2) - 1  # Subtract 1 for parameter column
    
    # Create dataframe with parameter and all variables
    df = DataFrame(parameter=param_vals)
    for i in 1:n_vars
        df[!, Symbol("var_$i")] = br.branch[:, i+1]
    end
    
    # Add stability information if available
    if hasfield(typeof(br), :stability)
        df.stable = br.stable
    end
    
    CSV.write(filename, df)
    println("  ✓ Saved branch data to: $filename")
end

"""
Create bifurcation plot for a given branch.
"""
function plot_bifurcation_branch(br, param_name, model_name; 
                                  ylabel="Activity", title_suffix="")
    p = plot(br, 
             xlabel=param_name,
             ylabel=ylabel,
             title="Bifurcation Diagram: $model_name $title_suffix",
             legend=:best,
             linewidth=2,
             size=(800, 600))
    return p
end

#=============================================================================
Analysis 1: Effect of Stimulus Strength
=============================================================================#

println("\n### Analysis 1: Effect of Stimulus Strength ###\n")

function analyze_stimulus_effect(model_name::Symbol, output_prefix::String)
    println("Analyzing $model_name with varying stimulus strength...")
    
    # Create parameter builder function
    base_params = create_point_model_full_dynamics(model_name)
    lattice = base_params.lattice
    
    function build_params_with_stimulus(stim_strength::Float64)
        stimulus = ConstantStimulus(
            strength=stim_strength,
            time_windows=[(0.0, Inf)],
            lattice=lattice,
            baseline=0.0
        )
        
        return WilsonCowanParameters{2}(
            α = base_params.α,
            β = base_params.β,
            τ = base_params.τ,
            connectivity = base_params.connectivity,
            nonlinearity = base_params.nonlinearity,
            stimulus = stimulus,
            lattice = lattice,
            pop_names = base_params.pop_names
        )
    end
    
    # Initial condition
    u0 = reshape([0.05, 0.05], 1, 2)
    
    # Create parameter wrapper with initial stimulus strength
    p_wrap = BifurcationParamWrapper(1.0, build_params_with_stimulus)
    
    # Create bifurcation problem with simple lens for the value field
    prob = BifurcationProblem(
        wcm_bifurcation_rhs!,
        vec(u0),
        p_wrap,
        @optic _.value
    )
    
    # Set up continuation parameters
    opts = create_default_continuation_opts(
        p_min=0.0,
        p_max=5.0,
        max_steps=300,
        dsmax=0.05,
        ds=0.01
    )
    
    println("  Running continuation...")
    try
        br = continuation(prob, PALC(), opts; bothside=true)
        
        # Save results
        csv_file = joinpath(output_dir, "$(output_prefix)_stimulus_strength.csv")
        save_branch_to_csv(br, csv_file, "stimulus_strength")
        
        # Create plot
        p = plot_bifurcation_branch(br, "Stimulus Strength", string(model_name),
                                     title_suffix="(Stimulus Effect)")
        png_file = joinpath(output_dir, "$(output_prefix)_stimulus_strength.png")
        savefig(p, png_file)
        println("  ✓ Saved plot to: $png_file")
        
        return br
    catch e
        println("  ✗ Continuation failed: $e")
        return nothing
    end
end

# Analyze both models
println("\n--- Full Dynamics Monotonic ---")
br_mono_stim = analyze_stimulus_effect(:full_dynamics_monotonic, "monotonic")

println("\n--- Full Dynamics Blocking ---")
br_block_stim = analyze_stimulus_effect(:full_dynamics_blocking, "blocking")

#=============================================================================
Analysis 2: Varying E→E Connectivity
=============================================================================#

println("\n### Analysis 2: E→E Connectivity Effect ###\n")

function analyze_connectivity_ee(model_name::Symbol, output_prefix::String)
    println("Analyzing $model_name with varying E→E connectivity...")
    
    # Get base parameters
    base_params = create_point_model_full_dynamics(model_name)
    
    # Extract base connectivity values
    base_conn_ei_w = base_params.connectivity.matrix[1,2].weight
    base_conn_ie_w = base_params.connectivity.matrix[2,1].weight
    base_conn_ii_w = base_params.connectivity.matrix[2,2].weight
    
    function build_params_with_ee(ee_weight::Float64)
        connectivity = ConnectivityMatrix{2}([
            ScalarConnectivity(ee_weight) ScalarConnectivity(base_conn_ei_w);
            ScalarConnectivity(base_conn_ie_w) ScalarConnectivity(base_conn_ii_w)
        ])
        
        return WilsonCowanParameters{2}(
            α = base_params.α,
            β = base_params.β,
            τ = base_params.τ,
            connectivity = connectivity,
            nonlinearity = base_params.nonlinearity,
            stimulus = nothing,
            lattice = base_params.lattice,
            pop_names = base_params.pop_names
        )
    end
    
    # Find better initial condition by solving to steady state at starting parameter
    initial_ee = base_params.connectivity.matrix[1,1].weight
    start_params = build_params_with_ee(initial_ee)
    
    println("  Finding steady state for initial condition...")
    u0 = nothing
    try
        # Solve system to steady state
        u0_test = reshape([0.05, 0.05], 1, 2)
        sol = solve_model(u0_test, (0.0, 200.0), start_params, saveat=1.0)
        u0 = sol.u[end]  # Use final state as initial condition
        println("  ✓ Found steady state: E=$(round(u0[1,1], digits=4)), I=$(round(u0[1,2], digits=4))")
    catch e
        println("  ⚠ Could not find steady state, using default: $e")
        u0 = reshape([0.1, 0.08], 1, 2)
    end
    
    # Create parameter wrapper
    p_wrap = BifurcationParamWrapper(initial_ee, build_params_with_ee)
    
    # Create bifurcation problem
    prob = BifurcationProblem(
        wcm_bifurcation_rhs!,
        vec(u0),
        p_wrap,
        @optic _.value
    )
    
    # Continuation parameters - smaller range and steps for stability
    opts = create_default_continuation_opts(
        p_min=0.5,
        p_max=2.0,
        max_steps=200,
        dsmax=0.02,
        ds=0.005
    )
    
    println("  Running continuation...")
    try
        br = continuation(prob, PALC(), opts; bothside=false)
        
        # Save results
        csv_file = joinpath(output_dir, "$(output_prefix)_ee_connectivity.csv")
        save_branch_to_csv(br, csv_file, "ee_connectivity")
        
        # Create plot
        p = plot_bifurcation_branch(br, "E→E Connectivity Weight", string(model_name),
                                     title_suffix="(E→E Connection)")
        png_file = joinpath(output_dir, "$(output_prefix)_ee_connectivity.png")
        savefig(p, png_file)
        println("  ✓ Saved plot to: $png_file")
        
        return br
    catch e
        println("  ✗ Continuation failed: $e")
        return nothing
    end
end

println("\n--- Full Dynamics Monotonic ---")
br_mono_ee = analyze_connectivity_ee(:full_dynamics_monotonic, "monotonic")

println("\n--- Full Dynamics Blocking ---")
br_block_ee = analyze_connectivity_ee(:full_dynamics_blocking, "blocking")

#=============================================================================
Analysis 3: Varying I→E Connectivity (Inhibitory Effect)
=============================================================================#

println("\n### Analysis 3: I→E Connectivity Effect ###\n")

function analyze_connectivity_ie(model_name::Symbol, output_prefix::String)
    println("Analyzing $model_name with varying I→E connectivity...")
    
    # Get base parameters
    base_params = create_point_model_full_dynamics(model_name)
    
    # Extract base connectivity values
    base_conn_ee_w = base_params.connectivity.matrix[1,1].weight
    base_conn_ie_w = base_params.connectivity.matrix[2,1].weight
    base_conn_ii_w = base_params.connectivity.matrix[2,2].weight
    
    function build_params_with_ie(ei_weight::Float64)
        connectivity = ConnectivityMatrix{2}([
            ScalarConnectivity(base_conn_ee_w) ScalarConnectivity(ei_weight);
            ScalarConnectivity(base_conn_ie_w) ScalarConnectivity(base_conn_ii_w)
        ])
        
        return WilsonCowanParameters{2}(
            α = base_params.α,
            β = base_params.β,
            τ = base_params.τ,
            connectivity = connectivity,
            nonlinearity = base_params.nonlinearity,
            stimulus = nothing,
            lattice = base_params.lattice,
            pop_names = base_params.pop_names
        )
    end
    
    # Find better initial condition
    initial_ei = base_params.connectivity.matrix[1,2].weight
    start_params = build_params_with_ie(initial_ei)
    
    println("  Finding steady state for initial condition...")
    u0 = nothing
    try
        u0_test = reshape([0.05, 0.05], 1, 2)
        sol = solve_model(u0_test, (0.0, 200.0), start_params, saveat=1.0)
        u0 = sol.u[end]
        println("  ✓ Found steady state: E=$(round(u0[1,1], digits=4)), I=$(round(u0[1,2], digits=4))")
    catch e
        println("  ⚠ Could not find steady state, using default: $e")
        u0 = reshape([0.1, 0.08], 1, 2)
    end
    
    # Create parameter wrapper
    p_wrap = BifurcationParamWrapper(initial_ei, build_params_with_ie)
    
    # Create bifurcation problem
    prob = BifurcationProblem(
        wcm_bifurcation_rhs!,
        vec(u0),
        p_wrap,
        @optic _.value
    )
    
    # Continuation parameters - varying inhibitory strength
    # Note: weight is negative, so we vary from -2.0 to -0.5
    opts = create_default_continuation_opts(
        p_min=-2.0,
        p_max=-0.5,
        max_steps=200,
        dsmax=0.02,
        ds=0.005
    )
    
    println("  Running continuation...")
    try
        br = continuation(prob, PALC(), opts; bothside=false)
        
        # Save results
        csv_file = joinpath(output_dir, "$(output_prefix)_ie_connectivity.csv")
        save_branch_to_csv(br, csv_file, "ie_connectivity")
        
        # Create plot
        p = plot_bifurcation_branch(br, "I→E Connectivity Weight", string(model_name),
                                     title_suffix="(Inhibitory Effect)")
        png_file = joinpath(output_dir, "$(output_prefix)_ie_connectivity.png")
        savefig(p, png_file)
        println("  ✓ Saved plot to: $png_file")
        
        return br
    catch e
        println("  ✗ Continuation failed: $e")
        return nothing
    end
end

println("\n--- Full Dynamics Monotonic ---")
br_mono_ie = analyze_connectivity_ie(:full_dynamics_monotonic, "monotonic")

println("\n--- Full Dynamics Blocking ---")
br_block_ie = analyze_connectivity_ie(:full_dynamics_blocking, "blocking")

#=============================================================================
Analysis 4: Varying Excitatory Nonlinearity Threshold
=============================================================================#

println("\n### Analysis 4: Excitatory Threshold Effect ###\n")

function analyze_threshold_e(model_name::Symbol, output_prefix::String)
    println("Analyzing $model_name with varying E threshold...")
    
    # Get base parameters
    base_params = create_point_model_full_dynamics(model_name)
    
    # Extract base values
    base_nonlin_i = base_params.nonlinearity[2]
    base_nonlin_e_a = base_params.nonlinearity[1].a
    
    function build_params_with_threshold(theta_e::Float64)
        # Reconstruct E nonlinearity with new threshold
        if base_params.nonlinearity[1] isa RectifiedZeroedSigmoidNonlinearity
            nonlin_e = RectifiedZeroedSigmoidNonlinearity(a=base_nonlin_e_a, θ=theta_e)
        else
            nonlin_e = SigmoidNonlinearity(a=base_nonlin_e_a, θ=theta_e)
        end
        
        return WilsonCowanParameters{2}(
            α = base_params.α,
            β = base_params.β,
            τ = base_params.τ,
            connectivity = base_params.connectivity,
            nonlinearity = (nonlin_e, base_nonlin_i),
            stimulus = nothing,
            lattice = base_params.lattice,
            pop_names = base_params.pop_names
        )
    end
    
    # Find better initial condition
    initial_theta = base_params.nonlinearity[1].θ
    start_params = build_params_with_threshold(initial_theta)
    
    println("  Finding steady state for initial condition...")
    u0 = nothing
    try
        u0_test = reshape([0.05, 0.05], 1, 2)
        sol = solve_model(u0_test, (0.0, 200.0), start_params, saveat=1.0)
        u0 = sol.u[end]
        println("  ✓ Found steady state: E=$(round(u0[1,1], digits=4)), I=$(round(u0[1,2], digits=4))")
    catch e
        println("  ⚠ Could not find steady state, using default: $e")
        u0 = reshape([0.1, 0.08], 1, 2)
    end
    
    # Create parameter wrapper
    p_wrap = BifurcationParamWrapper(initial_theta, build_params_with_threshold)
    
    # Create bifurcation problem
    prob = BifurcationProblem(
        wcm_bifurcation_rhs!,
        vec(u0),
        p_wrap,
        @optic _.value
    )
    
    # Continuation parameters - smaller range for stability
    opts = create_default_continuation_opts(
        p_min=0.10,
        p_max=0.20,
        max_steps=150,
        dsmax=0.005,
        ds=0.002
    )
    
    println("  Running continuation...")
    try
        br = continuation(prob, PALC(), opts; bothside=false)
        
        # Save results
        csv_file = joinpath(output_dir, "$(output_prefix)_e_threshold.csv")
        save_branch_to_csv(br, csv_file, "e_threshold")
        
        # Create plot
        p = plot_bifurcation_branch(br, "E Threshold (θ)", string(model_name),
                                     title_suffix="(Excitatory Threshold)")
        png_file = joinpath(output_dir, "$(output_prefix)_e_threshold.png")
        savefig(p, png_file)
        println("  ✓ Saved plot to: $png_file")
        
        return br
    catch e
        println("  ✗ Continuation failed: $e")
        return nothing
    end
end

println("\n--- Full Dynamics Monotonic ---")
br_mono_thresh = analyze_threshold_e(:full_dynamics_monotonic, "monotonic")

println("\n--- Full Dynamics Blocking ---")
br_block_thresh = analyze_threshold_e(:full_dynamics_blocking, "blocking")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^80)
println("Bifurcation Analysis Complete")
println("="^80)
println()
println("Generated bifurcation diagrams for:")
println("  • Full Dynamics Monotonic Model")
println("  • Full Dynamics Blocking Model")
println()
println("Parameter variations analyzed:")
println("  1. Stimulus strength (0.0 to 5.0)")
println("  2. E→E connectivity amplitude (0.1 to 3.0)")
println("  3. I→E connectivity amplitude (-3.0 to -0.1)")
println("  4. E population threshold θ (0.05 to 0.5)")
println()
println("Output files saved to: $output_dir")
println("  • PNG plots for visualization")
println("  • CSV data files for further analysis")
println()
println("="^80)

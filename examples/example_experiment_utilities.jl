#!/usr/bin/env julia

"""
Example demonstrating experiment utilities for reproducible research.

This example shows how to use the experiment management utilities to:
1. Create timestamped experiment directories
2. Save experiment metadata (git commit, parameters, etc.)
3. Organize plots and data outputs
4. Ensure reproducibility

All outputs from this example will be grouped in a timestamped directory with
complete metadata for reproducibility.
"""

using FailureOfInhibition2025
using Plots
using Printf

println("="^70)
println("Experiment Utilities Example")
println("="^70)

#=============================================================================
Example 1: Basic Experiment Organization
=============================================================================#

println("\n### Example 1: Basic Experiment with Manual Organization ###\n")

# Create a timestamped experiment directory
exp_dir = create_experiment_dir(
    base_dir="experiments",
    experiment_name="basic_simulation"
)
println("Created experiment directory: $exp_dir")

# Save experiment metadata with git information and description
metadata_file = save_experiment_metadata(
    exp_dir,
    description="Basic Wilson-Cowan simulation demonstrating experiment utilities",
    additional_info=Dict(
        "author" => "Example User",
        "purpose" => "Testing experiment organization",
        "notes" => "First run with new utilities"
    )
)
println("Saved metadata to: $metadata_file")

# Run a simple simulation
lattice = PointLattice()
conn = ScalarConnectivity(0.5)
connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1))

params = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (10.0,),
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E",)
)

A₀ = reshape([0.1], 1, 1)
tspan = (0.0, 50.0)
sol = solve_model(A₀, tspan, params, saveat=0.5)

println("Simulation complete: $(length(sol.t)) time points")

# Save simulation results
results_file = save_experiment_results(sol, exp_dir, "simulation_results", params=params)
println("Saved simulation results to: $results_file")

# Create and save a plot
p = plot(sol.t, [u[1,1] for u in sol.u],
         xlabel="Time", ylabel="Activity",
         title="Point Model Activity",
         linewidth=2, legend=false)
         
plot_file = save_plot(p, exp_dir, "activity_timeseries")
println("Saved plot to: $plot_file")

println("\n✓ Example 1 complete. All outputs saved to: $exp_dir")

#=============================================================================
Example 2: Using ExperimentContext for Convenient Organization
=============================================================================#

println("\n\n### Example 2: Using ExperimentContext ###\n")

# Create an experiment context - provides a convenient interface
exp = ExperimentContext("traveling_wave_analysis")
println("Created experiment context: $(exp.name)")
println("  Directory: $(exp.dir)")
println("  Timestamp: $(exp.timestamp)")

# Set up traveling wave simulation
lattice_tw = CompactLattice(extent=(20.0,), n_points=(101,))

conn_tw = GaussianConnectivityParameter(0.8, (2.5,))
connectivity_tw = ConnectivityMatrix{1}(reshape([conn_tw], 1, 1))

params_tw = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity_tw,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice_tw,
    pop_names = ("E",)
)

# Initial condition: localized bump
A₀_tw = zeros(101, 1)
A₀_tw[15:25, 1] .= 0.6

# Save metadata with parameters
save_experiment_metadata(
    exp.dir,
    params=params_tw,
    description="Traveling wave analysis with Gaussian connectivity",
    additional_info=Dict(
        "lattice_size" => 101,
        "connectivity_width" => 2.5,
        "initial_bump_location" => "points 15-25"
    )
)

println("Running traveling wave simulation...")
sol_tw = solve_model(A₀_tw, (0.0, 40.0), params_tw, saveat=0.2)
println("  ✓ Simulation complete")

# Save results using the context
save_experiment_results(sol_tw, exp.dir, "traveling_wave_data", params=params_tw)
println("  ✓ Saved data")

# Create spatiotemporal plot
n_spatial = length(sol_tw.u[1][:, 1])
n_time = length(sol_tw.t)
activity_matrix = zeros(n_time, n_spatial)
for (t_idx, state) in enumerate(sol_tw.u)
    activity_matrix[t_idx, :] = state[:, 1]
end

coords = coordinates(lattice_tw)
x_coords = [coord[1] for coord in coords]

p_tw = heatmap(x_coords, sol_tw.t, activity_matrix,
               xlabel="Space", ylabel="Time",
               title="Traveling Wave Dynamics",
               c=:viridis, clims=(0, 0.7),
               colorbar_title="Activity")

save_plot(p_tw, exp.dir, "spatiotemporal_plot")
println("  ✓ Saved plot")

# Compute and save metrics
has_peak, trajectory, peak_times = detect_traveling_peak(sol_tw, 1)
distance, _ = compute_distance_traveled(sol_tw, 1, lattice_tw)
decay_rate, amplitudes = compute_decay_rate(sol_tw, 1)

# Save metrics as additional metadata
metrics_file = joinpath(exp.dir, "metrics.txt")
open(metrics_file, "w") do io
    println(io, "Traveling Wave Metrics")
    println(io, "="^50)
    println(io, "Traveling peak detected: $has_peak")
    println(io, "Distance traveled: $(round(distance, digits=2)) units")
    if has_peak && length(peak_times) > 1
        speed = distance / (peak_times[end] - peak_times[1])
        println(io, "Average speed: $(round(speed, digits=3)) units/time")
    end
    if decay_rate !== nothing
        println(io, "Decay rate: $(round(decay_rate, digits=4)) /time")
        println(io, "Half-life: $(round(log(2)/decay_rate, digits=2)) time units")
    end
end
println("  ✓ Saved metrics to: $metrics_file")

println("\n✓ Example 2 complete. All outputs saved to: $(exp.dir)")

#=============================================================================
Example 3: Multiple Simulations in One Experiment
=============================================================================#

println("\n\n### Example 3: Multiple Parameter Sweeps in One Experiment ###\n")

# Create experiment for parameter sweep
exp_sweep = ExperimentContext("parameter_sweep_connectivity")
println("Created experiment: $(exp_sweep.name)")

# Save overall experiment metadata
save_experiment_metadata(
    exp_sweep.dir,
    description="Parameter sweep over connectivity strength",
    additional_info=Dict(
        "parameter" => "connectivity strength",
        "range" => "0.3 to 1.2",
        "n_values" => 5
    )
)

# Parameter sweep
connectivity_strengths = [0.3, 0.5, 0.7, 0.9, 1.2]
sweep_results = []

println("Running parameter sweep...")
for (idx, strength) in enumerate(connectivity_strengths)
    # Create parameters
    conn_sweep = GaussianConnectivityParameter(strength, (2.5,))
    connectivity_sweep = ConnectivityMatrix{1}(reshape([conn_sweep], 1, 1))
    
    params_sweep = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = connectivity_sweep,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus = nothing,
        lattice = lattice_tw,
        pop_names = ("E",)
    )
    
    # Run simulation
    sol_sweep = solve_model(A₀_tw, (0.0, 40.0), params_sweep, saveat=0.2)
    
    # Save results with unique names
    save_experiment_results(
        sol_sweep, 
        exp_sweep.dir, 
        "sim_strength_$(idx)_$(strength)",
        params=params_sweep
    )
    
    # Compute metrics
    distance_sweep, _ = compute_distance_traveled(sol_sweep, 1, lattice_tw)
    amplitude_sweep = compute_amplitude(sol_sweep, 1, method=:max)
    
    push!(sweep_results, (strength=strength, distance=distance_sweep, amplitude=amplitude_sweep))
    
    @printf("  [%d/%d] Strength=%.2f: distance=%.2f, amplitude=%.3f\n",
            idx, length(connectivity_strengths), strength, distance_sweep, amplitude_sweep)
end

# Create summary plot
distances = [r.distance for r in sweep_results]
amplitudes = [r.amplitude for r in sweep_results]

p_sweep = plot(
    connectivity_strengths, distances,
    xlabel="Connectivity Strength", 
    ylabel="Distance Traveled",
    title="Parameter Sweep Results",
    marker=:circle, markersize=6,
    linewidth=2,
    legend=false
)

save_plot(p_sweep, exp_sweep.dir, "sweep_summary")
println("\n✓ Example 3 complete. All outputs saved to: $(exp_sweep.dir)")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary: Experiment Utilities")
println("="^70)
println()
println("This example demonstrated three approaches to experiment organization:")
println()
println("1. Manual Organization:")
println("   • create_experiment_dir() - Create timestamped directory")
println("   • save_experiment_metadata() - Save reproducibility info")
println("   • save_plot() - Save figures to experiment directory")
println("   • save_experiment_results() - Save data to experiment directory")
println()
println("2. ExperimentContext:")
println("   • Convenient wrapper for experiment organization")
println("   • Automatic directory creation and timestamp tracking")
println("   • Easy file path construction with joinpath()")
println()
println("3. Parameter Sweeps:")
println("   • Multiple simulations in one experiment directory")
println("   • Systematic naming for multiple outputs")
println("   • Summary plots and metrics")
println()
println("Key Benefits:")
println("  ✓ Timestamped directories prevent overwriting")
println("  ✓ Git commit IDs ensure reproducibility")
println("  ✓ Metadata files document experiment parameters")
println("  ✓ Organized structure makes results easy to find")
println("  ✓ All information needed to reproduce results is saved")
println()
println("Experiment directories created:")
println("  1. $exp_dir")
println("  2. $(exp.dir)")
println("  3. $(exp_sweep.dir)")
println()
println("Each directory contains:")
println("  • metadata.json - Complete experiment information")
println("  • Data files (.csv) - Simulation results")
println("  • Plots (.png) - Visualizations")
println("  • Additional analysis files")
println()
println("="^70)

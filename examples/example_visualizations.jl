#!/usr/bin/env julia

"""
Comprehensive example demonstrating visualization capabilities for lattice simulations.

This example shows how to create:
1. Time series plots (for all lattice types)
2. Spatial snapshots (for 1D and 2D lattices)
3. Spatiotemporal plots (for 1D lattices)
4. Animations (for 1D and 2D lattices)
5. Multi-population plots
6. Phase portraits (for 2-population models)

Each section demonstrates visualization for a different lattice type:
- Point models (0D)
- 1D compact lattices
- 2D compact lattices
- 1D periodic lattices
"""

using FailureOfInhibition2025
using Plots

# Create output directory for visualizations
output_dir = joinpath(dirname(@__FILE__), "output", "visualizations")
mkpath(output_dir)

println("="^80)
println("Visualization Examples for Lattice Simulations")
println("="^80)
println("\nOutput directory: $output_dir\n")

#=============================================================================
Example 1: Point Model (0D) Visualizations
=============================================================================#

println("\n### Example 1: Point Model (0D) ###\n")

# Create oscillatory point model
params_point = create_point_model_wcm1973(:oscillatory)

# Initial condition
A₀_point = reshape([0.05, 0.05], 1, 2)

# Add sustained stimulus to trigger oscillations
# Create stimulus function
function sustained_stimulus(t)
    if 5.0 <= t < 15.0
        return 20.0
    else
        return 0.0
    end
end

# Time span
tspan = (0.0, 200.0)

println("Solving oscillatory point model...")
sol_point = solve_model(A₀_point, tspan, params_point, saveat=0.5)
println("  ✓ Simulation complete")

# 1. Time series plot
println("Creating time series plot...")
p1 = plot_time_series(sol_point, params_point, 
                     title="Oscillatory Mode: E-I Dynamics")
savefig(p1, joinpath(output_dir, "point_time_series.png"))
println("  ✓ Saved: point_time_series.png")

# 2. Phase portrait
println("Creating phase portrait...")
p2 = plot_phase_portrait(sol_point, params_point,
                        title="Oscillatory Mode: Phase Portrait")
savefig(p2, joinpath(output_dir, "point_phase_portrait.png"))
println("  ✓ Saved: point_phase_portrait.png")

#=============================================================================
Example 2: 1D Compact Lattice Visualizations
=============================================================================#

println("\n### Example 2: 1D Compact Lattice ###\n")

# Create 1D spatial lattice
lattice_1d = CompactLattice(extent=(20.0,), n_points=(101,))

# Single population with lateral connectivity
conn_1d = GaussianConnectivityParameter(0.8, (2.0,))
connectivity_1d = ConnectivityMatrix{1}(reshape([conn_1d], 1, 1))

params_1d = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity_1d,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice_1d,
    pop_names = ("E",)
)

# Initial condition: localized bump
A₀_1d = zeros(101, 1)
A₀_1d[45:55, 1] .= 0.6

tspan_1d = (0.0, 40.0)

println("Solving 1D spatial model...")
sol_1d = solve_model(A₀_1d, tspan_1d, params_1d, saveat=0.2)
println("  ✓ Simulation complete")

# 1. Time series (mean activity)
println("Creating time series plot...")
p3 = plot_time_series(sol_1d, params_1d,
                     title="1D Model: Mean Activity Over Time")
savefig(p3, joinpath(output_dir, "1d_time_series.png"))
println("  ✓ Saved: 1d_time_series.png")

# 2. Spatial snapshot at different times
println("Creating spatial snapshots...")
snapshot_times = [1, 50, 100, length(sol_1d.t)]
snapshots = []
for idx in snapshot_times
    p = plot_spatial_snapshot(sol_1d, params_1d, idx, pop_idx=1)
    push!(snapshots, p)
end
p4 = plot(snapshots..., layout=(2, 2), size=(800, 600))
savefig(p4, joinpath(output_dir, "1d_snapshots.png"))
println("  ✓ Saved: 1d_snapshots.png")

# 3. Spatiotemporal plot
println("Creating spatiotemporal plot...")
p5 = plot_spatiotemporal(sol_1d, params_1d,
                        title="1D Model: Spatiotemporal Activity")
savefig(p5, joinpath(output_dir, "1d_spatiotemporal.png"))
println("  ✓ Saved: 1d_spatiotemporal.png")

# 4. Animation
println("Creating 1D animation...")
anim_1d = animate_1d(sol_1d, params_1d,
                    filename=joinpath(output_dir, "1d_animation.gif"),
                    fps=10,
                    ylims=(0, 0.8))
println("  ✓ Saved: 1d_animation.gif")

#=============================================================================
Example 3: 2D Compact Lattice Visualizations
=============================================================================#

println("\n### Example 3: 2D Compact Lattice ###\n")

# Create 2D spatial lattice (smaller for speed)
lattice_2d = CompactLattice(extent=(10.0, 10.0), n_points=(31, 31))

# Single population with 2D Gaussian connectivity
conn_2d = GaussianConnectivityParameter(0.6, (1.5, 1.5))
connectivity_2d = ConnectivityMatrix{1}(reshape([conn_2d], 1, 1))

params_2d = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity_2d,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.3),
    stimulus = nothing,
    lattice = lattice_2d,
    pop_names = ("E",)
)

# Initial condition: central bump
A₀_2d = zeros(31 * 31, 1)
center_idx = 16  # Middle of 31x31 grid
for i in (center_idx-2):(center_idx+2)
    for j in (center_idx-2):(center_idx+2)
        idx = (i-1) * 31 + j
        A₀_2d[idx, 1] = 0.6
    end
end

tspan_2d = (0.0, 30.0)

println("Solving 2D spatial model...")
sol_2d = solve_model(A₀_2d, tspan_2d, params_2d, saveat=0.5)
println("  ✓ Simulation complete")

# 1. Time series (mean activity)
println("Creating time series plot...")
p6 = plot_time_series(sol_2d, params_2d,
                     title="2D Model: Mean Activity Over Time")
savefig(p6, joinpath(output_dir, "2d_time_series.png"))
println("  ✓ Saved: 2d_time_series.png")

# 2. Spatial heatmap snapshots
println("Creating spatial heatmap snapshots...")
snapshot_times_2d = [1, 20, 40, length(sol_2d.t)]
snapshots_2d = []
for idx in snapshot_times_2d
    p = plot_spatial_snapshot(sol_2d, params_2d, idx, pop_idx=1)
    push!(snapshots_2d, p)
end
p7 = plot(snapshots_2d..., layout=(2, 2), size=(800, 800))
savefig(p7, joinpath(output_dir, "2d_heatmap_snapshots.png"))
println("  ✓ Saved: 2d_heatmap_snapshots.png")

# 3. Animated heatmap
println("Creating 2D animation...")
anim_2d = animate_2d(sol_2d, params_2d,
                    filename=joinpath(output_dir, "2d_animation.gif"),
                    fps=8,
                    clims=(0, 0.8))
println("  ✓ Saved: 2d_animation.gif")

#=============================================================================
Example 4: Multi-Population 1D Model
=============================================================================#

println("\n### Example 4: Multi-Population 1D Model (E-I) ###\n")

# Create 1D lattice with E-I populations
lattice_ei = CompactLattice(extent=(20.0,), n_points=(101,))

# E-I connectivity
conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
conn_ei = GaussianConnectivityParameter(-0.6, (1.5,))
conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
conn_ii = GaussianConnectivityParameter(-0.4, (1.0,))

connectivity_ei = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

params_ei = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (8.0, 6.0),
    connectivity = connectivity_ei,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.3),
    stimulus = nothing,
    lattice = lattice_ei,
    pop_names = ("E", "I")
)

# Initial condition: localized bump in E
A₀_ei = zeros(101, 2)
A₀_ei[45:55, 1] .= 0.6  # E population
A₀_ei[45:55, 2] .= 0.2  # I population

tspan_ei = (0.0, 40.0)

println("Solving E-I spatial model...")
sol_ei = solve_model(A₀_ei, tspan_ei, params_ei, saveat=0.2)
println("  ✓ Simulation complete")

# 1. Time series for both populations
println("Creating multi-population time series...")
p8 = plot_time_series(sol_ei, params_ei,
                     title="E-I Model: Population Dynamics")
savefig(p8, joinpath(output_dir, "ei_time_series.png"))
println("  ✓ Saved: ei_time_series.png")

# 2. Multi-population spatial snapshot
println("Creating multi-population snapshot...")
p9 = plot_multi_population_snapshot(sol_ei, params_ei, 100,
                                   title="E-I Spatial Activity")
savefig(p9, joinpath(output_dir, "ei_multi_population_snapshot.png"))
println("  ✓ Saved: ei_multi_population_snapshot.png")

# 3. Spatiotemporal plots for both populations
println("Creating spatiotemporal plots for E and I...")
p10 = plot_spatiotemporal(sol_ei, params_ei, pop_idx=1,
                         title="E Population: Spatiotemporal Activity")
p11 = plot_spatiotemporal(sol_ei, params_ei, pop_idx=2,
                         title="I Population: Spatiotemporal Activity")
p12 = plot(p10, p11, layout=(2, 1), size=(600, 800))
savefig(p12, joinpath(output_dir, "ei_spatiotemporal.png"))
println("  ✓ Saved: ei_spatiotemporal.png")

# 4. Phase portrait (using spatial mean)
println("Creating phase portrait...")
p13 = plot_phase_portrait(sol_ei, params_ei,
                         title="E-I Model: Phase Portrait (Spatial Mean)")
savefig(p13, joinpath(output_dir, "ei_phase_portrait.png"))
println("  ✓ Saved: ei_phase_portrait.png")

#=============================================================================
Example 5: 1D Periodic Lattice
=============================================================================#

println("\n### Example 5: 1D Periodic Lattice ###\n")

# Create 1D periodic lattice
lattice_periodic = PeriodicLattice(extent=(20.0,), n_points=(101,))

# Single population with lateral connectivity
conn_periodic = GaussianConnectivityParameter(0.8, (2.0,))
connectivity_periodic = ConnectivityMatrix{1}(reshape([conn_periodic], 1, 1))

params_periodic = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity_periodic,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice_periodic,
    pop_names = ("E",)
)

# Initial condition: localized bump near edge to show wrapping
A₀_periodic = zeros(101, 1)
A₀_periodic[95:101, 1] .= 0.6  # Near right edge

tspan_periodic = (0.0, 40.0)

println("Solving 1D periodic model...")
sol_periodic = solve_model(A₀_periodic, tspan_periodic, params_periodic, saveat=0.2)
println("  ✓ Simulation complete")

# 1. Spatiotemporal plot showing periodic boundary
println("Creating spatiotemporal plot (periodic boundaries)...")
p14 = plot_spatiotemporal(sol_periodic, params_periodic,
                         title="Periodic Lattice: Activity Wraps at Boundaries")
savefig(p14, joinpath(output_dir, "periodic_spatiotemporal.png"))
println("  ✓ Saved: periodic_spatiotemporal.png")

# 2. Animation
println("Creating periodic lattice animation...")
anim_periodic = animate_1d(sol_periodic, params_periodic,
                          filename=joinpath(output_dir, "periodic_animation.gif"),
                          fps=10,
                          ylims=(0, 0.8))
println("  ✓ Saved: periodic_animation.gif")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^80)
println("Summary")
println("="^80)
println()
println("Created visualizations for all lattice types:")
println()
println("Point Models (0D):")
println("  ✓ Time series plots")
println("  ✓ Phase portraits")
println()
println("1D Lattices:")
println("  ✓ Time series (mean activity)")
println("  ✓ Spatial snapshots")
println("  ✓ Spatiotemporal plots (space × time)")
println("  ✓ Animations")
println()
println("2D Lattices:")
println("  ✓ Time series (mean activity)")
println("  ✓ Heatmap snapshots")
println("  ✓ Animated heatmaps")
println()
println("Multi-Population Models:")
println("  ✓ Combined time series")
println("  ✓ Multi-population snapshots")
println("  ✓ Individual spatiotemporal plots")
println("  ✓ Phase portraits")
println()
println("Periodic Lattices:")
println("  ✓ Spatiotemporal plots showing boundary wrapping")
println("  ✓ Animations")
println()
println("All visualizations saved to: $output_dir")
println()
println("Functions demonstrated:")
println("  - plot_time_series()")
println("  - plot_spatial_snapshot()")
println("  - plot_spatiotemporal()")
println("  - animate_1d()")
println("  - animate_2d()")
println("  - plot_multi_population_snapshot()")
println("  - plot_phase_portrait()")
println()
println("="^80)

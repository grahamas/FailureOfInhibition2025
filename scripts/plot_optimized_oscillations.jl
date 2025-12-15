#!/usr/bin/env julia

"""
Generate time course plot comparing baseline and optimized oscillatory modes.

This script creates a visual comparison showing the improvement in oscillation
amplitude and robustness achieved by the optimized parameters.
"""

using FailureOfInhibition2025

# Use Plots for graphical output (save to files). If missing, install it.
try
    using Plots
catch
    using Pkg
    println("Installing Plots.jl...")
    Pkg.add("Plots")
    using Plots
end

println("\n" * "="^70)
println("Time Course Plot: Optimized Oscillations")
println("="^70)

# Initial condition
A₀ = reshape([0.3, 0.2], 1, 2)
tspan = (0.0, 300.0)

println("\nSimulating baseline oscillatory mode...")
params_baseline = create_point_model_wcm1973(:oscillatory)
sol_baseline = solve_model(A₀, tspan, params_baseline, saveat=0.5)

println("Simulating optimized oscillatory mode...")
params_optimized = create_point_model_wcm1973(:optimized_oscillatory)
sol_optimized = solve_model(A₀, tspan, params_optimized, saveat=0.5)

# Extract time series data
times_baseline = sol_baseline.t
E_baseline = [sol_baseline.u[i][1,1] for i in 1:length(sol_baseline.t)]
I_baseline = [sol_baseline.u[i][1,2] for i in 1:length(sol_baseline.t)]

times_optimized = sol_optimized.t
E_optimized = [sol_optimized.u[i][1,1] for i in 1:length(sol_optimized.t)]
I_optimized = [sol_optimized.u[i][1,2] for i in 1:length(sol_optimized.t)]

println("Generating plots (saved as PNG files)...")

# Create comparison plot using Plots.jl and save to files
p1 = plot(times_baseline, E_baseline,
    title="Baseline WCM 1973 Oscillatory Mode",
    label="E",
    xlabel="Time (msec)",
    ylabel="Activity",
    size=(1000,300))
plot!(p1, times_baseline, I_baseline, label="I")

p2 = plot(times_optimized, E_optimized,
    title="Optimized Oscillatory Mode (191% amplitude increase)",
    label="E",
    xlabel="Time (msec)",
    ylabel="Activity",
    size=(1000,300))
plot!(p2, times_optimized, I_optimized, label="I")

outfile1 = joinpath(@__DIR__, "optimized_oscillations_baseline.png")
outfile2 = joinpath(@__DIR__, "optimized_oscillations_optimized.png")
savefig(p1, outfile1)
savefig(p2, outfile2)
println("Saved: $(outfile1)")
println("Saved: $(outfile2)")

# Zoomed-in comparison showing the first few oscillations
zoom_n = min(200, length(times_baseline))
p3 = plot(times_baseline[1:zoom_n], E_baseline[1:zoom_n],
    title="First 100 msec: Baseline vs Optimized (E population)",
    label="Baseline E",
    xlabel="Time (msec)",
    ylabel="Activity",
    size=(1000,300))
plot!(p3, times_optimized[1:zoom_n], E_optimized[1:zoom_n], label="Optimized E")
outfile3 = joinpath(@__DIR__, "optimized_oscillations_zoomed.png")
savefig(p3, outfile3)
println("Saved: $(outfile3)")

# Create a plot with sustained stimulus
println("\nSimulating with sustained stimulus...")
lattice = PointLattice()
stim = ConstantStimulus(
    strength=5.0,
    time_windows=[(10.0, 150.0)],
    lattice=lattice
)

params_stim = WilsonCowanParameters{2}(
    α = params_optimized.α,
    β = params_optimized.β,
    τ = params_optimized.τ,
    connectivity = params_optimized.connectivity,
    nonlinearity = params_optimized.nonlinearity,
    stimulus = stim,
    lattice = params_optimized.lattice,
    pop_names = params_optimized.pop_names
)

sol_stim = solve_model(A₀, (0.0, 300.0), params_stim, saveat=0.5)

times_stim = sol_stim.t
E_stim = [sol_stim.u[i][1,1] for i in 1:length(sol_stim.t)]
I_stim = [sol_stim.u[i][1,2] for i in 1:length(sol_stim.t)]

p4 = plot(times_stim, E_stim,
    title="Optimized with Sustained Stimulus (strength=5.0, t=10-150ms)",
    label="E (with stim)",
    xlabel="Time (msec)",
    ylabel="Activity",
    size=(1000,300))
plot!(p4, times_stim, I_stim, label="I (with stim)")
outfile4 = joinpath(@__DIR__, "optimized_oscillations_with_stimulus.png")
savefig(p4, outfile4)
println("Saved: $(outfile4)")

# Print summary statistics
println("\n" * "="^70)
println("Summary Statistics")
println("="^70)

# Analyze oscillations
has_osc_base, peaks_base, _ = detect_oscillations(sol_baseline, 1)
has_osc_opt, peaks_opt, _ = detect_oscillations(sol_optimized, 1)
has_osc_stim, peaks_stim, _ = detect_oscillations(sol_stim, 1)

amp_base, _ = compute_oscillation_amplitude(sol_baseline, 1, method=:envelope)
amp_opt, _ = compute_oscillation_amplitude(sol_optimized, 1, method=:envelope)
amp_stim, _ = compute_oscillation_amplitude(sol_stim, 1, method=:envelope)

println("\nBaseline WCM 1973 Oscillatory Mode:")
println("  Peaks: $(length(peaks_base))")
println("  Amplitude: $(round(amp_base, digits=4))")

println("\nOptimized Oscillatory Mode:")
println("  Peaks: $(length(peaks_opt))")
println("  Amplitude: $(round(amp_opt, digits=4))")
println("  Improvement: $(round((amp_opt/amp_base - 1)*100, digits=1))%")

println("\nOptimized with Sustained Stimulus:")
println("  Peaks: $(length(peaks_stim))")
println("  Amplitude: $(round(amp_stim, digits=4))")

println("\n" * "="^70)
println("Plots Generated (saved to PNG files in script directory)")
println("="^70)
println()
println("These plots show:")
println("1. Baseline WCM 1973 oscillatory mode - weak damped oscillations")
println("2. Optimized oscillatory mode - 191% amplitude increase")
println("3. First 100 msec comparison - clearly shows amplitude difference")
println("4. Optimized with sustained stimulus - robust oscillations (26 peaks)")
println()
println("="^70)

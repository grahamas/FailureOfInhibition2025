#!/usr/bin/env julia

"""
Example: GPU-Accelerated Parameter Search and Simulation

This example demonstrates how to use GPU acceleration for:
1. Solving Wilson-Cowan models
2. Performing global sensitivity analysis (Sobol and Morris methods)
3. Optimizing parameters for traveling waves

GPU acceleration can significantly speed up computationally intensive tasks,
especially for large-scale parameter searches and spatial models.

Prerequisites:
- CUDA-capable GPU
- CUDA.jl package installed: `using Pkg; Pkg.add("CUDA")`

Note: If CUDA is not available, the functions will automatically fall back to CPU versions.
"""

using FailureOfInhibition2025
using Statistics
using Optim

# Try to load CUDA
has_cuda = false
try
    using CUDA
    if CUDA.functional()
        has_cuda = true
        println("✓ CUDA is available and functional")
        println("GPU: ", CUDA.name(CUDA.device()))
        println()
    else
        println("⚠ CUDA is installed but not functional")
        println("Falling back to CPU versions")
        println()
    end
catch e
    println("⚠ CUDA.jl not installed or loaded")
    println("Install with: using Pkg; Pkg.add(\"CUDA\")")
    println("Falling back to CPU versions")
    println()
end

println("="^70)
println("Example 1: GPU-Accelerated Simulation")
println("="^70)
println()

# Create a spatial Wilson-Cowan model
lattice = CompactLattice(extent=(20.0,), n_points=(101,))
conn = GaussianConnectivityParameter(1.0, (2.0,))
params = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E",)
)

# Create initial condition with localized activity
A₀ = zeros(101, 1)
A₀[45:55, 1] .= 0.5

tspan = (0.0, 40.0)

if has_cuda
    println("Running simulation on GPU...")
    @time sol_gpu = solve_model_gpu(A₀, tspan, params, saveat=0.1)
    println("Final mean activity (GPU): ", mean(sol_gpu.u[end]))
    println()
else
    println("Running simulation on CPU (CUDA not available)...")
    @time sol_cpu = solve_model(A₀, tspan, params, saveat=0.1)
    println("Final mean activity (CPU): ", mean(sol_cpu.u[end]))
    println()
end

println("="^70)
println("Example 2: GPU-Accelerated Sobol Sensitivity Analysis")
println("="^70)
println()

# Create a point model for sensitivity analysis
lattice_point = PointLattice()
connectivity_point = ConnectivityMatrix{2}([
    ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
    ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
])

base_params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    connectivity = connectivity_point,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice_point,
    pop_names = ("E", "I")
)

# Define parameters to analyze
param_ranges = [
    ("α_E", 0.5, 2.0),       # Decay rate for E
    ("α_I", 0.5, 2.0),       # Decay rate for I
    ("τ_E", 5.0, 15.0),      # Time constant for E
    ("τ_I", 4.0, 12.0)       # Time constant for I
]

if has_cuda
    println("Running Sobol analysis on GPU with 500 samples...")
    @time result = sobol_sensitivity_analysis_gpu(
        base_params,
        param_ranges,
        500,
        tspan=(0.0, 100.0),
        output_metric=:final_mean
    )
    
    println("\nResults:")
    println("Method: ", result[:method])
    println("\nFirst-order indices (S1):")
    for (i, name) in enumerate(result[:param_names])
        println("  $name: ", round(result[:S1][i], digits=4))
    end
    println("\nTotal-order indices (ST):")
    for (i, name) in enumerate(result[:param_names])
        println("  $name: ", round(result[:ST][i], digits=4))
    end
    println()
else
    println("Running Sobol analysis on CPU with 500 samples...")
    @time result = sobol_sensitivity_analysis(
        base_params,
        param_ranges,
        500,
        tspan=(0.0, 100.0),
        output_metric=:final_mean
    )
    
    println("\nResults:")
    println("Method: ", result[:method])
    println("\nFirst-order indices (S1):")
    for (i, name) in enumerate(result[:param_names])
        println("  $name: ", round(result[:S1][i], digits=4))
    end
    println()
end

println("="^70)
println("Example 3: GPU-Accelerated Morris Screening")
println("="^70)
println()

if has_cuda
    println("Running Morris screening on GPU with 100 trajectories...")
    @time result_morris = morris_sensitivity_analysis_gpu(
        base_params,
        param_ranges,
        100,
        tspan=(0.0, 100.0),
        output_metric=:final_mean
    )
    
    println("\nResults:")
    println("Method: ", result_morris[:method])
    println("\nParameter importance (μ*):")
    for (i, name) in enumerate(result_morris[:param_names])
        println("  $name: ", round(result_morris[:means_star][i], digits=4))
    end
    println("\nParameter interactions (σ²):")
    for (i, name) in enumerate(result_morris[:param_names])
        println("  $name: ", round(result_morris[:variances][i], digits=4))
    end
    println()
else
    println("Running Morris screening on CPU with 100 trajectories...")
    @time result_morris = morris_sensitivity_analysis(
        base_params,
        param_ranges,
        100,
        tspan=(0.0, 100.0),
        output_metric=:final_mean
    )
    
    println("\nResults:")
    println("Method: ", result_morris[:method])
    println("\nParameter importance (μ*):")
    for (i, name) in enumerate(result_morris[:param_names])
        println("  $name: ", round(result_morris[:means_star][i], digits=4))
    end
    println()
end

println("="^70)
println("Example 4: GPU-Accelerated Parameter Optimization")
println("="^70)
println()

# Define parameter ranges to optimize
param_ranges_opt = (
    connectivity_width = (1.5, 3.5),
    sigmoid_a = (1.5, 3.0)
)

# Define optimization objective
objective = TravelingWaveObjective(
    target_distance=nothing,  # Maximize distance
    minimize_decay=true,
    require_traveling=false
)

if has_cuda
    println("Running parameter optimization on GPU...")
    @time result_opt, best_params = optimize_for_traveling_wave_gpu(
        params,
        param_ranges_opt,
        objective,
        A₀,
        (0.0, 40.0),
        maxiter=20
    )
    
    println("\nOptimization converged: ", Optim.converged(result_opt))
    println("Final objective value: ", round(Optim.minimum(result_opt), digits=4))
    println("Optimized connectivity width: ", round(best_params.connectivity[1,1].spread[1], digits=4))
    println("Optimized sigmoid a: ", round(best_params.nonlinearity.a, digits=4))
    println()
else
    println("Running parameter optimization on CPU...")
    @time result_opt, best_params = optimize_for_traveling_wave(
        params,
        param_ranges_opt,
        objective,
        A₀,
        (0.0, 40.0),
        maxiter=20
    )
    
    println("\nOptimization converged: ", Optim.converged(result_opt))
    println("Final objective value: ", round(Optim.minimum(result_opt), digits=4))
    println()
end

println("="^70)
println("Summary")
println("="^70)
println()

if has_cuda
    println("✓ GPU acceleration is enabled for:")
    println("  - Simulation (solve_model_gpu)")
    println("  - Sobol sensitivity analysis (sobol_sensitivity_analysis_gpu)")
    println("  - Morris screening (morris_sensitivity_analysis_gpu)")
    println("  - Parameter optimization (optimize_for_traveling_wave_gpu)")
    println()
    println("GPU acceleration provides significant speedup for:")
    println("  - Large spatial models (many grid points)")
    println("  - Extensive parameter searches (many samples)")
    println("  - Long-running simulations")
else
    println("⚠ GPU acceleration not available")
    println()
    println("To enable GPU acceleration:")
    println("  1. Ensure you have a CUDA-capable GPU")
    println("  2. Install CUDA.jl: using Pkg; Pkg.add(\"CUDA\")")
    println("  3. CUDA.jl will automatically detect and configure your GPU")
    println()
    println("All functions automatically fall back to CPU when GPU is not available.")
end

println()
println("Example completed successfully!")

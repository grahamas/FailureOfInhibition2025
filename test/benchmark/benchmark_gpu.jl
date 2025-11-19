"""
Benchmark GPU-accelerated functions vs CPU versions.

These benchmarks compare the performance of GPU-accelerated simulations,
sensitivity analysis, and optimization against their CPU counterparts.

Benchmarks only run if CUDA is available and functional.
"""

include("benchmark_utils.jl")

using FailureOfInhibition2025

# Check if CUDA is available
const has_cuda = let
    local available = false
    try
        using CUDA
        if CUDA.functional()
            available = true
            println("GPU detected: ", CUDA.name(CUDA.device()))
        else
            println("CUDA installed but not functional")
        end
    catch e
        println("CUDA not available: ", e)
    end
    available
end

"""
Benchmark GPU vs CPU simulations.
"""
function benchmark_gpu_simulations()
    if !has_cuda
        println("\n=== Skipping GPU Simulation Benchmarks (CUDA not available) ===")
        return []
    end
    
    println("\n=== Benchmarking GPU Simulations ===")
    results = []
    
    # Small spatial model (51 points)
    lattice_small = CompactLattice(extent=(10.0,), n_points=(51,))
    conn_small = GaussianConnectivityParameter(1.0, (2.0,))
    params_small = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = ConnectivityMatrix{1}(reshape([conn_small], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus = nothing,
        lattice = lattice_small,
        pop_names = ("E",)
    )
    
    A₀_small = zeros(51, 1)
    A₀_small[20:30, 1] .= 0.5
    tspan_small = (0.0, 10.0)
    
    # CPU version
    result = benchmark_function("Simulation CPU: 51 points, 1 population", samples=10) do
        solve_model(A₀_small, tspan_small, params_small, saveat=0.1)
    end
    push!(results, result)
    
    # GPU version
    result = benchmark_function("Simulation GPU: 51 points, 1 population", samples=10) do
        solve_model_gpu(A₀_small, tspan_small, params_small, saveat=0.1)
    end
    push!(results, result)
    
    # Medium spatial model (101 points)
    lattice_medium = CompactLattice(extent=(20.0,), n_points=(101,))
    conn_medium = GaussianConnectivityParameter(1.0, (2.0,))
    params_medium = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = ConnectivityMatrix{1}(reshape([conn_medium], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus = nothing,
        lattice = lattice_medium,
        pop_names = ("E",)
    )
    
    A₀_medium = zeros(101, 1)
    A₀_medium[45:55, 1] .= 0.5
    tspan_medium = (0.0, 20.0)
    
    # CPU version
    result = benchmark_function("Simulation CPU: 101 points, 1 population", samples=10) do
        solve_model(A₀_medium, tspan_medium, params_medium, saveat=0.1)
    end
    push!(results, result)
    
    # GPU version
    result = benchmark_function("Simulation GPU: 101 points, 1 population", samples=10) do
        solve_model_gpu(A₀_medium, tspan_medium, params_medium, saveat=0.1)
    end
    push!(results, result)
    
    # Large spatial model (201 points)
    lattice_large = CompactLattice(extent=(30.0,), n_points=(201,))
    conn_large = GaussianConnectivityParameter(1.0, (2.0,))
    params_large = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = ConnectivityMatrix{1}(reshape([conn_large], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus = nothing,
        lattice = lattice_large,
        pop_names = ("E",)
    )
    
    A₀_large = zeros(201, 1)
    A₀_large[95:105, 1] .= 0.5
    tspan_large = (0.0, 30.0)
    
    # CPU version
    result = benchmark_function("Simulation CPU: 201 points, 1 population", samples=5) do
        solve_model(A₀_large, tspan_large, params_large, saveat=0.1)
    end
    push!(results, result)
    
    # GPU version
    result = benchmark_function("Simulation GPU: 201 points, 1 population", samples=5) do
        solve_model_gpu(A₀_large, tspan_large, params_large, saveat=0.1)
    end
    push!(results, result)
    
    return results
end

"""
Benchmark GPU vs CPU sensitivity analysis.
"""
function benchmark_gpu_sensitivity_analysis()
    if !has_cuda
        println("\n=== Skipping GPU Sensitivity Analysis Benchmarks (CUDA not available) ===")
        return []
    end
    
    println("\n=== Benchmarking GPU Sensitivity Analysis ===")
    results = []
    
    # Create base parameters for point model
    lattice = PointLattice()
    connectivity = ConnectivityMatrix{2}([
        ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
        ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
    ])
    
    base_params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (10.0, 8.0),
        connectivity = connectivity,
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    # Define parameter ranges
    param_ranges = [
        ("α_E", 0.5, 2.0),
        ("α_I", 0.5, 2.0)
    ]
    
    # Sobol analysis - CPU
    result = benchmark_function("Sobol CPU: 100 samples, 2 params", samples=3) do
        sobol_sensitivity_analysis(
            base_params,
            param_ranges,
            100,
            tspan=(0.0, 10.0),
            output_metric=:final_mean
        )
    end
    push!(results, result)
    
    # Sobol analysis - GPU
    result = benchmark_function("Sobol GPU: 100 samples, 2 params", samples=3) do
        sobol_sensitivity_analysis_gpu(
            base_params,
            param_ranges,
            100,
            tspan=(0.0, 10.0),
            output_metric=:final_mean
        )
    end
    push!(results, result)
    
    # Morris analysis - CPU
    result = benchmark_function("Morris CPU: 20 trajectories, 2 params", samples=3) do
        morris_sensitivity_analysis(
            base_params,
            param_ranges,
            20,
            tspan=(0.0, 10.0),
            output_metric=:final_mean
        )
    end
    push!(results, result)
    
    # Morris analysis - GPU
    result = benchmark_function("Morris GPU: 20 trajectories, 2 params", samples=3) do
        morris_sensitivity_analysis_gpu(
            base_params,
            param_ranges,
            20,
            tspan=(0.0, 10.0),
            output_metric=:final_mean
        )
    end
    push!(results, result)
    
    return results
end

"""
Benchmark GPU vs CPU parameter optimization.
"""
function benchmark_gpu_optimization()
    if !has_cuda
        println("\n=== Skipping GPU Optimization Benchmarks (CUDA not available) ===")
        return []
    end
    
    println("\n=== Benchmarking GPU Optimization ===")
    results = []
    
    # Create spatial model for optimization
    lattice = CompactLattice(extent=(10.0,), n_points=(51,))
    conn = GaussianConnectivityParameter(1.0, (2.0,))
    params = WilsonCowanParameters{1}(
        α=(1.0,), β=(1.0,), τ=(8.0,),
        connectivity=ConnectivityMatrix{1}(reshape([conn], 1, 1)),
        nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus=nothing, lattice=lattice, pop_names=("E",)
    )
    
    # Parameter ranges to optimize
    param_ranges = (
        connectivity_width = (1.5, 3.0),
        sigmoid_a = (1.5, 2.5)
    )
    
    # Objective
    objective = TravelingWaveObjective(
        target_distance=nothing,
        minimize_decay=true,
        require_traveling=false
    )
    
    # Initial condition
    A₀ = zeros(51, 1)
    A₀[20:26, 1] .= 0.4
    
    # CPU optimization
    result = benchmark_function("Optimization CPU: 5 iterations", samples=2) do
        optimize_for_traveling_wave(
            params, param_ranges, objective, A₀, (0.0, 10.0),
            maxiter=5
        )
    end
    push!(results, result)
    
    # GPU optimization
    result = benchmark_function("Optimization GPU: 5 iterations", samples=2) do
        optimize_for_traveling_wave_gpu(
            params, param_ranges, objective, A₀, (0.0, 10.0),
            maxiter=5
        )
    end
    push!(results, result)
    
    return results
end

"""
Run all GPU benchmarks.
"""
function run_gpu_benchmarks()
    if !has_cuda
        println("\n" * "="^80)
        println("GPU BENCHMARKS - SKIPPED (CUDA not available)")
        println("="^80)
        println("\nTo enable GPU benchmarks:")
        println("  1. Install CUDA.jl: using Pkg; Pkg.add(\"CUDA\")")
        println("  2. Ensure you have a CUDA-capable GPU")
        println("="^80)
        return []
    end
    
    println("\n" * "="^80)
    println("GPU BENCHMARKS")
    println("GPU: ", CUDA.name(CUDA.device()))
    println("="^80)
    
    all_results = []
    
    # Benchmark each GPU function type
    append!(all_results, benchmark_gpu_simulations())
    append!(all_results, benchmark_gpu_sensitivity_analysis())
    append!(all_results, benchmark_gpu_optimization())
    
    # Print results
    print_benchmark_results(all_results)
    
    # Write to CSV
    output_file = joinpath(dirname(@__FILE__), "..", "..", "benchmark_results", "gpu_benchmarks.csv")
    write_benchmark_results(all_results, output_file)
    
    # Calculate and display speedups
    println("\n" * "="^80)
    println("GPU SPEEDUP ANALYSIS")
    println("="^80)
    
    # Group results by base name (CPU vs GPU)
    cpu_results = Dict{String, Any}()
    gpu_results = Dict{String, Any}()
    
    for result in all_results
        name = result[:benchmark_name]
        if occursin("CPU", name)
            base_name = replace(name, " CPU:" => ":")
            cpu_results[base_name] = result
        elseif occursin("GPU", name)
            base_name = replace(name, " GPU:" => ":")
            gpu_results[base_name] = result
        end
    end
    
    # Calculate speedups
    for base_name in sort(collect(keys(cpu_results)))
        if haskey(gpu_results, base_name)
            cpu_time = cpu_results[base_name][:mean_time_ms]
            gpu_time = gpu_results[base_name][:mean_time_ms]
            speedup = cpu_time / gpu_time
            
            println("\n$base_name")
            println("  CPU: $(round(cpu_time, digits=2)) ms")
            println("  GPU: $(round(gpu_time, digits=2)) ms")
            
            if speedup > 1.0
                println("  Speedup: $(round(speedup, digits=2))x (GPU faster)")
            else
                println("  Speedup: $(round(1/speedup, digits=2))x (CPU faster)")
            end
        end
    end
    
    println("\n" * "="^80)
    
    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_gpu_benchmarks()
end

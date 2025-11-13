"""
Tests for GPU-accelerated functions using CUDA.jl

These tests only run if CUDA is available and functional.
"""

using Test
using FailureOfInhibition2025

# Check if CUDA is available
has_cuda = false
try
    using CUDA
    if CUDA.functional()
        has_cuda = true
        @info "CUDA is available, running GPU tests"
    else
        @info "CUDA is not functional, skipping GPU tests"
    end
catch e
    @info "CUDA.jl not loaded, skipping GPU tests: $e"
end

if has_cuda
    using CUDA
    
    @testset "GPU Simulation Tests" begin
        @testset "GPU solve_model for point model" begin
            # Create a simple point model
            lattice = PointLattice()
            connectivity = ConnectivityMatrix{2}([
                ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
                ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
            ])
            
            params = WilsonCowanParameters{2}(
                α = (1.0, 1.5),
                β = (1.0, 1.0),
                τ = (10.0, 8.0),
                connectivity = connectivity,
                nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
                stimulus = nothing,
                lattice = lattice,
                pop_names = ("E", "I")
            )
            
            A₀ = reshape([0.1, 0.1], 1, 2)
            tspan = (0.0, 10.0)
            
            # Run on GPU
            sol_gpu = solve_model_gpu(A₀, tspan, params, saveat=0.1)
            
            # Run on CPU for comparison
            sol_cpu = solve_model(A₀, tspan, params, saveat=0.1)
            
            # Check that solutions are similar
            @test length(sol_gpu.t) == length(sol_cpu.t)
            @test length(sol_gpu.u) == length(sol_cpu.u)
            
            # Check that final states are close (allowing for numerical differences)
            final_diff = maximum(abs.(sol_gpu.u[end] .- sol_cpu.u[end]))
            @test final_diff < 1e-3
            
            println("✓ GPU point model simulation matches CPU version")
        end
        
        @testset "GPU solve_model for spatial model" begin
            # Create a spatial model
            lattice = CompactLattice(extent=(10.0,), n_points=(21,))
            connectivity = ConnectivityMatrix{1}([
                GaussianConnectivityParameter(1.0, (2.0,))
            ])
            
            params = WilsonCowanParameters{1}(
                α = (1.0,),
                β = (1.0,),
                τ = (8.0,),
                connectivity = connectivity,
                nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
                stimulus = nothing,
                lattice = lattice,
                pop_names = ("E",)
            )
            
            A₀ = fill(0.1, 21, 1)
            A₀[10:12, 1] .= 0.3
            tspan = (0.0, 10.0)
            
            # Run on GPU
            sol_gpu = solve_model_gpu(A₀, tspan, params, saveat=0.1)
            
            # Run on CPU for comparison
            sol_cpu = solve_model(A₀, tspan, params, saveat=0.1)
            
            # Check that solutions are similar
            @test length(sol_gpu.t) == length(sol_cpu.t)
            @test length(sol_gpu.u) == length(sol_cpu.u)
            
            # Check that final states are close
            final_diff = maximum(abs.(sol_gpu.u[end] .- sol_cpu.u[end]))
            @test final_diff < 1e-3
            
            println("✓ GPU spatial model simulation matches CPU version")
        end
    end
    
    @testset "GPU Sensitivity Analysis Tests" begin
        @testset "GPU Sobol analysis" begin
            # Create a simple point model for testing
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
            
            # Define small parameter ranges for quick testing
            param_ranges = [
                ("α_E", 0.5, 2.0),
                ("α_I", 0.5, 2.0)
            ]
            
            # Run GPU Sobol analysis with small sample size
            result_gpu = sobol_sensitivity_analysis_gpu(
                base_params,
                param_ranges,
                100,  # Small sample for testing
                tspan=(0.0, 10.0),
                output_metric=:final_mean
            )
            
            # Check that result has expected structure
            @test haskey(result_gpu, :S1)
            @test haskey(result_gpu, :ST)
            @test haskey(result_gpu, :param_names)
            @test result_gpu[:method] == "Sobol (GPU)"
            
            # Check that indices have correct length
            @test length(result_gpu[:S1]) == 2
            @test length(result_gpu[:ST]) == 2
            
            # Check that indices are in valid range
            @test all(0 .<= result_gpu[:S1] .<= 1.5)  # Allow some numerical slack
            @test all(0 .<= result_gpu[:ST] .<= 1.5)
            
            println("✓ GPU Sobol sensitivity analysis completed successfully")
        end
        
        @testset "GPU Morris analysis" begin
            # Create a simple point model for testing
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
            
            # Define small parameter ranges for quick testing
            param_ranges = [
                ("α_E", 0.5, 2.0),
                ("α_I", 0.5, 2.0)
            ]
            
            # Run GPU Morris analysis with small sample size
            result_gpu = morris_sensitivity_analysis_gpu(
                base_params,
                param_ranges,
                10,  # Small trajectory count for testing
                tspan=(0.0, 10.0),
                output_metric=:final_mean
            )
            
            # Check that result has expected structure
            @test haskey(result_gpu, :means)
            @test haskey(result_gpu, :means_star)
            @test haskey(result_gpu, :variances)
            @test haskey(result_gpu, :param_names)
            @test result_gpu[:method] == "Morris (GPU)"
            
            # Check that indices have correct length
            @test length(result_gpu[:means]) == 2
            @test length(result_gpu[:means_star]) == 2
            @test length(result_gpu[:variances]) == 2
            
            println("✓ GPU Morris sensitivity analysis completed successfully")
        end
    end
    
    @testset "GPU Optimization Tests" begin
        @testset "GPU traveling wave optimization" begin
            # Create a spatial model
            lattice = CompactLattice(extent=(10.0,), n_points=(21,))
            conn = GaussianConnectivityParameter(1.0, (2.0,))
            params = WilsonCowanParameters{1}(
                α=(1.0,), β=(1.0,), τ=(8.0,),
                connectivity=ConnectivityMatrix{1}(reshape([conn], 1, 1)),
                nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.25),
                stimulus=nothing, lattice=lattice, pop_names=("E",)
            )
            
            # Define parameter ranges to optimize
            param_ranges = (
                connectivity_width = (1.5, 3.0),
                sigmoid_a = (1.5, 2.5)
            )
            
            # Define objective
            objective = TravelingWaveObjective(
                target_distance=nothing,  # Maximize distance
                minimize_decay=true,
                require_traveling=false  # Don't require traveling for this quick test
            )
            
            # Initial condition
            A₀ = zeros(21, 1)
            A₀[8:13, 1] .= 0.4
            
            # Run GPU optimization with limited iterations
            result, best_params = optimize_for_traveling_wave_gpu(
                params, param_ranges, objective, A₀, (0.0, 10.0),
                maxiter=5  # Very few iterations for testing
            )
            
            # Check that optimization completed
            @test result isa Optim.MultivariateOptimizationResults
            @test best_params isa WilsonCowanParameters
            
            println("✓ GPU traveling wave optimization completed successfully")
        end
    end
    
    println("\n=== GPU Tests Passed! ===\n")
else
    @testset "GPU Tests Skipped" begin
        @test_skip false  # Mark as skipped
    end
    println("\n=== GPU Tests Skipped (CUDA not available) ===\n")
end

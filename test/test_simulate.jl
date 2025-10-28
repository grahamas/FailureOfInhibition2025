#!/usr/bin/env julia

"""
Tests for simulation functionality (solve_model, save_simulation_results, save_simulation_summary)
"""

using Test
using FailureOfInhibition2025
using CSV
using DataFrames
using Statistics
using DifferentialEquations
import DifferentialEquations.SciMLBase

"""
Test basic solve_model functionality with point model.
"""
function test_solve_model_point()
    println("Testing solve_model with point model...")
    
    # Create simple point model
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
    
    # Initial condition
    A₀ = reshape([0.1, 0.1], 1, 2)
    tspan = (0.0, 10.0)
    
    # Solve
    sol = solve_model(A₀, tspan, params, saveat=0.1)
    
    # Test solution properties
    @test SciMLBase.successful_retcode(sol)
    @test length(sol.t) >= 2
    @test sol.t[1] == 0.0
    @test sol.t[end] ≈ 10.0 atol=0.01
    @test size(sol.u[1]) == (1, 2)
    
    # Test that solution evolved
    @test sol.u[end][1, 1] != A₀[1, 1] || sol.u[end][1, 2] != A₀[1, 2]
    
    println("  ✓ Point model solution successful")
end

"""
Test solve_model with spatial model.
"""
function test_solve_model_spatial()
    println("Testing solve_model with spatial model...")
    
    # Create spatial model
    lattice = CompactLattice(extent=(10.0,), n_points=(11,))
    connectivity = ConnectivityMatrix{1}(reshape([GaussianConnectivityParameter(0.3, (2.0,))], 1, 1))
    
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (10.0,),
        connectivity = connectivity,
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice
    )
    
    # Initial condition
    A₀ = 0.1 .+ 0.05 .* rand(11, 1)
    tspan = (0.0, 10.0)
    
    # Solve
    sol = solve_model(A₀, tspan, params, saveat=0.5)
    
    # Test solution properties
    @test SciMLBase.successful_retcode(sol)
    @test length(sol.t) >= 2
    @test size(sol.u[1]) == (11, 1)
    
    println("  ✓ Spatial model solution successful")
end

"""
Test save_simulation_results for point model.
"""
function test_save_simulation_results_point()
    println("Testing save_simulation_results with point model...")
    
    # Create and solve a simple model
    lattice = PointLattice()
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (10.0, 8.0),
        connectivity = ConnectivityMatrix{2}([
            ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
            ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
        ]),
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    A₀ = reshape([0.1, 0.1], 1, 2)
    sol = solve_model(A₀, (0.0, 5.0), params, saveat=1.0)
    
    # Save to temporary file
    temp_file = tempname() * ".csv"
    df = save_simulation_results(sol, temp_file, params=params)
    
    # Test file was created
    @test isfile(temp_file)
    
    # Test DataFrame structure
    @test "time" in names(df)
    @test "E" in names(df)
    @test "I" in names(df)
    @test size(df, 1) == length(sol.t)
    @test df.time == sol.t
    
    # Test CSV can be read back
    df_read = CSV.read(temp_file, DataFrame)
    @test size(df_read) == size(df)
    
    # Clean up
    rm(temp_file)
    
    println("  ✓ Point model results saved correctly")
end

"""
Test save_simulation_results for spatial model.
"""
function test_save_simulation_results_spatial()
    println("Testing save_simulation_results with spatial model...")
    
    # Create and solve a spatial model
    lattice = CompactLattice(extent=(10.0,), n_points=(5,))
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (10.0,),
        connectivity = ConnectivityMatrix{1}(reshape([GaussianConnectivityParameter(0.3, (2.0,))], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    A₀ = 0.1 .+ 0.05 .* rand(5, 1)
    sol = solve_model(A₀, (0.0, 5.0), params, saveat=1.0)
    
    # Save to temporary file
    temp_file = tempname() * ".csv"
    df = save_simulation_results(sol, temp_file, params=params)
    
    # Test file was created
    @test isfile(temp_file)
    
    # Test DataFrame structure
    @test "time" in names(df)
    # Should have columns for each spatial point
    @test "E_point1" in names(df)
    @test "E_point5" in names(df)
    @test size(df, 1) == length(sol.t)
    
    # Clean up
    rm(temp_file)
    
    println("  ✓ Spatial model results saved correctly")
end

"""
Test save_simulation_summary for point model.
"""
function test_save_simulation_summary_point()
    println("Testing save_simulation_summary with point model...")
    
    # Create and solve a simple model
    lattice = PointLattice()
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (10.0, 8.0),
        connectivity = ConnectivityMatrix{2}([
            ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
            ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
        ]),
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    A₀ = reshape([0.1, 0.1], 1, 2)
    sol = solve_model(A₀, (0.0, 10.0), params, saveat=0.5)
    
    # Save summary to temporary file
    temp_file = tempname() * ".csv"
    df = save_simulation_summary(sol, temp_file, params=params)
    
    # Test file was created
    @test isfile(temp_file)
    
    # Test DataFrame structure
    @test "population" in names(df)
    @test "mean" in names(df)
    @test "std" in names(df)
    @test "min" in names(df)
    @test "max" in names(df)
    @test "final" in names(df)
    @test size(df, 1) == 2  # Two populations
    @test df.population == ["E", "I"]
    
    # Test summary statistics are reasonable
    @test all(df.mean .>= df.min)
    @test all(df.mean .<= df.max)
    @test all(df.std .>= 0.0)
    
    # Clean up
    rm(temp_file)
    
    println("  ✓ Point model summary saved correctly")
end

"""
Test save_simulation_summary for spatial model.
"""
function test_save_simulation_summary_spatial()
    println("Testing save_simulation_summary with spatial model...")
    
    # Create and solve a spatial model
    lattice = CompactLattice(extent=(10.0,), n_points=(7,))
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (10.0, 8.0),
        connectivity = ConnectivityMatrix{2}([
            GaussianConnectivityParameter(0.5, (2.0,)) GaussianConnectivityParameter(-0.3, (1.5,));
            GaussianConnectivityParameter(0.4, (2.5,)) GaussianConnectivityParameter(-0.2, (1.0,))
        ]),
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    A₀ = 0.1 .+ 0.05 .* rand(7, 2)
    sol = solve_model(A₀, (0.0, 10.0), params, saveat=0.5)
    
    # Save summary to temporary file
    temp_file = tempname() * ".csv"
    df = save_simulation_summary(sol, temp_file, params=params)
    
    # Test file was created
    @test isfile(temp_file)
    
    # Test DataFrame structure
    @test "population" in names(df)
    @test "spatial_mean_of_means" in names(df)
    @test "spatial_mean_of_stds" in names(df)
    @test "final_spatial_mean" in names(df)
    @test "final_spatial_std" in names(df)
    @test size(df, 1) == 2  # Two populations
    
    # Test spatial statistics are reasonable
    @test all(df.spatial_mean_of_stds .>= 0.0)
    @test all(df.final_spatial_std .>= 0.0)
    
    # Clean up
    rm(temp_file)
    
    println("  ✓ Spatial model summary with spatial stats saved correctly")
end

"""
Test solve_model with different solver algorithms.
"""
function test_solve_model_different_solvers()
    println("Testing solve_model with different solvers...")
    
    # Create simple model
    lattice = PointLattice()
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (10.0,),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice
    )
    
    A₀ = [0.1]
    tspan = (0.0, 5.0)
    
    # Test with Euler solver
    sol_euler = solve_model(A₀, tspan, params, solver=Euler(), dt=0.01)
    @test SciMLBase.successful_retcode(sol_euler)
    
    # Test with default Tsit5 solver
    sol_tsit5 = solve_model(A₀, tspan, params)
    @test SciMLBase.successful_retcode(sol_tsit5)
    
    println("  ✓ Different solvers work correctly")
end

"""
Run all simulation tests.
"""
function run_all_simulation_tests()
    println("\n" * "="^70)
    println("Running Simulation Tests")
    println("="^70 * "\n")
    
    @testset "Simulation Tests" begin
        test_solve_model_point()
        test_solve_model_spatial()
        test_save_simulation_results_point()
        test_save_simulation_results_spatial()
        test_save_simulation_summary_point()
        test_save_simulation_summary_spatial()
        test_solve_model_different_solvers()
    end
    
    println("\n" * "="^70)
    println("All Simulation Tests Passed!")
    println("="^70 * "\n")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_simulation_tests()
end

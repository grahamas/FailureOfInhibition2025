#!/usr/bin/env julia

"""
Test suite for sensitivity analysis functionality
"""

using Test
using FailureOfInhibition2025
using CSV
using DataFrames
using DifferentialEquations
using SciMLSensitivity

"""
    test_parameter_extraction()

Test extraction and reconstruction of parameters for sensitivity analysis.
"""
function test_parameter_extraction()
    println("\n=== Testing Parameter Extraction and Reconstruction ===\n")
    
    # Create test parameters
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
    
    # Test extraction
    wrapper = extract_parameters(params, include_params=[:α, :β, :τ])
    
    @test length(wrapper.param_names) == 6
    @test length(wrapper.param_values) == 6
    @test wrapper.param_names[1] == :α_1
    @test wrapper.param_names[2] == :α_2
    @test wrapper.param_values[1] == 1.0
    @test wrapper.param_values[2] == 1.5
    
    println("  ✓ Parameter extraction works correctly")
    
    # Test reconstruction
    reconstructed = reconstruct_parameters(wrapper, wrapper.param_values)
    
    @test reconstructed.α == params.α
    @test reconstructed.β == params.β
    @test reconstructed.τ == params.τ
    
    println("  ✓ Parameter reconstruction works correctly")
    
    # Test modification
    modified_values = copy(wrapper.param_values)
    modified_values[1] = 2.0  # Change α_1
    
    modified_params = reconstruct_parameters(wrapper, modified_values)
    
    @test modified_params.α[1] == 2.0
    @test modified_params.α[2] == 1.5
    @test modified_params.β == params.β
    
    println("  ✓ Parameter modification works correctly")
    
    # Test partial extraction
    wrapper_partial = extract_parameters(params, include_params=[:α])
    
    @test length(wrapper_partial.param_names) == 2
    @test wrapper_partial.param_names == [:α_1, :α_2]
    @test wrapper_partial.param_values == [1.0, 1.5]
    
    println("  ✓ Partial parameter extraction works correctly")
end

"""
    test_connectivity_nonlinearity_extraction()

Test extraction and reconstruction of connectivity and nonlinearity parameters.
"""
function test_connectivity_nonlinearity_extraction()
    println("\n=== Testing Connectivity and Nonlinearity Extraction ===\n")
    
    # Test ScalarConnectivity extraction
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
    
    # Test extraction with connectivity
    wrapper = extract_parameters(params, include_params=[:α, :connectivity])
    
    @test length(wrapper.param_names) == 6  # 2 α + 4 connectivity weights
    @test :b_1_1 in wrapper.param_names
    @test :b_1_2 in wrapper.param_names
    @test :b_2_1 in wrapper.param_names
    @test :b_2_2 in wrapper.param_names
    
    # Find indices
    b11_idx = findfirst(==(Symbol("b_1_1")), wrapper.param_names)
    b12_idx = findfirst(==(Symbol("b_1_2")), wrapper.param_names)
    
    @test wrapper.param_values[b11_idx] == 0.5
    @test wrapper.param_values[b12_idx] == -0.3
    
    println("  ✓ ScalarConnectivity extraction works correctly")
    
    # Test reconstruction with modified connectivity
    modified_values = copy(wrapper.param_values)
    modified_values[b11_idx] = 0.7  # Change b_1_1
    
    modified_params = reconstruct_parameters(wrapper, modified_values)
    
    @test modified_params.connectivity.matrix[1, 1].weight == 0.7
    @test modified_params.connectivity.matrix[1, 2].weight == -0.3
    
    println("  ✓ ScalarConnectivity reconstruction works correctly")
    
    # Test nonlinearity extraction
    wrapper_nl = extract_parameters(params, include_params=[:nonlinearity])
    
    @test length(wrapper_nl.param_names) == 2
    @test :a in wrapper_nl.param_names
    @test :θ in wrapper_nl.param_names
    
    a_idx = findfirst(==(:a), wrapper_nl.param_names)
    θ_idx = findfirst(==(:θ), wrapper_nl.param_names)
    
    @test wrapper_nl.param_values[a_idx] == 1.5
    @test wrapper_nl.param_values[θ_idx] == 0.3
    
    println("  ✓ Nonlinearity extraction works correctly")
    
    # Test nonlinearity reconstruction
    modified_nl_values = copy(wrapper_nl.param_values)
    modified_nl_values[a_idx] = 2.0
    
    modified_nl_params = reconstruct_parameters(wrapper_nl, modified_nl_values)
    
    @test modified_nl_params.nonlinearity.a == 2.0
    @test modified_nl_params.nonlinearity.θ == 0.3
    
    println("  ✓ Nonlinearity reconstruction works correctly")
    
    # Test GaussianConnectivityParameter extraction
    lattice_spatial = CompactLattice(extent=(10.0,), n_points=(11,))
    conn_spatial = GaussianConnectivityParameter(0.3, (2.0,))
    connectivity_spatial = ConnectivityMatrix{1}(reshape([conn_spatial], 1, 1))
    
    params_spatial = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (10.0,),
        connectivity = connectivity_spatial,
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice_spatial,
        pop_names = ("E",)
    )
    
    wrapper_spatial = extract_parameters(params_spatial, include_params=[:connectivity])
    
    @test length(wrapper_spatial.param_names) == 2  # amplitude + 1D spread
    @test :b_amplitude_1_1 in wrapper_spatial.param_names
    @test :b_spread_1_1_dim1 in wrapper_spatial.param_names
    
    amp_idx = findfirst(==(Symbol("b_amplitude_1_1")), wrapper_spatial.param_names)
    spread_idx = findfirst(==(Symbol("b_spread_1_1_dim1")), wrapper_spatial.param_names)
    
    @test wrapper_spatial.param_values[amp_idx] == 0.3
    @test wrapper_spatial.param_values[spread_idx] == 2.0
    
    println("  ✓ GaussianConnectivityParameter extraction works correctly")
    
    # Test GaussianConnectivityParameter reconstruction
    modified_spatial_values = copy(wrapper_spatial.param_values)
    modified_spatial_values[amp_idx] = 0.5
    modified_spatial_values[spread_idx] = 3.0
    
    modified_spatial_params = reconstruct_parameters(wrapper_spatial, modified_spatial_values)
    
    @test modified_spatial_params.connectivity.matrix[1, 1].amplitude == 0.5
    @test modified_spatial_params.connectivity.matrix[1, 1].spread[1] == 3.0
    
    println("  ✓ GaussianConnectivityParameter reconstruction works correctly")
end

"""
    test_local_sensitivity_point_model()

Test local sensitivity analysis setup for a point model.
"""
function test_local_sensitivity_point_model()
    println("\n=== Testing Local Sensitivity Setup - Point Model ===\n")
    
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
    
    A₀ = reshape([0.1, 0.1], 1, 2)
    tspan = (0.0, 10.0)
    
    # Compute sensitivity setup (solve the ODE with parameter structure)
    result = compute_local_sensitivities(
        A₀, tspan, params,
        include_params=[:α, :β, :τ],
        saveat=1.0
    )
    
    # Check result structure
    @test haskey(result, :solution)
    @test haskey(result, :param_names)
    @test haskey(result, :param_values)
    @test haskey(result, :times)
    
    println("  ✓ Sensitivity setup has correct structure")
    
    # Check that solution was computed
    @test length(result.times) == 11  # 0, 1, 2, ..., 10
    
    println("  ✓ Solution computed successfully")
    
    # Check parameter names
    @test result.param_names == [:α_1, :α_2, :β_1, :β_2, :τ_1, :τ_2]
    
    println("  ✓ Parameter names are correct")
    println("  ✓ Parametrized solution ready for sensitivity analysis")
end

"""
    test_sensitivity_summary()

Test basic sensitivity analysis workflow.
"""
function test_sensitivity_summary()
    println("\n=== Testing Sensitivity Analysis Workflow ===\n")
    
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
    
    A₀ = reshape([0.1, 0.1], 1, 2)
    tspan = (0.0, 10.0)
    
    # Solve with parameter structure
    result = compute_local_sensitivities(A₀, tspan, params, saveat=1.0)
    
    # Check that we got a solution
    @test result.solution !== nothing
    @test length(result.times) > 0
    
    println("  ✓ Parametrized solution computed")
    println("  ✓ Ready for sensitivity analysis with SciMLSensitivity.jl")
end

"""
    test_save_sensitivities()

Test saving parametrized solution for sensitivity analysis.
"""
function test_save_sensitivities()
    println("\n=== Testing Solution Save for Sensitivity Analysis ===\n")
    
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
    
    A₀ = reshape([0.1, 0.1], 1, 2)
    tspan = (0.0, 5.0)
    
    # Solve with parameter structure  
    result = compute_local_sensitivities(A₀, tspan, params, saveat=1.0)
    
    # Create temporary directory for test outputs
    test_dir = mktempdir()
    
    try
        # Create a simple parameter info DataFrame
        param_data = []
        for (i, pname) in enumerate(result.param_names)
            push!(param_data, Dict(
                "param_name" => string(pname),
                "param_value" => result.param_values[i]
            ))
        end
        df = DataFrame(param_data)
        
        # Test saving
        param_file = joinpath(test_dir, "test_parameters.csv")
        CSV.write(param_file, df)
        
        @test isfile(param_file)
        @test nrow(df) > 0
        @test hasproperty(df, :param_name)
        @test hasproperty(df, :param_value)
        
        println("  ✓ Parameter information saved correctly")
        
        # Check that we can read it back
        df_read = CSV.read(param_file, DataFrame)
        @test nrow(df_read) == nrow(df)
        
        println("  ✓ Saved CSV file can be read back")
        
    finally
        # Clean up
        rm(test_dir, recursive=true, force=true)
    end
end

"""
    test_spatial_sensitivity()

Test sensitivity analysis setup for a spatial model.
"""
function test_spatial_sensitivity()
    println("\n=== Testing Sensitivity Setup - Spatial Model ===\n")
    
    # Create small spatial model
    lattice = CompactLattice(extent=(5.0,), n_points=(5,))
    conn = GaussianConnectivityParameter(0.3, (1.0,))
    connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1))
    
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (5.0,),
        connectivity = connectivity,
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    # Initial condition
    A₀ = 0.1 .+ 0.05 .* rand(5, 1)
    tspan = (0.0, 5.0)
    
    # Solve with parameter structure
    result = compute_local_sensitivities(
        A₀, tspan, params,
        include_params=[:α, :β, :τ],
        saveat=1.0
    )
    
    # Check that we got results
    @test result.solution !== nothing
    @test length(result.param_names) == 3
    
    println("  ✓ Spatial model sensitivity setup works correctly")
    println("  ✓ Solution ready for sensitivity analysis")
end

"""
    test_different_sensitivity_methods()

Test different solvers for sensitivity analysis setup.
"""
function test_different_sensitivity_methods()
    println("\n=== Testing Different Solvers for Sensitivity Setup ===\n")
    
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
    
    A₀ = reshape([0.1, 0.1], 1, 2)
    tspan = (0.0, 5.0)
    
    # Test default solver
    result_default = compute_local_sensitivities(
        A₀, tspan, params,
        saveat=1.0
    )
    
    @test result_default.solution !== nothing
    println("  ✓ Default solver (Tsit5) works")
    
    # Test with explicit solver specification
    result_tsit5 = compute_local_sensitivities(
        A₀, tspan, params,
        solver=Tsit5(),
        saveat=1.0
    )
    
    @test result_tsit5.solution !== nothing
    println("  ✓ Explicit Tsit5 solver works")
end

"""
    run_all_local_sensitivity_tests()

Run all local sensitivity analysis tests.
"""
function run_all_local_sensitivity_tests()
    println("\n" * "="^70)
    println("Running Local Sensitivity Analysis Tests")
    println("="^70)
    
    test_parameter_extraction()
    test_connectivity_nonlinearity_extraction()
    test_local_sensitivity_point_model()
    test_sensitivity_summary()
    test_save_sensitivities()
    test_spatial_sensitivity()
    test_different_sensitivity_methods()
    
    println("\n" * "="^70)
    println("✓ All Local Sensitivity Analysis Tests Passed!")
    println("="^70)
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using FailureOfInhibition2025
    run_all_local_sensitivity_tests()
end

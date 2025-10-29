#!/usr/bin/env julia

"""
Test suite for sensitivity analysis functionality
"""

using Test
using FailureOfInhibition2025
using CSV
using DataFrames

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
    test_local_sensitivity_point_model()

Test local sensitivity analysis for a point model.
"""
function test_local_sensitivity_point_model()
    println("\n=== Testing Local Sensitivity Analysis - Point Model ===\n")
    
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
    
    # Compute sensitivities
    result = compute_local_sensitivities(
        A₀, tspan, params,
        include_params=[:α, :β, :τ],
        saveat=1.0
    )
    
    # Check result structure
    @test haskey(result, :solution)
    @test haskey(result, :sensitivities)
    @test haskey(result, :param_names)
    @test haskey(result, :param_values)
    @test haskey(result, :times)
    
    println("  ✓ Sensitivity result has correct structure")
    
    # Check dimensions
    n_times = length(result.times)
    n_states = length(A₀)
    n_params = length(result.param_names)
    
    @test size(result.sensitivities) == (n_times, n_states, n_params)
    @test n_times == 11  # 0, 1, 2, ..., 10
    @test n_states == 2
    @test n_params == 6  # 2 populations × 3 parameter types
    
    println("  ✓ Sensitivity array has correct dimensions: $(size(result.sensitivities))")
    
    # Check that sensitivities are finite
    @test all(isfinite.(result.sensitivities))
    
    println("  ✓ All sensitivities are finite")
    
    # Check parameter names
    @test result.param_names == [:α_1, :α_2, :β_1, :β_2, :τ_1, :τ_2]
    
    println("  ✓ Parameter names are correct")
end

"""
    test_sensitivity_summary()

Test sensitivity summary computation.
"""
function test_sensitivity_summary()
    println("\n=== Testing Sensitivity Summary ===\n")
    
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
    
    # Compute sensitivities
    result = compute_local_sensitivities(A₀, tspan, params, saveat=1.0)
    
    # Compute summary
    summary = summarize_sensitivities(result, params=params)
    
    # Check summary structure
    @test nrow(summary) == 12  # 6 parameters × 2 states
    @test hasproperty(summary, :param_name)
    @test hasproperty(summary, :state_idx)
    @test hasproperty(summary, :state_name)
    @test hasproperty(summary, :mean_abs_sensitivity)
    @test hasproperty(summary, :max_abs_sensitivity)
    @test hasproperty(summary, :final_sensitivity)
    
    println("  ✓ Summary DataFrame has correct structure")
    
    # Check that statistics are computed correctly
    for row in eachrow(summary)
        @test row.mean_abs_sensitivity >= 0
        @test row.max_abs_sensitivity >= 0
        @test isfinite(row.mean_sensitivity)
        @test isfinite(row.std_sensitivity)
    end
    
    println("  ✓ Summary statistics are valid")
    
    # Check state names
    @test unique(summary.state_name) == ["E", "I"]
    
    println("  ✓ State names are included")
end

"""
    test_save_sensitivities()

Test saving sensitivity results to CSV.
"""
function test_save_sensitivities()
    println("\n=== Testing Sensitivity Save Functions ===\n")
    
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
    
    # Compute sensitivities
    result = compute_local_sensitivities(A₀, tspan, params, saveat=1.0)
    
    # Create temporary directory for test outputs
    test_dir = mktempdir()
    
    try
        # Test save_local_sensitivities
        sens_file = joinpath(test_dir, "test_sensitivities.csv")
        df = save_local_sensitivities(result, sens_file, params=params)
        
        @test isfile(sens_file)
        @test nrow(df) > 0
        @test hasproperty(df, :time)
        @test hasproperty(df, :state_idx)
        @test hasproperty(df, :param_name)
        @test hasproperty(df, :sensitivity)
        
        println("  ✓ save_local_sensitivities works correctly")
        
        # Check that we can read it back
        df_read = CSV.read(sens_file, DataFrame)
        @test nrow(df_read) == nrow(df)
        
        println("  ✓ Saved CSV file can be read back")
        
    finally
        # Clean up
        rm(test_dir, recursive=true, force=true)
    end
end

"""
    test_spatial_sensitivity()

Test sensitivity analysis for a spatial model.
"""
function test_spatial_sensitivity()
    println("\n=== Testing Sensitivity Analysis - Spatial Model ===\n")
    
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
    
    # Compute sensitivities
    result = compute_local_sensitivities(
        A₀, tspan, params,
        include_params=[:α, :β, :τ],
        saveat=1.0
    )
    
    # Check dimensions
    n_times = length(result.times)
    n_states = 5  # 5 spatial points × 1 population
    n_params = 3  # α, β, τ for single population
    
    @test size(result.sensitivities) == (n_times, n_states, n_params)
    @test all(isfinite.(result.sensitivities))
    
    println("  ✓ Spatial sensitivity analysis works correctly")
    println("  ✓ Dimensions: $(size(result.sensitivities))")
end

"""
    test_different_sensitivity_methods()

Test different sensitivity computation methods (forward, adjoint, etc.).
"""
function test_different_sensitivity_methods()
    println("\n=== Testing Different Sensitivity Methods ===\n")
    
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
    
    # Test forward mode (already tested above, but explicitly)
    result_forward = compute_local_sensitivities(
        A₀, tspan, params,
        method=ForwardDiffSensitivity(),
        saveat=1.0
    )
    
    @test all(isfinite.(result_forward.sensitivities))
    println("  ✓ ForwardDiffSensitivity works")
    
    # Note: Other methods like InterpolatingAdjoint may require additional setup
    # and are commented out to avoid potential issues during testing
    
    # Test with different solver
    result_solver = compute_local_sensitivities(
        A₀, tspan, params,
        method=ForwardDiffSensitivity(),
        solver=Tsit5(),
        saveat=1.0
    )
    
    @test all(isfinite.(result_solver.sensitivities))
    println("  ✓ Different solver (Tsit5) works")
end

"""
    run_all_sensitivity_tests()

Run all sensitivity analysis tests.
"""
function run_all_sensitivity_tests()
    println("\n" * "="^70)
    println("Running Sensitivity Analysis Tests")
    println("="^70)
    
    test_parameter_extraction()
    test_local_sensitivity_point_model()
    test_sensitivity_summary()
    test_save_sensitivities()
    test_spatial_sensitivity()
    test_different_sensitivity_methods()
    
    println("\n" * "="^70)
    println("✓ All Sensitivity Analysis Tests Passed!")
    println("="^70)
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using FailureOfInhibition2025
    run_all_sensitivity_tests()
end

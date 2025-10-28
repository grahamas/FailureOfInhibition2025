"""
Benchmark individual components (nonlinearity, connectivity, stimulation).
"""

include("benchmark_utils.jl")

using FailureOfInhibition2025

"""
Benchmark nonlinearity components.
"""
function benchmark_nonlinearity_components()
    println("\n=== Benchmarking Nonlinearity Components ===")
    results = []
    
    # Setup test arrays
    n_points = 1000
    A = rand(n_points)
    dA = zeros(n_points)
    t = 0.0
    
    # Benchmark SigmoidNonlinearity
    sigmoid = SigmoidNonlinearity(a=2.0, θ=0.5)
    result = benchmark_function("SigmoidNonlinearity (n=$n_points)", n_runs=1000) do
        dA .= 0.0
        apply_nonlinearity!(dA, A, sigmoid, t)
    end
    push!(results, result)
    
    # Benchmark RectifiedZeroedSigmoidNonlinearity
    rect_sigmoid = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5)
    result = benchmark_function("RectifiedZeroedSigmoidNonlinearity (n=$n_points)", n_runs=1000) do
        dA .= 0.0
        apply_nonlinearity!(dA, A, rect_sigmoid, t)
    end
    push!(results, result)
    
    # Benchmark DifferenceOfSigmoidsNonlinearity
    diff_sigmoid = DifferenceOfSigmoidsNonlinearity(a_up=5.0, θ_up=0.3, a_down=3.0, θ_down=0.7)
    result = benchmark_function("DifferenceOfSigmoidsNonlinearity (n=$n_points)", n_runs=1000) do
        dA .= 0.0
        apply_nonlinearity!(dA, A, diff_sigmoid, t)
    end
    push!(results, result)
    
    # Benchmark simple_sigmoid function
    x = 0.5
    result = benchmark_function("simple_sigmoid (scalar)", n_runs=10000) do
        simple_sigmoid(x, 2.0, 0.5)
    end
    push!(results, result)
    
    # Benchmark rectified_zeroed_sigmoid function
    result = benchmark_function("rectified_zeroed_sigmoid (scalar)", n_runs=10000) do
        rectified_zeroed_sigmoid(x, 2.0, 0.5)
    end
    push!(results, result)
    
    return results
end

"""
Benchmark connectivity components.
"""
function benchmark_connectivity_components()
    println("\n=== Benchmarking Connectivity Components ===")
    results = []
    
    # 1D connectivity benchmark with ConnectivityMatrix (single population)
    lattice_1d = CompactLattice(extent=(10.0,), n_points=(101,))
    A_1d = reshape(rand(101), 101, 1)  # Shape (101, 1) for 1 population
    dA_1d = zeros(101, 1)
    conn_1d = GaussianConnectivityParameter(1.0, (2.0,))
    connectivity_1d = ConnectivityMatrix{1}(reshape([conn_1d], 1, 1))
    
    result = benchmark_function("GaussianConnectivity 1D (n=101)", n_runs=100) do
        dA_1d .= 0.0
        propagate_activation(dA_1d, A_1d, connectivity_1d, 0.0, lattice_1d)
    end
    push!(results, result)
    
    # ScalarConnectivity for point models (using ConnectivityMatrix)
    A_scalar = reshape([0.5], 1, 1)
    dA_scalar = zeros(1, 1)
    conn_scalar = ScalarConnectivity(1.5)
    connectivity_scalar = ConnectivityMatrix{1}(reshape([conn_scalar], 1, 1))
    
    result = benchmark_function("ScalarConnectivity (point model)", n_runs=10000) do
        dA_scalar .= 0.0
        propagate_activation(dA_scalar, A_scalar, connectivity_scalar, 0.0, PointLattice())
    end
    push!(results, result)
    
    # ConnectivityMatrix with 2 populations
    lattice_2pop = PointLattice()
    A_2pop = reshape([0.3, 0.5], 1, 2)
    dA_2pop = zeros(1, 2)
    
    conn_ee = ScalarConnectivity(1.0)
    conn_ei = ScalarConnectivity(-0.5)
    conn_ie = ScalarConnectivity(0.8)
    conn_ii = ScalarConnectivity(-0.3)
    
    connectivity_matrix = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    result = benchmark_function("ConnectivityMatrix 2x2 (point model)", n_runs=1000) do
        dA_2pop .= 0.0
        propagate_activation(dA_2pop, A_2pop, connectivity_matrix, 0.0, lattice_2pop)
    end
    push!(results, result)
    
    return results
end

"""
Benchmark stimulation components.
"""
function benchmark_stimulation_components()
    println("\n=== Benchmarking Stimulation Components ===")
    results = []
    
    # 1D stimulus (spatial dimensions only)
    lattice_1d = CompactLattice(extent=(10.0,), n_points=(101,))
    A_1d = rand(101)  # Shape (101,) - spatial only
    dA_1d = zeros(101)
    
    stimulus_1d = CircleStimulus(
        radius=2.0,
        strength=0.5,
        time_windows=[(0.0, 10.0)],
        lattice=lattice_1d
    )
    
    result = benchmark_function("CircleStimulus 1D (n=101, active)", n_runs=1000) do
        dA_1d .= 0.0
        stimulate!(dA_1d, A_1d, stimulus_1d, 5.0)
    end
    push!(results, result)
    
    result = benchmark_function("CircleStimulus 1D (n=101, inactive)", n_runs=1000) do
        dA_1d .= 0.0
        stimulate!(dA_1d, A_1d, stimulus_1d, 15.0)
    end
    push!(results, result)
    
    # No stimulus (nothing) - single value
    A_simple = [0.5]
    dA_simple = zeros(1)
    result = benchmark_function("No stimulus (nothing)", n_runs=10000) do
        dA_simple .= 0.0
        stimulate!(dA_simple, A_simple, nothing, 0.0)
    end
    push!(results, result)
    
    return results
end

"""
Run all component benchmarks.
"""
function run_component_benchmarks()
    println("\n" * "="^80)
    println("COMPONENT BENCHMARKS")
    println("="^80)
    
    all_results = []
    
    # Benchmark each component type
    append!(all_results, benchmark_nonlinearity_components())
    append!(all_results, benchmark_connectivity_components())
    append!(all_results, benchmark_stimulation_components())
    
    # Print results
    print_benchmark_results(all_results)
    
    # Write to CSV
    output_file = joinpath(dirname(@__FILE__), "..", "..", "benchmark_results", "component_benchmarks.csv")
    write_benchmark_results(all_results, output_file)
    
    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_component_benchmarks()
end

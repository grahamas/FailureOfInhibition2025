"""
Benchmark 0D (point) and 1D simulations.
"""

include("benchmark_utils.jl")

using FailureOfInhibition2025

"""
Benchmark 0D (point) simulations - simple ODEs without spatial structure.
"""
function benchmark_0d_simulations()
    println("\n=== Benchmarking 0D (Point) Simulations ===")
    results = []
    
    # Single population point model
    lattice = PointLattice()
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)
    
    params_1pop = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = nothing,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice
    )
    
    A_1pop = [0.5]
    dA_1pop = zeros(1)
    
    result = benchmark_function("0D: 1 population, no connectivity", samples=1000) do
        dA_1pop .= 0.0
        wcm1973!(dA_1pop, A_1pop, params_1pop, 0.0)
    end
    push!(results, result)
    
    # Two population point model with ScalarConnectivity
    conn_ee = ScalarConnectivity(1.0)
    conn_ei = ScalarConnectivity(-0.5)
    conn_ie = ScalarConnectivity(0.8)
    conn_ii = ScalarConnectivity(-0.3)
    
    connectivity_2pop = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    params_2pop = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = connectivity_2pop,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    A_2pop = reshape([0.3, 0.5], 1, 2)
    dA_2pop = zeros(1, 2)
    
    result = benchmark_function("0D: 2 populations (E,I), ScalarConnectivity", samples=1000) do
        dA_2pop .= 0.0
        wcm1973!(dA_2pop, A_2pop, params_2pop, 0.0)
    end
    push!(results, result)
    
    # Two population without connectivity (simpler)
    params_2pop_no_conn = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = nothing,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    A_2pop_simple = [0.3, 0.5]
    dA_2pop_simple = zeros(2)
    
    result = benchmark_function("0D: 2 populations, no connectivity", samples=1000) do
        dA_2pop_simple .= 0.0
        wcm1973!(dA_2pop_simple, A_2pop_simple, params_2pop_no_conn, 0.0)
    end
    push!(results, result)
    
    # Three population point model
    conn_3x3 = ConnectivityMatrix{3}([
        ScalarConnectivity(1.0) ScalarConnectivity(-0.5) ScalarConnectivity(-0.3);
        ScalarConnectivity(0.8) ScalarConnectivity(-0.2) ScalarConnectivity(-0.4);
        ScalarConnectivity(0.6) ScalarConnectivity(0.5) ScalarConnectivity(-0.1)
    ])
    
    params_3pop = WilsonCowanParameters{3}(
        α = (1.0, 1.2, 1.5),
        β = (1.0, 1.0, 1.0),
        τ = (1.0, 0.9, 0.8),
        connectivity = conn_3x3,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E1", "E2", "I")
    )
    
    A_3pop = reshape([0.3, 0.4, 0.5], 1, 3)
    dA_3pop = zeros(1, 3)
    
    result = benchmark_function("0D: 3 populations, full ConnectivityMatrix", samples=1000) do
        dA_3pop .= 0.0
        wcm1973!(dA_3pop, A_3pop, params_3pop, 0.0)
    end
    push!(results, result)
    
    return results
end

"""
Benchmark 1D simulations - PDEs with spatial structure.
"""
function benchmark_1d_simulations()
    println("\n=== Benchmarking 1D Simulations ===")
    results = []
    
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)
    
    # Small 1D lattice (51 points)
    lattice_small = CompactLattice(extent=(10.0,), n_points=(51,))
    
    conn_small = GaussianConnectivityParameter(1.0, (2.0,))
    connectivity_small = ConnectivityMatrix{1}(reshape([conn_small], 1, 1))
    
    params_small = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = connectivity_small,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice_small
    )
    
    A_small = rand(51, 1)
    dA_small = zeros(51, 1)
    
    result = benchmark_function("1D: 51 points, 1 population, Gaussian connectivity", samples=100) do
        dA_small .= 0.0
        wcm1973!(dA_small, A_small, params_small, 0.0)
    end
    push!(results, result)
    
    # Medium 1D lattice (101 points)
    lattice_medium = CompactLattice(extent=(10.0,), n_points=(101,))
    
    conn_medium = GaussianConnectivityParameter(1.0, (2.0,))
    connectivity_medium = ConnectivityMatrix{1}(reshape([conn_medium], 1, 1))
    
    params_medium = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = connectivity_medium,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice_medium
    )
    
    A_medium = rand(101, 1)
    dA_medium = zeros(101, 1)
    
    result = benchmark_function("1D: 101 points, 1 population, Gaussian connectivity", samples=100) do
        dA_medium .= 0.0
        wcm1973!(dA_medium, A_medium, params_medium, 0.0)
    end
    push!(results, result)
    
    # Large 1D lattice (201 points)
    lattice_large = CompactLattice(extent=(10.0,), n_points=(201,))
    
    conn_large = GaussianConnectivityParameter(1.0, (2.0,))
    connectivity_large = ConnectivityMatrix{1}(reshape([conn_large], 1, 1))
    
    params_large = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = connectivity_large,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice_large
    )
    
    A_large = rand(201, 1)
    dA_large = zeros(201, 1)
    
    result = benchmark_function("1D: 201 points, 1 population, Gaussian connectivity", samples=50) do
        dA_large .= 0.0
        wcm1973!(dA_large, A_large, params_large, 0.0)
    end
    push!(results, result)
    
    # 1D with 2 populations and ConnectivityMatrix (without stimulus for simplicity)
    lattice_2pop = CompactLattice(extent=(10.0,), n_points=(101,))
    
    conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
    conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
    conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
    conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
    
    connectivity_2pop = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    params_2pop = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = connectivity_2pop,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice_2pop,
        pop_names = ("E", "I")
    )
    
    A_2pop = rand(101, 2)
    dA_2pop = zeros(101, 2)
    
    result = benchmark_function("1D: 101 points, 2 populations (E,I), full model", samples=50) do
        dA_2pop .= 0.0
        wcm1973!(dA_2pop, A_2pop, params_2pop, 0.0)
    end
    push!(results, result)
    
    return results
end

"""
Run all simulation benchmarks.
"""
function run_simulation_benchmarks()
    println("\n" * "="^80)
    println("SIMULATION BENCHMARKS")
    println("="^80)
    
    all_results = []
    
    # Benchmark each simulation type
    append!(all_results, benchmark_0d_simulations())
    append!(all_results, benchmark_1d_simulations())
    
    # Print results
    print_benchmark_results(all_results)
    
    # Write to CSV
    output_file = joinpath(dirname(@__FILE__), "..", "..", "benchmark_results", "simulation_benchmarks.csv")
    write_benchmark_results(all_results, output_file)
    
    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulation_benchmarks()
end

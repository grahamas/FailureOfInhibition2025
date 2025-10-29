using Test
using FailureOfInhibition2025

println("\n" * "="^70)
println("Testing Parameter Optimization for Traveling Waves")
println("="^70)

@testset "Parameter Optimization Tests" begin
    
    @testset "TravelingWaveObjective Construction" begin
        println("\n=== Testing TravelingWaveObjective Construction ===")
        
        # Default construction
        obj1 = TravelingWaveObjective()
        @test obj1.target_distance === nothing
        @test obj1.target_amplitude === nothing
        @test obj1.target_width === nothing
        @test obj1.minimize_decay === true
        @test obj1.require_traveling === true
        @test obj1.threshold == 0.15
        println("   ✓ Default construction passed")
        
        # Custom construction
        obj2 = TravelingWaveObjective(
            target_distance=10.0,
            target_amplitude=0.8,
            minimize_decay=false,
            threshold=0.2
        )
        @test obj2.target_distance == 10.0
        @test obj2.target_amplitude == 0.8
        @test obj2.minimize_decay === false
        @test obj2.threshold == 0.2
        println("   ✓ Custom construction passed")
    end
    
    @testset "Parameter Updates" begin
        println("\n=== Testing Parameter Update Helper ===")
        
        # Create base parameters
        lattice = CompactLattice(extent=(20.0,), n_points=(101,))
        conn = GaussianConnectivityParameter(1.0, (2.0,))
        base_params = WilsonCowanParameters{1}(
            α = (1.0,),
            β = (1.0,),
            τ = (8.0,),
            connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
            nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        # Test updating connectivity width
        param_names = (:connectivity_width,)
        values = [3.5]
        updated_params = FailureOfInhibition2025._update_params(base_params, param_names, values)
        
        @test updated_params.connectivity[1,1].spread == (3.5,)
        @test updated_params.connectivity[1,1].amplitude == 1.0
        println("   ✓ Connectivity width update passed")
        
        # Test updating sigmoid parameters
        param_names = (:sigmoid_a, :sigmoid_θ)
        values = [2.5, 0.3]
        updated_params = FailureOfInhibition2025._update_params(base_params, param_names, values)
        
        @test updated_params.nonlinearity.a == 2.5
        @test updated_params.nonlinearity.θ == 0.3
        println("   ✓ Sigmoid parameter update passed")
        
        # Test updating time constant
        param_names = (:τ,)
        values = [10.0]
        updated_params = FailureOfInhibition2025._update_params(base_params, param_names, values)
        
        @test updated_params.τ == (10.0,)
        println("   ✓ Time constant update passed")
    end
    
    @testset "Basic Optimization" begin
        println("\n=== Testing Basic Optimization Setup ===")
        
        # Set up base parameters
        lattice = CompactLattice(extent=(20.0,), n_points=(21,))  # Much smaller for speed
        conn = GaussianConnectivityParameter(0.8, (2.0,))
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
        
        # Define optimization parameters
        param_ranges = (
            connectivity_width = (2.0, 3.5),
            sigmoid_a = (2.0, 2.8)
        )
        
        # Define objective
        objective = TravelingWaveObjective(
            minimize_decay = false,
            require_traveling = false,
            threshold = 0.1
        )
        
        # Initial condition
        A₀ = zeros(21, 1)
        A₀[5:8, 1] .= 0.6
        
        # Just test that the optimization function can be called
        # Use only 2 iterations for speed
        println("   Running optimization (minimal test)...")
        result, best_params = optimize_for_traveling_wave(
            params, param_ranges, objective, A₀, (0.0, 10.0),
            saveat=1.0,
            maxiter=2  # Minimal iterations
        )
        
        @test result !== nothing
        @test best_params !== nothing
        
        # Check that parameters are within bounds
        @test 2.0 <= best_params.connectivity[1,1].spread[1] <= 3.5
        @test 2.0 <= best_params.nonlinearity.a <= 2.8
        
        println("   ✓ Basic optimization completed")
        println("     - Optimized connectivity spread: $(round(best_params.connectivity[1,1].spread[1], digits=2))")
        println("     - Optimized sigmoid a: $(round(best_params.nonlinearity.a, digits=2))")
    end
    
    @testset "Optimization with Target Distance" begin
        println("\n=== Testing Optimization with Target Distance ===")
        
        # Set up parameters  
        lattice = CompactLattice(extent=(20.0,), n_points=(21,))
        conn = GaussianConnectivityParameter(1.0, (2.5,))
        params = WilsonCowanParameters{1}(
            α = (1.0,),
            β = (1.0,),
            τ = (8.0,),
            connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
            nonlinearity = SigmoidNonlinearity(a=2.5, θ=0.25),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        param_ranges = (connectivity_width = (2.0, 4.0),)
        
        objective = TravelingWaveObjective(
            target_distance = 5.0,
            require_traveling = false,
            threshold = 0.1
        )
        
        A₀ = zeros(21, 1)
        A₀[5:6, 1] .= 0.5
        
        println("   Running targeted optimization (minimal test)...")
        result, best_params = optimize_for_traveling_wave(
            params, param_ranges, objective, A₀, (0.0, 10.0),
            saveat=1.0,
            maxiter=2
        )
        
        @test result !== nothing
        @test 2.0 <= best_params.connectivity[1,1].spread[1] <= 4.0
        
        println("   ✓ Targeted optimization completed")
    end
    
end

println("\n" * "="^70)
println("All Parameter Optimization Tests Passed!")
println("="^70)

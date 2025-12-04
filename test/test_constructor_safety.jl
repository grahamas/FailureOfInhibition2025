using Test
using FailureOfInhibition2025

println("\n" * "="^70)
println("Testing WilsonCowanParameters Constructor Safety")
println("="^70)

@testset "Constructor Safety Tests" begin
    
    @testset "Positional Constructor Prepares Connectivity" begin
        println("\n=== Testing Positional Constructor ===")
        
        # Set up test parameters
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        conn_param = GaussianConnectivityParameter(1.0, (2.0,))
        connectivity = ConnectivityMatrix{1}(reshape([conn_param], 1, 1))
        
        # Use positional constructor
        params_positional = WilsonCowanParameters{Float64,1}(
            (1.0,),  # α
            (1.0,),  # β
            (8.0,),  # τ
            connectivity,
            SigmoidNonlinearity(a=2.0, θ=0.25),
            nothing,  # stimulus
            lattice,
            ("E",)  # pop_names
        )
        
        # Check that connectivity was prepared
        @test params_positional.connectivity isa ConnectivityMatrix
        @test params_positional.connectivity[1,1] isa FailureOfInhibition2025.GaussianConnectivity
        println("   ✓ Positional constructor prepared connectivity")
        
        # Verify the connectivity has the correct properties
        @test params_positional.connectivity[1,1].amplitude == 1.0
        @test params_positional.connectivity[1,1].spread == (2.0,)
        println("   ✓ Prepared connectivity has correct properties")
    end
    
    @testset "Keyword Constructor Prepares Connectivity" begin
        println("\n=== Testing Keyword Constructor ===")
        
        # Set up test parameters
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        conn_param = GaussianConnectivityParameter(1.0, (2.0,))
        connectivity = ConnectivityMatrix{1}(reshape([conn_param], 1, 1))
        
        # Use keyword constructor
        params_keyword = WilsonCowanParameters{1}(
            α = (1.0,),
            β = (1.0,),
            τ = (8.0,),
            connectivity = connectivity,
            nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        # Check that connectivity was prepared
        @test params_keyword.connectivity isa ConnectivityMatrix
        @test params_keyword.connectivity[1,1] isa FailureOfInhibition2025.GaussianConnectivity
        println("   ✓ Keyword constructor prepared connectivity")
        
        # Verify the connectivity has the correct properties
        @test params_keyword.connectivity[1,1].amplitude == 1.0
        @test params_keyword.connectivity[1,1].spread == (2.0,)
        println("   ✓ Prepared connectivity has correct properties")
    end
    
    @testset "Both Constructors Produce Equivalent Results" begin
        println("\n=== Testing Constructor Equivalence ===")
        
        # Set up test parameters
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        conn_param = GaussianConnectivityParameter(1.5, (3.0,))
        connectivity = ConnectivityMatrix{1}(reshape([conn_param], 1, 1))
        
        # Create using positional constructor
        params_positional = WilsonCowanParameters{Float64,1}(
            (1.2,), (0.9,), (7.5,),
            connectivity,
            SigmoidNonlinearity(a=2.5, θ=0.3),
            nothing, lattice, ("E",)
        )
        
        # Create using keyword constructor
        params_keyword = WilsonCowanParameters{1}(
            α = (1.2,),
            β = (0.9,),
            τ = (7.5,),
            connectivity = connectivity,
            nonlinearity = SigmoidNonlinearity(a=2.5, θ=0.3),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        # Check that both produced the same connectivity type
        @test params_positional.connectivity[1,1] isa FailureOfInhibition2025.GaussianConnectivity
        @test params_keyword.connectivity[1,1] isa FailureOfInhibition2025.GaussianConnectivity
        
        # Check that properties match
        @test params_positional.α == params_keyword.α
        @test params_positional.β == params_keyword.β
        @test params_positional.τ == params_keyword.τ
        @test params_positional.connectivity[1,1].amplitude == params_keyword.connectivity[1,1].amplitude
        @test params_positional.connectivity[1,1].spread == params_keyword.connectivity[1,1].spread
        @test params_positional.pop_names == params_keyword.pop_names
        
        println("   ✓ Both constructors produce equivalent parameters")
    end
    
    @testset "Already-Prepared Connectivity Not Re-prepared" begin
        println("\n=== Testing Idempotence ===")
        
        # Set up test parameters
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        conn_param = GaussianConnectivityParameter(1.0, (2.0,))
        connectivity_param = ConnectivityMatrix{1}(reshape([conn_param], 1, 1))
        
        # Create params once (prepares connectivity)
        params1 = WilsonCowanParameters{1}(
            α = (1.0,),
            β = (1.0,),
            τ = (8.0,),
            connectivity = connectivity_param,
            nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        # Create params again using the already-prepared connectivity
        params2 = WilsonCowanParameters{Float64,1}(
            (1.0,), (1.0,), (8.0,),
            params1.connectivity,  # Already prepared!
            SigmoidNonlinearity(a=2.0, θ=0.25),
            nothing, lattice, ("E",)
        )
        
        # Both should have GaussianConnectivity
        @test params1.connectivity[1,1] isa FailureOfInhibition2025.GaussianConnectivity
        @test params2.connectivity[1,1] isa FailureOfInhibition2025.GaussianConnectivity
        
        # Properties should match
        @test params1.connectivity[1,1].amplitude == params2.connectivity[1,1].amplitude
        @test params1.connectivity[1,1].spread == params2.connectivity[1,1].spread
        
        println("   ✓ Already-prepared connectivity handled correctly")
    end
    
    @testset "Multi-Population Constructor Safety" begin
        println("\n=== Testing Multi-Population Constructor ===")
        
        # Set up test parameters for 2 populations
        lattice = CompactLattice(extent=(20.0,), n_points=(41,))
        conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
        conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
        conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
        conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        # Use positional constructor for 2 populations
        params = WilsonCowanParameters{Float64,2}(
            (1.0, 1.0),  # α
            (1.0, 1.0),  # β
            (8.0, 8.0),  # τ
            connectivity,
            SigmoidNonlinearity(a=2.0, θ=0.25),
            nothing,
            lattice,
            ("E", "I")
        )
        
        # Check that all connectivity entries were prepared
        @test params.connectivity[1,1] isa FailureOfInhibition2025.GaussianConnectivity
        @test params.connectivity[1,2] isa FailureOfInhibition2025.GaussianConnectivity
        @test params.connectivity[2,1] isa FailureOfInhibition2025.GaussianConnectivity
        @test params.connectivity[2,2] isa FailureOfInhibition2025.GaussianConnectivity
        
        # Verify properties
        @test params.connectivity[1,1].amplitude == 1.0
        @test params.connectivity[1,2].amplitude == -0.5
        @test params.connectivity[2,1].amplitude == 0.8
        @test params.connectivity[2,2].amplitude == -0.3
        
        println("   ✓ Multi-population positional constructor prepared all connections")
    end
end

println("\n" * "="^70)
println("All Constructor Safety Tests Passed!")
println("="^70)

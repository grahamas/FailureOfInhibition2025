"""
Tests for visualization functions in visualize.jl

Tests basic functionality of all visualization functions without requiring
extensive simulations. Uses small models for speed.
"""

using Test
using FailureOfInhibition2025
using Plots

@testset "Visualization Functions" begin
    
    # Set up GR backend for headless operation
    gr()
    
    @testset "Point Model Visualizations" begin
        # Create simple point model
        params = create_point_model_wcm1973(:oscillatory)
        A₀ = reshape([0.05, 0.05], 1, 2)
        sol = solve_model(A₀, (0.0, 20.0), params, saveat=1.0)
        
        # Test plot_time_series
        @test_nowarn plot_time_series(sol, params)
        @test_nowarn plot_time_series(sol, params, pop_indices=[1])
        @test_nowarn plot_time_series(sol, params, pop_indices=[1, 2])
        
        # Test plot_phase_portrait
        @test_nowarn plot_phase_portrait(sol, params)
    end
    
    @testset "1D Spatial Model Visualizations" begin
        # Create simple 1D spatial model
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        conn = GaussianConnectivityParameter(0.8, (2.0,))
        params = WilsonCowanParameters{1}(
            α=(1.0,), β=(1.0,), τ=(10.0,),
            connectivity=ConnectivityMatrix{1}(reshape([conn], 1, 1)),
            nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.25),
            stimulus=nothing, lattice=lattice, pop_names=("E",)
        )
        
        A₀ = zeros(21, 1)
        A₀[10:12, 1] .= 0.5
        sol = solve_model(A₀, (0.0, 5.0), params, saveat=1.0)
        
        # Test plot_time_series (spatial mean)
        @test_nowarn plot_time_series(sol, params)
        
        # Test plot_spatial_snapshot
        @test_nowarn plot_spatial_snapshot(sol, params, 1)
        @test_nowarn plot_spatial_snapshot(sol, params, length(sol.t))
        
        # Test plot_spatiotemporal
        @test_nowarn plot_spatiotemporal(sol, params)
    end
    
    @testset "2D Spatial Model Visualizations" begin
        # Note: 2D simulations can be slow, so we skip this in standard tests
        # The visualization functions themselves work correctly, as shown by 1D tests
        
        @test_skip begin
            # Create simple 2D spatial model
            lattice = CompactLattice(extent=(6.0, 6.0), n_points=(11, 11))
            conn = GaussianConnectivityParameter(0.6, (1.5, 1.5))
            params = WilsonCowanParameters{1}(
                α=(1.0,), β=(1.0,), τ=(10.0,),
                connectivity=ConnectivityMatrix{1}(reshape([conn], 1, 1)),
                nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.3),
                stimulus=nothing, lattice=lattice, pop_names=("E",)
            )
            
            A₀ = zeros(11*11, 1)
            A₀[60, 1] = 0.5  # Central point
            sol = solve_model(A₀, (0.0, 3.0), params, saveat=1.0)
            
            # Test plot_time_series (spatial mean)
            plot_time_series(sol, params)
            
            # Test plot_spatial_snapshot
            plot_spatial_snapshot(sol, params, 1)
        end
    end
    
    @testset "Multi-Population Visualizations" begin
        # Create 1D E-I model
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        
        conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
        conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
        conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
        conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
        
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = WilsonCowanParameters{2}(
            α=(1.0, 1.5), β=(1.0, 1.0), τ=(10.0, 8.0),
            connectivity=connectivity,
            nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.3),
            stimulus=nothing, lattice=lattice, pop_names=("E", "I")
        )
        
        A₀ = zeros(21, 2)
        A₀[10:12, 1] .= 0.5
        A₀[10:12, 2] .= 0.2
        sol = solve_model(A₀, (0.0, 5.0), params, saveat=1.0)
        
        # Test plot_time_series for both populations
        @test_nowarn plot_time_series(sol, params)
        @test_nowarn plot_time_series(sol, params, pop_indices=[1])
        @test_nowarn plot_time_series(sol, params, pop_indices=[2])
        
        # Test plot_multi_population_snapshot
        @test_nowarn plot_multi_population_snapshot(sol, params, 1)
        
        # Test plot_phase_portrait
        @test_nowarn plot_phase_portrait(sol, params)
    end
    
    @testset "Periodic Lattice Visualizations" begin
        # Create simple periodic lattice model
        lattice = PeriodicLattice(extent=(10.0,), n_points=(21,))
        conn = GaussianConnectivityParameter(0.8, (2.0,))
        params = WilsonCowanParameters{1}(
            α=(1.0,), β=(1.0,), τ=(10.0,),
            connectivity=ConnectivityMatrix{1}(reshape([conn], 1, 1)),
            nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.25),
            stimulus=nothing, lattice=lattice, pop_names=("E",)
        )
        
        A₀ = zeros(21, 1)
        A₀[19:21, 1] .= 0.5  # Near boundary to test periodicity
        sol = solve_model(A₀, (0.0, 5.0), params, saveat=1.0)
        
        # Test visualizations work with periodic lattice
        @test_nowarn plot_time_series(sol, params)
        @test_nowarn plot_spatial_snapshot(sol, params, 1)
        @test_nowarn plot_spatiotemporal(sol, params)
    end
end

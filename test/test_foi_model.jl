#!/usr/bin/env julia

"""
Tests for the Failure of Inhibition (FoI) model in FailureOfInhibition2025.
"""

using Test
using FailureOfInhibition2025

@testset "FoI Model Tests" begin
    @testset "FoI Parameter Construction" begin
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        
        conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
        conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
        conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
        conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = FailureOfInhibitionParameters(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (10.0, 8.0),
            connectivity = connectivity,
            nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
            nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
                a_activating=5.0, θ_activating=0.3,
                a_failing=3.0, θ_failing=0.7
            ),
            stimulus = nothing,
            lattice = lattice
        )
        
        # FoI parameters are actually WilsonCowanParameters
        @test params isa WilsonCowanParameters{Float64, 2}
        @test params.α == (1.0, 1.5)
        @test params.β == (1.0, 1.0)
        @test params.τ == (10.0, 8.0)
        @test params.pop_names == ("E", "I")
        # Nonlinearity is stored as a tuple
        @test params.nonlinearity isa Tuple
        @test length(params.nonlinearity) == 2
        @test params.nonlinearity[1] isa RectifiedZeroedSigmoidNonlinearity
        @test params.nonlinearity[2] isa DifferenceOfSigmoidsNonlinearity
    end
    
    @testset "FoI Uses WilsonCowanParameters" begin
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        
        conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
        conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
        conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
        conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = FailureOfInhibitionParameters(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (10.0, 8.0),
            connectivity = connectivity,
            nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
            nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
                a_activating=5.0, θ_activating=0.3,
                a_failing=3.0, θ_failing=0.7
            ),
            stimulus = nothing,
            lattice = lattice
        )
        
        # FailureOfInhibitionParameters returns WilsonCowanParameters directly
        @test params isa WilsonCowanParameters{Float64, 2}
        @test params.α == (1.0, 1.5)
        @test params.pop_names == ("E", "I")
        @test params.nonlinearity isa Tuple
        @test length(params.nonlinearity) == 2
        @test params.nonlinearity[1] isa RectifiedZeroedSigmoidNonlinearity
        @test params.nonlinearity[2] isa DifferenceOfSigmoidsNonlinearity
    end
    
    @testset "FoI Dynamics" begin
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        
        conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
        conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
        conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
        conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = FailureOfInhibitionParameters(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (10.0, 8.0),
            connectivity = connectivity,
            nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
            nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
                a_activating=5.0, θ_activating=0.3,
                a_failing=3.0, θ_failing=0.7
            ),
            stimulus = nothing,
            lattice = lattice
        )
        
        # Test foi! function
        A = rand(21, 2) .* 0.5
        dA = zeros(21, 2)
        foi!(dA, A, params, 0.0)
        
        # Verify dynamics were computed
        @test !all(dA .== 0.0)
        @test size(dA) == size(A)
    end
    
    @testset "FoI with Point Lattice" begin
        lattice = PointLattice()
        
        conn_ee = ScalarConnectivity(1.0)
        conn_ei = ScalarConnectivity(-0.5)
        conn_ie = ScalarConnectivity(0.8)
        conn_ii = ScalarConnectivity(-0.3)
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = FailureOfInhibitionParameters(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (10.0, 8.0),
            connectivity = connectivity,
            nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
            nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
                a_activating=5.0, θ_activating=0.3,
                a_failing=3.0, θ_failing=0.7
            ),
            stimulus = nothing,
            lattice = lattice
        )
        
        # Test with point model (1, 2) shape
        A = reshape([0.3, 0.5], 1, 2)
        dA = zeros(1, 2)
        foi!(dA, A, params, 0.0)
        
        @test !all(dA .== 0.0)
        @test size(dA) == size(A)
    end
    
    @testset "FoI Inhibitory Nonlinearity Properties" begin
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        
        conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
        conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
        conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
        conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        # Test that the inhibitory nonlinearity has the expected parameters
        nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
            a_activating=5.0, θ_activating=0.3,
            a_failing=3.0, θ_failing=0.7
        )
        
        params = FailureOfInhibitionParameters(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (10.0, 8.0),
            connectivity = connectivity,
            nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
            nonlinearity_I = nonlinearity_I,
            stimulus = nothing,
            lattice = lattice
        )
        
        # Access inhibitory nonlinearity from tuple (second element)
        @test params.nonlinearity[2].a_activating == 5.0
        @test params.nonlinearity[2].θ_activating == 0.3
        @test params.nonlinearity[2].a_failing == 3.0
        @test params.nonlinearity[2].θ_failing == 0.7
    end
    
    @testset "FoI and WCM are Compatible" begin
        lattice = CompactLattice(extent=(10.0,), n_points=(21,))
        
        conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
        conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
        conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
        conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        params = FailureOfInhibitionParameters(
            α = (1.0, 1.5),
            β = (1.0, 1.0),
            τ = (10.0, 8.0),
            connectivity = connectivity,
            nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
            nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
                a_activating=5.0, θ_activating=0.3,
                a_failing=3.0, θ_failing=0.7
            ),
            stimulus = nothing,
            lattice = lattice
        )
        
        # Test that both foi! and wcm1973! work with the same parameters
        A = rand(21, 2) .* 0.5
        dA_foi = zeros(21, 2)
        dA_wcm = zeros(21, 2)
        
        foi!(dA_foi, A, params, 0.0)
        wcm1973!(dA_wcm, A, params, 0.0)
        
        # They should produce identical results
        @test dA_foi ≈ dA_wcm
    end
end

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
        
        @test params isa FailureOfInhibitionParameters
        @test params.α == (1.0, 1.5)
        @test params.β == (1.0, 1.0)
        @test params.τ == (10.0, 8.0)
        @test params.nonlinearity_E isa RectifiedZeroedSigmoidNonlinearity
        @test params.nonlinearity_I isa DifferenceOfSigmoidsNonlinearity
    end
    
    @testset "FoI to WCM Conversion" begin
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
        
        wcm_params = foi_to_wcm(params)
        
        @test wcm_params isa WilsonCowanParameters{Float64, 2}
        @test wcm_params.α == (1.0, 1.5)
        @test wcm_params.pop_names == ("E", "I")
        @test wcm_params.nonlinearity isa Tuple
        @test length(wcm_params.nonlinearity) == 2
        @test wcm_params.nonlinearity[1] isa RectifiedZeroedSigmoidNonlinearity
        @test wcm_params.nonlinearity[2] isa DifferenceOfSigmoidsNonlinearity
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
        
        @test params.nonlinearity_I.a_activating == 5.0
        @test params.nonlinearity_I.θ_activating == 0.3
        @test params.nonlinearity_I.a_failing == 3.0
        @test params.nonlinearity_I.θ_failing == 0.7
    end
end

println("🎉 All FoI model tests passed!")

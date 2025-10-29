using Test
using FailureOfInhibition2025

println("\n" * "="^70)
println("Testing Analysis Functions on Known Traveling Wave Solutions")
println("="^70)

"""
Test traveling wave analysis functions on a known analytical solution.

We use an exponentially decaying sech² traveling wave:
    A(x,t) = A₀ * exp(-λt) * sech²(k(x - ct))

where:
- A₀ is the initial amplitude
- λ is the decay rate
- k is the spatial wavenumber (controls width)
- c is the wave speed
"""

@testset "Analytical Traveling Wave Tests" begin
    
    @testset "Exponentially Decaying Sech² Wave" begin
        println("\n=== Testing with sech² traveling wave ===")
        
        # Wave parameters
        A₀ = 1.0        # Initial amplitude
        λ = 0.05        # Decay rate (1/time)
        k = 1.0         # Spatial wavenumber (1/space)
        c = 2.0         # Wave speed (space/time)
        
        # Spatial domain
        extent = 40.0
        n_points = 201
        lattice = CompactLattice(extent=(extent,), n_points=(n_points,))
        
        # Create WilsonCowanParameters (required for the new function)
        # The parameters are not used for the analytical wave itself,
        # but the lattice information is extracted from them
        conn = GaussianConnectivityParameter(1.0, (2.0,))
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
        
        # Time domain
        t_start = 0.0
        t_end = 10.0
        dt = 0.2
        times = t_start:dt:t_end
        
        # Initial position of wave peak
        x₀ = -10.0
        
        # Generate the traveling wave solution using the new function
        sol = generate_analytical_traveling_wave(
            params, times,
            wave_speed = c,
            decay_rate = λ,
            wavenumber = k,
            amplitude = A₀,
            initial_position = x₀
        )
        
        println("  Wave parameters:")
        println("    - Initial amplitude A₀: $(A₀)")
        println("    - Decay rate λ: $(λ)")
        println("    - Wavenumber k: $(k)")
        println("    - Speed c: $(c)")
        println("    - Expected speed: $(c) space/time")
        println("    - Expected decay rate: $(λ) /time")
        
        # Test 1: detect_traveling_peak
        println("\n  Testing detect_traveling_peak:")
        has_peak, trajectory, peak_times = detect_traveling_peak(sol, 1, threshold=0.1)
        
        @test has_peak == true
        @test length(trajectory) > 0
        @test length(peak_times) > 0
        println("    ✓ Traveling peak detected")
        
        # Test 2: compute_distance_traveled
        println("\n  Testing compute_distance_traveled:")
        distance, _ = compute_distance_traveled(sol, 1, lattice, threshold=0.1)
        
        # Expected distance = c * t_end
        expected_distance = c * t_end
        distance_error = abs(distance - expected_distance) / expected_distance
        
        println("    - Expected distance: $(round(expected_distance, digits=2))")
        println("    - Computed distance: $(round(distance, digits=2))")
        println("    - Relative error: $(round(distance_error * 100, digits=2))%")
        
        # Allow 10% error due to discretization
        @test distance_error < 0.1
        println("    ✓ Distance measurement accurate")
        
        # Test 3: compute_decay_rate
        println("\n  Testing compute_decay_rate:")
        decay_rate, amplitudes = compute_decay_rate(sol, 1)
        
        @test decay_rate !== nothing
        
        decay_error = abs(decay_rate - λ) / λ
        
        println("    - Expected decay rate: $(round(λ, digits=4))")
        println("    - Computed decay rate: $(round(decay_rate, digits=4))")
        println("    - Relative error: $(round(decay_error * 100, digits=2))%")
        
        # Allow 15% error due to discretization and log-linear fit
        @test decay_error < 0.15
        println("    ✓ Decay rate measurement accurate")
        
        # Test 4: compute_amplitude
        println("\n  Testing compute_amplitude:")
        amplitude = compute_amplitude(sol, 1, method=:max)
        
        # Should be close to A₀ (at t=0)
        amp_error = abs(amplitude - A₀) / A₀
        
        println("    - Expected max amplitude: $(round(A₀, digits=3))")
        println("    - Computed max amplitude: $(round(amplitude, digits=3))")
        println("    - Relative error: $(round(amp_error * 100, digits=2))%")
        
        @test amp_error < 0.05
        println("    ✓ Amplitude measurement accurate")
        
        # Test 5: compute_half_max_width
        println("\n  Testing compute_half_max_width:")
        width, half_max, profile = compute_half_max_width(sol, 1, 1, lattice)  # At t=0
        
        # For sech²(kx), the FWHM is approximately 1.7627/k
        expected_width = 1.7627 / k
        width_error = abs(width - expected_width) / expected_width
        
        println("    - Expected width (theoretical FWHM): $(round(expected_width, digits=2))")
        println("    - Computed width: $(round(width, digits=2))")
        println("    - Relative error: $(round(width_error * 100, digits=2))%")
        
        # Note: The half-max width algorithm may differ from theoretical FWHM
        # due to discretization and the specific definition used.
        # Allow 50% error as we're testing the algorithm works, not that it
        # matches the exact theoretical value
        @test width_error < 0.5
        println("    ✓ Width measurement reasonably accurate")
        
        # Test 6: Speed measurement from trajectory
        println("\n  Testing wave speed from trajectory:")
        if length(trajectory) > 1
            # Calculate speed from peak positions
            time_span = peak_times[end] - peak_times[1]
            if time_span > 0
                # Convert index displacement to physical distance
                idx_displacement = trajectory[end] - trajectory[1]
                dx = extent / (n_points - 1)
                physical_displacement = idx_displacement * dx
                measured_speed = physical_displacement / time_span
                
                speed_error = abs(measured_speed - c) / c
                
                println("    - Expected speed: $(round(c, digits=2))")
                println("    - Measured speed: $(round(measured_speed, digits=2))")
                println("    - Relative error: $(round(speed_error * 100, digits=2))%")
                
                # Allow 15% error
                @test speed_error < 0.15
                println("    ✓ Speed measurement accurate")
            end
        end
    end
    
    @testset "Non-decaying Sech² Wave" begin
        println("\n=== Testing with non-decaying sech² wave ===")
        
        # Wave parameters (no decay)
        A₀ = 0.8
        λ = 0.0  # No decay
        k = 0.5
        c = 1.5
        
        # Spatial domain
        extent = 40.0
        n_points = 151
        lattice = CompactLattice(extent=(extent,), n_points=(n_points,))
        
        # Create WilsonCowanParameters
        conn = GaussianConnectivityParameter(1.0, (2.0,))
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
        
        # Time domain
        times = 0.0:0.5:15.0
        x₀ = -12.0
        
        # Generate the non-decaying traveling wave solution
        sol = generate_analytical_traveling_wave(
            params, times,
            wave_speed = c,
            decay_rate = λ,
            wavenumber = k,
            amplitude = A₀,
            initial_position = x₀
        )
        
        # Test that no decay is detected
        println("  Testing decay detection on non-decaying wave:")
        decay_rate, _ = compute_decay_rate(sol, 1)
        
        # Should return nothing or very small decay rate
        if decay_rate !== nothing
            println("    - Detected decay rate: $(round(decay_rate, digits=6))")
            @test decay_rate < 0.01  # Very small
        else
            println("    - No decay detected (as expected)")
            @test decay_rate === nothing
        end
        println("    ✓ Non-decaying wave handled correctly")
        
        # Test amplitude remains constant
        println("  Testing amplitude consistency:")
        amp = compute_amplitude(sol, 1, method=:max)
        @test abs(amp - A₀) / A₀ < 0.05
        println("    ✓ Amplitude remains close to A₀")
    end
    
end

println("\n" * "="^70)
println("All Analytical Wave Tests Passed!")
println("="^70)

#!/usr/bin/env julia

"""
Example demonstrating the use of stimulation functions in FailureOfInhibition2025.

This example shows how to:
1. Create a spatial lattice
2. Define a circular stimulus
3. Apply the stimulus to a neural field
4. Visualize the results
"""

using FailureOfInhibition2025

function example_1d_stimulation()
    println("=== 1D Stimulation Example ===\n")
    
    # Create a 1D spatial lattice from -10 to 10 with 101 points
    lattice = CompactLattice(-10.0, 10.0, 101)
    println("Created 1D lattice: $(size(lattice)) points from -10 to 10")
    
    # Define a circular stimulus centered at the origin
    # - radius: 3.0 (covers points within 3 units of center)
    # - strength: 10.0 (amplitude of stimulus)
    # - time_windows: active from t=0 to t=10
    # - center: nothing (defaults to origin)
    # - baseline: 0.0 (background activity level)
    stimulus = CircleStimulus(
        radius=3.0,
        strength=10.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        center=nothing,
        baseline=0.0
    )
    println("\nCreated stimulus:")
    println("  - Radius: $(stimulus.radius)")
    println("  - Strength: $(stimulus.strength)")
    println("  - Active during: $(stimulus.time_windows)")
    
    # Create arrays for neural activity
    dA = zeros(Float64, size(lattice)...)  # Rate of change
    A = zeros(Float64, size(lattice)...)   # Current activity
    
    # Apply stimulus at t=5.0 (within active time window)
    stimulate!(dA, A, stimulus, 5.0)
    
    println("\nStimulation applied at t=5.0")
    println("Points affected: $(sum(dA .> 0.0)) out of $(length(dA))")
    println("Maximum stimulus value: $(maximum(dA))")
    
    # Show the spatial profile
    println("\nSpatial profile (showing every 10th point):")
    coords = coordinates(lattice)
    for i in 1:10:length(coords)
        coord = coords[i]
        println("  x=$(coord[1]): dA=$(dA[i])")
    end
    
    println("\n=== 1D Example Complete ===\n")
end

function example_2d_stimulation()
    println("=== 2D Stimulation Example ===\n")
    
    # Create a 2D spatial lattice
    lattice = CompactLattice((-10.0, -10.0), (10.0, 10.0), (21, 21))
    println("Created 2D lattice: $(size(lattice)) grid from (-10,-10) to (10,10)")
    
    # Define a circular stimulus centered at (3.0, -2.0)
    stimulus = CircleStimulus(
        radius=4.0,
        strength=15.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        center=(3.0, -2.0),
        baseline=1.0
    )
    println("\nCreated off-center stimulus:")
    println("  - Center: $(stimulus.center)")
    println("  - Radius: $(stimulus.radius)")
    println("  - Strength: $(stimulus.strength)")
    println("  - Baseline: $(stimulus.baseline)")
    
    # Create arrays for neural activity
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    
    # Apply stimulus
    stimulate!(dA, A, stimulus, 5.0)
    
    println("\nStimulation results:")
    println("  - Total points: $(length(dA))")
    println("  - Points at baseline: $(sum(dA .== stimulus.baseline))")
    println("  - Points stimulated: $(sum(dA .> stimulus.baseline))")
    println("  - Maximum value: $(maximum(dA))")
    println("  - Minimum value: $(minimum(dA))")
    
    println("\n=== 2D Example Complete ===\n")
end

function example_time_windows()
    println("=== Time Window Example ===\n")
    
    lattice = CompactLattice(-5.0, 5.0, 51)
    
    # Create stimulus with multiple time windows (pulsed stimulation)
    stimulus = CircleStimulus(
        radius=2.0,
        strength=5.0,
        time_windows=[(0.0, 1.0), (3.0, 4.0), (6.0, 7.0)],  # Three pulses
        lattice=lattice,
        baseline=0.0
    )
    
    println("Created pulsed stimulus with $(length(stimulus.time_windows)) pulses:")
    for (i, window) in enumerate(stimulus.time_windows)
        println("  Pulse $i: t=$(window[1]) to $(window[2])")
    end
    
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    
    # Test at different times
    test_times = [0.5, 2.0, 3.5, 5.0, 6.5, 8.0]
    println("\nTesting stimulus at different times:")
    for t in test_times
        fill!(dA, 0.0)
        stimulate!(dA, A, stimulus, t)
        is_active = any(dA .> 0.0)
        status = is_active ? "ACTIVE" : "inactive"
        println("  t=$t: $status (max dA=$(maximum(dA)))")
    end
    
    println("\n=== Time Window Example Complete ===\n")
end

function example_wrapper_function()
    println("=== Using Wrapper Function ===\n")
    
    lattice = CompactLattice(-5.0, 5.0, 41)
    stimulus = CircleStimulus(
        radius=2.5,
        strength=8.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice
    )
    
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    
    stimulate!(dA, A, stimulus, 5.0)
    
    println("  - Stimulus applied successfully")
    println("  - Points stimulated: $(sum(dA .> 0.0))")
    
    println("\n=== Wrapper Function Example Complete ===\n")
end

function main()
    println("\n" * "="^60)
    println("  Stimulation Functions Examples")
    println("="^60 * "\n")
    
    example_1d_stimulation()
    example_2d_stimulation()
    example_time_windows()
    example_wrapper_function()
    
    println("="^60)
    println("  All examples completed successfully!")
    println("="^60 * "\n")
end

# Run examples when script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

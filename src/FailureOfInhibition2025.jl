module FailureOfInhibition2025

greet() = print("Hello World!")

# Include space/lattice/coordinates functionality
include("space.jl")

# Export main space types and functions
export AbstractSpace, AbstractLattice
export AbstractCompactLattice, CompactLattice
export AbstractPeriodicLattice, PeriodicLattice
export AbstractAugmentedLattice, AbstractEmbeddedLattice, RandomlyEmbeddedLattice
export coordinates, differences, difference
export discrete_segment, discrete_lattice, coordinate_axes
export abs_difference, abs_difference_periodic
export start, stop, extent, simpson_weights
export fft_center_idx

end # module FailureOfInhibition2025

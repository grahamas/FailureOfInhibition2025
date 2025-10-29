module FailureOfInhibition2025

greet() = print("Hello World!")

# Include space/lattice/coordinates functionality
include("space.jl")

# Include connectivity functionality
include("connect.jl")

# Include stimulus functionality  
include("stimulate.jl")

# Include nonlinearity functionality
include("nonlinearity.jl")

# Include model functionality
include("models.jl")

# Include simulation functionality
include("simulate.jl")

# Include bifurcation analysis functionality
include("bifurcation.jl")

# Export main space types and functions
export AbstractSpace, AbstractLattice
export AbstractCompactLattice, CompactLattice
export AbstractPeriodicLattice, PeriodicLattice
export AbstractPointLattice, PointLattice
export AbstractAugmentedLattice, AbstractEmbeddedLattice, RandomlyEmbeddedLattice
export coordinates, differences, difference
export discrete_segment, discrete_lattice, coordinate_axes
export abs_difference, abs_difference_periodic
export start, stop, extent, simpson_weights
export fft_center_idx

# Export nonlinearity types and functions
export SigmoidNonlinearity, RectifiedZeroedSigmoidNonlinearity, DifferenceOfSigmoidsNonlinearity, simple_sigmoid, rectified_zeroed_sigmoid, difference_of_simple_sigmoids, difference_of_rectified_zeroed_sigmoids, apply_nonlinearity!

# Export connectivity types and functions
export GaussianConnectivityParameter, ScalarConnectivity, ConnectivityMatrix, propagate_activation

# Export stimulation types and functions
export CircleStimulus, stimulate!

# Export model functions
export WilsonCowanParameters, wcm1973!, population

# Export simulation functions
export solve_model, save_simulation_results, save_simulation_summary

# Export bifurcation analysis functions (BifurcationKit integration)
export create_bifurcation_problem, wcm_rhs!

end # module FailureOfInhibition2025

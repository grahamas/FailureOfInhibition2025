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

# Include sensitivity analysis functionality
include("local_sensitivity.jl")
include("sensitivity.jl")

# Include analysis functionality
include("analysis.jl")

# Include optimization functionality
include("optimize.jl")

# Include canonical model parameter sets
include("canonical.jl")

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
export GaussianConnectivityParameter, ScalarConnectivity, ConnectivityMatrix, propagate_activation, prepare_connectivity

# Export stimulation types and functions
export CircleStimulus, stimulate!

# Export model functions
export WilsonCowanParameters, wcm1973!, population
export FailureOfInhibitionParameters, foi!, foi_to_wcm

# Export simulation functions
export solve_model, save_simulation_results, save_simulation_summary

# Export GPU simulation functions (implemented in CUDA extension)
function solve_model_gpu end
export solve_model_gpu

# Export bifurcation analysis functions (BifurcationKit integration)
export create_bifurcation_problem, wcm_rhs!

# Export sensitivity analysis functions
export compute_local_sensitivities, save_local_sensitivities, summarize_sensitivities
export compute_sensitivity_indices
export extract_parameters, reconstruct_parameters, ODEParameterWrapper
export sobol_sensitivity_analysis, morris_sensitivity_analysis
export create_parameter_builder, create_output_function

# Export GPU sensitivity analysis functions (implemented in CUDA extension)
function sobol_sensitivity_analysis_gpu end
function morris_sensitivity_analysis_gpu end
export sobol_sensitivity_analysis_gpu, morris_sensitivity_analysis_gpu

# Export analysis functions
export detect_traveling_peak, compute_decay_rate, compute_amplitude
export compute_distance_traveled, compute_half_max_width
export detect_oscillations, compute_oscillation_frequency, compute_oscillation_amplitude
export compute_oscillation_decay, compute_oscillation_duration
export generate_analytical_traveling_wave
export coordinates

# Export optimization functions
export TravelingWaveObjective, optimize_for_traveling_wave, _update_params

# Export GPU optimization functions (implemented in CUDA extension)
function optimize_for_traveling_wave_gpu end
export optimize_for_traveling_wave_gpu

# Export canonical model parameter functions
export create_wcm1973_parameters, create_point_model_wcm1973

end # module FailureOfInhibition2025

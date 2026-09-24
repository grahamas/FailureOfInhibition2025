module FailureOfInhibition2025

include("responses.jl")
include("drives.jl")
include("point_model.jl")
include("stability.jl")
include("equilibria.jl")
include("configurations.jl")
include("simulation.jl")
include("diagnostics.jl")
include("continuation.jl")
include("periodic_orbits.jl")
include("hopf_diagnostics.jl")
include("pulse_experiments.jl")

export LogisticResponse
export FailureOfInhibitionResponse
export RectifiedZeroedLogisticResponse
export DifferenceOfLogisticsCandidate
export DifferenceOfRectifiedZeroedLogisticsCandidate
export response, response_derivative

export NoDrive, DrivePulse, DriveInterpretation
export AfferentExcitation, AbstractIntervention
export PiecewiseConstantDrive, drive_value

export PopulationParameters, PointCoupling, PointModelParameters
export point_balance!, point_balance_jacobian!, point_rhs!, point_jacobian!
export matched_point_models

export StabilityOptions, StabilityClassification, SpectralGeometry
export Attracting, Repelling, Saddle, StabilityUnresolved
export RealDistinctSpectrum, RealRepeatedSpectrum, ComplexConjugateSpectrum
export LocalStabilityResult, classify_local_stability

export EquilibriumOptions, CandidateValidation, SearchCompleteness
export AdmissibleCandidate, BoundaryAmbiguousCandidate, RejectedCandidate
export CompletenessNotCertified
export EquilibriumAttempt, EquilibriumSolveResult, Equilibrium, EquilibriumSearchResult
export default_equilibrium_seeds, solve_equilibrium, find_equilibria

export solve_point_model, write_trajectory_csv

export DiagnosticOptions, TrajectoryDiagnostics, TrajectoryClassification
export EquilibriumCompatible, TrajectoryUnresolved, diagnose_trajectory

export ContinuationOptions, ContinuationAttempt, ContinuationPoint
export ContinuationBranch, ContinuationCandidate, EquilibriumContinuationResult
export continue_equilibria

export PeriodicOrbitOptions, PeriodicOrbitValidation, PeriodicOrbitStability
export NumericallyValidatedPeriodicOrbit, PeriodicOrbitUnresolved
export PeriodicOrbitAttracting, PeriodicOrbitRepelling, PeriodicOrbitStabilityUnresolved
export PeriodicOrbitResult, solve_periodic_orbit, periodic_orbit_phases

export Figure5bTopologyOptions, Figure5bTopologyResult
export classify_figure5b_topology
export HopfDiagnosticOptions, HopfDiagnosticResult, hopf_diagnostics

export PulseExperimentOptions, PulseTrialResult, PulseExperimentResult
export run_pulse_trial, run_pulse_experiments

end

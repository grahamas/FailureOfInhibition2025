module FailureOfInhibition2025

include("responses.jl")
include("drives.jl")
include("point_model.jl")
include("stability.jl")
include("equilibria.jl")
include("configurations.jl")
include("simulation.jl")

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

end

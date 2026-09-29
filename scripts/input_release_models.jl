module InputReleaseModels
using FailureOfInhibition2025

"""
    release_models(case; e_to_e, on_input=8.0)

Pair autonomous FoI models differing only in E drive, with identical population
and coupling objects. The off model represents permanent release to zero.
"""
function release_models(case; e_to_e, on_input=8.0)
    for (value, name, positive) in ((e_to_e, "e_to_e", false), (on_input, "on_input", true))
        value isa Real && !(value isa Bool) && isfinite(value) &&
            (positive ? value > 0 : value >= 0) && isfinite(Float64(value)) ||
            throw(ArgumentError("$name must be finite, representable as Float64 and $(positive ? "positive" : "nonnegative")"))
    end
    off = matched_point_models(
        excitatory=PopulationParameters(timescale=case["tau_e"],
            response=LogisticResponse(slope=case["slope_e"], threshold=case["theta_e"])),
        inhibitory_control=PopulationParameters(timescale=case["tau_e"] * case["tau_ratio"],
            response=LogisticResponse(slope=case["slope_i"], threshold=case["theta_on"])),
        failure_threshold=case["theta_off"],
        coupling=PointCoupling(e_to_e=Float64(e_to_e), i_to_e=case["i_to_e"],
            e_to_i=case["e_to_i"], i_to_i=case["i_to_i"]),
        drive=PiecewiseConstantDrive(baseline=(0.0, 0.0), pulses=(),
            interpretation=AfferentExcitation)).failure_of_inhibition
    on = PointModelParameters(excitatory=off.excitatory, inhibitory=off.inhibitory,
        coupling=off.coupling, drive=PiecewiseConstantDrive(baseline=(Float64(on_input), 0.0),
            pulses=(), interpretation=AfferentExcitation))
    return (; off, on)
end
end

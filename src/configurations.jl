"""
    matched_point_models(;
        excitatory,
        inhibitory_control,
        failure_threshold,
        coupling,
        drive=NoDrive(),
    )

Construct matched control and failure-of-inhibition point models. Both models
share the supplied excitatory population, coupling, and drive objects. The
control model also uses the supplied inhibitory population unchanged; the
failure-of-inhibition model preserves its time constant and replaces only its
logistic response with an equal-slope `FailureOfInhibitionResponse` beginning
at the same onset threshold.

Both supplied population responses must be `LogisticResponse`s. The existing
response and model constructors validate the failure threshold and remaining
model parameters.
"""
function matched_point_models(;
    excitatory::PopulationParameters,
    inhibitory_control::PopulationParameters,
    failure_threshold,
    coupling::PointCoupling,
    drive::AbstractPointDrive=NoDrive(),
)
    excitatory.response isa LogisticResponse || throw(
        ArgumentError("excitatory.response must be LogisticResponse"),
    )
    inhibitory_control.response isa LogisticResponse || throw(
        ArgumentError("inhibitory_control.response must be LogisticResponse"),
    )

    inhibitory_failure = PopulationParameters(
        timescale=inhibitory_control.timescale,
        response=FailureOfInhibitionResponse(
            inhibitory_control.response;
            failure_threshold=failure_threshold,
        ),
    )

    return (
        control=PointModelParameters(
            excitatory=excitatory,
            inhibitory=inhibitory_control,
            coupling=coupling,
            drive=drive,
        ),
        failure_of_inhibition=PointModelParameters(
            excitatory=excitatory,
            inhibitory=inhibitory_failure,
            coupling=coupling,
            drive=drive,
        ),
    )
end

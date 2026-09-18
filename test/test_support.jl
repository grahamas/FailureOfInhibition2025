function synthetic_model(;
    excitatory_response=LogisticResponse(slope=2.0, threshold=0.1),
    inhibitory_response=LogisticResponse(slope=1.5, threshold=0.2),
    drive=NoDrive(),
)
    excitatory = PopulationParameters(
        decay=0.7,
        saturation=1.1,
        timescale=2.0,
        response=excitatory_response,
    )
    inhibitory = PopulationParameters(
        decay=0.9,
        saturation=0.8,
        timescale=1.5,
        response=inhibitory_response,
    )
    coupling = PointCoupling(
        e_to_e=1.2,
        i_to_e=-0.6,
        e_to_i=0.7,
        i_to_i=-0.4,
    )
    return PointModelParameters(
        excitatory=excitatory,
        inhibitory=inhibitory,
        coupling=coupling,
        drive=drive,
    )
end

function central_difference(parameters, x; step=1.0e-6)
    return (response(parameters, x + step) - response(parameters, x - step)) / (2step)
end

import TOML

"""Check that a captured runner can load its literal include closure from source/."""
function assert_archived_include_closure(output, entrypoint)
    metadata = TOML.parsefile(joinpath(output, "metadata.toml"))
    declared = Set(keys(metadata["source_sha256"]))
    visited = Set{String}()
    function visit(relative)
        relative in visited && return
        push!(visited, relative)
        @test relative in declared
        source = joinpath(output, "source", relative)
        @test isfile(source)
        isfile(source) || return
        for match in eachmatch(r"include\(\"([^\"]+\.jl)\"\)", read(source, String))
            visit(normpath(joinpath(dirname(relative), match.captures[1])))
        end
    end
    visit(joinpath("scripts", entrypoint))
end

function synthetic_model(;
    excitatory_response=LogisticResponse(slope=2.0, threshold=0.1),
    inhibitory_response=LogisticResponse(slope=1.5, threshold=0.2),
    excitatory_timescale=2.0,
    inhibitory_timescale=1.5,
    coupling=PointCoupling(
        e_to_e=1.2,
        i_to_e=0.6,
        e_to_i=0.7,
        i_to_i=0.4,
    ),
    drive=NoDrive(),
)
    excitatory = PopulationParameters(
        timescale=excitatory_timescale,
        response=excitatory_response,
    )
    inhibitory = PopulationParameters(
        timescale=inhibitory_timescale,
        response=inhibitory_response,
    )
    return PointModelParameters(
        excitatory=excitatory,
        inhibitory=inhibitory,
        coupling=coupling,
        drive=drive,
    )
end

function synthetic_foi_model(; drive=NoDrive())
    return synthetic_model(
        inhibitory_response=FailureOfInhibitionResponse(
            slope=1.5,
            onset_threshold=0.2,
            failure_threshold=1.0,
        ),
        drive=drive,
    )
end

function synthetic_matched_models(; drive=NoDrive(), failure_threshold=1.0)
    excitatory_response = LogisticResponse(slope=2.0, threshold=0.1)
    inhibitory_onset = LogisticResponse(slope=1.5, threshold=0.2)
    excitatory = PopulationParameters(timescale=2.0, response=excitatory_response)
    inhibitory_control = PopulationParameters(timescale=1.5, response=inhibitory_onset)
    inhibitory_failure = PopulationParameters(
        timescale=inhibitory_control.timescale,
        response=FailureOfInhibitionResponse(
            inhibitory_onset;
            failure_threshold=failure_threshold,
        ),
    )
    coupling = PointCoupling(
        e_to_e=1.2,
        i_to_e=0.6,
        e_to_i=0.7,
        i_to_i=0.4,
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

function central_difference(parameters, x; step=1.0e-6)
    return (response(parameters, x + step) - response(parameters, x - step)) / (2step)
end

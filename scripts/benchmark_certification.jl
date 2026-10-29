# Bounded engineering fixtures only; this is not a parameter study.
using FailureOfInhibition2025
import TOML
const FC = FailureOfInhibition2025

function certification_benchmark_models()
    function model(threshold, coupling; foi=false)
        PointModelParameters(
            excitatory=PopulationParameters(timescale=1.0,
                response=LogisticResponse(slope=1.0, threshold=threshold)),
            inhibitory=PopulationParameters(timescale=2.0,
                response=foi ? FailureOfInhibitionResponse(slope=1.0,
                    onset_threshold=0.0, failure_threshold=2.0) :
                    LogisticResponse(slope=1.0, threshold=0.0)),
            coupling=PointCoupling(e_to_e=coupling, i_to_e=0.0,
                e_to_i=0.0, i_to_i=0.0))
    end
    return [("regular", model(0.3, 0.0; foi=true)),
        ("three_roots", model(3 + log(2), 12.0)),
        ("saturated", model(-100.0, 0.0))]
end

function benchmark_certification()
    records = Dict[]
    for (name, model) in certification_benchmark_models()
        options = CertificationOptions(time_limit=30.0, maximum_boxes=10_000)
        cold = @timed certify_equilibria(model; options)
        @assert cold.value.status == :certified
        @assert verify_equilibrium_certificate(cold.value)
        field, jacobian = FC._certification_callbacks(model)
        for (variant, range_exclusion, contraction) in
            (("default", true, true), ("no_range_exclusion", false, true),
             ("no_contraction", true, false))
            run() = FC._certify_equilibria(field, jacobian; options, model,
                precision_hint=box -> FC._certification_saturation(model, box),
                use_range_exclusion=range_exclusion, use_contraction=contraction)
            run() # compile this invocation before measurement
            samples = [@timed(run()) for _ in 1:3]
            for sample in samples
                @assert sample.value.status == :certified
                @assert sample.value.equilibrium_count == cold.value.equilibrium_count
                @assert verify_equilibrium_certificate(sample.value)
            end
            best = samples[argmin([s.time for s in samples])]
            result = best.value
            push!(records, Dict("fixture"=>name, "variant"=>variant,
                "first_call_seconds"=>cold.time,
                "warmed_min_seconds"=>best.time, "allocated_bytes"=>best.bytes,
                "processed_boxes"=>result.processed_boxes,
                "evaluations"=>result.evaluations,
                "root_count"=>result.equilibrium_count,
                "maximum_used_precision"=>maximum(n.precision for n in result.nodes)))
        end
    end
    TOML.print(stdout, Dict("julia_version"=>string(VERSION), "measurements"=>records); sorted=true)
end

abspath(PROGRAM_FILE) == (@__FILE__) && benchmark_certification()

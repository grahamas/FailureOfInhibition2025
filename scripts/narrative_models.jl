"""Model construction, provisional coordinate roles, and bounded sampling for the narrative study."""
module NarrativeModels
using FailureOfInhibition2025

function finite_number(x, name; lower=0.0, positive=false)
    x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)) &&
        (positive ? x > lower : x >= lower) || throw(ArgumentError("invalid $name"))
    return Float64(x)
end

"""Construct matched models at the declared tonic input; never require a zero-input source."""
function models(p)
    for key in ("e_to_e", "i_to_e", "e_to_i", "i_to_i", "B_E")
        finite_number(p[key], key)
    end
    finite_number(p["tau_ratio"], "tau_ratio"; positive=true)
    finite_number(p["theta_off"], "theta_off"; lower=4.0, positive=true)
    return matched_point_models(
        excitatory=PopulationParameters(timescale=7.8,
            response=LogisticResponse(slope=5.0, threshold=1.5)),
        inhibitory_control=PopulationParameters(timescale=7.8*p["tau_ratio"],
            response=LogisticResponse(slope=5.0, threshold=4.0)),
        failure_threshold=p["theta_off"],
        coupling=PointCoupling(e_to_e=p["e_to_e"], i_to_e=p["i_to_e"],
            e_to_i=p["e_to_i"], i_to_i=p["i_to_i"]),
        drive=PiecewiseConstantDrive(baseline=(p["B_E"], 0.0), pulses=(),
            interpretation=AfferentExcitation))
end

"""Return distinct relative-coordinate role assignments, retaining ambiguity explicitly.

No absolute biological activity threshold is imposed. A solitary sink remains
unassigned: its rank alone cannot identify rest. The resting branch can be
matched separately from a known multi-state context when input is changed.
"""
function roles(model, search)
    sinks = [i for (i,r) in enumerate(search.equilibria)
        if r.stability.classification == Attracting && !r.near_singular]
    sort!(sinks; by=i -> search.equilibria[i].state[1])
    result = Dict{String,Int}()
    length(sinks) >= 2 || return result
    state(i) = search.equilibria[i].state
    margin = 1e-6
    midpoint = (model.inhibitory.response.onset_threshold +
        model.inhibitory.response.failure_threshold)/2
    input(i) = model.coupling.e_to_i*state(i)[1] - model.coupling.i_to_i*state(i)[2]
    high = sinks
    # A descending, lower-I state must be higher-E than another more inhibited sink.
    seizures = [s for s in high if input(s)>midpoint && any(h ->
        h != s && state(h)[2] > state(s)[2]+margin &&
        state(h)[1] <= state(s)[1]+margin, high)]
    length(seizures) == 1 || return result
    seizure = only(seizures)
    result["seizure"] = seizure
    rest = first(sinks)
    others = [i for i in sinks if i != seizure]
    # Rest requires separation from an additional nonquiescent sink; two
    # almost equal-E high sinks must never be called rest and seizure.
    if length(sinks)>=3 && all(i -> i==rest ||
            state(rest)[1]+margin<state(i)[1], others)
        result["rest"] = rest
    end
    # The intermediate (inhibition-stabilized) E-nullcline arm has positive
    # E self-derivative. Saturated final-arm high-I states retain the herald
    # role even if their E coordinate is microscopically below the low-I state.
    function middle_arm(i)
        jac=zeros(2,2)
        point_jacobian!(jac,state(i),model,0.0)
        return jac[1,1]>0
    end
    active = [i for i in high if i != seizure && i != get(result,"rest",0) && input(i)<midpoint && middle_arm(i) &&
        state(i)[1]+margin < state(seizure)[1] && state(i)[2]>state(seizure)[2]+margin]
    length(active) == 1 && (result["active"] = only(active))
    herald = [i for i in high if i != seizure && i != get(result,"rest",0) && !(i in active) &&
        state(i)[2]>state(seizure)[2]+margin &&
        (!haskey(result,"active") || state(i)[1]>state(result["active"])[1]+margin)]
    length(herald) == 1 && (result["herald"] = only(herald))
    return result
end

role_name(assignments, index) = something(findfirst(==(index), assignments), "unassigned")

"""Unique nearest-coordinate correspondence; ties and collisions remain unassigned."""
function match_roles(old_search, old_roles, search; atol=0.15)
    sinks = [i for (i,r) in enumerate(search.equilibria)
        if r.stability.classification == Attracting && !r.near_singular]
    result = Dict{String,Int}()
    for (name, source) in old_roles
        distances = sort([(maximum(abs.(old_search.equilibria[source].state .-
            search.equilibria[i].state)), i) for i in sinks])
        isempty(distances) && continue
        distances[1][1] <= atol || continue
        length(distances)>1 && distances[2][1]-distances[1][1]<=1e-6 && continue
        result[name] = distances[1][2]
    end
    collisions = [i for i in values(result) if count(==(i), values(result))>1]
    filter!(pair -> !(pair.second in collisions), result)
    return result
end

function radical_inverse(n, base)
    value, factor = 0.0, 1.0/base
    while n > 0
        value += (n % base)*factor
        n = div(n, base)
        factor /= base
    end
    return value
end

"""Visit endpoints and interior probes, refining every sampled outcome change.

Equal-outcome cells remain finite sampling, not an exclusion of narrow islands.
The caller owns the immutable trial results cached by this function.
"""
function sample_line(evaluate, stop; step, width)
    finite_number(stop,"stop"); finite_number(step,"step"; positive=true)
    finite_number(width,"width"; positive=true)
    cache = Dict{Float64,Any}()
    at(x) = get!(() -> evaluate(x), cache, Float64(x))
    key(x) = (at(x).status, at(x).destination)
    axis = unique(vcat(collect(0.0:step:stop), [Float64(stop)]))
    at(0.0)
    function refine(a,b)
        m = (a+b)/2
        ka,kb,km = key(a),key(b),key(m)
        b-a <= width && return
        if ka != kb || ka != km || ka[1] != "compatible"
            refine(a,m); refine(m,b)
        end
    end
    for (a,b) in zip(axis,axis[2:end])
        refine(a,b)
    end
    return cache
end
end

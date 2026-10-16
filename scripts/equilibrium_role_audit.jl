"""Explain existing equilibrium roles without changing the role classifier."""
module EquilibriumRoleAudit
using FailureOfInhibition2025, LinearAlgebra
import TOML
include("input_response_models.jl")
const M=InputResponseModels

"""Read only the retained roots, avoiding potentially large solver-attempt arrays."""
function retained_context(text)
    firstroot=findfirst("[[equilibria]]",text)
    firstroot===nothing && return Dict{String,Any}("equilibria"=>Any[])
    TOML.parse(text[first(firstroot):end])
end

function search_view(records)
    (;equilibria=[(;state=r["state"],near_singular=r["near_singular"],
        stability=(;classification=getproperty(FailureOfInhibition2025,Symbol(r["stability"]["classification"])))) for r in records])
end

"""Intrinsic geometry plus exact eligibility failures under the existing contextual roles.

The response-side sign uses the analytical peak input as well as the computed
derivative: floating-point underflow of a tail derivative is recorded, not
silently interpreted as being at the peak. No biological threshold is added.
"""
function explain(model,records)
    search=search_view(records);assigned=M.roles(model,search)
    sinks=[k for (k,r) in enumerate(search.equilibria) if r.stability.classification==Attracting && !r.near_singular]
    a=model.coupling.e_to_e;s=model.excitatory.response.slope
    turns=a*s>8
    lo=turns ? (1-sqrt(1-8/(a*s)))/4 : NaN
    hi=turns ? (1+sqrt(1-8/(a*s)))/4 : NaN
    c=model.coupling;be,bi=model.drive.baseline
    peak=(model.inhibitory.response.onset_threshold+model.inhibitory.response.failure_threshold)/2
    state(k)=records[k]["state"]
    vi(k)=bi+c.e_to_i*state(k)[1]-c.i_to_i*state(k)[2]
    di(k)=response_derivative(model.inhibitory.response,vi(k))
    rest=turns ? filter(k->state(k)[1]<lo,sinks) : Int[]
    active=turns ? filter(k->lo<state(k)[1]<hi && di(k)>0,sinks) : Int[]
    high=turns ? filter(k->state(k)[1]>hi,sinks) : Int[]
    comparisons(k)=filter(h->state(h)[2]>state(k)[2]+1e-6 && state(h)[1]<=state(k)[1]+1e-6,sinks)
    seizures=filter(k->di(k)<0 && !isempty(comparisons(k)),high)
    herald=length(seizures)==1 ? filter(k->k!=only(seizures) && state(k)[2]>state(only(seizures))[2]+1e-6,high) : Int[]
    rows=NamedTuple[]
    for (k,r) in enumerate(records)
        e,i=state(k);u=be+c.e_to_e*e-c.i_to_e*i;v=vi(k)
        arm=!turns ? "no_turning_points" : e<lo ? "lower" : e>hi ? "upper" : lo<e<hi ? "middle" : "turning_point"
        side=v<peak ? "ascending" : v>peak ? "descending" : "peak"
        role=something(findfirst(==(k),assigned),"unassigned");reasons=String[]
        if role=="unassigned"
            r["stability"]["classification"]!="Attracting" && push!(reasons,"non_attracting")
            r["near_singular"] && push!(reasons,"near_singular")
            if k in sinks
                if !turns
                    push!(reasons,"no_E_turning_points")
                elseif arm=="lower"
                    push!(reasons,"multiple_rest_candidates")
                elseif arm=="middle"
                    push!(reasons,di(k)>0 ? "multiple_active_candidates" : "middle_without_positive_I_slope")
                elseif arm=="turning_point"
                    push!(reasons,"on_E_turning_point")
                else
                    di(k)<0 && isempty(comparisons(k)) && push!(reasons,"missing_higher_I_sink")
                    length(seizures)>1 && push!(reasons,"multiple_seizure_candidates")
                    isempty(seizures) && push!(reasons,"no_unique_seizure_for_herald")
                    k in herald && length(herald)>1 && push!(reasons,"multiple_herald_candidates")
                    di(k)==0 && side!="peak" && push!(reasons,"I_slope_underflow")
                    isempty(reasons) && push!(reasons,"outside_high_role_predicates")
                end
            end
            isempty(reasons) && error("unexplained root $k")
        end
        jac=zeros(2,2);point_jacobian!(jac,state(k),model,0.)
        eig=sort(eigvals(jac);by=x->(real(x),imag(x)))
        push!(rows,(;root=k,E=e,I=i,role,stability=r["stability"]["classification"],near_singular=r["near_singular"],
            arm,I_side=side,total_E_input=u,total_I_input=v,E_response=response(model.excitatory.response,u),
            I_response=response(model.inhibitory.response,v),E_slope=response_derivative(model.excitatory.response,u),I_slope=di(k),
            lower_turn=lo,upper_turn=hi,lambda1_real=real(eig[1]),lambda1_imag=imag(eig[1]),
            lambda2_real=real(eig[2]),lambda2_imag=imag(eig[2]),alpha=maximum(real.(eig)),
            residual=maximum(abs.(r["balance_residual"])),sinks=length(sinks),higher_I_sinks=length(comparisons(k)),
            rest_candidates=length(rest),active_candidates=length(active),seizure_candidates=length(seizures),herald_candidates=length(herald),
            reasons=join(reasons,"|"),descriptor=arm*"/"*side))
    end
    rows,assigned
end

"""Reproduce the conservative reference-label fallback and keep its origin explicit."""
function destination_label(local_search,local_roles,index;reference_search=nothing,reference_roles=Dict{String,Int}())
    local_role=something(findfirst(==(index),local_roles),"unassigned")
    local_role!="unassigned" && return (;label=local_role,origin="local",reference_root=0)
    if reference_search!==nothing
        matches=M.correspondence(reference_search,local_search;atol=.05)
        old=findfirst(==(index),matches)
        if old!==nothing
            role=something(findfirst(==(old),reference_roles),"unassigned")
            role!="unassigned" && return (;label=role,origin="reference_coordinate_match",reference_root=old)
        end
    end
    (;label="equilibrium_$index",origin="numbered_local_root",reference_root=0)
end
end

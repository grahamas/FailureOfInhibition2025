# Private reproducibility helpers. Coordinates describe zero-drive systems only.
if !isdefined(@__MODULE__, :ATLAS_LIGHTWEIGHT_COORDINATOR) || !ATLAS_LIGHTWEIGHT_COORDINATOR
    using FailureOfInhibition2025
end
using LinearAlgebra
const ATLAS_COORDINATES = ("A", "B", "C", "D", "a", "b", "h", "r")

function validate_parameters(p)
    all(k -> haskey(p,k) && p[k] isa Real && !(p[k] isa Bool) && isfinite(p[k]), ATLAS_COORDINATES) || throw(ArgumentError("eight finite reduced coordinates required"))
    all(k -> p[k]>=0, ATLAS_COORDINATES[1:4]) && p["h"]>0 && p["r"]>0 || throw(ArgumentError("invalid reduced parameter domain"))
    p
end
function reduced_model(p)
    validate_parameters(p)
    PointModelParameters(excitatory=PopulationParameters(timescale=1.0,response=LogisticResponse(slope=1.0,threshold=p["a"])), inhibitory=PopulationParameters(timescale=p["r"],response=FailureOfInhibitionResponse(slope=1.0,onset_threshold=p["b"],failure_threshold=p["b"]+p["h"])),coupling=PointCoupling(e_to_e=p["A"],i_to_e=p["B"],e_to_i=p["C"],i_to_i=p["D"]))
end
function physical_model(p;slope_e=1.0,slope_i=1.0,tau_e=1.0)
    validate_parameters(p)
    all(x->isfinite(x)&&x>0,(slope_e,slope_i,tau_e)) || throw(ArgumentError("positive physical scales required"))
    PointModelParameters(excitatory=PopulationParameters(timescale=tau_e,response=LogisticResponse(slope=slope_e,threshold=p["a"]/slope_e)),inhibitory=PopulationParameters(timescale=p["r"]*tau_e,response=FailureOfInhibitionResponse(slope=slope_i,onset_threshold=p["b"]/slope_i,failure_threshold=(p["b"]+p["h"])/slope_i)),coupling=PointCoupling(e_to_e=p["A"]/slope_e,i_to_e=p["B"]/slope_e,e_to_i=p["C"]/slope_i,i_to_i=p["D"]/slope_i))
end
atlas_logistic(x) = x>=0 ? inv(1+exp(-x)) : exp(x)/(1+exp(x))
function occupancy(u,v,h)
    h isa Real && isfinite(h) && h>0 || throw(ArgumentError("positive finite inhibitory width required"))
    fe=atlas_logistic(u)
    fi=atlas_logistic(v)*atlas_logistic(h-v)*(-expm1(-h))
    [fe/(1+fe),fi/(1+fi)]
end
function equilibrium_chart(A,B,C,D,u,v,h;r=1.0)
    x=occupancy(u,v,h)
    p=Dict{String,Any}("A"=>A,"B"=>B,"C"=>C,"D"=>D,"a"=>A*x[1]-B*x[2]-u,"b"=>C*x[1]-D*x[2]-v,"h"=>h,"r"=>r)
    validate_parameters(p)
    Dict{String,Any}("params"=>p,"state"=>x,"u"=>u,"v"=>v)
end
function balance_value(p,x)
    out=zeros(2);point_balance!(out,x,reduced_model(p),0.0);out
end
function balance_jacobian(p,x)
    out=zeros(2,2);point_balance_jacobian!(out,x,reduced_model(p),0.0);out
end
function positive_quadratic_roots(a,b,c;tol=1e-12)
    aa,bb,cc=BigFloat.((a,b,c))
    roots=if iszero(aa)
        iszero(bb) ? BigFloat[] : [-cc/bb]
    else
        discriminant=bb*bb-4aa*cc
        discriminant<0 && return Float64[]
        q=-big"0.5"*(bb+copysign(sqrt(discriminant),bb))
        iszero(q) ? [-bb/(2aa)] : [q/aa,cc/q]
    end
    positive=filter(x->isfinite(x)&&x>0,roots)
    all(x->isfinite(Float64(x))&&Float64(x)>0,positive) || throw(ArgumentError("positive boundary outside Float64 range; unresolved numerical tail"))
    sort!(unique(Float64.(positive)))
end
function ratio_partition(K;tol=1e-10)
    size(K)==(2,2) && all(isfinite,K) || throw(ArgumentError("finite 2x2 K required"))
    determinant=det(K);t1=K[1,1];t0=K[2,2]
    hopf=iszero(t1) ? -1.0 : -t0/t1
    bt1,bt0,bd=BigFloat.((t1,t0,determinant))
    cuts=positive_quadratic_roots(bt1*bt1,2bt1*bt0-4bd,bt0*bt0)
    hopf>0 && isfinite(hopf) && push!(cuts,hopf)
    cuts=sort!(unique(cuts));ends=vcat(0.0,cuts,Inf)
    intervals=Dict{String,Any}[]
    for i in 1:length(ends)-1
        l,u=ends[i:i+1];r=l==0 ? (isfinite(u) ? u/2 : 1.0) : (isfinite(u) ? sqrt(l)*sqrt(u) : 2l)
        isfinite(r) && l<r<u || throw(ArgumentError("interval has no representable probe; unresolved numerical tail"))
        tr=BigFloat(r)*bt1+bt0;d=bd;dis=tr*tr-4BigFloat(r)*d
        stability=d < -tol ? "saddle" : abs(d)<=tol || abs(tr)<=tol*max(abs(r*t1),abs(t0),1.0) ? "unresolved" : tr<0 ? "attracting" : "repelling"
        push!(intervals,Dict("lower"=>l,"upper"=>u,"probe"=>r,"stability"=>stability,"spectrum"=>abs(dis)<=tol ? "unresolved" : dis<0 ? "complex" : "real"))
    end
    Dict("intervals"=>intervals,"boundaries"=>cuts,"hopf_candidate"=>hopf>0 && determinant>tol,"hopf_ratio"=>hopf>0 ? hopf : 0.0,"determinant"=>determinant,"completeness"=>"analytic linear classification only")
end
function radical_inverse(n,b)
    n>0 || throw(ArgumentError("positive sequence index required"));x=0.0;f=1.0/b
    while n>0
        x+=f*(n%b);n÷=b;f/=b
    end
    x
end
function in_window(p,k;chart="parameter")
    k<1 && return false
    R=exp2(k)
    all(j->0<=p[j]<=R,1:4) && abs(p[5])<=R && abs(p[6])<=R && inv(R)<=p[7]<=R
end
function shell_sample(k,index;chart="parameter",mask=0)
    k>=1 && k<=1023 && index>0 && 0<=mask<=15 || throw(ArgumentError("invalid shell/index/mask"))
    chart in ("parameter","equilibrium") || throw(ArgumentError("unknown chart"))
    R=exp2(k);accepted=0;primes=(2,3,5,7,11,13,17)
    # index is the accepted ordinal in a separate shell/chart/stratum stream.
    n=0
    while n<typemax(Int)
        n+=1
        q=[radical_inverse(n,b) for b in primes]
        vals=vcat(R.*q[1:4], R.*(2 .*q[5:6].-1), exp2((2q[7]-1)*k))
        for j in 1:4
            mask & (1<<(j-1)) !=0 && (vals[j]=0.0)
        end
        if k==1 || !in_window(vals,k-1)
            accepted+=1
            accepted==index || continue
            p=if chart=="parameter"
                Dict{String,Any}(zip(ATLAS_COORDINATES,vcat(vals,1.0)))
            else
                equilibrium_chart(vals[1:7]...)["params"]
            end
            return Dict{String,Any}("params"=>p,"chart"=>chart,"shell"=>k,"sequence_index"=>index,"raw_sequence_index"=>n,"boundary_mask"=>mask,"chart_coordinates"=>vals,"coverage"=>"sampled_point")
        end
    end
    error("shell rejection limit; unresolved sampling job")
end
function discovery_job(index::Int,chart::String,boundary::Bool)
    index>0 || throw(ArgumentError("positive index required"))
    k=1+trailing_zeros(index);ordinal=((index>>(k-1))+1)÷2
    mask=boundary ? 1+mod(ordinal-1,15) : 0
    stream_index=boundary ? 1+div(ordinal-1,15) : ordinal
    merge(shell_sample(k,stream_index;chart=chart,mask=mask),Dict("kind"=>"discovery","queue"=>"discovery"))
end
function analytic_seed()
    h=log(9.0);base=equilibrium_chart(648/49,8.0,8.0,0.0,-log(2.0),0.0,h;r=1029/640)
    merge(base,Dict("kind"=>"nexus","event_type"=>"bt","queue"=>"events","origin"=>"analytic_double_zero","coverage"=>"local_seed_only"))
end
function initial_jobs(config)
    p=analytic_seed()["params"]
    cusp=merge(equilibrium_chart(8.,0.,0.,0.,-log(2.),0.,log(9.)),Dict("kind"=>"nexus","event_type"=>"cusp","queue"=>"events","origin"=>"analytic_cusp","coverage"=>"local_seed_only"))
    jobs=Dict{String,Any}[analytic_seed(),cusp,Dict{String,Any}("kind"=>"certification","queue"=>"certification","params"=>discovery_job(1,"parameter",false)["params"])]
    get(config,"mode","benchmark")=="benchmark" && push!(jobs,Dict{String,Any}("kind"=>"periodic","queue"=>"cycles","fixture"=>true,"origin"=>"benchmark_fixture"))
    jobs
end

"""Algebraic fold chart; a singular denominator is an unresolved chart, not an exclusion."""
function fold_chart(B,C,D,u,v,h;r=1.0)
    x=occupancy(u,v,h);fe=atlas_logistic(u)
    ge=fe*(1-fe)/(1+fe)^2
    response_i=FailureOfInhibitionResponse(slope=1.0,onset_threshold=0.0,failure_threshold=h)
    fi=response(response_i,v);gi=response_derivative(response_i,v)/(1+fi)^2
    denominator=1+D*gi
    abs(denominator)>1e-12 && ge>0 || throw(ArgumentError("singular fold chart; retain augmented determinant obligation"))
    A=inv(ge)+B*C*gi/denominator
    result=equilibrium_chart(A,B,C,D,u,v,h;r=r)
    K=balance_jacobian(result["params"],x)
    result["fold_residual"]=abs(det(K))
    result
end

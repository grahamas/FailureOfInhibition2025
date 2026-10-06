"""Two-input equilibrium charts and numerical study policies; no biological certification."""
module InputResponseModels
using FailureOfInhibition2025, LinearAlgebra

function number(x,name;positive=false)
    x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)) &&
        (positive ? x>0 : x>=0) || throw(ArgumentError("invalid $name"))
    Float64(x)
end

function model(p,be=0.,bi=0.;control=false)
    for key in ("e_to_e","i_to_e","e_to_i","i_to_i")
        number(p[key],key)
    end
    number(be,"B_E");number(bi,"B_I")
    number(p["tau_ratio"],"tau_ratio";positive=true)
    pair=matched_point_models(
        excitatory=PopulationParameters(timescale=get(p,"tau_e",7.8),
            response=LogisticResponse(slope=get(p,"slope_e",5.),threshold=get(p,"theta_e",1.5))),
        inhibitory_control=PopulationParameters(timescale=get(p,"tau_e",7.8)*p["tau_ratio"],
            response=LogisticResponse(slope=get(p,"slope_i",5.),threshold=get(p,"theta_on",4.))),
        failure_threshold=p["theta_off"],
        coupling=PointCoupling(; (Symbol(k)=>p[k] for k in ("e_to_e","i_to_e","e_to_i","i_to_i"))...),
        drive=PiecewiseConstantDrive(baseline=(be,bi),pulses=(),interpretation=AfferentExcitation))
    control ? pair.control : pair.failure_of_inhibition
end

occupancy(f,u)=response(f,u)/(1+response(f,u))
occupancy_derivative(f,u)=response_derivative(f,u)/(1+response(f,u))^2

"""Map total population inputs to an exact equilibrium and its external inputs.

External inputs returned here may be negative; the study clips to its declared
nonnegative domain. The chart is exact, its numerical enumeration is not exhaustive.
"""
function chart(m,u,v)
    fe,fi=m.excitatory.response,m.inhibitory.response
    e,i=occupancy(fe,u),occupancy(fi,v)
    ep,ip=occupancy_derivative(fe,u),occupancy_derivative(fi,v)
    c=m.coupling
    inputs=[u-c.e_to_e*e+c.i_to_e*i,v-c.e_to_i*e+c.i_to_i*i]
    derivative=[1-c.e_to_e*ep c.i_to_e*ip; -c.e_to_i*ep 1+c.i_to_i*ip]
    state=[e,i]
    # Evaluate response derivatives at u,v directly to avoid reconstructing
    # saturated inputs from rounded state coordinates.
    f,g=response(fe,u),response(fi,v)
    fp,gp=response_derivative(fe,u),response_derivative(fi,v)
    balance_jacobian=[-1-f+(1-e)*fp*c.e_to_e -(1-e)*fp*c.i_to_e;
        (1-i)*gp*c.e_to_i -1-g-(1-i)*gp*c.i_to_i]
    jacobian=balance_jacobian./[m.excitatory.timescale,m.inhibitory.timescale]
    (;state,inputs,derivative,jacobian,balance_jacobian)
end

function state_bounds(m)
    f=m.inhibitory.response
    peak=f isa FailureOfInhibitionResponse ? response(f,(f.onset_threshold+f.failure_threshold)/2) : 1.
    (0.5,peak/(1+peak))
end

"""Expand axes until uniform response-tail bounds hold on the invariant box.

These bounds concern response values, not attractor uniqueness or pulse outcomes.
"""
function input_bounds(m;initial=16.,epsilon=1e-6,max_expansions=12)
    number(initial,"initial bound";positive=true)
    number(epsilon,"tail tolerance";positive=true)
    epsilon<0.5 || throw(ArgumentError("tail tolerance must be below 0.5"))
    max_expansions isa Integer && !(max_expansions isa Bool) && max_expansions>=0 ||
        throw(ArgumentError("invalid expansion budget"))
    _,imax=state_bounds(m);c=m.coupling;f=m.inhibitory.response
    f isa FailureOfInhibitionResponse || throw(ArgumentError("failure-tail bound requires FoI response"))
    be=Float64(initial);bi=Float64(initial);ne=ni=0
    eok()=response(m.excitatory.response,be-c.i_to_e*imax)>=1-epsilon
    iok()=bi-c.i_to_i*imax>=(f.onset_threshold+f.failure_threshold)/2 &&
        response(f,bi-c.i_to_i*imax)<=epsilon
    while !eok() && ne<max_expansions;be*=2;ne+=1;end
    while !iok() && ni<max_expansions;bi*=2;ni+=1;end
    (;B_E=be,B_I=bi,E_tail_resolved=eok(),I_tail_resolved=iok(),
        E_tail_value=response(m.excitatory.response,be-c.i_to_e*imax),
        I_tail_value=response(f,bi-c.i_to_i*imax),E_expansions=ne,I_expansions=ni,epsilon)
end

function total_input_bounds(m,bounds)
    emax,imax=state_bounds(m);c=m.coupling
    ((-c.i_to_e*imax,bounds.B_E+c.e_to_e*emax),
     (-c.i_to_i*imax,bounds.B_I+c.e_to_i*emax))
end

"""Local input susceptibility; singular/ill-conditioned cases stay unresolved."""
function susceptibility(m,state;condition_limit=1e10)
    j=zeros(2,2);point_jacobian!(j,state,m,0.)
    e,i=state;c=m.coupling;be,bi=m.drive.baseline
    d=Diagonal([(1-e)*response_derivative(m.excitatory.response,be+c.e_to_e*e-c.i_to_e*i)/m.excitatory.timescale,
        (1-i)*response_derivative(m.inhibitory.response,bi+c.e_to_i*e-c.i_to_i*i)/m.inhibitory.timescale])
    condition=cond(j)
    condition<=condition_limit ? (;status="resolved",gain=-(j\Matrix(d)),condition) :
        (;status="singular_or_ill_conditioned",gain=fill(NaN,2,2),condition)
end

"""Provisional shape roles, using total I input including B_I.

Lower and intermediate E-nullcline arms are defined by turning points of the
logistic E-nullcline. High-state names require relative I separation; neither
the sole sink nor the lowest-I sink is automatically a seizure.
"""
function roles(m,search)
    sinks=[k for (k,x) in enumerate(search.equilibria) if x.stability.classification==Attracting && !x.near_singular]
    result=Dict{String,Int}();a=m.coupling.e_to_e;s=m.excitatory.response.slope
    a*s>8 || return result
    lower=(1-sqrt(1-8/(a*s)))/4;upper=(1+sqrt(1-8/(a*s)))/4
    st(k)=search.equilibria[k].state
    vi(k)=m.drive.baseline[2]+m.coupling.e_to_i*st(k)[1]-m.coupling.i_to_i*st(k)[2]
    rest=filter(k->st(k)[1]<lower,sinks)
    active=filter(k->lower<st(k)[1]<upper && response_derivative(m.inhibitory.response,vi(k))>0,sinks)
    length(rest)==1 && (result["rest"]=only(rest))
    length(active)==1 && (result["active"]=only(active))
    high=filter(k->st(k)[1]>upper,sinks)
    seizure=filter(k->response_derivative(m.inhibitory.response,vi(k))<0 &&
        any(h->st(h)[2]>st(k)[2]+1e-6 && st(h)[1]<=st(k)[1]+1e-6,sinks),high)
    length(seizure)==1 && (result["seizure"]=only(seizure))
    herald=filter(k->haskey(result,"seizure") && k!=result["seizure"] &&
        st(k)[2]>st(result["seizure"])[2]+1e-6,high)
    length(herald)==1 && (result["herald"]=only(herald))
    result
end

"""Follow a unique nearby root, rejecting ties and many-to-one matches."""
function correspondence(old,new;atol=0.05)
    matches=Dict{Int,Int}()
    for (i,r) in enumerate(old.equilibria)
        candidates=sort([(maximum(abs.(r.state-x.state)),j) for (j,x) in enumerate(new.equilibria)])
        isempty(candidates) && continue
        candidates[1][1]<=atol || continue
        length(candidates)>1 && candidates[2][1]-candidates[1][1]<1e-6 && continue
        matches[i]=candidates[1][2]
    end
    collisions=[j for j in values(matches) if count(==(j),values(matches))>1]
    filter!(p->!(p.second in collisions),matches)
end

"""Exact scalar seed formulas for det(J)=0 or trace(J)=0 in the input chart.

They locate numerical critical-set candidates, not classified bifurcations.
"""
function critical_states(m,v,kind)
    c=m.coupling;s=m.excitatory.response.slope;a=c.e_to_e
    fi=m.inhibitory.response;i=occupancy(fi,v);ip=occupancy_derivative(fi,v)
    es=Float64[]
    if kind=="fold"
        denominator=a*(1+c.i_to_i*ip)-c.i_to_e*c.e_to_i*ip
        if abs(denominator)>1e-14
            k=(1+c.i_to_i*ip)/denominator
            if 0<k<=s/8
                append!(es,[(1-sqrt(max(0.,1-8*k/s)))/4,(1+sqrt(max(0.,1-8*k/s)))/4])
            end
        end
    elseif kind=="hopf"
        ratio=m.inhibitory.timescale/m.excitatory.timescale
        q=(1+c.i_to_i*ip)/((1-i)*ratio)
        A=2a*s;B=-(a*s+q);C=1+q
        if A==0
            B!=0 && push!(es,-C/B)
        elseif B^2-4A*C>=0
            append!(es,[(-B-sqrt(B^2-4A*C))/(2A),(-B+sqrt(B^2-4A*C))/(2A)])
        end
    else
        throw(ArgumentError("unknown critical set"))
    end
    rows=NamedTuple[]
    for (sheet,e) in enumerate(es)
        0<e<0.5 || continue
        u=m.excitatory.response.threshold+log(e/(1-2e))/s
        x=chart(m,u,v)
        kind=="hopf" && det(x.jacobian)<=1e-12 && continue
        push!(rows,(;kind,sheet,u,v,E=x.state[1],I=x.state[2],B_E=x.inputs[1],B_I=x.inputs[2],
            trace=tr(x.jacobian),determinant=det(x.jacobian)))
    end
    rows
end

"""Adaptive line sampling with interior probes; no monotone-success assumption."""
function sample_line(f,lo,hi;step=1.,width=.01,key=identity,max_evaluations=10000)
    lo<=hi || throw(ArgumentError("unordered line"))
    number(step,"step";positive=true);number(width,"width";positive=true)
    cache=Dict{Float64,Any}();unresolved=Tuple{Float64,Float64}[]
    at(x)=get!(() -> f(x),cache,Float64(x))
    function visit(a,b)
        if length(cache)+3>max_evaluations
            push!(unresolved,(a,b));return
        end
        m=(a+b)/2;ka,kb,km=key(at(a)),key(at(b)),key(at(m))
        if b-a>width && (ka!=kb || ka!=km || occursin("unresolved",string(ka)))
            visit(a,m);visit(m,b)
        end
    end
    at(lo)
    axis=unique(vcat(collect(lo:step:hi),[Float64(hi)]))
    for (a,b) in zip(axis,axis[2:end]);visit(a,b);end
    (;cache,unresolved)
end

"""Sample a rectangle, refining around known observations as well as new probes."""
function sample_rectangle(f,xs,ys;width=.01,key=identity,max_evaluations=20000,known_samples=())
    cache=Dict{Tuple{Float64,Float64},Any}();leaves=NamedTuple[]
    known_keys=[(Float64(x),Float64(y),key(value)) for ((x,y),value) in known_samples]
    at(x,y)=get!(() -> f(x,y),cache,(Float64(x),Float64(y)))
    queue=[(Float64(x0),Float64(x1),Float64(y0),Float64(y1))
        for (x0,x1) in zip(xs,xs[2:end]) for (y0,y1) in zip(ys,ys[2:end])]
    index=1
    while index<=length(queue)
        x0,x1,y0,y1=queue[index];index+=1
        if length(cache)+5>max_evaluations
            push!(leaves,(;x0,x1,y0,y1,status="budget_unresolved"));continue
        end
        xm,ym=(x0+x1)/2,(y0+y1)/2
        ks=[key(at(x,y)) for (x,y) in ((x0,y0),(x1,y0),(x0,y1),(x1,y1),(xm,ym))]
        append!(ks,(known_key for (x,y,known_key) in known_keys if x0<=x<=x1 && y0<=y<=y1))
        changed=length(unique(ks))>1 || any(k->occursin("unresolved",string(k)),ks)
        if changed && max(x1-x0,y1-y0)>width
            if x1-x0>=y1-y0
                push!(queue,(x0,xm,y0,y1),(xm,x1,y0,y1))
            else
                push!(queue,(x0,x1,y0,ym),(x0,x1,ym,y1))
            end
        else
            push!(leaves,(;x0,x1,y0,y1,status=changed ? "boundary_bracket" : "sampled_homogeneous"))
        end
    end
    # A later neighboring cell may sample the interior of an earlier cell's
    # edge. Reconcile labels against every observation before callers use
    # homogeneous cells to establish connectivity.
    observed=Dict((x,y)=>value for (x,y,value) in known_keys)
    for (point,value) in cache
        observed[point]=key(value)
    end
    observed_points=sort!(collect(keys(observed)))
    observed_x=first.(observed_points)
    for (i,cell) in enumerate(leaves)
        cell.status=="sampled_homogeneous" || continue
        reference=observed[(cell.x0,cell.y0)]
        lo=searchsortedfirst(observed_x,cell.x0)
        hi=searchsortedlast(observed_x,cell.x1)
        if any(point->cell.y0<=point[2]<=cell.y1 && observed[point]!=reference,
                @view observed_points[lo:hi])
            leaves[i]=merge(cell,(status="boundary_bracket",))
        end
    end
    (;cache,leaves)
end

function nondominated(rows)
    [r for r in rows if !any(q->q.E_withdrawal<=r.E_withdrawal && q.I_stimulation<=r.I_stimulation &&
        (q.E_withdrawal<r.E_withdrawal || q.I_stimulation<r.I_stimulation),rows)]
end
end

# Scientific jobs retain numerical evidence and never promote a finite search.
function atlas_attempt(a)
    Dict("seed"=>a.seed,"candidate"=>a.candidate,"residual"=>a.residual_norm,"validation"=>string(a.validation),"solver_status"=>string(a.solver_status),"reasons"=>string.(a.reasons))
end
function atlas_search(p,config;extra_seeds=Vector{Float64}[])
    n=Int(get(config,"seed_grid",5));2<=n<=31 || throw(ArgumentError("seed_grid must be 2:31"))
    seeds=[[e,i] for e in range(0,0.5;length=n) for i in range(0,0.5;length=n)]
    find_equilibria(reduced_model(p);seeds=vcat(extra_seeds,seeds),options=EquilibriumOptions(maxiters=Int(get(config,"max_corrector_iters",100))))
end
function atlas_point(p,config;id="",extra_seeds=Vector{Float64}[])
    found=atlas_search(p,config;extra_seeds=extra_seeds)
    roots=[Dict{String,Any}("state"=>x.state,"residual"=>maximum(abs,x.balance_residual),"stability"=>string(x.stability.classification),"spectrum"=>string(x.stability.geometry),"eigenvalue_real"=>real.(x.stability.eigenvalues),"eigenvalue_imag"=>imag.(x.stability.eigenvalues),"spectral_abscissa"=>x.stability.spectral_abscissa,"trace"=>x.stability.trace,"determinant"=>x.stability.determinant,"ratio_partition"=>ratio_partition(x.balance_jacobian)) for x in found.equilibria]
    for root in roots
        x=root["state"];v=p["C"]*x[1]-p["D"]*x[2]-p["b"]
        root["inhibitory_response_branch"]=abs(v-p["h"]/2)<=1e-8 ? "unresolved" : v<p["h"]/2 ? "rising" : "falling"
        e=x[1];ae=p["A"];derivative=1-ae*e*(1-2e)
        root["excitatory_chart_branch"]=abs(derivative)<1e-8 ? "unresolved" : ae<=8 ? "single" : e<(1-sqrt(1-8/ae))/4 ? "left" : e>(1+sqrt(1-8/ae))/4 ? "right" : "middle"
        resp=FailureOfInhibitionResponse(slope=1.,onset_threshold=0.,failure_threshold=p["h"])
        fi=response(resp,v);gi=response_derivative(resp,v)/(1+fi)^2
        hh=p["h"];ee=exp(-hh);tt=(1+sqrt(1+8ee))/2
        crit=(tt+1)^2/((-expm1(-hh)/2)*sqrt(tt*tt-ee))
        minimum_position=hh>100 ? hh+log(2tt) : hh/2+acosh(exp(hh/2)*tt)
        root["inhibitory_self_chart_branch"]=abs(1+p["D"]*gi)<1e-8 ? "unresolved" : p["D"]<=crit ? "single" : 1+p["D"]*gi<0 ? "middle" : v<minimum_position ? "left" : "right"
    end
    # State-order-independent inventory; identical signatures do not prove connectivity.
    ie=exp(-p["h"]);t=(1+sqrt(1+8ie))/2
    dcrit=(t+1)^2/((-expm1(-p["h"])/2)*sqrt(t*t-ie))
    eturn=abs(p["A"]-8)<1e-8 ? "unresolved" : p["A"]>8 ? "two" : "none"
    iturn=abs(p["D"]-dcrit)<1e-8 ? "unresolved" : p["D"]>dcrit ? "two" : "none"
    geometry=Dict("excitatory_global_chart_turns"=>eturn,"inhibitory_global_chart_turns"=>iturn,"inhibitory_turn_threshold"=>dcrit,"coupling_zero_mask"=>sum(p[k]==0 ? 1<<(i-1) : 0 for (i,k) in enumerate(("A","B","C","D"))))
    signature="Eturn:$eturn/Iturn:$iturn/zero:$(geometry["coupling_zero_mask"])/"*join(sort([r["stability"]*":"*r["spectrum"]*":"*r["inhibitory_response_branch"]*":"*r["excitatory_chart_branch"]*":"*r["inhibitory_self_chart_branch"] for r in roots]),"|")
    Dict{String,Any}("id"=>id,"parameters"=>p,"roots"=>roots,"signature"=>signature,"geometry"=>geometry,"signature_scope"=>"sampled_root_inventory_and_nullcline_chart_branches","attempts"=>atlas_attempt.(found.attempts),"completeness"=>"CompletenessNotCertified","coverage"=>"sampled_point")
end
function atlas_replace(p,name,value)
    q=copy(p);q[name]=value;validate_parameters(q);q
end
function localize_event(p,x,name,kind;iterations=20,tolerance=1e-9)
    z=vcat(x,p[name]);history=Dict{String,Any}[]
    function residual(z)
        q=atlas_replace(p,name,z[3]);K=balance_jacobian(q,z[1:2]);J=Diagonal([1.0,1/q["r"]])*K
        vcat(balance_value(q,z[1:2]),kind=="fold" ? det(K) : tr(J))
    end
    for iteration in 1:iterations
        f=try residual(z) catch; return Dict("status"=>"unresolved","reason"=>"inadmissible_correction","attempts"=>history) end
        push!(history,Dict("state_parameter"=>copy(z),"residual"=>maximum(abs,f)))
        if all(isfinite,f) && maximum(abs,f)<=tolerance && all(v->0<=v<=0.5,z[1:2])
            q=atlas_replace(p,name,z[3]);K=balance_jacobian(q,z[1:2]);J=Diagonal([1.0,1/q["r"]])*K
            kind=="hopf" && det(J)<=tolerance && return Dict("status"=>"unresolved","reason"=>"nonpositive_hopf_determinant","attempts"=>history)
            return Dict("status"=>"completed","params"=>q,"state"=>z[1:2],"kind"=>kind,"residual"=>maximum(abs,f),"attempts"=>history)
        end
        M=zeros(3,3)
        for j in 1:3
            h=1e-5*max(1,abs(z[j]));zz=copy(z);zz[j]+=h
            M[:,j]=(residual(zz)-f)/h
        end
        step=try M\f catch; break end
        all(isfinite,step) || break
        accepted=false
        for power in 0:12
            zz=z-step/2.0^power
            all(v->0<=v<=0.5,zz[1:2]) || continue
            ff=try residual(zz) catch; continue end
            if all(isfinite,ff) && norm(ff)<norm(f)
                z=zz;accepted=true;break
            end
        end
        accepted || break
    end
    Dict("status"=>"unresolved","reason"=>"event_corrector_limit","attempts"=>history)
end
function execute_discovery(job,config)
    p=job["params"];extra=Vector{Float64}[]
    if get(job,"chart","")=="equilibrium"
        c=job["chart_coordinates"];push!(extra,occupancy(c[5],c[6],c[7]))
    end
    point=atlas_point(p,config;id=get(job,"id","discovery"),extra_seeds=extra);follow=Dict{String,Any}[]
    for root in point["roots"]
        push!(follow,Dict("kind"=>"continuation","queue"=>"events","params"=>p,"state"=>root["state"],"coordinate"=>"A","origin_point"=>point["id"]))
        rp=root["ratio_partition"]
        if rp["hopf_candidate"]
            q=atlas_replace(p,"r",rp["hopf_ratio"])
            push!(follow,Dict("kind"=>"periodic","queue"=>"cycles","params"=>q,"state"=>root["state"],"origin"=>"analytic_trace_zero"))
        end
    end
    seed_events=Dict{String,Any}[]
    if get(job,"chart","")=="equilibrium"
        c=job["chart_coordinates"]
        candidate=try
            fold=fold_chart(c[2],c[3],c[4],c[5],c[6],c[7])
            Dict{String,Any}("status"=>"completed","params"=>fold["params"],"state"=>fold["state"],"residual"=>fold["fold_residual"],"origin"=>"algebraic_fold_chart")
        catch err
            # Singular/invalid algebraic charts leave a real augmented-solver obligation.
            augmented=localize_event(p,extra[1],"A","fold")
            augmented["algebraic_chart_failure"]=sprint(showerror,err)
            augmented["origin"]="augmented_fold_correction"
            augmented
        end
        push!(seed_events,candidate)
        if candidate["status"]=="completed"
            q=candidate["params"];x=candidate["state"];K=balance_jacobian(q,x)
            push!(follow,Dict("kind"=>"nexus","queue"=>"events","params"=>q,"state"=>x,"event_type"=>"cusp","origin"=>candidate["origin"]))
            if K[1,1]!=0
                r=-K[2,2]/K[1,1]
                isfinite(r) && r>0 && push!(follow,Dict("kind"=>"nexus","queue"=>"events","params"=>atlas_replace(q,"r",r),"state"=>x,"event_type"=>"bt","origin"=>"algebraic_fold_trace_zero"))
            end
        end
    end
    Dict("status"=>isempty(point["roots"]) ? "unresolved" : "completed","evidence"=>Dict("points"=>[point],"events"=>seed_events,"chart"=>get(job,"chart","parameter"),"shell"=>get(job,"shell",0)),"followups"=>follow)
end
function execute_continuation(job,config)
    p=job["params"];name=get(job,"coordinate","A");center=p[name];radius=max(1.0,abs(center))*0.5
    lower=name in ("a","b") ? center-radius : max(name in ("h","r") ? 1e-6 : 0.0,center-radius)
    upper=center+radius
    factory=t->reduced_model(atlas_replace(p,name,t))
    curve=continue_equilibria(factory,job["state"],center;parameter_bounds=(lower,upper),options=ContinuationOptions(max_steps=Int(get(config,"continuation_steps",12))))
    points=Dict{String,Any}[];paths=Dict{String,Any}[];events=Dict{String,Any}[];follow=Dict{String,Any}[]
    for branch in (curve.negative,curve.positive)
        ids=String[]
        for (i,pt) in enumerate(branch.points)
            q=atlas_replace(p,name,pt.parameter);id="$(get(job,"id","branch"))-$(branch.direction)-$i"
            point=atlas_point(q,config;id=id,extra_seeds=[pt.state]);push!(points,point);push!(ids,id)
            if i>1
                previous=first(filter(z->z["id"]==ids[end-1],points))
                samples=Dict{String,Any}[]
                for (j,t) in enumerate((0.25,0.5,0.75))
                    qmid=Dict{String,Any}(k=>(1-t)*previous["parameters"][k]+t*q[k] for k in ATLAS_COORDINATES)
                    push!(samples,atlas_point(qmid,config;id="$id-mid$j"))
                end
                same=all(z->z["signature"]==point["signature"] && !isempty(z["roots"]) && !occursin("unresolved",lowercase(z["signature"])),vcat([previous,point],samples))
                append!(points,samples)
                push!(paths,Dict("from"=>previous["id"],"to"=>id,"signature_preserved"=>same,"validated"=>true,"status"=>"completed","coverage"=>"five_sample_full_inventory_path","sample_ids"=>[previous["id"],getindex.(samples,"id")...,id],"continuous_coverage_certified"=>false))
            end
        end
        for candidate in branch.candidates
            pt=branch.points[candidate.first_point];q=atlas_replace(p,name,pt.parameter)
            event=localize_event(q,pt.state,name,string(candidate.kind));push!(events,event)
            if event["status"]=="completed"
                push!(follow,Dict("kind"=>"nexus","queue"=>"events","params"=>event["params"],"state"=>event["state"],"event_type"=>candidate.kind==:fold ? "cusp" : "bt","origin"=>"localized_$(candidate.kind)"))
                candidate.kind==:hopf && push!(follow,Dict("kind"=>"periodic","queue"=>"cycles","params"=>event["params"],"state"=>event["state"]))
            end
        end
        push!(events,Dict("kind"=>"branch_termination","direction"=>branch.direction,"termination"=>string(branch.termination),"attempts"=>[Dict("candidate"=>a.candidate,"status"=>string(a.status),"accepted"=>a.accepted,"iterations"=>a.iterations) for a in branch.attempts]))
    end
    Dict("status"=>"completed","evidence"=>Dict("points"=>points,"paths"=>paths,"events"=>events,"completeness"=>"CompletenessNotCertified"),"followups"=>follow)
end
function execute_certification(job,config)
    cert=certify_equilibria(reduced_model(job["params"]);options=CertificationOptions(maximum_boxes=Int(get(config,"certification_boxes",500)),maximum_evaluations=Int(get(config,"certification_evaluations",5000)),time_limit=Float64(get(config,"certification_seconds",10.0))))
    record=Dict{String,Any}("parameters"=>job["params"],"status"=>string(cert.status),"processed_boxes"=>cert.processed_boxes,"evaluations"=>cert.evaluations,"domain"=>[0.0,0.5],"scope"=>"fixed_parameter_equilibria_only","root_enclosures"=>[Dict("lower"=>string.(r.enclosure.lower),"upper"=>string.(r.enclosure.upper)) for r in cert.roots],"proof_nodes"=>[Dict("lower"=>string.(n.box.lower),"upper"=>string.(n.box.upper),"kind"=>string(n.kind),"reason"=>string(n.reason)) for n in cert.nodes])
    cert.equilibrium_count!==nothing && (record["equilibrium_count"]=cert.equilibrium_count)
    Dict("status"=>cert.status==:certified ? "completed" : "unresolved","evidence"=>Dict("certifications"=>[record]),"followups"=>Dict{String,Any}[])
end
function atlas_normalize_job(job)
    result=copy(job)
    if haskey(result,"parameters")
        haskey(result,"params") && result["params"]!=result["parameters"] && throw(ArgumentError("conflicting job parameters aliases"))
        get!(result,"params",result["parameters"])
    end
    get!(result,"queue",result["kind"]=="periodic" ? "cycles" : result["kind"]=="certification" ? "certification" : result["kind"]=="discovery" ? "discovery" : "events")
    result
end
function atlas_anchor_points(raw,kind,job,config)
    witnesses=Dict{String,Any}[]
    if kind=="periodic" && get(raw,"validated",false) && haskey(raw,"parameters")
        extra=haskey(raw,"hopf_state") ? [raw["hopf_state"]] : Vector{Float64}[]
        push!(witnesses,atlas_point(raw["parameters"],config;id="$(get(job,"id","cycle"))-anchor",extra_seeds=extra))
    elseif kind=="section"
        for (i,sample) in enumerate(get(raw,"section_samples",[]))
            haskey(sample,"parameters") || continue
            extra=[root["state"] for root in get(sample,"equilibria",[]) if haskey(root,"state")]
            point=atlas_point(sample["parameters"],config;id="$(get(job,"id","section"))-side$i",extra_seeds=extra)
            isempty(point["roots"]) || push!(witnesses,point)
        end
    end
    witnesses
end
function execute_job(job::Dict{String,Any},config::Dict{String,Any})
    job=atlas_normalize_job(job)
    kind=job["kind"]
    kind=="discovery" && return execute_discovery(job,config)
    kind=="continuation" && return execute_continuation(job,config)
    kind=="certification" && return execute_certification(job,config)
    if kind in ("nexus","section","periodic")
        raw=execute_event_job(job,config)
        follow=get(raw,"followups",Dict{String,Any}[])
        follow=atlas_normalize_job.(follow)
        status=get(raw,"status","unresolved")
        accepted=status in ("completed","numerically_validated","fixture_exercised","sampled")
        key=kind=="section" ? "sections" : kind=="periodic" ? "cycles" : "events"
        witnesses=atlas_anchor_points(raw,kind,job,config)
        raw["witness_point_ids"]=[p["id"] for p in witnesses]
        kind=="section" && (raw["validated"]=!isempty(witnesses))
        evidence=Dict{String,Any}(key=>[raw],"points"=>witnesses)
        kind=="section" && (evidence["events"]=[Dict("kind"=>"section_side_witnesses","validated"=>!isempty(witnesses),"witness_point_ids"=>raw["witness_point_ids"],"scope"=>"sampled_section_neighbors_only")])
        return Dict("status"=>accepted ? "completed" : "unresolved","evidence"=>evidence,"followups"=>follow)
    end
    throw(ArgumentError("unknown scientific job kind: $kind"))
end

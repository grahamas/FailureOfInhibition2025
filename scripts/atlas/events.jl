include("derivatives.jl")

# Numerical local bifurcation evidence. These routines never certify a
# neighborhood or infer absence from a failed correction/shooting attempt.

function atlas_rhs(p, x)
    y = zeros(2)
    point_rhs!(y, x, reduced_model(p), 0.0)
    y
end
function atlas_jac(p, x)
    J = zeros(2, 2)
    point_jacobian!(J, x, reduced_model(p), 0.0)
    J
end

function atlas_tensors(jac, x; step=1e-3)
    n = length(x)
    H = zeros(n, n, n)
    C = zeros(n, n, n, n)
    basis = Matrix{Float64}(I, n, n)
    difference = (f,y,e) -> (f(y-2step*e)-8f(y-step*e)+
        8f(y+step*e)-f(y+2step*e))/(12step)
    hessian = y -> cat((difference(jac,y,basis[:,j]) for j in 1:n)...; dims=3)
    H .= hessian(x)
    for k in 1:n
        C[:,:,:,k] .= difference(hessian,x,basis[:,k])
    end
    B = (u,v) -> [sum(H[l,i,j]*u[i]*v[j] for i in 1:n, j in 1:n) for l in 1:n]
    T = (u,v,w) -> [sum(C[l,i,j,k]*u[i]*v[j]*w[k]
        for i in 1:n, j in 1:n, k in 1:n) for l in 1:n]
    B, T
end

function atlas_orient(v)
    v[argmax(abs.(v))] < 0 ? -v : v
end

# Resolve each claimed nonzero coefficient against its own refinement error.
# A unit-scale absolute allowance would falsely validate quintic degeneracies.
function atlas_coefficient_resolved(coarse,fine;floor=1e-5,rtol=0.02)
    isfinite(coarse) && isfinite(fine) && min(abs(coarse),abs(fine))>floor &&
        sign(coarse)==sign(fine) && abs(coarse-fine)<=rtol*min(abs(coarse),abs(fine))
end

"""Cusp/BT coefficients, with the raw degeneracy residuals retained.

The coefficient convention uses F=J*x+B(x,x)/2+C(x,x,x)/6+... .
BT coordinates obey J*q1=q0 and duality p_i'*q_j=delta_ij.
"""
function atlas_nexus_coefficients(jac, x, kind; step=1e-3, tensors=nothing)
    A = jac(x)
    S = svd(A)
    q = atlas_orient(S.V[:,end])
    B, C = tensors===nothing ? atlas_tensors(jac, x; step=step) : tensors
    if kind == "cusp"
        p = S.U[:,end]
        abs(dot(p,q)) > 1e-7 || throw(ArgumentError("zero eigenvalue is not simple"))
        p /= dot(p,q)
        quadratic = dot(p,B(q,q))/2
        h = ([A q; p' 0.0] \ vcat(-B(q,q) + 2quadratic*q, 0.0))[1:2]
        cubic = dot(p,C(q,q,q) + 3B(q,h))/6
        return Dict{String,Any}("quadratic"=>quadratic, "cubic"=>cubic,
            "null_residual"=>norm(A*q,Inf), "simple_pairing"=>abs(dot(S.U[:,end],q)),
            "transverse_eigenvalue_magnitude"=>abs(tr(A)),"largest_singular_value"=>S.S[1])
    elseif kind == "bt"
        q1 = pinv(A; rtol=1e-7)*q
        norm(q1) > 1e-8 || throw(ArgumentError("double-zero Jordan chain unresolved"))
        dual = inv(hcat(q,q1))'
        p0, p1 = dual[:,1], dual[:,2]
        return Dict{String,Any}("quadratic"=>dot(p1,B(q,q))/2,
            "mixed"=>dot(p1,B(q,q1)) + dot(p0,B(q,q)),
            "jordan_residual"=>max(norm(A*q,Inf),norm(A*q1-q,Inf)),
            "jacobian_norm"=>norm(A))
    end
    throw(ArgumentError("unknown nexus type $kind"))
end

const ATLAS_EVENT_DIFFERENCE_STEP = 1e-5
const ATLAS_EVENT_DIFFERENCE_MAX_HALVINGS = 20

struct AtlasDifferenceUnresolved <: Exception
    coordinate::Int
    reason::String
end
Base.showerror(io::IO,error::AtlasDifferenceUnresolved)=print(io,error.reason," at coordinate ",error.coordinate)

# Three-point derivative using the actual represented offsets from x. This
# remains second order for either sign of a,b and does not assume x+2h-x=2h.
function atlas_three_point_difference(y,first,second,a,b)
    ((b/a).*(first-y)-(a/b).*(second-y))/(b-a)
end

function atlas_difference(f, x; step=ATLAS_EVENT_DIFFERENCE_STEP,
        admissible=coordinates->true,max_halvings=ATLAS_EVENT_DIFFERENCE_MAX_HALVINGS,diagnostics=nothing)
    step isa Real && !(step isa Bool) && isfinite(step) && step>0 || throw(ArgumentError("positive finite derivative step required"))
    max_halvings isa Integer && !(max_halvings isa Bool) && 0<=max_halvings<=64 || throw(ArgumentError("derivative halvings must be 0:64"))
    all(isfinite,x)&&admissible(x) || throw(AtlasDifferenceUnresolved(0,"inadmissible_derivative_center"))
    y = f(x)
    all(isfinite,y) || throw(AtlasDifferenceUnresolved(0,"nonfinite_derivative_center"))
    M = zeros(length(y),length(x))
    for j in eachindex(x)
        initial_h = step*max(1,abs(x[j]))
        resolved=false
        for halving in 0:max_halvings
            h=initial_h/2.0^halving
            function probe(multiplier)
                trial=copy(x);trial[j]=x[j]+multiplier*h
                offset=trial[j]-x[j]
                isfinite(offset)&&offset!=0&&sign(offset)==sign(multiplier)&&admissible(trial) || return nothing
                value=f(trial)
                length(value)==length(y) || throw(ArgumentError("derivative probe output size changed"))
                all(isfinite,value) ? (value,offset) : nothing
            end
            plus,minus=probe(1),probe(-1)
            column=nothing;scheme=""
            if plus!==nothing && minus!==nothing
                # Preserve the established centered interior arithmetic when
                # nominal offsets faithfully represent the floating probes.
                column=if isapprox(plus[2],h;rtol=sqrt(eps(Float64)),atol=0)&&
                        isapprox(minus[2],-h;rtol=sqrt(eps(Float64)),atol=0)
                    (plus[1]-minus[1])/(2h)
                else
                    atlas_three_point_difference(y,minus[1],plus[1],minus[2],plus[2])
                end
                scheme="centered_second_order"
            elseif plus!==nothing
                farther=probe(2)
                if farther!==nothing && farther[2]!=plus[2]
                    column=atlas_three_point_difference(y,plus[1],farther[1],plus[2],farther[2])
                    scheme="forward_second_order"
                end
            elseif minus!==nothing
                farther=probe(-2)
                if farther!==nothing && farther[2]!=minus[2]
                    column=atlas_three_point_difference(y,minus[1],farther[1],minus[2],farther[2])
                    scheme="backward_second_order"
                end
            end
            if column!==nothing && all(isfinite,column)
                M[:,j].=column
                diagnostics===nothing || push!(diagnostics,Dict{String,Any}("coordinate"=>j,"scheme"=>scheme,
                    "nominal_step"=>h,"halvings"=>halving))
                resolved=true;break
            end
        end
        resolved || throw(AtlasDifferenceUnresolved(j,"no_admissible_finite_representable_derivative_stencil"))
    end
    M
end

# Configured tolerances cap residual acceptance; rank and nondegeneracy
# diagnostics retain their separately documented numerical thresholds.
function atlas_residual_tolerance(config,key,default)
    value=get(config,key,default)
    value isa Real && !(value isa Bool) && isfinite(value) && value>0 ||
        throw(ArgumentError("$key must be a finite positive number"))
    converted=Float64(value)
    isfinite(converted) && converted>0 || throw(ArgumentError("$key is outside the numerical range"))
    converted
end
atlas_balance_tolerance(config)=atlas_residual_tolerance(config,"balance_atol",1e-9)
atlas_event_tolerance(config,stage)=min(stage,atlas_residual_tolerance(config,"event_atol",1e-7))
function atlas_equilibrium_options(config)
    tolerance=atlas_balance_tolerance(config)
    EquilibriumOptions(residual_atol=tolerance,solver_abstol=min(1e-12,tolerance/10),
        solver_reltol=min(1e-10,tolerance/10),maxiters=Int(get(config,"max_corrector_iters",100)))
end
function atlas_saved_event_residuals(p,x,event_components,config;stage=1e-8)
    balance_residual=norm(balance_value(p,x),Inf)
    event_residual=norm(event_components,Inf)
    event_tolerance=atlas_event_tolerance(config,stage)
    balance_tolerance=atlas_balance_tolerance(config)
    Dict{String,Any}("balance_residual"=>balance_residual,"event_condition_residual"=>event_residual,
        "balance_atol"=>balance_tolerance,"event_atol"=>event_tolerance,
        "residuals_accepted"=>isfinite(balance_residual)&&isfinite(event_residual)&&
            balance_residual<=balance_tolerance&&event_residual<=event_tolerance)
end
function atlas_newton(f, x, admissible; iterations=16, atol=1e-8, residual_tolerances=nothing)
    x = Float64.(x)
    attempts = Dict{String,Any}[]
    admissible(x) || return x,false,attempts
    for iteration in 0:iterations
        y = f(x)
        residual = norm(y,Inf)
        tolerances=residual_tolerances===nothing ? fill(atol,length(y)) : residual_tolerances
        length(tolerances)==length(y) && all(t->isfinite(t)&&t>0,tolerances) || throw(ArgumentError("positive component tolerances required"))
        scaled_residual=norm(y ./ tolerances,Inf)
        push!(attempts,Dict("iteration"=>iteration,"coordinates"=>copy(x),"residual"=>residual,"scaled_residual"=>scaled_residual))
        isfinite(residual) || return x,false,attempts
        scaled_residual <= 1 && return x,true,attempts
        iteration == iterations && break
        stencils=Dict{String,Any}[]
        M = try
            atlas_difference(f,x;admissible=admissible,diagnostics=stencils)
        catch error
            error isa AtlasDifferenceUnresolved || rethrow()
            attempts[end]["derivative_status"]="unresolved"
            attempts[end]["derivative_reason"]=error.reason
            attempts[end]["derivative_coordinate"]=error.coordinate
            attempts[end]["derivative_stencils"]=stencils
            return x,false,attempts
        end
        attempts[end]["derivative_stencils"]=stencils
        minimum(svdvals(M)) > 1e-12 || break
        correction = M \ y
        accepted = false
        for reduction in 0:10
            trial = x - correction/2.0^reduction
            if admissible(trial) && norm(f(trial) ./ tolerances,Inf) < scaled_residual
                x = trial; accepted = true; break
            end
        end
        accepted || break
    end
    x,false,attempts
end

const ATLAS_EVENT_PARAMETERS = ("A","B","C","D","a","b","h","r")
function atlas_parameter_admissible(p)
    all(isfinite, values(p)) && all(p[k]>=0 for k in ("A","B","C","D")) &&
        p["h"]>0 && p["r"]>0
end
function atlas_event_vector(rhs, jac, x, kind; tensors=nothing)
    A=jac(x)
    kind == "bt" ? vcat(rhs(x), tr(A),det(A)) :
        vcat(rhs(x),det(A),atlas_nexus_coefficients(jac,x,"cusp";tensors=tensors)["quadratic"])
end

# Keep reconstruction in its own function scope: a callback-local assignment
# must never capture and overwrite a later saved output binding in its caller.
function atlas_parameters_at(base, pair, coordinates; scales=(1.0,1.0))
    reconstructed=copy(base)
    reconstructed[pair[1]]=coordinates[3]*scales[1]
    reconstructed[pair[2]]=coordinates[4]*scales[2]
    reconstructed
end
function atlas_nexus_vector_at(base,pair,coordinates,kind)
    candidate_parameters=atlas_parameters_at(base,pair,coordinates)
    atlas_parameter_admissible(candidate_parameters) ||
        throw(ArgumentError("parameter perturbation outside domain"))
    vector=atlas_event_vector(state->atlas_rhs(candidate_parameters,state),
        state->atlas_jac(candidate_parameters,state),coordinates[1:2],kind;
        tensors=atlas_reduced_tensors(candidate_parameters,coordinates[1:2]))
    vector[1:2]=balance_value(candidate_parameters,coordinates[1:2])
    vector
end

function atlas_cusp_simple_zero_resolved(coefficients,event_tolerance)
    coefficients["null_residual"]<=event_tolerance &&
        coefficients["transverse_eigenvalue_magnitude"]>1e-7 && coefficients["largest_singular_value"]>1e-7
end

function atlas_nexus_admissible(base,pair,coordinates)
    all(isfinite,coordinates) && all(0 .<= coordinates[1:2] .<= 0.5) &&
        atlas_parameter_admissible(atlas_parameters_at(base,pair,coordinates))
end

function atlas_nexus_job(job, config;differentiate=atlas_difference)
    p = Dict{String,Any}(get(job,"parameters",get(job,"params",Dict())))
    x = Float64.(job["state"])
    kind = lowercase(get(job,"nexus_type",get(job,"event_type","bt")))
    kind in ("bt","cusp") || throw(ArgumentError("nexus_type must be bt or cusp"))
    scored = Tuple{Float64,Tuple{String,String}}[]
    pair_diagnostics=Dict{String,Any}[]
    # A full-rank augmented event system is the numerical unfolding test.
    for i in 1:7, j in i+1:8
        pair=(ATLAS_EVENT_PARAMETERS[i],ATLAS_EVENT_PARAMETERS[j])
        f=coordinates->atlas_nexus_vector_at(p,pair,coordinates,kind)
        z=vcat(x,p[pair[1]],p[pair[2]])
        admissible=coordinates->atlas_nexus_admissible(p,pair,coordinates)
        stencils=Dict{String,Any}[]
        record=Dict{String,Any}("parameter_pair"=>collect(pair),"derivative_stencils"=>stencils)
        try
            M=differentiate(f,z;admissible=admissible,diagnostics=stencils)
            scale=[1.0,1.0,max(1,abs(z[3])),max(1,abs(z[4]))]
            score=minimum(svdvals(M*Diagonal(scale)))
            push!(scored,(score,pair))
            merge!(record,Dict("status"=>"scored","smallest_singular_value"=>score))
        catch err
            if err isa AtlasDifferenceUnresolved
                merge!(record,Dict("status"=>"unresolved_derivative_stencil","reason"=>err.reason,"coordinate"=>err.coordinate))
            elseif err isa SingularException
                merge!(record,Dict("status"=>"unresolved_singular_event_system","reason"=>sprint(showerror,err)))
            else
                rethrow()
            end
        end
        push!(pair_diagnostics,record)
    end
    isempty(scored) && return Dict{String,Any}("status"=>"unresolved","validated"=>false,
        "reason"=>"no_admissible_unfolding_pair",
        "attempts"=>[Dict("stage"=>"unfolding_pair_selection","pair_selection_attempts"=>pair_diagnostics)],"followups"=>Any[])
    sort!(scored;by=z->(-z[1],z[2]))
    score,pair=first(scored)
    params=coordinates->atlas_parameters_at(p,pair,coordinates)
    f=coordinates->atlas_nexus_vector_at(p,pair,coordinates,kind)
    admissible=coordinates->atlas_nexus_admissible(p,pair,coordinates)
    z,converged,attempts=atlas_newton(f,vcat(x,p[pair[1]],p[pair[2]]),admissible;
        iterations=get(config,"nexus_max_iterations",16),residual_tolerances=[min(1e-8,atlas_balance_tolerance(config)),min(1e-8,atlas_balance_tolerance(config)),atlas_event_tolerance(config,1e-8),atlas_event_tolerance(config,1e-8)])
    if isempty(attempts)
        push!(attempts,Dict("stage"=>"unfolding_pair_selection","pair_selection_attempts"=>pair_diagnostics))
    else
        attempts[1]["pair_selection_attempts"]=pair_diagnostics
    end
    # Finish derivative probes before creating the saved parameter/state tuple.
    conditioning_stencils=Dict{String,Any}[]
    M=try
        differentiate(f,z;admissible=admissible,diagnostics=conditioning_stencils)
    catch err
        err isa AtlasDifferenceUnresolved || rethrow()
        return Dict{String,Any}("status"=>"unresolved","validated"=>false,
            "classification"=>"candidate_$(kind)","reason"=>"unresolved_conditioning_derivative_stencil",
            "parameters"=>params(z),"state"=>copy(z[1:2]),"parameter_pair"=>collect(pair),
            "converged_coordinates"=>copy(z),"attempts"=>attempts,
            "conditioning"=>Dict("status"=>"unresolved","reason"=>err.reason,"coordinate"=>err.coordinate,
                "derivative_stencils"=>conditioning_stencils),"followups"=>Any[],
            "completeness"=>"CompletenessNotCertified")
    end
    t=params(z); y=copy(z[1:2])
    coefficients=atlas_nexus_coefficients(q->atlas_jac(t,q),y,kind;tensors=atlas_reduced_tensors(t,y))
    reference=atlas_nexus_coefficients(q->atlas_jac(t,q),y,kind;tensors=atlas_reference_tensors(t,y))
    refined=atlas_nexus_coefficients(q->atlas_jac(t,q),y,kind;
        tensors=atlas_reference_tensors(t,y;step=ATLAS_DERIVATIVE_REFERENCE_STEP/2))
    keys_to_check=kind=="bt" ? ("quadratic","mixed") : ("quadratic","cubic")
    difference=maximum(abs(coefficients[k]-refined[k]) for k in keys_to_check)
    nonzero_keys=kind=="bt" ? ("quadratic","mixed") : ("cubic",)
    checks=Dict(k=>atlas_coefficient_resolved(coefficients[k],refined[k]) &&
        atlas_coefficient_resolved(reference[k],refined[k]) for k in nonzero_keys)
    stable=all(values(checks))
    nondegenerate=stable
    unfolding=minimum(svdvals(M*Diagonal([1.0,1.0,max(1,abs(z[3])),max(1,abs(z[4]))])))
    chain=kind=="bt" ? coefficients["jordan_residual"]<=1e-7 :
        atlas_cusp_simple_zero_resolved(coefficients,atlas_event_tolerance(config,1e-7)) &&
        all(abs(c["quadratic"])<=atlas_event_tolerance(config,1e-7) for c in (coefficients,reference,refined))
    saved_vector=atlas_nexus_vector_at(t,pair,vcat(y,t[pair[1]],t[pair[2]]),kind)
    residual_validation=atlas_saved_event_residuals(t,y,saved_vector[3:end],config)
    saved_residual=norm(saved_vector,Inf)
    saved_coordinates=vcat(y,t[pair[1]],t[pair[2]])
    saved_tuple_matches=isequal(saved_coordinates,z)
    validated=converged && stable && nondegenerate && chain && unfolding>1e-7 &&
        saved_tuple_matches && residual_validation["residuals_accepted"]
    followups=validated ? Any[Dict("kind"=>"section","parameters"=>t,"state"=>y,
        "parameter_pair"=>collect(pair),"nexus_type"=>kind)] : Any[]
    Dict{String,Any}("status"=>validated ? "numerically_validated" : "unresolved","validated"=>validated,
        "classification"=>validated ? kind : "candidate_$(kind)","parameters"=>t,"state"=>y,
        "parameter_pair"=>collect(pair),"coefficients"=>coefficients,"independent_reference_coefficients"=>refined,"coefficient_refinement_difference"=>difference,
        "coefficient_refinement_checks"=>checks,
        "coefficient_comparison"=>"analytic versus independent direct-RHS highprecision differences; reference checked at two steps",
        "reference_coefficient_refinement_difference"=>maximum(abs(reference[k]-refined[k]) for k in keys_to_check),"refined_degeneracy_residual"=>kind=="cusp" ? abs(refined["quadratic"]) : 0.0,
        "analytic_degeneracy_residual"=>kind=="cusp" ? abs(coefficients["quadratic"]) : 0.0,
        "derivative_scheme"=>"analytic_original_time_state_tensors_with_independent_256bit_rhs_differences",
        "numerical_policy_version"=>ATLAS_NUMERICAL_POLICY_VERSION,
        "derivative_diagnostics"=>atlas_derivative_diagnostics(t,y),
        "conditioning"=>merge(atlas_event_conditioning(M,z,saved_vector),
            Dict("derivative_stencils"=>conditioning_stencils)),
        "coefficient_convention"=>"F=J*x+B(x,x)/2+C(x,x,x)/6; normalized right null vector; dual Jordan basis for BT",
        "unfolding_smallest_singular_value"=>unfolding,"initial_pair_score"=>score,
        "event_residual"=>saved_residual,"residual_validation"=>residual_validation,"converged_coordinates"=>copy(z),
        "saved_tuple_matches_converged_coordinates"=>saved_tuple_matches,
        "attempts"=>attempts,"followups"=>followups,
        "completeness"=>"CompletenessNotCertified")
end

function atlas_section_job(job, config)
    p=Dict{String,Any}(job["parameters"]); center=Float64.(job["state"])
    pair=String.(job["parameter_pair"])
    rows=Dict{String,Any}[]
    angles=get(config,"section_angles",16)
    4<=angles<=128 || throw(ArgumentError("section_angles must be 4:128"))
    for radius in (0.125,0.0625,0.03125), angle_index in 0:angles-1
        angle=2pi*angle_index/angles
        t=copy(p)
        t[pair[1]]+=radius*max(1,abs(p[pair[1]]))*cos(angle)
        t[pair[2]]+=radius*max(1,abs(p[pair[2]]))*sin(angle)
        if !atlas_parameter_admissible(t)
            push!(rows,Dict("radius"=>radius,"angle"=>angle,"status"=>"outside_admissible_domain"));continue
        end
        model=reduced_model(t)
        search=find_equilibria(model;seeds=vcat([center],default_equilibrium_seeds(model)),options=atlas_equilibrium_options(config))
        roots=[Dict("state"=>root.state,"residual"=>maximum(abs,root.balance_residual),
            "stability"=>string(root.stability.classification)) for root in search.equilibria]
        push!(rows,Dict("radius"=>radius,"angle"=>angle,"parameters"=>t,"status"=>"sampled",
            "equilibria"=>roots,"balance_atol"=>atlas_balance_tolerance(config),"failed_attempts"=>count(a->a.validation!=AdmissibleCandidate,search.attempts),
            "attempts"=>[Dict("seed"=>a.seed,"candidate"=>a.candidate,"residual"=>a.residual_norm,
                "solver_status"=>string(a.solver_status),"validation"=>string(a.validation),
                "reasons"=>string.(a.reasons)) for a in search.attempts]))
    end
    traces,followups=atlas_section_traces(p,center,pair,config)
    Dict{String,Any}("status"=>"sampled","parameter_pair"=>pair,"parameters"=>p,"state"=>center,
        "section_samples"=>rows,"completeness"=>"CompletenessNotCertified",
        "transition_curves"=>traces,"followups"=>followups,
        "scope"=>"Three sampled circles and bounded corrected event curves in the named section; unsampled sectors and intervening paths remain unresolved")
end

function atlas_section_traces(p,center,pair,config;differentiate=atlas_difference)
    scale=[1.0,1.0,max(1,abs(p[pair[1]])),max(1,abs(p[pair[2]]))]
    initial=vcat(center,p[pair[1]],p[pair[2]])./scale
    params=coordinates->atlas_parameters_at(p,pair,coordinates;scales=(scale[3],scale[4]))
    valid=z->all(isfinite,z) && all(0 .<= z[1:2] .<= 0.5) && atlas_parameter_admissible(params(z))
    traces=Dict{String,Any}[]; followups=Any[]
    for event in ("fold","hopf"), direction in (-1,1)
        condition=z->begin
            local candidate_state=z[1:2]
            local candidate_parameters=params(z)
            local candidate_jacobian=atlas_jac(candidate_parameters,candidate_state)
            vcat(balance_value(candidate_parameters,candidate_state),
                event=="fold" ? det(candidate_jacobian) : tr(candidate_jacobian))
        end
        z=copy(initial); points=Dict{String,Any}[]; attempts=Any[]
        oldtangent=nothing; termination="step_limit"; cycle_enqueued=false
        # A cusp is on the fold locus, but generally not on the trace-zero
        # locus. Do not jump to a remote curve and call it a local unfolding.
        if !valid(z)
            push!(traces,Dict("event"=>event,"direction"=>direction,"points"=>points,
                "attempts"=>attempts,"termination"=>"unresolved_initial_point_outside_admissible_domain"))
            continue
        end
        if !all(abs.(condition(z)) .<= [min(1e-7,atlas_balance_tolerance(config)),min(1e-7,atlas_balance_tolerance(config)),atlas_event_tolerance(config,1e-7)])
            push!(traces,Dict("event"=>event,"direction"=>direction,"points"=>points,
                "attempts"=>attempts,"termination"=>"initial_point_not_on_event_locus"))
            continue
        end
        for index in 1:get(config,"section_curve_max_steps",12)
            stencils=Dict{String,Any}[]
            M=try
                differentiate(condition,z;admissible=valid,diagnostics=stencils)
            catch err
                err isa AtlasDifferenceUnresolved || rethrow()
                termination="unresolved_derivative_stencil"
                push!(attempts,Dict("step"=>index,"accepted"=>false,"reason"=>err.reason,
                    "derivative_coordinate"=>err.coordinate,"derivative_stencils"=>stencils))
                break
            end
            S=svd(M;full=true)
            if minimum(S.S)<1e-10
                termination="unresolved_event_rank";break
            end
            tangent=S.V[:,end]
            if oldtangent===nothing
                tangent=atlas_orient(tangent)*direction
            elseif dot(tangent,oldtangent)<0
                tangent=-tangent
            end
            predictor=z+0.02tangent
            augmented=y->vcat(condition(y),dot(y-predictor,tangent))
            corrected,ok,history=atlas_newton(augmented,predictor,valid;iterations=12,residual_tolerances=[min(1e-8,atlas_balance_tolerance(config)),min(1e-8,atlas_balance_tolerance(config)),atlas_event_tolerance(config,1e-8),atlas_event_tolerance(config,1e-8)])
            push!(attempts,Dict("step"=>index,"accepted"=>ok,"iterations"=>history,"tangent_derivative_stencils"=>stencils))
            if !ok
                termination="unresolved_corrector";break
            end
            z=corrected;oldtangent=tangent;t=params(z);x=copy(z[1:2]);A=atlas_jac(t,x)
            saved_condition=vcat(balance_value(t,x),event=="fold" ? det(A) : tr(A))
            residual_validation=atlas_saved_event_residuals(t,x,saved_condition[3:end],config)
            saved_residual=norm(saved_condition,Inf)
            if !residual_validation["residuals_accepted"]
                termination="unresolved_saved_tuple_residual";break
            end
            classification=event=="fold" ? "fold_candidate" : det(A)>1e-8 ? "hopf_candidate" : "trace_zero_without_positive_determinant"
            push!(points,Dict("parameters"=>t,"state"=>x,"classification"=>classification,
                "trace"=>tr(A),"determinant"=>det(A),"residual"=>saved_residual,"residual_validation"=>residual_validation))
            if classification=="hopf_candidate" && !cycle_enqueued
                push!(followups,Dict("kind"=>"periodic","queue"=>"cycles","parameters"=>t,"state"=>x,"origin"=>"section_hopf"))
                cycle_enqueued=true
            end
        end
        push!(traces,Dict("event"=>event,"direction"=>direction,"points"=>points,
            "attempts"=>attempts,"termination"=>termination))
    end
    traces,followups
end

function atlas_hopf_coefficient(jac, x;step=1e-3,tensors=nothing)
    A=jac(x); E=eigen(A)
    index=argmax(imag.(E.values)); lambda=E.values[index]; omega=imag(lambda)
    omega>1e-7 && abs(real(lambda))<1e-6 || throw(ArgumentError("not a resolved Hopf candidate"))
    q=ComplexF64.(E.vectors[:,index]); q/=norm(q)
    P=eigen(complex.(A'))
    p=P.vectors[:,argmin(abs.(P.values .+ im*omega))]
    p/=conj(dot(p,q))
    B,C=tensors===nothing ? atlas_tensors(jac,x;step=step) : tensors
    G=dot(p,C(q,q,conj(q))-2B(q,A\B(q,conj(q)))+
        B(conj(q),(2im*omega*I-A)\B(q,q)))
    real(G)/(2omega),omega,q
end

function atlas_orbit_evidence(orbit)
    Dict{String,Any}("state"=>orbit.initial_state,"period"=>orbit.period,"amplitudes"=>orbit.amplitudes,
        "times"=>orbit.times,"states"=>orbit.states,
        "closure_residual"=>orbit.closure_residual,"phase_residual"=>orbit.phase_residual,
        "equation_residual"=>orbit.equation_residual,"validation"=>string(orbit.validation),
        "stability"=>string(orbit.stability),"floquet_real"=>real(orbit.transverse_multiplier),
        "floquet_imag"=>imag(orbit.transverse_multiplier),"reasons"=>string.(orbit.reasons))
end

function atlas_periodic_attempt_evidence(a)
    Dict{String,Any}("direction"=>a.direction,"step"=>a.step,"predictor"=>a.predictor,
        "candidate"=>a.candidate,"iterations"=>a.iterations,
        "corrector_arclength_residual"=>a.corrector_arclength_residual,
        "corrector_constraint_residual"=>a.corrector_constraint_residual,
        "post_shoot_arclength_residual"=>a.post_shoot_arclength_residual,
        "post_shoot_constraint_residual"=>a.post_shoot_constraint_residual,
        "status"=>string(a.status),"accepted"=>a.accepted,
        "shooting_evidence"=>a.orbit===nothing ? Dict{String,Any}() : atlas_orbit_evidence(a.orbit))
end

function atlas_periodic_job(job, config)
    get(job,"fixture",false) && return atlas_periodic_fixture(config)
    p=Dict{String,Any}(get(job,"parameters",get(job,"params",Dict())))
    x=Float64.(job["state"])
    hopf_validation=atlas_saved_event_residuals(p,x,[tr(atlas_jac(p,x))],config)
    hopf_validation["residuals_accepted"] || return Dict{String,Any}("status"=>"unresolved",
        "reason"=>"hopf_tuple_outside_residual_tolerances","residual_validation"=>hopf_validation,"followups"=>Any[])
    l1,omega,q=atlas_hopf_coefficient(y->atlas_jac(p,y),x;tensors=atlas_reduced_tensors(p,x))
    l1_reference,_,_=atlas_hopf_coefficient(y->atlas_jac(p,y),x;tensors=atlas_reference_tensors(p,x))
    l1_fine,_,_=atlas_hopf_coefficient(y->atlas_jac(p,y),x;
        tensors=atlas_reference_tensors(p,x;step=ATLAS_DERIVATIVE_REFERENCE_STEP/2))
    (atlas_coefficient_resolved(l1,l1_fine) && atlas_coefficient_resolved(l1_reference,l1_fine)) ||
        return Dict{String,Any}("status"=>"unresolved","reason"=>"generalized_hopf_or_unresolved_coefficient",
            "first_lyapunov_coefficient"=>l1_fine,"followups"=>Any[])
    # The r derivative is analytic because J=diag(1,1/r)*K.
    A=atlas_jac(p,x); derivative=-A[2,2]/(2p["r"])
    abs(derivative)>1e-10 || return Dict{String,Any}("status"=>"unresolved","reason"=>"time_ratio_not_transverse","followups"=>Any[])
    # For w'= (beta+i*omega)w+l1*w*|w|^2, beta/l1<0.
    direction=-sign(l1*derivative)
    delta=get(config,"hopf_parameter_step",1e-3)*max(1,p["r"])
    t=copy(p);t["r"]+=direction*delta
    atlas_parameter_admissible(t) || return Dict{String,Any}("status"=>"unresolved","reason"=>"initial_cycle_outside_domain","followups"=>Any[])
    amplitude=min(0.02,sqrt(abs(derivative*delta/(omega*l1))))
    seed=x+2amplitude*real(q)
    options=PeriodicOrbitOptions(maxiters=get(config,"periodic_shooting_iterations",15),samples=129,
        ode_maxiters=get(config,"periodic_ode_maxiters",100000))
    orbit=solve_periodic_orbit(reduced_model(t),seed,2pi/omega;options=options)
    result=Dict{String,Any}("status"=>orbit.validation==NumericallyValidatedPeriodicOrbit ? "numerically_validated" : "unresolved",
        "validated"=>orbit.validation==NumericallyValidatedPeriodicOrbit,
        "first_lyapunov_coefficient"=>l1,"independent_reference_coefficient"=>l1_fine,
        "coefficient_refinement_difference"=>abs(l1-l1_fine),
        "reference_coefficient_refinement_difference"=>abs(l1_reference-l1_fine),
        "numerical_policy_version"=>ATLAS_NUMERICAL_POLICY_VERSION,"derivative_diagnostics"=>atlas_derivative_diagnostics(p,x),
        "parameters"=>t,"hopf_parameters"=>p,"hopf_state"=>x,"hopf_residual_validation"=>hopf_validation,"initial_side"=>Int(direction),
        "orbit"=>atlas_orbit_evidence(orbit),"followups"=>Any[],"completeness"=>"CompletenessNotCertified")
    if orbit.validation==NumericallyValidatedPeriodicOrbit
        factory=r->begin
            local orbit_parameters=copy(t)
            orbit_parameters["r"]=r
            reduced_model(orbit_parameters)
        end
        bounds=(max(1e-6,t["r"]-0.25max(1,t["r"])),t["r"]+0.25max(1,t["r"]))
        branch=continue_periodic_orbit(factory,orbit.initial_state,orbit.period,t["r"];
            parameter_bounds=bounds,options=PeriodicContinuationOptions(max_steps=get(config,"periodic_max_steps",12)),
            periodic_options=options)
        result["branches"]=[Dict("direction"=>b.direction,"termination"=>string(b.termination),
            "orbits"=>[merge(Dict("parameter"=>pt.parameter),atlas_orbit_evidence(pt.orbit)) for pt in b.points],
            "attempts"=>atlas_periodic_attempt_evidence.(b.attempts)) for b in (branch.negative,branch.positive)]
    end
    result
end

function atlas_periodic_fixture(config)
    evidence=Any[]
    for cubic in (-1.0,1.0)
        rhs! = (out,x,p)->begin
            local fixture_beta,fixture_cubic=p
            local radius2=sum(abs2,x)
            out[1]=fixture_beta*x[1]-x[2]+fixture_cubic*x[1]*radius2
            out[2]=x[1]+fixture_beta*x[2]+fixture_cubic*x[2]*radius2
        end
        jac! = (J,x,p)->begin
            local fixture_beta,fixture_cubic=p
            local first_state,second_state=x
            local radius2=first_state^2+second_state^2
            J[1,1]=fixture_beta+fixture_cubic*(radius2+2first_state^2)
            J[1,2]=-1+2fixture_cubic*first_state*second_state
            J[2,1]=1+2fixture_cubic*first_state*second_state
            J[2,2]=fixture_beta+fixture_cubic*(radius2+2second_state^2)
        end
        beta=-0.04cubic
        options=PeriodicOrbitOptions(samples=129,maxiters=12,ode_maxiters=100000)
        orbit=solve_periodic_orbit(rhs!,jac!,(beta,cubic),[0.2,0.0],2pi;options=options)
        row=atlas_orbit_evidence(orbit);row["cubic"]=cubic;row["beta"]=beta
        if orbit.validation==NumericallyValidatedPeriodicOrbit
            branch=continue_periodic_orbit(rhs!,jac!,mu->(mu,cubic),orbit.initial_state,
                orbit.period,beta;parameter_bounds=cubic<0 ? (0.01,0.08) : (-0.08,-0.01),
                options=PeriodicContinuationOptions(max_steps=get(config,"periodic_max_steps",4)),
                periodic_options=options)
            row["branches"]=[Dict("direction"=>b.direction,"termination"=>string(b.termination),
                "orbits"=>[merge(Dict("parameter"=>pt.parameter),atlas_orbit_evidence(pt.orbit)) for pt in b.points],
                "attempts"=>atlas_periodic_attempt_evidence.(b.attempts))
                for b in (branch.negative,branch.positive)]
        end
        push!(evidence,row)
    end
    Dict{String,Any}("status"=>"fixture_exercised","origin"=>"synthetic_normal_form",
        "scope"=>"Synthetic stable and unstable oscillator shooting and continuation; no evidence about the inhibition model",
        "fixtures"=>evidence,"followups"=>Any[],"completeness"=>"CompletenessNotCertified")
end

function execute_event_job(job,config)
    kind=job["kind"]
    kind=="nexus" && return atlas_nexus_job(job,config)
    kind=="section" && return atlas_section_job(job,config)
    kind=="periodic" && return atlas_periodic_job(job,config)
    throw(ArgumentError("unsupported event job $kind"))
end

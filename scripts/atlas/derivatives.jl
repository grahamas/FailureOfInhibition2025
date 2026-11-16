# Private atlas derivatives; no public package API or scientific convention changes.
const ATLAS_NUMERICAL_POLICY_VERSION = "analytic-state-derivatives-v2"
const ATLAS_DERIVATIVE_REFERENCE_BITS = 256
const ATLAS_DERIVATIVE_REFERENCE_STEP = 2.0^-24

function atlas_logistic_jet(z)
    s, t = atlas_logistic(z), atlas_logistic(-z)
    d = s*t
    (s, d, d*(t-s), d*(one(z)-6d))
end

"""Original-time state Hessian/third tensor for the supported reduced model.

The prefactor (1-x_i) is differentiated as well as the response. Inhibitory
jets use the stable product representation; the independent reference below
uses a difference of logistic values instead. Arrays retain the input precision.
"""
function atlas_reduced_tensor_arrays(p, x)
    validate_parameters(p)
    length(x)==2 && all(isfinite,x) || throw(ArgumentError("two finite state coordinates required"))
    T = promote_type(eltype(x), (typeof(p[k]) for k in ATLAS_COORDINATES)...)
    H, C = zeros(T,2,2,2), zeros(T,2,2,2,2)
    w = ((p["A"],-p["B"]),(p["C"],-p["D"]))
    failure = p["b"]+p["h"]
    isfinite(failure) && failure>p["b"] || throw(ArgumentError("inhibitory thresholds are not representable and ordered"))
    u = w[1][1]*x[1]+w[1][2]*x[2]-p["a"]
    v = w[2][1]*x[1]+w[2][2]*x[2]-p["b"]
    e = atlas_logistic_jet(u)
    onset = atlas_logistic_jet(v)
    complement = atlas_logistic_jet(failure-(w[2][1]*x[1]+w[2][2]*x[2]))
    factor = -expm1(-(failure-p["b"]))
    inhibitory = ntuple(n->factor*sum(binomial(n-1,k)*onset[k+1]*
        complement[n-k]*(-1)^(n-1-k) for k in 0:n-1),4)
    for l in 1:2
        jet = l==1 ? e : inhibitory
        tau = l==1 ? one(p["r"]) : p["r"]
        for i in 1:2, j in 1:2
            H[l,i,j] = ((1-x[l])*jet[3]*w[l][i]*w[l][j] -
                jet[2]*((l==i)*w[l][j]+(l==j)*w[l][i]))/tau
            for k in 1:2
                C[l,i,j,k] = ((1-x[l])*jet[4]*w[l][i]*w[l][j]*w[l][k] -
                    jet[3]*((l==i)*w[l][j]*w[l][k]+(l==j)*w[l][i]*w[l][k]+
                        (l==k)*w[l][i]*w[l][j]))/tau
            end
        end
    end
    H,C
end

function atlas_tensor_forms(H,C)
    n=size(H,1)
    B=(u,v)->[sum(H[l,i,j]*u[i]*v[j] for i in 1:n,j in 1:n) for l in 1:n]
    T=(u,v,w)->[sum(C[l,i,j,k]*u[i]*v[j]*w[k] for i in 1:n,j in 1:n,k in 1:n) for l in 1:n]
    B,T
end
atlas_reduced_tensors(p,x)=atlas_tensor_forms(atlas_reduced_tensor_arrays(p,x)...)

# Independent direct equations, deliberately separate from package kernels and
# the analytic jet/product implementation. Only used at high precision.
function atlas_reference_rhs(p,x)
    se=inv(1+exp(-(p["A"]*x[1]-p["B"]*x[2]-p["a"])))
    v=p["C"]*x[1]-p["D"]*x[2]-p["b"]
    failure=get(p,"_failure_threshold",p["b"]+p["h"])
    off=p["C"]*x[1]-p["D"]*x[2]-failure
    si=inv(1+exp(-v))-inv(1+exp(-off))
    [-x[1]+(1-x[1])*se,(-x[2]+(1-x[2])*si)/p["r"]]
end

function atlas_reference_tensor_arrays(p,x;bits=ATLAS_DERIVATIVE_REFERENCE_BITS,
        step=ATLAS_DERIVATIVE_REFERENCE_STEP)
    bits>=192 || throw(ArgumentError("reference requires at least 192 bits"))
    isfinite(step)&&step>0 || throw(ArgumentError("positive reference step required"))
    setprecision(BigFloat,bits) do
        failure=p["b"]+p["h"]
        isfinite(failure)&&failure>p["b"] || throw(ArgumentError("inhibitory thresholds are not representable and ordered"))
        q=Dict(k=>BigFloat(p[k]) for k in ATLAS_COORDINATES)
        q["_failure_threshold"]=BigFloat(failure)
        y=BigFloat.(x)
        # Keep all logits in the same small perturbation range even at large coupling.
        h=BigFloat(step)/max(one(BigFloat),maximum(abs(q[k]) for k in ATLAS_COORDINATES[1:4]))
        offsets=(-2,-1,1,2);weights=(1,-8,8,-1)
        function difference(f,z,j)
            delta=zeros(BigFloat,2);delta[j]=h
            sum(weights[k]*f(z+offsets[k]*delta) for k in 1:4)/(12h)
        end
        field=z->atlas_reference_rhs(q,z)
        H=zeros(BigFloat,2,2,2);C=zeros(BigFloat,2,2,2,2)
        for i in 1:2,j in 1:2
            second=z->difference(t->difference(field,t,i),z,j)
            H[:,i,j]=second(y)
            for k in 1:2
                C[:,i,j,k]=difference(second,y,k)
            end
        end
        H,C
    end
end
function atlas_reference_tensors(p,x;kwargs...)
    H,C=atlas_reference_tensor_arrays(p,x;kwargs...)
    atlas_tensor_forms(Float64.(H),Float64.(C))
end

function atlas_reference_equation_values(p,x)
    setprecision(BigFloat,ATLAS_DERIVATIVE_REFERENCE_BITS) do
        q=Dict(k=>BigFloat(p[k]) for k in ATLAS_COORDINATES)
        q["_failure_threshold"]=BigFloat(p["b"]+p["h"])
        y=BigFloat.(x)
        h=BigFloat(ATLAS_DERIVATIVE_REFERENCE_STEP)/max(one(BigFloat),maximum(abs(q[k]) for k in ATLAS_COORDINATES[1:4]))
        J=zeros(BigFloat,2,2)
        for i in 1:2
            d=zeros(BigFloat,2);d[i]=h
            J[:,i]=(atlas_reference_rhs(q,y-2d)-8atlas_reference_rhs(q,y-d)+
                8atlas_reference_rhs(q,y+d)-atlas_reference_rhs(q,y+2d))/(12h)
        end
        atlas_reference_rhs(q,y),J
    end
end

function atlas_derivative_diagnostics(p,x)
    H,C=atlas_reduced_tensor_arrays(p,x)
    Hr,Cr=atlas_reference_tensor_arrays(p,x;step=ATLAS_DERIVATIVE_REFERENCE_STEP/2)
    Hc,Cc=atlas_reference_tensor_arrays(p,x)
    reference_difference=[Float64(maximum(abs.(a.-b))) for (a,b) in ((Hc,Hr),(Cc,Cr))]
    difference=[Float64(maximum(abs.(BigFloat.(a).-b))) for (a,b) in ((H,Hr),(C,Cr))]
    scale=[Float64(maximum(abs,b)) for b in (Hr,Cr)]
    field,jac=atlas_reference_equation_values(p,x)
    rhs_difference=Float64(norm(BigFloat.(atlas_rhs(p,x)).-field,Inf))
    jac_difference=Float64(maximum(abs.(BigFloat.(atlas_jac(p,x)).-jac)))
    Dict{String,Any}("policy_version"=>ATLAS_NUMERICAL_POLICY_VERSION,
        "primary"=>"analytic_original_time_state_tensors",
        "package_rhs_reference_difference"=>rhs_difference,
        "package_jacobian_reference_difference"=>jac_difference,
        "package_rhs_reference_scale"=>Float64(norm(field,Inf)),
        "package_jacobian_reference_scale"=>Float64(maximum(abs,jac)),
        "independent_reference"=>"nested_fourth_order_centered_direct_rhs_differences",
        "reference_bits"=>ATLAS_DERIVATIVE_REFERENCE_BITS,"reference_step"=>ATLAS_DERIVATIVE_REFERENCE_STEP,
        "reference_refinement_step"=>ATLAS_DERIVATIVE_REFERENCE_STEP/2,
        "hessian_reference_refinement_difference"=>reference_difference[1],
        "third_reference_refinement_difference"=>reference_difference[2],
        "reference_model"=>"exact rounded input tuple with failure threshold reconstructed in input precision before BigFloat conversion",
        "reference_step_normalization"=>"divide by max(1,A,B,C,D)",
        "hessian_maximum_absolute_difference"=>difference[1],"third_maximum_absolute_difference"=>difference[2],
        "hessian_relative_difference"=>difference[1]/max(scale[1],eps(Float64)),
        "third_relative_difference"=>difference[2]/max(scale[2],eps(Float64)),
        "scope"=>"Finite precision crosscheck at this tuple; not an interval error bound")
end

# These estimates use the scaled augmented event system; they describe local
# linear sensitivity and are not validated coordinate-error bounds.
function atlas_event_conditioning(M,z,residual)
    scales=[1.0,1.0,max(1,abs(z[3])),max(1,abs(z[4]))]
    scaled=M*Diagonal(scales)
    singular=svdvals(scaled)
    sigma=minimum(singular)
    Dict{String,Any}("coordinate_scales"=>scales,"scaled_matrix_norm"=>opnorm(scaled),
        "scaled_smallest_singular_value"=>sigma,
        "scaled_condition_estimate"=>sigma>0 ? maximum(singular)/sigma : Inf,
        "linearized_scaled_coordinate_residual_estimate"=>sigma>0 ? norm(residual)/sigma : Inf,
        "scope"=>"Local finite-difference Jacobian diagnostics; not rigorous coordinate-error bounds")
end

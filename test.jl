import InfrastructureModels
import Memento

import PowerModels
const _PMs = PowerModels

import PowerModelsDistribution
const _PMD = PowerModelsDistribution

# Suppress warnings during testing.
Memento.setlevel!(Memento.getlogger(InfrastructureModels), "error")
PowerModels.logger_config!("error")

import Cbc
import Ipopt
import SCS
import Juniper

import JuMP
import JSON

import LinearAlgebra
using Test

# default setup for solvers
ipopt_solver = JuMP.with_optimizer(Ipopt.Optimizer, tol=1e-6, print_level=0)
ipopt_ws_solver = JuMP.with_optimizer(Ipopt.Optimizer, tol=1e-6, mu_init=1e-4, print_level=0)

cbc_solver = JuMP.with_optimizer(Cbc.Optimizer, logLevel=0)
juniper_solver = JuMP.with_optimizer(Juniper.Optimizer, nl_solver=JuMP.with_optimizer(Ipopt.Optimizer, tol=1e-6, print_level=0), mip_solver=cbc_solver, log_levels=[])
scs_solver = JuMP.with_optimizer(SCS.Optimizer, max_iters=500000, acceleration_lookback=1, verbose=0)

###########################################################################################################################################################################
function ref_add_transformer_imbalance!(pm::_PMs.AbstractPowerModel; nw::Int=pm.cnw, limit::Float64 = .15)
    _PMs.ref(pm, nw)[:trans_bal] = limit
end


function ref_add_vm_imbalance!(pm::_PMs.AbstractPowerModel; nw::Int=pm.cnw, limit::Float64 = .025)
    for i in _PMs.ids(pm, :bus)
        _PMs.ref(pm, 0, :bus, i)["vm_vuf_max"] = limit
    end
end

function ref_add_transformer_imbalance_arcs!(pm::_PMs.AbstractPowerModel; nw::Int=pm.cnw)
    _PMs.ref(pm, nw)[:arcs_bal] = Tuple{Int64,Int64, Int64}[]
    for arc in _PMs.ref(pm, nw, :arcs_trans)
        tr_id = arc[1]
        if length(_PMs.ref(pm, pm.cnw, :transformer, tr_id, "active_phases")) == 3
            push!(_PMs.ref(pm, nw)[:arcs_bal], arc) 
        end
    end
end

function variable_branch_be(pm::_PMs.AbstractPowerModel; nw::Int=pm.cnw)
    variable_branch_be_p(pm)
    variable_branch_be_q(pm)
end 

function variable_branch_be_p(pm::_PMs.AbstractPowerModel; nw::Int=pm.cnw)  
    _PMs.var(pm, nw)[:be_p] = JuMP.@variable(pm.model,  
    [(l,i,j) in _PMs.ref(pm, nw, :arcs_bal)], 
    base_name = "$(nw)_branch_be_p",
    binary = true, 
    start = 0)  
end  

function variable_branch_be_q(pm::_PMs.AbstractPowerModel; nw::Int=pm.cnw)  
    _PMs.var(pm, nw)[:be_q] = JuMP.@variable(pm.model,  
    [(l,i,j) in _PMs.ref(pm, nw, :arcs_bal)],  
    base_name = "$(nw)_branch_be_q",  
    binary = true,  
    start = 0) 
end

function constraint_branch_be(pm::_PMs.AbstractPowerModel, i::Int) 
    constraint_branch_be_p(pm, i)
    constraint_branch_be_q(pm, i)
end


function constraint_branch_be_p(pm::_PMs.AbstractPowerModel, i::Int; nw::Int=pm.cnw) 
    arcs = _PMs.ref(pm, nw, :arcs_bal)[i]
    be = _PMs.var(pm, nw, :be_p, (arcs)) 
    for cnd in _PMs.conductor_ids(pm)
        pt = _PMs.var(pm, nw, 1, :pt, arcs) 
        t = _PMs.ref(pm, nw, :transformer, arcs[1], "rate_a")[cnd]
        JuMP.@constraint(pm.model, pt <= t * (1 - be)) 
        JuMP.@constraint(pm.model, pt >= -t * be)
    end
end 

function constraint_branch_be_q(pm::_PMs.AbstractPowerModel, i::Int; nw::Int=pm.cnw) 
    arcs = _PMs.ref(pm, nw, :arcs_bal, i)
    be = _PMs.var(pm, nw, :be_q, arcs) 
    for cnd in _PMs.conductor_ids(pm)
        qt = _PMs.var(pm, nw, cnd, :qt, arcs) 
        t = _PMs.ref(pm, nw, :transformer, arcs[1], "rate_a")[cnd]
        JuMP.@constraint(pm.model, qt <= t * (1 - be)) 
        JuMP.@constraint(pm.model, qt >= -t * be)
    end
end 

function constraint_balance_p_flow(pm::_PMs.AbstractPowerModel, i::Int; nw::Int=pm.cnw)
    arcs = _PMs.ref(pm, nw, :arcs_bal, i)
    be = _PMs.var(pm, nw, :be_p, arcs)
    limit = _PMs.ref(pm, nw)[:trans_bal]
    lb_beta = 1 - limit
    ub_beta = 1 + limit
    pt = [ _PMs.var(pm, nw, c, :pt, arcs) for c in _PMs.conductor_ids(pm)]
    for cnd in _PMs.conductor_ids(pm)
        JuMP.@constraint(pm.model, pt[cnd] <= (lb_beta * be + ub_beta * (1 - be)) * sum(pt[c] for c in _PMs.conductor_ids(pm))/3)
        JuMP.@constraint(pm.model, pt[cnd] >= (lb_beta * (1 - be) + ub_beta * be) * sum(pt[c] for c in _PMs.conductor_ids(pm))/3)
    end
end
    
function constraint_balance_q_flow(pm::_PMs.AbstractPowerModel, i::Int; nw::Int=pm.cnw)
    arcs = _PMs.ref(pm, nw, :arcs_bal, i)
    be = _PMs.var(pm, nw, :be_q, arcs)
    limit = _PMs.ref(pm, nw)[:trans_bal]
    lb_beta = 1 - limit
    ub_beta = 1 + limit
    qt = [ _PMs.var(pm, nw, c, :qt, arcs) for c in _PMs.conductor_ids(pm)]
    for cnd in _PMs.conductor_ids(pm)
        JuMP.@constraint(pm.model, qt[cnd] <= (lb_beta * be + ub_beta * (1 - be)) * sum(qt[c] for c in _PMs.conductor_ids(pm))/3)
        JuMP.@constraint(pm.model, qt[cnd] >= (lb_beta * (1 - be) + ub_beta * be) * sum(qt[c] for c in _PMs.conductor_ids(pm))/3)
    end
end

function run_pf(data::Dict{String,Any}, model_type, solver; kwargs...)
    return _PMs.run_model(data, model_type, solver, build_mc_mld; multiconductor=true, ref_extensions=[_PMD.ref_add_arcs_trans!, ref_add_transformer_imbalance!, ref_add_vm_imbalance!, ref_add_transformer_imbalance_arcs!], kwargs...) 
 end
 
function build_mc_mld(pm::_PMs.AbstractPowerModel)   
     # copied from PowerModelsDistribution with operational variables and constraints added 
     _PMD.variable_mc_voltage(pm, bounded=false)
     _PMD.variable_mc_branch_flow(pm, bounded=false)
     _PMD.variable_mc_transformer_flow(pm, bounded=false)
     _PMD.variable_mc_generation(pm, bounded=false)  

    for i in _PMs.ids(pm, :bus)
        _PMD.constraint_mc_voltage_balance(pm, i)
    end

    # variable_branch_be(pm)
 
    _PMD.constraint_mc_model_voltage(pm)
         
    for (i,bus) in _PMs.ref(pm, :ref_buses)
        @assert bus["bus_type"] == 3
        _PMD.constraint_mc_theta_ref(pm, i)
        _PMD.constraint_mc_voltage_magnitude_setpoint(pm, i)
    end
    for (i,bus) in _PMs.ref(pm, :bus)
        _PMD.constraint_mc_power_balance(pm, i)
 
         # PV Bus Constraints
        if length(_PMs.ref(pm, :bus_gens, i)) > 0 && !(i in _PMs.ids(pm,:ref_buses))
             # this assumes inactive generators are filtered out of bus_gens
            @assert bus["bus_type"] == 2
            _PMD.constraint_mc_voltage_magnitude_setpoint(pm, i)
            for j in _PMs.ref(pm, :bus_gens, i)
                _PMD.constraint_mc_active_gen_setpoint(pm, j)
            end
        end
    end
    for i in _PMs.ids(pm, :branch)
        _PMD.constraint_mc_ohms_yt_from(pm, i)
        _PMD.constraint_mc_ohms_yt_to(pm, i)
    end
 
    for i in _PMs.ids(pm, :transformer)
        _PMD.constraint_mc_trans(pm, i)
    end
    
    # for i in _PMs.ids(pm, :arcs_bal)
    #     constraint_branch_be_p(pm, i)
    #     constraint_branch_be_q(pm, i)
    #     constraint_balance_p_flow(pm, i) 
    #     constraint_balance_q_flow(pm, i)
    # end
    println(pm)
    println(pm.model)
    # println(jjj)
end


data = _PMD.parse_file("../test/data/t_trans_2w_dy_lag.dss")
result = run_pf(data, _PMs.ACPPowerModel, juniper_solver)
println(result)

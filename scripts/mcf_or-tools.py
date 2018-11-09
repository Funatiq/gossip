#!/usr/bin/env python3

import numpy as np
from ortools.linear_solver import pywraplp
import json

from topology_parser import get_topology_matrix

capacities = get_topology_matrix("dgx1_topology.txt")

num_gpus = capacities.shape[0]

# T: num timesteps
T_max = num_gpus

# num_gpus = 8

# # capacities: num_gpus**2 
# if(num_gpus == 4):
#     capacities = np.eye(num_gpus) * num_gpus*2
#     capacities += np.array([[0,2,1,1],
#                             [2,0,1,1],
#                             [1,1,0,2],
#                             [1,1,2,0]])
# if(num_gpus == 8):
#     capacities = np.eye(num_gpus) * num_gpus*2
#     capacities += np.array([[0,1,1,2,2,0,0,0],
#                             [1,0,2,1,0,2,0,0],
#                             [1,2,0,2,0,0,1,0],
#                             [2,1,2,0,0,0,0,1],
#                             [2,0,0,0,0,1,1,2],
#                             [0,2,0,0,1,0,2,1],
#                             [0,0,1,0,1,2,0,2],
#                             [0,0,0,1,2,1,2,0]])

for T in range(1, T_max+1):
    print("Creating flow problem for %i timesteps" %(T))

    edges_per_timestep = num_gpus*num_gpus
    num_edges = edges_per_timestep*T

    num_commodities = num_gpus
    flows_per_gpu = num_gpus*num_commodities
    flows_per_timestep = flows_per_gpu*num_gpus
    num_flows = flows_per_timestep*T

    # x: gpu x
    # s: source
    # t: target
    # tau: timestep


    # enumeration of vertices V
    # x,tau: tau*num_gpus + x

    # enumeration of flows
    # commodity i
    # vertex v
    # v,i: v*num_gpus + i

    # enumeration of edges E
    # s, t, tau: tau*num_gpus**2 + s*num_gpus + t

    # commidities K_i = (S_i, t_i, d_i)
    # K_i = (range(num_gpus), i, num_gpus) for i in range(num_gpus)

    # (1) Link capacity: The sum of all flows routed over a link does not exceed its capacity.

    edge_capacities = capacities.reshape(1,-1)
    # print(edge_capacities)
    edge_capacities = np.repeat(edge_capacities, T, axis=0).flatten()
    # print(edge_capacities)
    # print(edge_capacities.shape, num_edges)
    assert(edge_capacities.shape[0] == num_edges)

    edge_coeffs = np.kron(np.eye(num_edges), np.ones(num_gpus))
    # print(edge_coeffs.shape)
    assert(edge_coeffs.shape[0] == num_edges)
    assert(edge_coeffs.shape[1] == num_flows)


    # (2) Flow conservation on transit nodes: The amount of a flow entering an intermediate node u {\displaystyle u} u is the same that exits the node.
    # transit is both source and destination
    # (3) Flow conservation at the source: A flow must exit its source node completely.
    out_flow_coeff = np.kron(np.eye(num_gpus), np.kron(np.ones(num_gpus), np.eye(num_commodities)))
    # print(out_flow_coeff.shape)
    assert(out_flow_coeff.shape[0] == flows_per_gpu)
    assert(out_flow_coeff.shape[1] == flows_per_timestep)
    # print(out_flow_coeff)

    # (4) Flow conservation at the destination: A flow must enter its sink node completely.
    in_flow_coeff = np.kron(np.ones(num_gpus), -1*np.eye(num_gpus**2))
    # print(in_flow_coeff.shape)
    assert(in_flow_coeff.shape[0] == flows_per_gpu)
    assert(in_flow_coeff.shape[1] == flows_per_timestep)
    # print(in_flow_coeff)

    # coeffs for all timesteps
    out_flow_coeffs = np.kron(np.eye(T), out_flow_coeff)
    in_flow_coeffs = np.kron(np.eye(T), in_flow_coeff)
    # print(out_flow_coeffs)
    # print(in_flow_coeffs)
    assert(out_flow_coeffs.shape[1] == num_flows)
    assert(in_flow_coeffs.shape[1] == num_flows)

    # combine coeffs in one matrix
    in_out_coeffs = np.zeros((flows_per_gpu*(T+1),num_flows))
    # print(in_out_coeffs.shape)
    # first <flows_per_gpu> flows only outgoing = sources
    in_out_coeffs[:flows_per_gpu*T,:] = out_flow_coeffs
    # last <flows_per_gpu> flows only incoming = destinations
    in_out_coeffs[flows_per_gpu:,:] += in_flow_coeffs
    # print(in_out_coeffs)
    # print(np.where(in_out_coeffs>0))

    in_out_bounds = np.zeros(flows_per_gpu*(T+1))
    # each source starts with 1 commodity of each type
    in_out_bounds[:flows_per_gpu] = 1
    # each destination expects <num_gpu> commodities of its own type
    for i in range(num_gpus):
        in_out_bounds[-flows_per_gpu+i*num_gpus+i] = -1*num_gpus
    # in_out_bounds[-flows_per_gpu:] = -1
    # print(in_out_bounds)


    # () space constraints: each gpu can hold exactly <num_gpus> commodities
    space_coeff = np.kron(np.eye(T * num_gpus), np.ones(num_gpus * num_commodities))
    space_bounds = [num_gpus] * (T * num_gpus)


    # costs
    # costs = np.ones(num_flows)
    # costs antiproportinal to edge_capacities (bandwidth)
    costs = np.where(edge_capacities > 0, 1 / edge_capacities, np.ones(len(edge_capacities)))
    costs = np.kron(costs, np.ones(num_commodities))
    # zero cost for copy on same gpu
    for i in range(num_gpus):
        begin = i*num_gpus*num_commodities + i*num_commodities
        end = begin + num_commodities
        for t in range(T):
            offset = flows_per_timestep*t
            costs[offset+begin:offset+end] = np.zeros(num_commodities)


    # solver = pywraplp.Solver('mcf', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    solver = pywraplp.Solver('mcf', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    # solver = pywraplp.Solver('mcf', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
    # solver = pywraplp.Solver('mcf', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)

    flows = []
    for t in range(T):
        for src in range(num_gpus):
            for trg in range(num_gpus):
                for c in range(num_commodities):
                    name = 't'+str(t)+'_'+str(src)+'to'+str(trg)+'_c'+str(c)
                    # flows.append(solver.NumVar(0.0, solver.infinity(), name))
                    flows.append(solver.IntVar(0.0, solver.infinity(), name))

    objective = solver.Objective()
    for i in range(num_flows):
        objective.SetCoefficient(flows[i], costs[i])
    objective.SetMinimization()

    edge_constraints = [0] * num_edges
    for i in range(num_edges):
        edge_constraints[i] = solver.Constraint(-solver.infinity(), edge_capacities[i])
        for j in range(num_flows):
            edge_constraints[i].SetCoefficient(flows[j], edge_coeffs[i,j])

    conservation_constraints = [0] * (flows_per_gpu * (T+1))
    for i in range(len(conservation_constraints)):
        conservation_constraints[i] = solver.Constraint(in_out_bounds[i], in_out_bounds[i])
        for j in range(num_flows):
            conservation_constraints[i].SetCoefficient(flows[j], in_out_coeffs[i,j])

    space_constraints = [0] * (num_gpus * T)
    for i in range(len(space_constraints)):
        space_constraints[i] = solver.Constraint(-solver.infinity(), space_bounds[i])
        for j in range(num_flows):
            space_constraints[i].SetCoefficient(flows[j], space_coeff[i,j])

    status = solver.Solve()

    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())


    if status != solver.OPTIMAL:
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found\n.')
        else:
            print('The solver could not solve the problem in %i timesteps.\n' % (T))
        continue
    else:
        print('A solution was found:')
        print("transfers:", sum([flows[i].solution_value() for i in range(num_flows)]))
        print("cost:", sum([flows[i].solution_value()*costs[i] for i in range(num_flows)]))

        flows_array = np.array([flows[i].solution_value() for i in range(num_flows)])
        flows_array = flows_array.reshape((T, num_gpus, num_gpus, num_commodities))

        # print all flows
        for t in range(T):
            print("\ntimestep",t)
            for gpu in range(num_gpus):
                print("form gpus",gpu,"send commodity (column) to gpu (row)")
                print(flows_array[t,gpu])

        # print sequence of owners per commodity
        plan = []
        for gpu in range(num_gpus):
            for c in range(num_commodities):
                owners = [gpu]
                for t in range(T):
                    owner = owners[-1]
                    new_owner = np.where(flows_array[t,owner,:,c] > 0)[0][0]
                    flows_array[t,owner,new_owner,c] -= 1
                    owners.append(int(new_owner))
                # print(owners)
                plan.append(owners)
        # for p in plan:
        #     print(p)

        # topo_hash = hash(capacities.tostring())

        data = {
            "num_gpus" : num_gpus,
            "num_steps" : T,
            # "topo_hash" : topo_hash,
            "plan" : plan
        }

        json_string = json.dumps(data)
        print(json_string)
        with open("plan.json", "w") as file:
            json.dump(data, file)

        break


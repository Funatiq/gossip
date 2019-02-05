#!/usr/bin/env python3

import numpy as np
from ortools.linear_solver import pywraplp
import json

from topology_parser import get_topology_matrix

# dgx1 volta: 6 nvlink per gpu
capacities = get_topology_matrix("dgx1_topology.txt")
parts_per_commodity = 3

# ps0001 pascal: 4 nvlink per gpu
# capacities = get_topology_matrix("ps0001_topology.txt")
# parts_per_commodity = 4


# ps0001 pascal: 4 nvlink per gpu
# num_gpus = 4
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,2,1,1],
#                         [2,0,1,1],
#                         [1,1,0,2],
#                         [1,1,2,0]])

# like ps0001 but volta: 6 nvlink per gpu
# num_gpus = 4
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,2,2,2],
#                         [2,0,2,2],
#                         [2,2,0,2],
#                         [2,2,2,0]])

# dgx1 volta: 6 nvlink per gpu
# num_gpus = 8
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,1,1,2,2,0,0,0],
#                         [1,0,2,1,0,2,0,0],
#                         [1,2,0,2,0,0,1,0],
#                         [2,1,2,0,0,0,0,1],
#                         [2,0,0,0,0,1,1,2],
#                         [0,2,0,0,1,0,2,1],
#                         [0,0,1,0,1,2,0,2],
#                         [0,0,0,1,2,1,2,0]])

# like dgx1 volta: 6 nvlink per gpu, different ring structure
# num_gpus = 8
# parts_per_commodity = 1
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,2,1,1,2,0,0,0],
#                         [2,0,1,1,0,2,0,0],
#                         [1,1,0,2,0,0,2,0],
#                         [1,1,2,0,0,0,0,2],
#                         [2,0,0,0,0,1,1,2],
#                         [0,2,0,0,1,0,2,1],
#                         [0,0,2,0,1,2,0,1],
#                         [0,0,0,2,2,1,1,0]])
# capacities += np.array([[0,1,1,1,2,0,0,0],
#                         [1,0,1,1,0,2,0,0],
#                         [1,1,0,1,0,0,2,0],
#                         [1,1,1,0,0,0,0,2],
#                         [2,0,0,0,0,1,1,1],
#                         [0,2,0,0,1,0,1,1],
#                         [0,0,2,0,1,1,0,1],
#                         [0,0,0,2,1,1,1,0]])
# capacities += np.array([[0,1,1,1,1,1,0,1],
#                         [1,0,1,1,1,1,1,0],
#                         [1,1,0,1,0,1,1,1],
#                         [1,1,1,0,1,0,1,1],
#                         [1,1,0,1,0,1,1,1],
#                         [1,1,1,0,1,0,1,1],
#                         [0,1,1,1,1,1,0,1],
#                         [1,0,1,1,1,1,1,0]])

num_gpus = capacities.shape[0]

print("topology:")
print(capacities)

num_commodities = num_gpus
max_steps = num_gpus

modes = {"scatter":0, "gather":1, "all2all":2}
mode = "all2all"
if modes[mode] == 0: # scatter
    # one gpu starts with one of each commodity
    src = 0
    commodities_in = np.zeros((num_gpus,num_gpus))
    commodities_in[src,:] += np.ones(num_gpus) * parts_per_commodity
elif modes[mode] == 1: # gather
    trg = 0
    commodities_in = np.zeros((num_gpus,num_gpus))
    commodities_in[:,trg] += np.ones(num_gpus) * parts_per_commodity
elif modes[mode] == 2: # all-to-all
    # each gpu starts with one of each commodity
    commodities_in = np.ones((num_gpus,num_gpus)) * parts_per_commodity
else:
    raise SystemExit()

# each gpu wants to have all of its own commodity
commodities_out = np.diagflat( np.sum(commodities_in, axis=0) )
print("commodities at begin:")
print(commodities_in)
print("commodities at end:")
print(commodities_out)
# max number of items that can be transferred per edge
max_multiplicity = num_commodities * parts_per_commodity
# storage size of each gpu in number of items
max_space_per_gpu = num_commodities * parts_per_commodity


for steps in range(1, max_steps+1):
    print("Creating flow problem for %i timesteps" %(steps))

    # edges_per_timestep = num_gpus*num_gpus
    # num_edges = edges_per_timestep*steps

    flows_per_gpu = num_gpus*num_commodities
    # flows_per_timestep = flows_per_gpu*num_gpus
    # num_flows = flows_per_timestep*steps

    # solver = pywraplp.Solver('mcf', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    solver = pywraplp.Solver('mcf', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    # solver = pywraplp.Solver('mcf', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
    # solver = pywraplp.Solver('mcf', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)

    objective = solver.Objective()
    objective.SetMinimization()

    # (1) conservation_constraints for each commodity flow at each gpu
    conservation_constraints = [0] * (flows_per_gpu * (steps+1))
    # Flow conservation on transit nodes: The amount of a flow entering is the same that exits the node.
    in_out_bounds = np.zeros(flows_per_gpu*(steps+1))
    # Flow conservation at the source: A flow must exit its source node completely.
    in_out_bounds[:flows_per_gpu] = commodities_in.flatten()
    # Flow conservation at the destination: A flow must enter its sink node completely.
    in_out_bounds[-flows_per_gpu:] = -1*commodities_out.flatten()

    for i in range(len(conservation_constraints)):
        conservation_constraints[i] = solver.Constraint(in_out_bounds[i], in_out_bounds[i])

    # (2) space constraints: each gpu can hold exactly <num_gpus> commodities
    space_constraints = [0] * (num_gpus * steps)
    for i in range(len(space_constraints)):
        space_constraints[i] = solver.Constraint(-solver.infinity(), max_space_per_gpu)

    flows = []
    # create flows for each edge for each step
    for step in range(steps):
        for src in range(num_gpus):
            for trg in range(num_gpus):
                if capacities[src][trg] == 0:
                    continue

                # copying edges
                if src == trg:
                    # copy at most num_commodities items
                    edge_capacity = max_multiplicity
                    cost = 0
                    multiplicity = 1
                # sending edges
                if (src != trg):
                    edge_capacity = 1
                    cost = 1/capacities[src][trg]
                    multiplicity = max_multiplicity

                for m in range(multiplicity):
                    # (3) Link capacity: The sum of all flows routed over a link does not exceed its capacity.
                    edge_constraint = solver.Constraint(-solver.infinity(), edge_capacity)
                    for c in range(num_commodities):
                        name = 't'+str(step)+' '+str(src)+'to'+str(trg)+' i'+str(m)+' c'+str(c)
                        flow = solver.IntVar(0, edge_capacity, name)
                        flows.append(flow)
                        # increase cost with multiplicity
                        if(cost > 0):
                            objective.SetCoefficient(flow, cost*(m+1) + step / 100)
                        else:
                            objective.SetCoefficient(flow, 0)
                        # sum flows of same edge
                        edge_constraint.SetCoefficient(flow, 1)
                        # sum flows of same gpu
                        space_constraints[step*num_gpus+src].SetCoefficient(flow, 1)
                        # outgoing flow at src
                        conservation_constraints[step*flows_per_gpu+src*num_commodities+c].SetCoefficient(flow, 1)
                        # incoming flow at trg
                        conservation_constraints[(step+1)*flows_per_gpu+trg*num_commodities+c].SetCoefficient(flow, -1)


    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())

    status = solver.Solve()

    if status != solver.OPTIMAL:
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found\n.')
        else:
            print('The solver could not solve the problem in %i timesteps.\n' % (steps))
        continue
    else:
        print('A solution was found:')

        copies = 0
        transfers = 0

        flows_array = np.zeros((steps, num_gpus, num_gpus, num_commodities))

        step_time = np.zeros(steps)

        for flow in flows:
            # print(flow, flow.solution_value(), objective.GetCoefficient(flow))
            step, edge, _, commodity = str(flow).split()
            step = int(step[1:])
            src, trg = edge.split("to")
            src = int(src)
            trg = int(trg)
            commodity = int(commodity[1:])
            value = flow.solution_value()
            if (src == trg) and (value > 0):
                copies += 1
            if (src != trg) and (value > 0):
                transfers += 1
                step_time[step] = max(step_time[step], objective.GetCoefficient(flow) )
            flows_array[step, src, trg, commodity] += value

        step_time /= parts_per_commodity
        print("copies:", copies)
        print("transfers:", transfers)
        print("time for each step:", step_time)
        print("total time:", np.sum(step_time))

        # print all flows
        for step in range(steps):
            print("\nstep",step)
            print(np.sum(flows_array[step], axis=2))
            # for gpu in range(num_gpus):
                # print("from gpus",gpu,"send commodity (column) to gpu (row)")
                # print(flows_array[step,gpu])


        # trace sequence of owners per commodity
        plan = []
        while np.any(flows_array[0] > 0):
            owner, new_owner, commodity = np.transpose(np.nonzero(flows_array[0]))[0]
            flows_array[0,owner,new_owner,commodity] -= 1
            owners = [owner,new_owner]

            for step in range(1,steps):
                owner = owners[-1]
                new_owner = np.nonzero(flows_array[step,owner,:,commodity])[0][0]
                flows_array[step,owner,new_owner,commodity] -= 1
                owners.append(int(new_owner))
            # print(owners)
            plan.append(owners)
        # for p in plan:
        #     print(p)

        plan_unique, counts = np.unique(plan, return_counts=True, axis=0)
        # plan_reduced = {}
        # for v,c in zip(plan_unique.tolist(), counts.tolist()):
        #     plan_reduced[tuple(v)] = c
        # for p in plan_reduced:
        #     print(p)
        print("num paths:", len(counts))

        count_usage = np.zeros((steps,num_gpus,num_gpus),dtype=int)

        for v in plan_unique:
            for i in range(len(v)-1):
                count_usage[i,v[i],v[i+1]] += 1

        max_usage = np.max(count_usage-np.eye(num_gpus)*num_gpus, axis=(1,2)).astype(int)

        # print(count_usage)
        # print(max_usage)

        steps_expanded = [np.zeros((m+1,num_gpus,num_gpus),dtype=bool) for m in max_usage]
        plan_expanded = []

        for v in plan_unique.tolist():
            v_expanded = [v[0]]
            for i in range(steps):
                if v[i] != v[i+1]:
                    for j in range(max_usage[i]):
                        if not steps_expanded[i][j,v[i],v[i+1]]:
                            steps_expanded[i][j,v[i],v[i+1]] = True
                            v_expanded.append(v[i+1])
                            break
                        else:
                            v_expanded.append(v[i])
                while len(v_expanded) < np.sum(max_usage[:i+1])+1:
                    v_expanded.append(v_expanded[-1])
            plan_expanded.append(v_expanded)

        # for v in plan_expanded:
        #     print(v)

        data = {
            "type" : mode,
            "num_gpus" : num_gpus,
            # "num_steps" : steps,
            "num_steps" : int(np.sum(max_usage)),
            "num_chunks" : parts_per_commodity,
            # "topo_hash" : topo_hash,
            "plan" : plan_unique.tolist(),
            # "plan" : plan_expanded,
            "chunks" : counts.tolist(),
            "sync_steps" : max_usage[:-1].tolist()
        }

        json_string = json.dumps(data)
        print(json_string)
        json_name = mode+"_plan.json"
        print("saving json to '%s'" %(json_name))
        with open(json_name, "w") as file:
            json.dump(data, file)

        break


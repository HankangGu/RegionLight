from gurobipy import *
import networkx as nx
import itertools
import numpy as np

def get_neighboring_itsx(x, y, row, col):
    """
    :param x: x-coordinate for the target node
    :param y: y-coordinate for the target node
    :param row: boundary of x
    :param col: boundary of y

    :return: list of neighboring intersections in boudary
    """
    x_incre_list = [0, -1, 1, 0]
    y_incre_list = [1, 0, 0, -1]
    ans = []
    for x_incre, y_incre in zip(x_incre_list, y_incre_list):
        neighbour_x = x + x_incre
        neighbour_y = y + y_incre
        if neighbour_x < 0 or neighbour_x > row - 1 or neighbour_y < 0 or neighbour_y > col - 1:
            pass
        else:
            ans.append(neighbour_x * col + neighbour_y)
    return ans


def minimum_dominating_set(row, col):
    sol = linear_opt_prog(row, col)
    centers = [itsx for itsx, v in sol.items() if v == 0]
    print(centers)
    centers_coordinate = []
    for center in centers:
        x = int(center.split('_')[1])
        y = int(center.split('_')[2])
        centers_coordinate.append((x,y))
    return centers_coordinate

def linear_opt_prog(row, col):
    """
    Now this function only support grid network, find !one! minimum dominating set
    :param row: how many row
    :param col: how many col

    """

    m = Model()  # gurobi model
    m.setParam('OutputFlag', 0)
    var_list = []
    for x, y in itertools.product(range(row), range(col)):  # initialise decision variable
        v = x * col + y
        var = m.addVar(vtype=GRB.BINARY, name='intersection_' + str(x + 1) + '_' + str(y + 1))
        var_list.append(var)
        m.update()
    m.setObjective(quicksum(var_list), GRB.MAXIMIZE)  # the sum of all variable
    for x, y in itertools.product(range(row), range(col)):  # build constraint
        neigh_nodes = get_neighboring_itsx(x, y, row, col)
        var_related = []  # get neighboring varibles
        var_related.append(var_list[x * col + y])
        for node in neigh_nodes:
            var_related.append(var_list[node])
        m.addConstr(quicksum(var_related) <= len(neigh_nodes))  # constraint
    m.optimize()
    print("domination number ", row*col-m.ObjVal)  # the size of minimum dominating set
    variable_dict = {v.VarName: v.X for v in var_list}
    # print(m.Solution)
    return variable_dict

def verify_region_assignment(row,col,assignment):
    """
    verify whether the assignemtn fit constraint: union and disjoint
    :param row:
    :param col:
    :param assignment:
    :return:
    """
    itsx_union=set()
    for region in assignment:
        for itsx in region:
            if itsx=='dummy':
                continue
            if itsx_union.__contains__(itsx):
                raise Exception("Joint Region")
            else:
                itsx_union.add(itsx)

    if len(itsx_union)!=row*col:
        raise Exception("Missing itsx")
    print("verified good")

def construct_configuration(centers_coordinates, row, col, shuffle=False):
    """

    :param centers_coordinates: centers_coordinates for each region
    :param row: how many row
    :param col: how many col

    :return: region configuration
    """

    if shuffle: # for centers failed to satisfy theorem1 in manuscript
        # we can get different region configuration by trying different iteration order
        shuffle_order = np.arange(0, len(centers_coordinates))
        np.random.shuffle(shuffle_order)
        centers_coordinates=[centers_coordinates[i] for i in shuffle_order]
    assignment = []
    assigned_track = []
    for (i, j) in centers_coordinates:
        itsx_id = 'intersection_' + str(i) + '_' + str(j)
        assigned_track.append(itsx_id)
    cre = [[0, 1], [-1, 0], [0, 0], [1, 0], [0, -1]]

    for (i, j) in centers_coordinates:
        itsx = []
        for x, y in cre:
            itsx_id = 'intersection_' + str(i + x) + '_' + str(j + y)
            if x == 0 and y == 0:
                itsx.append(itsx_id)
            else:
                if i + x < 1 or i + x > row or j + y < 1 or j + y > col:
                    itsx.append('dummy')
                else:

                    if itsx_id in assigned_track:
                        itsx.append('dummy')
                    else:
                        itsx.append(itsx_id)
                        assigned_track.append(itsx_id)
        assignment.append(itsx)
    # print(assignment)
    return assignment


if __name__ == "__main__":
    row = 16
    col = 3
    centers_coordinate = minimum_dominating_set(row, col)
    print(centers_coordinate)
    region_assignment = construct_configuration(centers_coordinate, row, col,shuffle=True)
    print(region_assignment)
    verify_region_assignment(row,col,region_assignment)

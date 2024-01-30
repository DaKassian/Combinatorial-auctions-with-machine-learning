import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, PULP_CBC_CMD


def parse_bids(bids, num_bids, num_goods):
    parsed_bids = []
    for i in range(num_bids):
        price = bids[i * (num_goods + 1)]
        goods = bids[i * (num_goods + 1) + 1: (i + 1) * (num_goods + 1)]
        parsed_bids.append((price, goods))
    return parsed_bids


def solve(bids, num_goods, num_bids, debug=False):
    if debug:
        print(bids)
    model = LpProblem(name="Combinatorial_Auction", sense=LpMaximize)
    bid_vars = LpVariable.dicts("Bid", range(num_bids), cat='Binary')
    model += lpSum([bids[i][0] * bid_vars[i] for i in range(num_bids)])
    for i in range(num_goods):
        model += lpSum([bids[j][i + 1] * bid_vars[j] for j in range(num_bids)]) <= 1

    if debug:
        model.solve()
    else:
        model.solve(PULP_CBC_CMD(msg=False))  # Suppress solver messages
    chosen_bids = [1 if bid_vars[i].value() == 1 else 0 for i in range(num_bids)]

    if debug:
        print(f"Status: {LpStatus[model.status]}")
        print(f"Objective: {model.objective.value()}")
        print(f"Bids:")
        for i in range(num_bids):
            print(f"{bid_vars[i].name}: {bid_vars[i].value()}")
            if bid_vars[i].value() == 1:
                print(f"Price: {bids[i][0]}")
                print(f"Goods: {bids[i][1:]}")
            else:
                print(f"Goods: {bids[i][1:]}")
        print(f"Number of selected bids: {sum(chosen_bids)}")

    return np.array(chosen_bids)

def compute_costs(revenue, var_pct, fixed, inflation=0.0):
    var_cost = revenue * var_pct
    fixed_cost = fixed * (1 + inflation)
    return var_cost + fixed_cost

def calculate_wacc(equity, debt, ke, kd, tax):
    return equity * ke + debt * kd * (1 - tax)

import os
import sys

import pandas as pd

DEMAND_TYPES = ["Arrival", "Departure"]

def main():
    if len(sys.argv) != 2:
        print("Please provide a dataset name. python3 scripts/create_ar_rules.py DATASET_NAME")

    lags_df = pd.read_csv("data/" + sys.argv[1] + "/local_models/global_AR_coefs.txt", sep="\t")

    coef_cols = lags_df.columns[1:-1]

    # 0.05322886435424773: DepartureDemand(S, T) - DepartureDemand(S, T_Lag2) + 0.0 * Lag2(T, T_Lag2) = 0.0 ^2
    for demand_type in DEMAND_TYPES:
        print("#" + demand_type)

        demand_lag_df = lags_df[lags_df["DemandType"] == demand_type]

        for col in coef_cols:
            weight = str(demand_lag_df[col].values[0])
            # Minus sign in rule for positive coefficients, flipped weight and plus sign for negative coefficients.
            sign = "-"

            if weight[0] == "-":
                sign = "+"
                weight = weight[1:]

            # Omit 0-weight rules
            if weight == "0.0":
                continue

            print(weight + ": " + demand_type + "Demand(S, T) " + sign + " " + demand_type + "Demand(S, T_" + col + ") + 0.0 * " + str(col) + "(T, T_" + col + ") = 0.0 ^2")

        bias = float(demand_lag_df["Bias"])
        if bias > 0:
            print(str(bias/2) + ": " + demand_type + "Demand(S, T) = 1 ^2")
        elif bias < 0:
            print(str(-1 * bias / 2) + ": " + demand_type + "Demand(S, T) = 0 ^2")

        print("\n")


if __name__ == '__main__':
    main()



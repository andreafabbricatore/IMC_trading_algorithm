import pandas as pd

df = pd.read_csv(
    "island-data-bottle-round-1/island-data-bottle-round-1/prices_round_1_day_0.csv",
    delimiter=";",
)

pearls_df = df[df["product"] == "PEARLS"]
print(min(pearls_df["ask_price_1"]), max(pearls_df["bid_price_1"]))

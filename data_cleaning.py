from pathlib import Path
import pandas as pd

data_folder = Path(r"C:\Users\angka\OneDrive\Documents\lol_worlds_predictor")

df_24 = pd.read_csv(data_folder / "2024_LoL_esports_match_data_from_OraclesElixir.csv", low_memory = False)
df_25 = pd.read_csv(data_folder / "2025_LoL_esports_match_data_from_OraclesElixir.csv", low_memory = False)
df_26 = pd.read_csv(data_folder / "2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory = False)
df = pd.concat([df_24, df_25, df_26], ignore_index = True)


df = df[
    (df["position"] == "team") &
    (df["league"].isin(["LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL", "DCGI", "EWC", "MSI", "WLDs"])) &

    ]



print(df.head())
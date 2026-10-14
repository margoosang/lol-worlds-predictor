from pathlib import Path
import pandas as pd

data_folder = Path(__file__).resolve().parent

df_24 = pd.read_csv(data_folder / "2024_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
df_25 = pd.read_csv(data_folder / "2025_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
df_26 = pd.read_csv(data_folder / "2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
df = pd.concat([df_24, df_25, df_26], ignore_index=True)
df = df[
    (df["position"] == "team") &
    (df["league"].isin(["LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL", "DCGI", "EWC", "MSI", "WLDs"]))
]
df["teamname"] = df["teamname"].replace({r"LØS": "LOS", r"^QT D.*": "QT DIG"}, regex=True)
df["year"] = df["year"].replace({2027: 2026})


blue = df[df["side"] == "Blue"].copy()
red = df[df["side"] == "Red"].copy()
blue = blue[["gameid", "league", "year", "playoffs", "date", "game", "teamname", "result"]].rename(
    columns={
    "teamname": "team_a",
    "result": "result_a"
})
red = red[["gameid", "teamname", "result"]].rename(columns={
    "teamname": "team_b",
    "result": "result_b"
})

clean_df = blue.merge(red, on="gameid", how="inner", validate="one_to_one")
def classify_stage(row):
    if row["league"] == "WLDs":
        return "worlds"
    elif row["league"] == "MSI":
        return "msi"
    elif row["league"] == "DCGI":
        return "dcgi"
    elif row["league"] == "EWC":
        return "ewc"
    elif row["playoffs"] == 1:
        return "playoffs"
    else:
        return "reg_szn"
clean_df["stage"] = clean_df.apply(classify_stage, axis=1)
clean_df["date"] = pd.to_datetime(clean_df["date"])
clean_df = clean_df.sort_values("date", kind="stable").reset_index(drop=True)

clean_df.to_csv(data_folder / "cleaned_data.csv", index=False)


from pathlib import Path
import pandas as pd

data_folder = Path(__file__).resolve().parent
games = pd.read_csv(data_folder / "cleaned_data.csv")

base_rating = 1500 # every team starts at this elo when initialised
off_szn_retention = 0.8  # accounting for roster changes and lack of practice in the off-season
k_factor = 24
stage_mults = {
    "reg_szn": 1.0,
    "playoffs": 1.1,
    "dcgi": 1.2,
    "ewc": 1.2,
    "msi": 1.25,
    "worlds": 1.3,
    } 

def expected_score(rating_a, rating_b):
    """Team A's probability of winning the game against Team B."""

    x = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return x

def off_szn_regression(ratings, retention):
    """Regress existing team ratings toward the base rating."""

    for team in ratings:
        ratings[team] = (base_rating + retention * (ratings[team] - base_rating))

def update_elo(rating_a, rating_b, result_a, stage, k_factor, stage_mults):
    """Update ELOs of both teams after 1 game. result_a = 1 if Team A won, 0 if Team A lost."""

    expected_a = expected_score(rating_a, rating_b)
    mult = stage_mults[stage]
    eff_k = k_factor * mult

    rating_change = eff_k * (result_a - expected_a) # rating change with respect to Team A
    new_a = rating_a + rating_change
    new_b = rating_b - rating_change

    return new_a, new_b

def run_elo(games):
    """Process games chronologically and update team Elo ratings."""

    ratings = {}
    predictions = []
    current_year = None

    for _, game in games.iterrows():
        team_a = game["team_a"]
        team_b = game["team_b"]
        result_a = game["result_a"]
        stage = game["stage"]
        year = int(game["year"])

        if current_year is not None and year != current_year:
            off_szn_regression(ratings, off_szn_retention)
        current_year = year

        ratings.setdefault(team_a, base_rating)
        ratings.setdefault(team_b, base_rating)
        rating_a = ratings[team_a]
        rating_b = ratings[team_b]

        actual_winner = team_a if result_a == 1 else team_b
        expected_a = expected_score(rating_a, rating_b)
        if expected_a > 0.5:
            predicted_winner = team_a
        elif expected_a < 0.5:
            predicted_winner = team_b
        else:
            predicted_winner = "Evenly Matched"
        if predicted_winner == "Evenly Matched":
            success_predict = pd.NA
        elif predicted_winner == actual_winner:
            success_predict = 1
        else:
            success_predict = 0

        predictions.append({
            "gameid": game["gameid"],
            "stage": game["stage"],
            "year": game["year"],
            "date": game["date"],
            "game": game["game"],
            "team_a": team_a,
            "team_b": team_b,
            "predicted_result": predicted_winner,
            "actual_result": actual_winner,
            "correct_prediction": success_predict,
            "expected_a": expected_a,
            "result_a": game["result_a"]
        })

        new_a, new_b = update_elo(rating_a, rating_b, result_a, stage, k_factor, stage_mults)
        ratings[team_a] = new_a
        ratings[team_b] = new_b

    return ratings, predictions


ratings, predictions = run_elo(games)
ratings_df = pd.DataFrame(ratings.items(), columns=["team", "elo"])
predictions_df = pd.DataFrame(predictions)
ratings_df.to_csv(data_folder / "team_ratings.csv", index=False)
predictions_df.to_csv(data_folder / "historical_predictions.csv", index=False)


def predict_match(team_a, team_b, ratings, best_of=5):
    """Predict Team A's probability of winning a Bo1, Bo3 or Bo5."""

    rating_a = ratings[team_a]
    rating_b = ratings[team_b]

    p = expected_score(rating_a, rating_b)

    if best_of == 1:
        return p

    elif best_of == 3:
        return 3 * p**2 - 2 * p**3

    elif best_of == 5:
        return 10 * p**3 * (1-p)**2 + 5 * p**4 * (1-p) + p**5

    else:
        raise ValueError("best_of must be 1, 3 or 5")

def predict_score(team_a, team_b, best_of):
    """Return the most probable exact series score."""

    p = expected_score(
        ratings[team_a],
        ratings[team_b]
    )

    q = 1 - p

    if best_of == 1:
        scores = {
            (1, 0): p,
            (0, 1): q
        }

    elif best_of == 3:
        scores = {
            (2, 0): p**2,
            (2, 1): 2 * p**2 * q,
            (0, 2): q**2,
            (1, 2): 2 * q**2 * p
        }

    elif best_of == 5:
        scores = {
            (3, 0): p**3,
            (3, 1): 3 * p**3 * q,
            (3, 2): 6 * p**3 * q**2,
            (0, 3): q**3,
            (1, 3): 3 * q**3 * p,
            (2, 3): 6 * q**3 * p**2
        }

    else:
        raise ValueError("Invalid match format")

    predicted_score = max(scores, key=scores.get)

    return predicted_score, scores[predicted_score]
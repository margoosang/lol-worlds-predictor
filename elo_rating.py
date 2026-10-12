import pandas as pd


### shared
base_rating = 1500


### elo rating system
def expected_score(rating_a, rating_b):
    """ Team A's probability of winning 1 game against Team B. """
    x = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return x


k_factor = 32
stage_mults = {
    "reg_szn" : 1.0,
    "playoffs" : 1.2,
    "msi" : 1.4,
    "worlds" : 1.5,
    } 

def update_elo(rating_a, rating_b, result_a, stage, k_factor, stage_mults):
    """ Update ELOs of both teams after 1 game. result_a = 1 if Team A won, 0 if Team A lost. """
    exp_a = expected_score(rating_a, rating_b)
    mult = stage_mults[stage]
    eff_k = k_factor * mult

    rating_change = eff_k * (result_a - exp_a) # with respect to Team A
    new_a = rating_a + rating_change
    new_b = rating_b - rating_change

    return new_a, new_b

def run_elo(games):
    """Process games chronologically and update team Elo ratings."""

    ratings = {}
    predictions = []

    games = games.sort_values("date", kind = "stable")

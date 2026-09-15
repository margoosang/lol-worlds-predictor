gitimport pandas as pd


### configs
base_rating = 1500

models = {
    "model_1" : {
        "k_factor" : 32,
        "stage_mults" : {
            "reg_szn" : 1.0,
            "playoffs" : 1.2,
            "msi" : 1.4,
            "worlds" : 1.5,
        },
        "mov" : False,
    },

    
}

### elo rating system
def expected_score(rating_a, rating_b):
    x = 1 / (1 + 10**((rating_b - rating_a) / 400))
    return x 
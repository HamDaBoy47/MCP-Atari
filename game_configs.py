ATARI_CONFIGS = {
    "MsPacman": {
        "random_score": 307.3,
        "human_score": 15693,
        "success_threshold": 0.75,
        "completion_metric": None
    },
    "Breakout": {
        "random_score": 1.7,
        "human_score": 30.5,
        "success_threshold": 0.75,
        "completion_metric": "lives"
    },
    "Gravitar": {
        "random_score": 173,
        "human_score": 3351,
        "success_threshold": 0.75,
        "completion_metric": "lives"
    },
    "MarioBros": {
        "random_score": 308,
        "human_score": 28800,
        "success_threshold": 0.75,
        "completion_metric": "lives"
    },
    "DonkeyKong": {
        "random_score": 475,
        "human_score": 33000,
        "success_threshold": 0.75,
        "completion_metric": "lives"
    },
    "Asteroids": {
        "random_score": 719,
        "human_score": 13157,
        "success_threshold": 0.75,
        "completion_metric": "lives"
    },
    "Seaquest": {
        "random_score": 68,
        "human_score": 20182,
        "success_threshold": 0.75,
        "completion_metric": "lives"
    }
}

def get_game_config(game_name):
    return ATARI_CONFIGS.get(game_name, {
        "random_score": 0,
        "human_score": 1,
        "success_threshold": 0.75,
        "completion_metric": None
    })
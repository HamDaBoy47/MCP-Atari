ATARI_CONFIGS = {
    "MsPacman": {
        "random_score": 307.3,
        "human_score": 15693,
        "success_threshold": 0.75,
        "completion_metric": None  # Ms. Pacman doesn't have a clear "completion" state
    },
    "Breakout": {
        "random_score": 1.7,
        "human_score": 30.5,
        "success_threshold": 0.75,
        "completion_metric": "lives"  # Consider an episode complete when all lives are lost
    },
    # Add more games as needed
}

def get_game_config(game_name):
    return ATARI_CONFIGS.get(game_name, {
        "random_score": 0,
        "human_score": 1,
        "success_threshold": 0.75,
        "completion_metric": None
    })
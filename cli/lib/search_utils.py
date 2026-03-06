import json

DEFAULT_SEARCH_LIMIT = 5

def load_movies():
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return data["movies"]
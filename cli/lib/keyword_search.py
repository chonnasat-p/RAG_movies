from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def search_command(query, limit = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    results = []
    new_query = preprocessing(query)
    for movie in movies:
        title = preprocessing(movie["title"])
        if matching_token(new_query, title):
            results.append(movie)
            if len(results) >= limit:
                break
                    
    return results

def preprocessing(text):
        mapping = str.maketrans("", "", string.punctuation)
        result = text.lower()

        # Remove Punctuation
        rm_punc = result.translate(mapping)

        # Tokenization
        tokens = rm_punc.split()
        tokenize = [token for token in tokens if token]
        return tokenize

def matching_token(query_tokens, title_tokens):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
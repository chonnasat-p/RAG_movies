from .search_utils import (
    DEFAULT_SEARCH_LIMIT, 
    load_movies, 
    load_stopswords,
    format_search_result,
    BM25_K1,
    BM25_B
)
import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import defaultdict, Counter
import math

def search_command(query, limit = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    seen, results = set(), []
    query_tokens = preprocessing(query)
    for query in query_tokens:
        for doc_id in idx.get_documents(query):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
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

        # Check Stopword
        filtered_words = check_stopwords(tokenize)

        # Stemming
        stemmed_words = stemming(filtered_words)

        return stemmed_words

def matching_token(query_tokens, title_tokens):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def check_stopwords(token_list):
    stopwords = load_stopswords()
    filtered_words = []
    for token in token_list:
        if token not in stopwords:
            filtered_words.append(token)

    return filtered_words

def stemming(filtered_words):
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


class InvertedIndex():
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.tf = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocessing(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.tf[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term: str) -> list[int]:
        tokens = preprocessing(term)
        if not tokens:
            return []
        doc_ids = self.index.get(tokens[0], set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = preprocessing(term)
        if len(token) > 1:
            raise Exception("Number of tokens is more than one")
        if not token:
            return 0       
        term_frequencies = self.tf[doc_id][token[0]]
        return term_frequencies

    def get_idf(self, term: str) -> float:
        doc_id = self.get_documents(term)
        return math.log((len(self.docmap) + 1) / (len(doc_id) + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)

        # Length normalization factor
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def __get_avg_doc_length(self) -> float:
        sum_lengths = 0
        for length in self.doc_lengths.values():
            sum_lengths += length
        return sum_lengths / len(self.doc_lengths)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            movie_description = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_description)
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open('cache/index.pkl', 'wb') as file:
            pickle.dump(self.index, file)
        with open('cache/docmap.pkl', 'wb') as file:
            pickle.dump(self.docmap, file)
        with open('cache/term_frequencies.pkl', 'wb') as file:
            pickle.dump(self.tf, file)
        with open(self.doc_lengths_path, 'wb') as file:
            pickle.dump(self.doc_lengths, file)

    def load(self) -> None:
        with open('cache/index.pkl', 'rb') as file:
            self.index = pickle.load(file)
        with open('cache/docmap.pkl', 'rb') as file:
            self.docmap = pickle.load(file)
        with open('cache/term_frequencies.pkl', 'rb') as file:
            self.tf = pickle.load(file)
        with open(self.doc_lengths_path, 'rb') as file:
            self.doc_lengths = pickle.load(file)

    def bm25(self, doc_id, term) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):
        query_tokens = preprocessing(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

def build_command() -> None:
    Inverted_Index = InvertedIndex()
    Inverted_Index.build()
    Inverted_Index.save()

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: int, b:float) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
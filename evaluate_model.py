import ast
import os
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# 1. Load and preprocess the dataset identically to train_model.py
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")
movies = movies.merge(credits, on="title")

movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
movies.dropna(subset=["overview"], inplace=True)
movies.drop_duplicates(subset=["title"], inplace=True)
movies.reset_index(drop=True, inplace=True)


def extract_names(obj_str):
    return [item["name"] for item in ast.literal_eval(obj_str)]


def extract_top_cast(obj_str):
    return [item["name"] for item in ast.literal_eval(obj_str)[:3]]


def extract_director(obj_str):
    return [item["name"] for item in ast.literal_eval(obj_str) if item["job"] == "Director"]


movies["genres_clean"] = movies["genres"].apply(extract_names)
movies["keywords_clean"] = movies["keywords"].apply(extract_names)
movies["cast_clean"] = movies["cast"].apply(extract_top_cast)
movies["director_clean"] = movies["crew"].apply(extract_director)

# Build ground-truth relevance sets (shared genre or shared director)
relevance_sets = [
    set(g) | set(d)
    for g, d in zip(movies["genres_clean"], movies["director_clean"])
]

ps = PorterStemmer()


def build_tags(row):
    tokens = (
        row["overview"].split()
        + [x.replace(" ", "") for x in row["genres_clean"]]
        + [x.replace(" ", "") for x in row["keywords_clean"]]
        + [x.replace(" ", "") for x in row["cast_clean"]]
        + [x.replace(" ", "") for x in row["director_clean"]]
    )
    return " ".join(ps.stem(word.lower()) for word in tokens)


movies["tags"] = movies.apply(build_tags, axis=1)


def get_top_k_indices(score_matrix, k=5, higher_is_better=True):
    """Return an (N, k) array of top-k neighbor indices excluding self."""
    n = score_matrix.shape[0]
    top_k = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        row = score_matrix[i].copy()
        if higher_is_better:
            row[i] = -np.inf
            top_k[i] = np.argsort(row)[::-1][:k]
        else:
            row[i] = np.inf
            top_k[i] = np.argsort(row)[:k]
    return top_k


def compute_metrics(cos_top_k, euc_top_k, k=5):
    n = len(cos_top_k)
    overlaps = []
    cos_mrr, euc_mrr = [], []
    cos_prec, euc_prec = [], []

    for i in range(n):
        c_recs = cos_top_k[i]
        e_recs = euc_top_k[i]

        # 1. Top-K Overlap Consistency between Cosine and Euclidean
        overlap = len(set(c_recs) & set(e_recs)) / k
        overlaps.append(overlap)

        # 2. Mean Reciprocal Rank (MRR) & Precision@K using shared genre/director
        query_labels = relevance_sets[i]

        c_hits = [1 if (relevance_sets[idx] & query_labels) else 0 for idx in c_recs]
        e_hits = [1 if (relevance_sets[idx] & query_labels) else 0 for idx in e_recs]

        cos_prec.append(sum(c_hits) / k)
        euc_prec.append(sum(e_hits) / k)

        cos_mrr.append(1.0 / (c_hits.index(1) + 1) if 1 in c_hits else 0.0)
        euc_mrr.append(1.0 / (e_hits.index(1) + 1) if 1 in e_hits else 0.0)

    return {
        "Top-5 Overlap Consistency": round(float(np.mean(overlaps)), 4),
        "Cosine MRR": round(float(np.mean(cos_mrr)), 4),
        "Euclidean MRR": round(float(np.mean(euc_mrr)), 4),
        "Cosine Precision@5": round(float(np.mean(cos_prec)), 4),
        "Euclidean Precision@5": round(float(np.mean(euc_prec)), 4),
    }


print("Running Cosine Similarity vs. Euclidean Distance evaluation across all movies...\n")

results = []
vectorizers = {
    "TF-IDF (L2-Normalized)": TfidfVectorizer(max_features=5000, stop_words="english", norm="l2"),
    "Count Vectorizer (Unnormalized)": CountVectorizer(max_features=5000, stop_words="english"),
}

for name, vec in vectorizers.items():
    matrix = vec.fit_transform(movies["tags"]).toarray().astype("float32")
    cos_sim = cosine_similarity(matrix)
    euc_dist = euclidean_distances(matrix)

    cos_top5 = get_top_k_indices(cos_sim, k=5, higher_is_better=True)
    euc_top5 = get_top_k_indices(euc_dist, k=5, higher_is_better=False)

    metrics = compute_metrics(cos_top5, euc_top5, k=5)
    metrics["Vector Representation"] = name
    results.append(metrics)

results_df = pd.DataFrame(results)[
    [
        "Vector Representation",
        "Top-5 Overlap Consistency",
        "Cosine MRR",
        "Euclidean MRR",
        "Cosine Precision@5",
        "Euclidean Precision@5",
    ]
]

print(results_df.to_string(index=False))

os.makedirs("models", exist_ok=True)
results_df.to_csv("models/evaluation_metrics.csv", index=False)
print("\nSaved benchmark results to models/evaluation_metrics.csv")
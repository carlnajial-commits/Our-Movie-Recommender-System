import ast
import os
import pickle
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load and merge the two CSV files on 'title'
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")
movies = movies.merge(credits, on="title")

# Keep only the relevant features defined in your proposal
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
movies.dropna(subset=["overview"], inplace=True)
movies.drop_duplicates(subset=["title"], inplace=True)
movies.reset_index(drop=True, inplace=True)

# 2. Helper functions to parse JSON-formatted string columns
def extract_names(obj_str):
    return [item["name"] for item in ast.literal_eval(obj_str)]

def extract_top_cast(obj_str):
    return [item["name"] for item in ast.literal_eval(obj_str)[:3]]

def extract_director(obj_str):
    return [item["name"] for item in ast.literal_eval(obj_str) if item["job"] == "Director"]

movies["genres"] = movies["genres"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)
movies["cast"] = movies["cast"].apply(extract_top_cast)
movies["crew"] = movies["crew"].apply(extract_director)

# Keep clean overview text for UI display later, and split a copy for NLP tags
movies["overview_list"] = movies["overview"].apply(lambda x: x.split())

# Remove spaces inside multi-word names/genres (e.g., "Sam Worthington" -> "SamWorthington")
for col in ["genres", "keywords", "cast", "crew"]:
    movies[col] = movies[col].apply(lambda items: [i.replace(" ", "") for i in items])

# 3. Combine features into a single 'tags' corpus and apply Porter Stemming
movies["tags"] = (
    movies["overview_list"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

ps = PorterStemmer()

def stem_text(word_list):
    return " ".join(ps.stem(word.lower()) for word in word_list)

movies["tags"] = movies["tags"].apply(stem_text)

# 4. Vectorize text using TF-IDF and compute Cosine Similarity
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
vectors = vectorizer.fit_transform(movies["tags"]).toarray()

# Cast to float32 to cut memory and disk size in half
similarity = cosine_similarity(vectors).astype("float32")

# 5. Export trained artifacts to the models/ directory
os.makedirs("models", exist_ok=True)

final_df = movies[["movie_id", "title", "overview"]].copy()
with open("models/movies_dict.pkl", "wb") as f:
    pickle.dump(final_df.to_dict(), f)

with open("models/similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print(f"Training complete! Processed {len(final_df)} movies.")
print("Saved: models/movies_dict.pkl and models/similarity.pkl")
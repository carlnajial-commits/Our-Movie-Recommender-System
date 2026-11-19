import json
import os
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# 1. Page Configuration & Backend API Key Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Stream | Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# Hide only the Deploy button and bottom footer while keeping the sidebar toggle and theme menu
st.markdown(
    """
    <style>
        footer {visibility: hidden;}
        .stAppDeployButton {display: none;}
        [data-testid="stDecoration"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Optional local fallback (keep your actual key in .streamlit/secrets.toml before pushing to GitHub)
HARDCODED_TMDB_API_KEY = "PASTE_YOUR_TMDB_API_KEY_HERE"


def get_tmdb_api_key() -> str:
    try:
        if "TMDB_API_KEY" in st.secrets:
            return str(st.secrets["TMDB_API_KEY"]).strip()
    except Exception:
        pass

    if HARDCODED_TMDB_API_KEY and HARDCODED_TMDB_API_KEY != "PASTE_YOUR_TMDB_API_KEY_HERE":
        return HARDCODED_TMDB_API_KEY.strip()

    return ""


TMDB_API_KEY = get_tmdb_api_key()
DB_PATH = "database/movie_app.db"
MOVIES_PKL = "models/movies_dict.pkl"
SIMILARITY_PKL = "models/similarity.pkl"
PLACEHOLDER_POSTER = "https://placehold.co/500x750?text=No+Poster+Available"


# ---------------------------------------------------------
# 2. Cached SQLite Connection & HTTP Session
# ---------------------------------------------------------
@st.cache_resource
def get_db_connection():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tmdb_cache (
            movie_id INTEGER PRIMARY KEY,
            poster_url TEXT,
            youtube_key TEXT,
            release_year TEXT,
            rating REAL,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            selected_movie TEXT NOT NULL,
            recommended_movies TEXT NOT NULL,
            searched_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


@st.cache_resource
def get_http_session():
    return requests.Session()


def get_cached_tmdb_batch(movie_ids: list[int]) -> dict[int, dict]:
    if not movie_ids:
        return {}
    conn = get_db_connection()
    placeholders = ",".join("?" for _ in movie_ids)
    cursor = conn.execute(
        f"SELECT movie_id, poster_url, youtube_key, release_year, rating FROM tmdb_cache WHERE movie_id IN ({placeholders})",
        [int(mid) for mid in movie_ids],
    )
    cached_map = {}
    for mid, poster_url, youtube_key, release_year, rating in cursor.fetchall():
        if (
            poster_url
            and poster_url != PLACEHOLDER_POSTER
            and not poster_url.startswith("https://via.placeholder.com")
        ):
            cached_map[int(mid)] = {
                "poster_url": poster_url,
                "youtube_key": youtube_key,
                "release_year": release_year,
                "rating": rating,
            }
    return cached_map


def save_cached_tmdb_batch(items: dict[int, dict]):
    if not items:
        return
    conn = get_db_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (
            int(mid),
            info["poster_url"],
            info["youtube_key"],
            info["release_year"],
            info["rating"],
            now,
        )
        for mid, info in items.items()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO tmdb_cache
        (movie_id, poster_url, youtube_key, release_year, rating, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def log_search(selected_movie: str, recommended_titles: list[str]):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO search_logs (selected_movie, recommended_movies, searched_at) VALUES (?, ?, ?)",
        (
            selected_movie,
            json.dumps(recommended_titles),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()


def get_recent_logs(limit: int = 8):
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT selected_movie, recommended_movies FROM search_logs ORDER BY log_id DESC LIMIT ?",
        (limit,),
    )
    return cursor.fetchall()


def clear_search_logs():
    conn = get_db_connection()
    conn.execute("DELETE FROM search_logs")
    conn.commit()


# ---------------------------------------------------------
# 3. Load ML Artifacts into RAM (@st.cache_resource)
# ---------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    if not os.path.exists(MOVIES_PKL) or not os.path.exists(SIMILARITY_PKL):
        return None, None
    with open(MOVIES_PKL, "rb") as f:
        movies_df = pd.DataFrame(pickle.load(f))
    with open(SIMILARITY_PKL, "rb") as f:
        similarity_matrix = pickle.load(f)
    return movies_df, similarity_matrix


# ---------------------------------------------------------
# 4. Parallel TMDB API v3 Fetcher
# ---------------------------------------------------------
def _fetch_single_tmdb_from_api(movie_id: int) -> tuple[int, dict, bool]:
    default_data = {
        "poster_url": PLACEHOLDER_POSTER,
        "youtube_key": None,
        "release_year": "N/A",
        "rating": 0.0,
    }
    if not TMDB_API_KEY:
        return movie_id, default_data, False

    session = get_http_session()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
        "append_to_response": "videos",
    }

    try:
        response = session.get(url, params=params, timeout=5)
        if response.status_code != 200:
            return movie_id, default_data, False

        data = response.json()
        poster_path = data.get("poster_path")
        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path
            else PLACEHOLDER_POSTER
        )

        release_date = data.get("release_date", "")
        release_year = release_date[:4] if release_date else "N/A"
        rating = round(float(data.get("vote_average", 0.0)), 1)

        youtube_key = None
        videos = data.get("videos", {}).get("results", [])
        for v in videos:
            if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                youtube_key = v.get("key")
                break
        if not youtube_key:
            for v in videos:
                if v.get("site") == "YouTube":
                    youtube_key = v.get("key")
                    break

        details = {
            "poster_url": poster_url,
            "youtube_key": youtube_key,
            "release_year": release_year,
            "rating": rating,
        }
        return movie_id, details, True

    except requests.RequestException:
        return movie_id, default_data, False


def fetch_tmdb_details_batch(movie_ids: list[int]) -> dict[int, dict]:
    """Check SQLite cache in one query, then fetch any missing movies concurrently."""
    unique_ids = list(dict.fromkeys(int(mid) for mid in movie_ids))
    results = get_cached_tmdb_batch(unique_ids)
    missing_ids = [mid for mid in unique_ids if mid not in results]

    if missing_ids:
        newly_fetched_valid = {}
        with ThreadPoolExecutor(max_workers=len(missing_ids)) as executor:
            for mid, details, is_valid in executor.map(_fetch_single_tmdb_from_api, missing_ids):
                results[mid] = details
                if is_valid:
                    newly_fetched_valid[mid] = details
        save_cached_tmdb_batch(newly_fetched_valid)

    return results


# ---------------------------------------------------------
# 5. Fast Vectorized Content-Based Recommendation Engine
# ---------------------------------------------------------
def get_movie_and_recommendations(selected_title: str, movies_df: pd.DataFrame, similarity_matrix):
    movie_idx = int(movies_df.index[movies_df["title"] == selected_title][0])
    selected_row = movies_df.iloc[movie_idx]

    # Exclude self and partition top 5 in O(N) using NumPy instead of sorting all 4,803 items
    scores = similarity_matrix[movie_idx].copy()
    scores[movie_idx] = -1.0
    top_5_unsorted = np.argpartition(-scores, range(5))[:5]
    top_5_indices = top_5_unsorted[np.argsort(-scores[top_5_unsorted])]

    # Batch-fetch TMDB details for the searched movie + all 5 recommendations in parallel
    selected_id = int(selected_row["movie_id"])
    rec_rows = [movies_df.iloc[int(idx)] for idx in top_5_indices]
    all_ids = [selected_id] + [int(r["movie_id"]) for r in rec_rows]
    tmdb_map = fetch_tmdb_details_batch(all_ids)

    searched_movie_data = {
        "movie_id": selected_id,
        "title": selected_row["title"],
        "overview": selected_row["overview"],
        **tmdb_map[selected_id],
    }

    recommendations = []
    for idx, row in zip(top_5_indices, rec_rows):
        mid = int(row["movie_id"])
        recommendations.append(
            {
                "movie_id": mid,
                "title": row["title"],
                "overview": row["overview"],
                "similarity_score": round(float(scores[idx]) * 100, 1),
                **tmdb_map[mid],
            }
        )

    return searched_movie_data, recommendations


# ---------------------------------------------------------
# 6. Clean User Interface
# ---------------------------------------------------------
movies_df, similarity_matrix = load_ml_artifacts()

with st.sidebar:
    st.header("Recent Searches")
    logs = get_recent_logs(limit=8)
    if logs:
        for searched_title, recs_json in logs:
            with st.expander(searched_title):
                for r_title in json.loads(recs_json):
                    st.caption(f"• {r_title}")
        if st.button("Clear History", width="stretch"):
            clear_search_logs()
            st.rerun()
    else:
        st.caption("No recent searches.")

st.title("🎬 Movie Stream")
st.caption("Discover movies tailored to your taste and watch official trailers.")

if not TMDB_API_KEY:
    st.warning(
        "TMDB API key not found. Add `TMDB_API_KEY` in `.streamlit/secrets.toml` to load movie posters and trailers."
    )

if movies_df is None or similarity_matrix is None:
    st.error("Model artifacts not found! Run `python train_model.py` first.")
    st.stop()

default_title = (
    "Inception"
    if "Inception" in movies_df["title"].values
    else movies_df["title"].values[0]
)
if "movie_selector" not in st.session_state:
    st.session_state["movie_selector"] = default_title


def clear_search_input():
    st.session_state["movie_selector"] = None


search_col, clear_col = st.columns([6, 1], vertical_alignment="bottom")
with search_col:
    selected_movie = st.selectbox(
        "Search for a movie:",
        options=movies_df["title"].values,
        index=None,
        placeholder="Type a movie title...",
        key="movie_selector",
    )
with clear_col:
    st.button("Clear", on_click=clear_search_input, width="stretch")

if selected_movie and st.session_state.get("active_movie") != selected_movie:
    with st.spinner("Finding similar movies..."):
        searched_data, recs = get_movie_and_recommendations(
            selected_movie, movies_df, similarity_matrix
        )
        st.session_state["active_movie"] = selected_movie
        st.session_state["searched_movie_data"] = searched_data
        st.session_state["recommendations"] = recs
        st.session_state["selected_trailer"] = None
        log_search(selected_movie, [r["title"] for r in recs])

if "searched_movie_data" in st.session_state:
    searched_movie_data = st.session_state["searched_movie_data"]
    st.divider()

    hero_poster_col, hero_info_col = st.columns([1, 3], gap="large")
    with hero_poster_col:
        st.image(searched_movie_data["poster_url"], width="stretch")

    with hero_info_col:
        st.markdown(f"## {searched_movie_data['title']} ({searched_movie_data['release_year']})")
        st.markdown(f"**Rating:** ⭐ {searched_movie_data['rating']} / 10")
        st.write(searched_movie_data["overview"])

        if searched_movie_data["youtube_key"]:
            if st.button(
                "▶ Watch Trailer",
                key=f"hero_trailer_{searched_movie_data['movie_id']}",
                type="primary",
            ):
                st.session_state["selected_trailer"] = searched_movie_data
        else:
            st.button("No Trailer Available", key="hero_no_trailer", disabled=True)

    st.divider()
    st.subheader(f"More Like *{searched_movie_data['title']}*")

    cols = st.columns(5)
    for i, rec in enumerate(st.session_state["recommendations"]):
        with cols[i]:
            st.image(rec["poster_url"], width="stretch")
            st.markdown(f"**{rec['title']}**")
            st.caption(
                f"Match: **{rec['similarity_score']}%** | ⭐ {rec['rating']} | {rec['release_year']}"
            )

            with st.expander("Overview"):
                st.write(rec["overview"])

            if rec["youtube_key"]:
                if st.button("▶ Watch Trailer", key=f"trailer_{rec['movie_id']}", width="stretch"):
                    st.session_state["selected_trailer"] = rec
            else:
                st.button(
                    "No Trailer",
                    key=f"no_trailer_{rec['movie_id']}",
                    disabled=True,
                    width="stretch",
                )

    if st.session_state.get("selected_trailer"):
        trailer_movie = st.session_state["selected_trailer"]
        st.divider()
        st.subheader(f"🎥 Official Trailer: {trailer_movie['title']}")
        st.video(f"https://www.youtube.com/watch?v={trailer_movie['youtube_key']}")
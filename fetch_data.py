import os
import requests
import pandas as pd
import argparse
from dotenv import load_dotenv

load_dotenv()

DATASET_FILE = "movies_dataset.csv"

def get_access_token():
    token = os.getenv("TMDB_API_READ_ACCESS_TOKEN")
    if not token:
        raise ValueError("Missing TMDB_API_READ_ACCESS_TOKEN in .env")
    return token

def get_genres(access_token):
    """Fetch genre mapping from TMDB."""
    url = "https://api.themoviedb.org/3/genre/movie/list"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    genres = response.json().get("genres", [])
    return {g["id"]: g["name"] for g in genres}

def fetch_endpoint(endpoint, access_token, genre_mapping, limit):
    """Fetches movies from a specific TMDB endpoint."""
    movies_data = []
    page = 1
    total_fetched = 0
    
    print(f"Fetching up to {limit} movies from /{endpoint}...")
    
    while total_fetched < limit:
        url = f"https://api.themoviedb.org/3/movie/{endpoint}"
        params = {"language": "en-US", "page": page}
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        results = response.json().get("results", [])
        if not results:
            break
            
        for movie in results:
            if total_fetched >= limit:
                break
                
            genre_names = [genre_mapping.get(gid, "Unknown") for gid in movie.get("genre_ids", [])]
            
            movies_data.append({
                "id": movie["id"],
                "title": movie["title"],
                "overview": movie.get("overview", ""),
                "genres": ", ".join(genre_names),
                "vote_average": movie.get("vote_average", 0.0),
                "original_language": movie.get("original_language", "en")
            })
            total_fetched += 1
            
        page += 1
        
    return movies_data

def fetch_movies(limit=1000, force_refresh=False):
    """
    Fetches movies from TMDB (split between popular and top_rated) to build the dataset.
    """
    if os.path.exists(DATASET_FILE) and not force_refresh:
        print(f"{DATASET_FILE} already exists. Skipping fetch. Use --refresh to rebuild.")
        return DATASET_FILE

    access_token = get_access_token()
    genre_mapping = get_genres(access_token)
    
    # Split the limit between the two endpoints
    limit_per_endpoint = max(1, limit // 2)
    
    popular_movies = fetch_endpoint("popular", access_token, genre_mapping, limit_per_endpoint)
    top_rated_movies = fetch_endpoint("top_rated", access_token, genre_mapping, limit_per_endpoint)
    
    # Combine and drop duplicates
    df = pd.DataFrame(popular_movies + top_rated_movies)
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    df.to_csv(DATASET_FILE, index=False)
    print(f"Successfully saved {len(df)} unique movies to {DATASET_FILE}")
    
    return DATASET_FILE

def search_and_append_movie(search_title):
    """
    Searches TMDB for a specific movie title, and if found, appends it to the local CSV.
    Returns the exact matched title, or None if not found.
    """
    print(f"\n[Dynamic Fetch] '{search_title}' not found locally. Searching TMDB...")
    access_token = get_access_token()
    genre_mapping = get_genres(access_token)
    
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"query": search_title, "language": "en-US", "page": 1}
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    results = response.json().get("results", [])
    if not results:
        print(f"[Dynamic Fetch] Movie '{search_title}' does not exist on TMDB.")
        return None
        
    # Smart Auto-Selection for API results
    best_match = None
    
    def normalize_title(t):
        import re
        t = re.sub(r'[^\w\s]', '', t.lower()).strip()
        if t.startswith("the "): return t[4:]
        if t.startswith("a "): return t[2:]
        if t.startswith("an "): return t[3:]
        return t
        
    normalized_search = normalize_title(search_title)
    exact_matches = [res for res in results if normalize_title(res['title']) == normalized_search]
    
    # Prevent obscure exact-match trap: Only auto-select if the exact match is the #1 most popular result.
    # ALSO prevent Remake Trap: If there are MULTIPLE exact matches (like Dune 1984 and 2021), force disambiguation!
    if len(exact_matches) == 1 and exact_matches[0]['id'] == results[0]['id']:
        best_match = exact_matches[0]
        print(f"  (Auto-matched to exact title: {best_match['title']})")
            
    elif len(results) == 1:
        best_match = results[0]
        print(f"  (Auto-matched to: {best_match['title']})")
        
    else:
        # Disambiguation: present up to 5 choices
        print(f"\n[Dynamic Fetch] Found matches for '{search_title}':")
        display_results = results[:5]
        for i, res in enumerate(display_results, 1):
            year = res.get('release_date', 'Unknown')[:4] if res.get('release_date') else 'Unknown'
            print(f"  {i}. {res['title']} ({year})")
        print("  0. None of these / Cancel")
        
        while True:
            choice = input(f"Which one did you mean? (0-{len(display_results)}): ").strip()
            if choice.isdigit():
                idx = int(choice)
                if idx == 0:
                    print("Search cancelled.")
                    return None
                elif 1 <= idx <= len(display_results):
                    best_match = display_results[idx - 1]
                    break
            print("Invalid choice, please try again.")
        
    matched_title = best_match["title"]
    
    genre_names = [genre_mapping.get(gid, "Unknown") for gid in best_match.get("genre_ids", [])]
    
    new_movie = pd.DataFrame([{
        "id": best_match["id"],
        "title": matched_title,
        "overview": best_match.get("overview", ""),
        "genres": ", ".join(genre_names),
        "vote_average": best_match.get("vote_average", 0.0),
        "original_language": best_match.get("original_language", "en")
    }])
    
    # Append to existing CSV
    if os.path.exists(DATASET_FILE):
        existing_df = pd.read_csv(DATASET_FILE)
        # Check if we accidentally already have it (maybe under a slightly different name)
        if best_match["id"] not in existing_df["id"].values:
            new_movie.to_csv(DATASET_FILE, mode='a', header=False, index=False)
            print(f"[Dynamic Fetch] Added '{matched_title}' to local dataset!")
        else:
            print(f"[Dynamic Fetch] '{matched_title}' is already in the dataset under its ID.")
    else:
        new_movie.to_csv(DATASET_FILE, index=False)
        print(f"[Dynamic Fetch] Created dataset with '{matched_title}'.")
        
    return matched_title

def check_movie_collection(movie_id):
    """Checks if a movie belongs to a collection and returns (collection_id, collection_name)."""
    access_token = get_access_token()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {access_token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        collection = data.get("belongs_to_collection")
        if collection:
            return collection["id"], collection["name"]
    return None, None

def fetch_collection_movies(collection_id):
    """Fetches all movies in a collection and maps their genres."""
    access_token = get_access_token()
    genre_mapping = get_genres(access_token)
    
    url = f"https://api.themoviedb.org/3/collection/{collection_id}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {access_token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
        
    parts = response.json().get("parts", [])
    collection_movies = []
    
    for part in parts:
        genre_names = [genre_mapping.get(gid, "Unknown") for gid in part.get("genre_ids", [])]
        collection_movies.append({
            "id": part["id"],
            "title": part["title"],
            "overview": part.get("overview", ""),
            "genres": ", ".join(genre_names),
            "vote_average": part.get("vote_average", 0.0),
            "original_language": part.get("original_language", "en"),
            "release_date": part.get("release_date", "")
        })
        
    return collection_movies

def append_multiple_movies(movies_list):
    """Appends multiple movies to the local dataset CSV safely."""
    if not movies_list: return
    
    new_movies_df = pd.DataFrame(movies_list)
    
    if os.path.exists(DATASET_FILE):
        existing_df = pd.read_csv(DATASET_FILE)
        # Filter out movies we already have
        new_movies_df = new_movies_df[~new_movies_df['id'].isin(existing_df['id'])]
        
        # Drop release_date column so it doesn't break the CSV parser (which only expects 5 columns)
        if 'release_date' in new_movies_df.columns:
            new_movies_df = new_movies_df.drop(columns=['release_date'])
        if not new_movies_df.empty:
            new_movies_df.to_csv(DATASET_FILE, mode='a', header=False, index=False)
            print(f"[Dynamic Fetch] Appended {len(new_movies_df)} new franchise movies to dataset.")
    else:
        new_movies_df.to_csv(DATASET_FILE, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch movie dataset from TMDB")
    parser.add_argument("--limit", type=int, default=1000, help="Total number of movies to fetch")
    parser.add_argument("--refresh", action="store_true", help="Force refresh the dataset if it exists")
    args = parser.parse_args()
    
    fetch_movies(limit=args.limit, force_refresh=args.refresh)

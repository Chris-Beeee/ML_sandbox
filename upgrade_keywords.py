import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import time

load_dotenv()
DATASET_FILE = "movies_dataset.csv"

def get_access_token():
    token = os.getenv("TMDB_API_READ_ACCESS_TOKEN")
    if not token:
        raise ValueError("Missing TMDB_API_READ_ACCESS_TOKEN in .env")
    return token

def fetch_keywords_for_movie(movie_id, access_token):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                # Rate limited, back off
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            keywords = response.json().get("keywords", [])
            keyword_names = [k["name"] for k in keywords]
            return movie_id, ", ".join(keyword_names)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch keywords for {movie_id}: {e}")
                return movie_id, ""
            time.sleep(1)
    
    return movie_id, ""

def main():
    if not os.path.exists(DATASET_FILE):
        print(f"{DATASET_FILE} not found!")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILE)
    
    # If keywords column already exists, only fetch for those that are missing
    if 'keywords' not in df.columns:
        df['keywords'] = ""
        
    access_token = get_access_token()
    
    movies_to_fetch = df[df['keywords'] == ""]['id'].tolist()
    if not movies_to_fetch:
        print("All movies already have keywords fetched.")
        return
        
    print(f"Fetching keywords for {len(movies_to_fetch)} movies. This will take a minute or two...")
    
    keyword_dict = {}
    fetched = 0
    total = len(movies_to_fetch)
    
    # Use 20 threads to speed up the network requests
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_id = {executor.submit(fetch_keywords_for_movie, m_id, access_token): m_id for m_id in movies_to_fetch}
        
        for future in as_completed(future_to_id):
            m_id, keywords = future.result()
            keyword_dict[m_id] = keywords
            fetched += 1
            if fetched % 500 == 0:
                print(f"  Progress: {fetched} / {total} movies fetched...")

    print("Merging keywords into dataset...")
    # Update the dataframe
    for m_id, keywords in keyword_dict.items():
        df.loc[df['id'] == m_id, 'keywords'] = keywords
        
    df.to_csv(DATASET_FILE, index=False)
    print("Migration complete! Keywords column added to movies_dataset.csv.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

from fetch_data import search_and_append_movie

class ProfileRecommender:
    def __init__(self, dataset_path="movies_dataset.csv", history_file="user_history.json"):
        self.dataset_path = dataset_path
        self.history_file = history_file
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.history = []
        
        self.load_data()
        self.load_history()
        
    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"{self.dataset_path} not found. Please run fetch_data.py first.")
            
        self.df = pd.read_csv(self.dataset_path)
        self.df['overview'] = self.df['overview'].fillna('')
        self.df['genres'] = self.df['genres'].fillna('')
        
        # Handle backwards compatibility if dataset hasn't been upgraded yet
        if 'keywords' not in self.df.columns:
            self.df['keywords'] = ""
        self.df['keywords'] = self.df['keywords'].fillna('')
        
        # Weight human-assigned tags higher than plot overview text
        self.df['combined_features'] = (self.df['genres'] + " ") * 2 + (self.df['keywords'] + " ") * 2 + self.df['overview']
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                raw_history = json.load(f)
                
            # Check for migration
            if raw_history and isinstance(raw_history[0], str):
                print("\n[Migration] Upgrading user_history.json to ID-based schema...")
                new_history = []
                for title in raw_history:
                    match = self.df[self.df['title'] == title]
                    if not match.empty:
                        m_id = int(match.iloc[0]['id'])
                        new_history.append({"id": m_id, "title": title})
                self.history = new_history
                self.save_history()
                print(f"[Migration] Upgrade complete! Converted {len(new_history)} movies.")
            else:
                self.history = raw_history
        else:
            self.history = []

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)

    def clear_profile(self):
        self.history = []
        self.save_history()

    def add_to_profile(self, movie_title):
        """Finds the movie and adds it to the user history."""
        # Fix TMDB dictionary collisions (e.g., 'et' means 'and' in French)
        aliases = {"et": "e.t."}
        if movie_title.lower().strip() in aliases:
            movie_title = aliases[movie_title.lower().strip()]
            
        def normalize_title(t):
            import re
            t = re.sub(r'[^\w\s]', '', t.lower()).strip()
            if t.startswith("the "): return t[4:]
            if t.startswith("a "): return t[2:]
            if t.startswith("an "): return t[3:]
            return t
            
        normalized_search = normalize_title(movie_title)
        
        exact_match = self.df[self.df['title'].apply(normalize_title) == normalized_search]
        
        # Skip local partial matching for tiny 1 or 2 letter acronyms to avoid 
        # "ai" matching "Captain" or "et" matching "Secrets".
        if len(movie_title) > 2:
            partial_match = self.df[self.df['title'].str.contains(movie_title, case=False, na=False, regex=False)]
        else:
            partial_match = pd.DataFrame()
        
        matched_title = None
        
        if not exact_match.empty and len(exact_match) == 1:
            # Auto-select exact match ONLY if there is exactly one (prevents silent auto-selection of remakes)
            matched_title = exact_match.iloc[0]['title']
            
            # Print FYI for sequels so the user knows they exist without being blocked
            others = partial_match[partial_match['title'] != matched_title].head(3)
            if not others.empty:
                other_names = ", ".join(others['title'].tolist())
                print(f"  (FYI: I also found related movies you can type: {other_names})")
        elif len(exact_match) > 1:
            # Multiple identical titles locally! Since we don't store years locally, the local menu is useless.
            # Force it to fall through to the TMDB fetch which has release years.
            print(f"  (Multiple local matches for '{movie_title}'. Fetching from TMDB to show release years...)")
            partial_match = pd.DataFrame()
                
        elif not partial_match.empty:
            # Disambiguation prompt
            print(f"\n[Local Search] Found matches for '{movie_title}':")
            display_results = partial_match.head(6)
            for i, (_, row) in enumerate(display_results.iterrows(), 1):
                print(f"  {i}. {row['title']}")
            print("  0. None of these / Search TMDB instead")
            
            while True:
                choice = input(f"Which one did you mean? (0-{len(display_results)}): ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if idx == 0:
                        break
                    elif 1 <= idx <= len(display_results):
                        matched_title = display_results.iloc[idx - 1]['title']
                        break
                print("Invalid choice, please try again.")
                
        if not matched_title:
            # Fallback to Dynamic Fetch if no local matches or user selected 0
            from fetch_data import search_and_append_movie
            fetched_title = search_and_append_movie(movie_title)
            
            if fetched_title == "CANCELLED":
                return "Search cancelled by user."
            elif not fetched_title:
                return f"Movie '{movie_title}' not found on TMDB."
                
            self.load_data()
            matched_title = fetched_title
        # Get movie ID for matched_title
        matched_row = self.df[self.df['title'] == matched_title]
        movie_id = None
        if not matched_row.empty:
            movie_id = int(matched_row.iloc[0]['id'])
            
        history_ids = [m['id'] for m in self.history]
        
        if movie_id and movie_id not in history_ids:
            # Check for franchise
            from fetch_data import check_movie_collection, fetch_collection_movies, append_multiple_movies
            
            col_id, col_name = check_movie_collection(movie_id)
            if col_id:
                print(f"\n[Franchise Detected] '{matched_title}' belongs to '{col_name}'.")
                print("Fetching franchise details...")
                franchise_movies = fetch_collection_movies(col_id)
                
                if franchise_movies:
                    print(f"This collection contains {len(franchise_movies)} movies:")
                    for fm in franchise_movies:
                        year = fm.get('release_date', 'Unknown')[:4] if fm.get('release_date') else 'Unknown'
                        print(f"  - {fm['title']} ({year})")
                        
                    add_franchise = input(f"\nWould you like to add all {len(franchise_movies)} movies to your profile? (y/n): ").strip().lower()
                    if add_franchise == 'y':
                        append_multiple_movies(franchise_movies)
                        self.load_data() # Reload so the new movies are in the TF-IDF matrix
                        
                        added_count = 0
                        for fm in franchise_movies:
                            if fm['id'] not in [m['id'] for m in self.history]:
                                self.history.append({"id": fm['id'], "title": fm['title']})
                                added_count += 1
                                
                        self.save_history()
                        return f"Added '{matched_title}' and {added_count-1} other franchise films to your profile!"
        
            # Standard single movie add
            self.history.append({"id": movie_id, "title": matched_title})
            self.save_history()
            return f"Added '{matched_title}' to your profile!"
        else:
            return f"'{matched_title}' is already in your profile."
    def add_to_profile_online(self, movie_title):
        """Bypasses local CSV search and queries TMDB directly for disambiguation."""
        movie_title = movie_title.lower().strip()
        
        from fetch_data import search_and_append_movie
        fetched_title = search_and_append_movie(movie_title)
        
        if fetched_title == "CANCELLED":
            return "Search cancelled by user."
        elif not fetched_title:
            return f"Movie '{movie_title}' not found on TMDB."
            
        self.load_data()
        matched_title = fetched_title
        
        # Get movie ID for matched_title
        matched_row = self.df[self.df['title'] == matched_title]
        movie_id = None
        if not matched_row.empty:
            movie_id = int(matched_row.iloc[0]['id'])
            
        history_ids = [m['id'] for m in self.history]
        
        if movie_id and movie_id not in history_ids:
            # Check for franchise
            from fetch_data import check_movie_collection, fetch_collection_movies, append_multiple_movies
            
            col_id, col_name = check_movie_collection(movie_id)
            if col_id:
                print(f"\n[Franchise Detected] '{matched_title}' belongs to '{col_name}'.")
                print("Fetching franchise details...")
                franchise_movies = fetch_collection_movies(col_id)
                
                if franchise_movies:
                    print(f"This collection contains {len(franchise_movies)} movies:")
                    for fm in franchise_movies:
                        year = fm.get('release_date', 'Unknown')[:4] if fm.get('release_date') else 'Unknown'
                        print(f"  - {fm['title']} ({year})")
                        
                    add_franchise = input(f"\nWould you like to add all {len(franchise_movies)} movies to your profile? (y/n): ").strip().lower()
                    if add_franchise == 'y':
                        append_multiple_movies(franchise_movies)
                        self.load_data() # Reload so the new movies are in the TF-IDF matrix
                        
                        added_count = 0
                        for fm in franchise_movies:
                            if fm['id'] not in [m['id'] for m in self.history]:
                                self.history.append({"id": fm['id'], "title": fm['title']})
                                added_count += 1
                                
                        self.save_history()
                        return f"Added '{matched_title}' and {added_count-1} other franchise films to your profile!"
        
            # Standard single movie add
            self.history.append({"id": movie_id, "title": matched_title})
            self.save_history()
            return f"Added '{matched_title}' to your profile!"
        else:
            return f"'{matched_title}' is already in your profile."


    def get_profile_recommendations(self, top_n=5, filter_genres=None):
        if not self.history:
            return "Profile is empty. Add some movies first!"
            
        history_indices = []
        for m in self.history:
            match = self.df[self.df['id'] == m['id']]
            if not match.empty:
                history_indices.append(match.index[0])
                
        if not history_indices:
             return "Could not find your history movies in the dataset. Try clearing your profile."

        # Extract the TF-IDF vectors for the liked movies
        liked_vectors = self.tfidf_matrix[history_indices]
        
        # Calculate maximum pooling across all liked movies
        # This prevents specific plot keywords (like 'spy') from being diluted 
        # when a user has a massive 26-movie franchise in their profile.
        max_vector = liked_vectors.max(axis=0)
        
        # Convert back from a sparse matrix to a 2D array for sklearn if necessary
        max_vector_array = np.asarray(max_vector.todense() if hasattr(max_vector, 'todense') else max_vector)
        
        # Compute cosine similarity between the Profile Vector and ALL movies
        cosine_similarities = cosine_similarity(max_vector_array, self.tfidf_matrix).flatten()
        
        # Extract ratings, treating unrated movies (0.0) as average (5.0) to prevent unfair penalization
        if 'vote_average' in self.df.columns:
            vote_averages = self.df['vote_average'].fillna(0.0).astype(float).values
            vote_averages = np.where(vote_averages == 0.0, 5.0, vote_averages)
        else:
            vote_averages = np.full(len(self.df), 5.0)
            
        # Extract original languages
        if 'original_language' in self.df.columns:
            langs = self.df['original_language'].fillna('en').values
        else:
            langs = np.full(len(self.df), 'en')
            
        # Apply weighting formula: Final Score = Similarity Score * (Vote Average / 10)
        weighted_scores = cosine_similarities * (vote_averages / 10.0)
        
        # Apply 50% mathematical penalty to foreign language films
        lang_penalty = np.where(langs == 'en', 1.0, 0.5)
        weighted_scores = weighted_scores * lang_penalty
        
        # Apply "Discovery Temperature" to prevent repetitiveness. 
        # Multiplies every score by a random factor between 0.85 and 1.0, gently shuffling close matches.
        temperature = np.random.uniform(0.85, 1.0, size=weighted_scores.shape)
        weighted_scores = weighted_scores * temperature
        
        # Sort indices to get most similar based on the NEW weighted score
        similar_indices = weighted_scores.argsort()[::-1]
        
        # Create a normalized set of history titles for robust checking
        def normalize_for_check(t):
            import re
            return re.sub(r'[^\w\s]', '', t.lower()).strip()
            
        normalized_history = {normalize_for_check(t) for t in self.history}
        
        recommendations = []
        for idx in similar_indices:
            movie = self.df.iloc[idx]
            movie_title = movie['title']
            
            # Filter out non-Latin characters (CJK, Cyrillic, Arabic, etc.)
            import re
            is_latin = bool(re.match(r'^[\u0000-\u024F\u2000-\u206F]+$', movie_title))
            if not is_latin:
                continue
                
            # Filter by specific requested genres
            if filter_genres:
                movie_genres = str(movie.get('genres', '')).lower()
                has_genres = all(g.lower() in movie_genres for g in filter_genres)
                if not has_genres:
                    continue
                
            # Robustly exclude movies already in the user's history
            if normalize_for_check(movie_title) not in normalized_history:
                recommendations.append({
                    "title": movie_title,
                    "genres": movie['genres'],
                    "similarity_score": round(cosine_similarities[idx], 3),
                    "vote_average": round(vote_averages[idx], 1),
                    "final_score": round(weighted_scores[idx], 3)
                })
                if len(recommendations) == top_n:
                    break
                    
        return recommendations

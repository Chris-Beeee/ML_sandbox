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

    def add_to_profile_online(self, movie_title):
        """Bypasses local CSV search and queries TMDB directly for disambiguation."""
        movie_title = movie_title.lower().strip()
        
        from fetch_data import search_and_append_movie, check_movie_collection, fetch_collection_movies, append_multiple_movies
        fetched_movies = search_and_append_movie(movie_title)
        
        if fetched_movies == "CANCELLED":
            return "Search cancelled by user."
        elif not fetched_movies:
            return f"Could not find '{movie_title}' on TMDB."
            
        added_total = 0
        added_names = []
        
        for movie_id, matched_title, year in fetched_movies:
            movie_id = int(movie_id)
            if any(m['id'] == movie_id for m in self.history):
                continue
                
            added_franchise = False
            col_id, col_name = check_movie_collection(movie_id)
            if col_id:
                print(f"\\n[Franchise Detected] '{matched_title}' belongs to '{col_name}'.")
                print("Fetching franchise details...")
                franchise_movies = fetch_collection_movies(col_id)
                if franchise_movies:
                    print(f"This collection contains {len(franchise_movies)} movies:")
                    for fm in franchise_movies:
                        fm_year = fm.get('release_date', 'Unknown')[:4] if fm.get('release_date') else 'Unknown'
                        print(f"  - {fm['title']} ({fm_year})")
                        
                    add_all = input(f"\\nWould you like to add all {len(franchise_movies)} movies to your profile? (y/n): ").strip().lower()
                    if add_all == 'y':
                        append_multiple_movies(franchise_movies)
                        self.load_data()
                        for fm in franchise_movies:
                            if not any(hist_m['id'] == fm['id'] for hist_m in self.history):
                                self.history.append({"id": int(fm['id']), "title": fm['title'], "year": fm.get('release_date', 'Unknown')[:4]})
                                added_total += 1
                        self.save_history()
                        added_franchise = True
                        added_names.append(col_name)
                        
            if not added_franchise:
                self.history.append({"id": movie_id, "title": matched_title, "year": year})
                self.save_history()
                added_total += 1
                added_names.append(matched_title)
                
        if added_total > 1:
            return f"Added {added_total} movies to your profile!"
        elif added_total == 1:
            return f"Added '{added_names[0]}' to your profile!"
        else:
            return "No new movies were added."

    def remove_from_profile(self, search_term):
        if not self.history:
            return "Your profile is already empty."
            
        search_term = search_term.lower().strip()
        
        # Find matches in history
        matches = [m for m in self.history if search_term in m['title'].lower()]
        
        if not matches:
            return f"No movies matching '{search_term}' found in your profile."
            
        target_movie = None
        if len(matches) == 1:
            target_movie = matches[0]
        else:
            print(f"\nFound multiple matches for '{search_term}' in your profile:")
            print("  (Hint: To remove an entire franchise, just select any movie from it first)")
            for i, m in enumerate(matches, 1):
                year = m.get('year', 'Unknown')
                print(f"  {i}. {m['title']} ({year})")
            print("  0. Cancel")
            
            while True:
                choice = input(f"Which one do you want to remove? (0-{len(matches)}): ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if idx == 0:
                        return "Removal cancelled."
                    elif 1 <= idx <= len(matches):
                        target_movie = matches[idx - 1]
                        break
                print("Invalid choice, please try again.")
                
        movie_id = target_movie['id']
        matched_title = target_movie['title']
        
        # Check if it belongs to a franchise
        from fetch_data import check_movie_collection, fetch_collection_movies
        col_id, col_name = check_movie_collection(movie_id)
        
        to_remove_ids = [movie_id]
        
        if col_id:
            # Get all movies in that collection
            franchise_movies = fetch_collection_movies(col_id)
            if franchise_movies:
                franchise_ids = {fm['id'] for fm in franchise_movies}
                
                # Check how many from this franchise are actually IN the profile
                profile_franchise_movies = [m for m in self.history if m['id'] in franchise_ids]
                
                if len(profile_franchise_movies) > 1:
                    print(f"\n[Franchise Detected] '{matched_title}' belongs to '{col_name}'.")
                    print(f"You have {len(profile_franchise_movies)} movies from this franchise in your profile:")
                    for m in profile_franchise_movies:
                        year = m.get('year', 'Unknown')
                        print(f"  - {m['title']} ({year})")
                        
                    remove_all = input("\nWould you like to remove ALL of these franchise movies from your profile? (y/n): ").strip().lower()
                    if remove_all == 'y':
                        to_remove_ids = [m['id'] for m in profile_franchise_movies]
                        
        # Perform removal
        initial_count = len(self.history)
        self.history = [m for m in self.history if m['id'] not in to_remove_ids]
        self.save_history()
        
        removed_count = initial_count - len(self.history)
        if removed_count > 1:
            return f"Removed '{matched_title}' and {removed_count - 1} other franchise movies."
        else:
            return f"Removed '{matched_title}' from your profile."

    def add_to_profile(self, movie_title):
        """Searches the dataset for a movie and adds it to the user's history."""
        movie_title = movie_title.lower().strip()
        
        def normalize_title(t):
            import re
            t = re.sub(r'[^\w\s]', '', str(t).lower()).strip()
            if t.startswith("the "): return t[4:]
            if t.startswith("a "): return t[2:]
            if t.startswith("an "): return t[3:]
            return t
            
        normalized_search = normalize_title(movie_title)
        
        exact_match = self.df[self.df['title'].apply(normalize_title) == normalized_search]
        
        if len(movie_title) > 2:
            partial_match = self.df[self.df['title'].str.contains(movie_title, case=False, na=False, regex=False)]
        else:
            partial_match = pd.DataFrame()
            
        selected_movies = []
        
        if not partial_match.empty:
            import requests
            from fetch_data import get_access_token
            token = get_access_token()
            headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
            history_ids = {m['id'] for m in self.history}
            
            page = 0
            page_size = 5
            
            while True:
                start_p = page * page_size
                end_p = start_p + page_size
                display_results = partial_match.iloc[start_p:end_p]
                
                if display_results.empty:
                    print("No more local results.")
                    page = 0
                    continue
                    
                print(f"\n[Local Search] Found matches for '{movie_title}' (Page {page+1}/{(len(partial_match) + page_size - 1) // page_size}):")
                print("  (Hint: To add multiple, enter comma-separated numbers like 1,3,4)")
                
                valid_indices = []
                for i, (_, row) in enumerate(display_results.iterrows(), 1):
                    m_id = int(row['id'])
                    year = "Unknown"
                    if token:
                        try:
                            resp = requests.get(f"https://api.themoviedb.org/3/movie/{m_id}", headers=headers, timeout=2)
                            if resp.status_code == 200:
                                date = resp.json().get('release_date', '')
                                if date:
                                    year = date[:4]
                        except:
                            pass
                            
                    title_str = f"  {i}. {row['title']} ({year})"
                    if m_id in history_ids:
                        print(f"{title_str} (Already in profile)")
                    else:
                        print(title_str)
                        valid_indices.append(i)
                        
                print("  N. Next Page")
                print("  0. None of these / Search TMDB instead")
                
                choice = input(f"Which one(s) did you mean? (e.g. 1,3 or N or 0): ").strip().lower()
                
                if choice == 'n':
                    page += 1
                    continue
                elif choice == '0':
                    break
                else:
                    parts = [p.strip() for p in choice.split(',')]
                    valid = True
                    for p in parts:
                        if not p.isdigit():
                            valid = False
                            break
                        idx = int(p)
                        if idx not in valid_indices:
                            valid = False
                            break
                        
                        row = display_results.iloc[idx - 1]
                        m_id = int(row['id'])
                        
                        # Fetch year
                        year = "Unknown"
                        try:
                            resp = requests.get(f"https://api.themoviedb.org/3/movie/{m_id}", headers=headers, timeout=2)
                            if resp.status_code == 200:
                                date = resp.json().get('release_date', '')
                                if date:
                                    year = date[:4]
                        except:
                            pass
                        selected_movies.append((m_id, row['title'], year))
                        
                    if valid and selected_movies:
                        break
                    else:
                        print("Invalid choice or movie already in profile. Please try again.")
                        selected_movies = []
                        
        if not selected_movies:
            # Fallback to TMDB
            from fetch_data import search_and_append_movie
            fetched = search_and_append_movie(movie_title)
            
            if fetched == "CANCELLED":
                return "Search cancelled by user."
            elif not fetched:
                return f"Movie '{movie_title}' not found on TMDB."
                
            self.load_data()
            selected_movies = fetched
            
        added_total = 0
        added_names = []
        from fetch_data import check_movie_collection, fetch_collection_movies, append_multiple_movies
        
        for movie_id, matched_title, year in selected_movies:
            movie_id = int(movie_id)
            if any(m['id'] == movie_id for m in self.history):
                continue
                
            added_franchise = False
            col_id, col_name = check_movie_collection(movie_id)
            if col_id:
                print(f"\n[Franchise Detected] '{matched_title}' belongs to '{col_name}'.")
                print("Fetching franchise details...")
                franchise_movies = fetch_collection_movies(col_id)
                if franchise_movies:
                    print(f"This collection contains {len(franchise_movies)} movies:")
                    for fm in franchise_movies:
                        fm_year = fm.get('release_date', 'Unknown')[:4] if fm.get('release_date') else 'Unknown'
                        print(f"  - {fm['title']} ({fm_year})")
                        
                    add_all = input(f"\nWould you like to add all {len(franchise_movies)} movies to your profile? (y/n): ").strip().lower()
                    if add_all == 'y':
                        append_multiple_movies(franchise_movies)
                        self.load_data()
                        for fm in franchise_movies:
                            if not any(hist_m['id'] == fm['id'] for hist_m in self.history):
                                self.history.append({"id": int(fm['id']), "title": fm['title'], "year": fm.get('release_date', 'Unknown')[:4]})
                                added_total += 1
                        self.save_history()
                        added_franchise = True
                        added_names.append(col_name)
                        
            if not added_franchise:
                self.history.append({"id": movie_id, "title": matched_title, "year": year})
                self.save_history()
                added_total += 1
                added_names.append(matched_title)
                
        if added_total > 1:
            return f"Added {added_total} movies to your profile!"
        elif added_total == 1:
            return f"Added '{added_names[0]}' to your profile!"
        else:
            return "No new movies were added."
    def add_to_profile_online(self, movie_title):
        """Bypasses local CSV search and queries TMDB directly for disambiguation."""
        movie_title = movie_title.lower().strip()
        
        from fetch_data import search_and_append_movie
        fetched = search_and_append_movie(movie_title)
        
        if fetched == "CANCELLED":
            return "Search cancelled by user."
        elif not fetched:
            return f"Movie '{movie_title}' not found on TMDB."
            
        self.load_data()
        movie_id, matched_title, year = fetched
        movie_id = int(movie_id)
            
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
                                self.history.append({"id": int(fm['id']), "title": fm['title'], "year": fm.get('release_date', 'Unknown')[:4]})
                                added_count += 1
                                
                        self.save_history()
                        return f"Added '{matched_title}' and {added_count-1} other franchise films to your profile!"
        
            # Standard single movie add
            self.history.append({"id": movie_id, "title": matched_title, "year": year})
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

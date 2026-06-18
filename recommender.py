import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

from fetch_data import search_and_append_movie

class MovieRecommender:
    def __init__(self, dataset_path="movies_dataset.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
        self.load_data()
        
    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"{self.dataset_path} not found. Please run fetch_data.py first.")
            
        self.df = pd.read_csv(self.dataset_path)
        
        # Fill missing values
        self.df['overview'] = self.df['overview'].fillna('')
        self.df['genres'] = self.df['genres'].fillna('')
        
        # Combine text features to create a single "content" string for each movie
        self.df['combined_features'] = self.df['genres'] + " " + self.df['overview']
        
        self._build_model()
        
    def _build_model(self):
        # Stop words help ignore common words like 'the', 'is', 'in'
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the text data into a TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
        
    def get_recommendations(self, movie_title, top_n=5):
        # Find the index of the movie that matches the title
        # We do a case-insensitive search
        exact_match = self.df[self.df['title'].str.lower() == movie_title.lower()]
        
        if exact_match.empty:
            # If not found, try a partial match
            partial_match = self.df[self.df['title'].str.contains(movie_title, case=False, na=False)]
            if partial_match.empty:
                # DYNAMIC FETCHING FALLBACK
                fetched_title = search_and_append_movie(movie_title)
                if not fetched_title:
                    return f"Movie '{movie_title}' not found in the local dataset or on TMDB."
                    
                # Reload the dataset and rebuild the model!
                self.load_data()
                
                # Now the movie is guaranteed to be in self.df, we can find it
                exact_match = self.df[self.df['title'] == fetched_title]
                if exact_match.empty:
                     return "Error loading newly fetched movie."
                     
                movie_idx = exact_match.index[0]
                matched_title = fetched_title
            else:
                movie_idx = partial_match.index[0]
                matched_title = self.df.iloc[movie_idx]['title']
                print(f"Exact match not found. Using partial match: '{matched_title}'")
        else:
            movie_idx = exact_match.index[0]
            matched_title = self.df.iloc[movie_idx]['title']
            
        # Calculate cosine similarity for this specific movie against all others
        # (This is faster than computing the entire N x N matrix if N is large)
        cosine_sim = cosine_similarity(self.tfidf_matrix[movie_idx], self.tfidf_matrix).flatten()
        
        # Get the indices of the top N most similar movies
        # We use [::-1] to reverse the sort order (descending), and skip the first one [1:top_n+1] 
        # because the most similar movie to X is X itself.
        similar_indices = cosine_sim.argsort()[-(top_n+1):][::-1][1:]
        
        recommendations = []
        for i in similar_indices:
            recommendations.append({
                "title": self.df.iloc[i]['title'],
                "similarity_score": round(cosine_sim[i], 3),
                "genres": self.df.iloc[i]['genres']
            })
            
        return {"target": matched_title, "recommendations": recommendations}


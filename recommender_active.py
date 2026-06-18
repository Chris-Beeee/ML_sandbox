import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

class ActiveRecommender:
    def __init__(self, dataset_path="movies_dataset.csv", feedback_file="user_feedback.csv"):
        self.dataset_path = dataset_path
        self.feedback_file = feedback_file
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feedback_df = None
        self.model = None
        self.is_trained = False
        
        self.load_data()
        self.load_feedback()
        self.train_model()
        
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

    def load_feedback(self):
        if os.path.exists(self.feedback_file):
            self.feedback_df = pd.read_csv(self.feedback_file)
        else:
            self.feedback_df = pd.DataFrame(columns=['movie_id', 'liked'])

    def save_feedback(self):
        self.feedback_df.to_csv(self.feedback_file, index=False)

    def add_feedback(self, movie_id, liked):
        """Adds a rating (1 for like, 0 for dislike) for a movie_id."""
        # Remove old rating if it exists
        self.feedback_df = self.feedback_df[self.feedback_df['movie_id'] != movie_id]
        
        # Append new rating
        new_row = pd.DataFrame([{'movie_id': movie_id, 'liked': liked}])
        self.feedback_df = pd.concat([self.feedback_df, new_row], ignore_index=True)
        self.save_feedback()
        
        # Retrain instantly
        self.train_model()

    def clear_feedback(self):
        """Clears all feedback and resets the model."""
        self.feedback_df = pd.DataFrame(columns=['movie_id', 'liked'])
        if os.path.exists(self.feedback_file):
            os.remove(self.feedback_file)
        self.is_trained = False
        self.model = None

    def train_model(self):
        """Trains the Logistic Regression model if there are enough diverse ratings."""
        if len(self.feedback_df) == 0:
            self.is_trained = False
            return
            
        # Filter out skips to see how many real classes we have
        valid_ratings = self.feedback_df[self.feedback_df['liked'] != -1]
        unique_classes = valid_ratings['liked'].unique()
        if len(unique_classes) < 2:
            # We need at least one 1 and one 0 to train a classifier
            self.is_trained = False
            return
            
        # Get indices of rated movies (excluding skips)
        rated_indices = []
        labels = []
        
        for _, row in self.feedback_df.iterrows():
            if row['liked'] == -1:
                continue # Ignore skips for ML training
                
            match = self.df[self.df['id'] == row['movie_id']]
            if not match.empty:
                rated_indices.append(match.index[0])
                labels.append(row['liked'])
                
        if len(set(labels)) < 2:
            self.is_trained = False
            return

        # Prepare Training Data (X=TF-IDF vectors of rated movies, y=liked labels)
        X_train = self.tfidf_matrix[rated_indices]
        y_train = np.array(labels)
        
        # We use class_weight='balanced' to handle if you have 10 likes and 1 dislike
        self.model = LogisticRegression(class_weight='balanced', random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def get_next_movie(self):
        """Returns the best next movie to present to the user."""
        rated_ids = self.feedback_df['movie_id'].tolist()
        
        # Find all unrated movies
        unrated_mask = ~self.df['id'].isin(rated_ids)
        unrated_indices = self.df.index[unrated_mask].tolist()
        
        if not unrated_indices:
            return None # You've rated the entire database!
            
        if not self.is_trained:
            # Cold Start: Pick a random popular movie from the unrated list
            # We'll just pick randomly from the top 100 remaining to get good mainstream samples
            import random
            random_idx = random.choice(unrated_indices[:100] if len(unrated_indices) > 100 else unrated_indices)
            movie = self.df.iloc[random_idx]
            return {"id": movie['id'], "title": movie['title'], "genres": movie['genres'], "overview": movie['overview'], "is_ml": False}
            
        # ML Mode: Predict probabilities for ALL unrated movies
        X_unrated = self.tfidf_matrix[unrated_indices]
        
        # predict_proba returns an array of shape (n_samples, n_classes). 
        # Column 1 is the probability of class 1 (Liked)
        probabilities = self.model.predict_proba(X_unrated)[:, 1]
        
        is_exploration = False
        epsilon = np.random.rand()
        
        # 20% Exploration (Epsilon-Greedy)
        if epsilon < 0.20:
            # Find wildcard movies in the middle of the decision boundary (40% to 60%)
            wildcard_mask = (probabilities >= 0.40) & (probabilities <= 0.60)
            wildcard_indices = np.where(wildcard_mask)[0]
            
            # If none found, expand the boundary slightly
            if len(wildcard_indices) == 0:
                wildcard_mask = (probabilities >= 0.30) & (probabilities <= 0.70)
                wildcard_indices = np.where(wildcard_mask)[0]
                
            if len(wildcard_indices) > 0:
                best_local_idx = np.random.choice(wildcard_indices)
                is_exploration = True
            else:
                # Fallback to exploitation if no middle ground exists
                best_local_idx = np.argmax(probabilities)
        else:
            # 80% Exploitation
            best_local_idx = np.argmax(probabilities)
            
        best_global_idx = unrated_indices[best_local_idx]
        
        movie = self.df.iloc[best_global_idx]
        return {
            "id": movie['id'], 
            "title": movie['title'], 
            "genres": movie['genres'], 
            "overview": movie['overview'],
            "probability": round(probabilities[best_local_idx] * 100, 2),
            "is_ml": True,
            "is_exploration": is_exploration
        }

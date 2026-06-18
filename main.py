import sys
import os
from recommender import MovieRecommender

def main():
    print("Welcome to the TMDB Content-Based Recommender System!")
    print("-----------------------------------------------------")
    
    if not os.path.exists("movies_dataset.csv"):
        print("Dataset not found. Please run 'python fetch_data.py' first.")
        sys.exit(1)
        
    print("Loading Machine Learning model and dataset...")
    try:
        recommender = MovieRecommender()
    except Exception as e:
        print(f"Error loading recommender: {e}")
        sys.exit(1)
        
    print(f"Model loaded successfully with {len(recommender.df)} movies.\n")
    
    while True:
        try:
            user_input = input("\nEnter a movie you like (or type 'quit' to exit): ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            results = recommender.get_recommendations(user_input, top_n=5)
            
            if isinstance(results, str):
                # If it's a string, it's an error message (movie not found)
                print(f"-> {results}")
            else:
                target_movie = results['target']
                print(f"\nSince you liked '{target_movie}', you might also enjoy:")
                for i, rec in enumerate(results['recommendations'], 1):
                    print(f"  {i}. {rec['title']} (Score: {rec['similarity_score']}) - {rec['genres']}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()

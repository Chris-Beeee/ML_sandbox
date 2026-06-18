import sys
import os
from recommender_active import ActiveRecommender

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("Welcome to the TMDB Active Learning Recommender!")
    print("------------------------------------------------")
    
    if not os.path.exists("movies_dataset.csv"):
        print("Dataset not found. Please run 'python fetch_data.py' first.")
        sys.exit(1)
        
    print("Loading Machine Learning model and dataset...")
    try:
        recommender = ActiveRecommender()
    except Exception as e:
        print(f"Error loading recommender: {e}")
        sys.exit(1)
        
    # Main Loop
    while True:
        try:
            clear_screen()
            print("="*60)
            if recommender.is_trained:
                print("🧠 ML MODEL ACTIVE: Showing highest probability matches")
            else:
                print("🎲 COLD START: Rate some movies to train the model!")
            print("="*60)
            
            # Get the next movie to show
            movie = recommender.get_next_movie()
            
            if not movie:
                print("You have rated every single movie in the dataset! Wow!")
                break
                
            # Display Movie
            print(f"\n🎥 TITLE  : {movie['title']}")
            print(f"🎭 GENRES : {movie['genres']}")
            if movie['is_ml']:
                print(f"🎯 ML PREDICTED MATCH : {movie['probability']}%")
            print(f"\n📖 OVERVIEW:\n{movie['overview']}")
            print("\n" + "-"*60)
            
            # Get Feedback
            while True:
                user_input = input("Do you like this kind of movie? (y = Yes / n = No / s = Skip / r = Reset / q = Quit): ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    print("\nSaving feedback and exiting. Goodbye!")
                    sys.exit(0)
                elif user_input in ['r', 'reset', 'clear']:
                    print("\nScrubbing your feedback history. Neural network reset!")
                    recommender.clear_feedback()
                    break
                elif user_input in ['s', 'skip']:
                    # We skip by just ignoring it and not calling add_feedback
                    # But wait, if we don't add feedback, get_next_movie might return it again next time.
                    # Let's add a dummy rating of -1 to 'ignore' it, or just pass. 
                    # For simplicity, we can just give it a temporary "skip" state, but since we are picking the max prob, 
                    # it will keep showing up if it's the max prob. 
                    # Let's just add it to feedback as a 'skipped' (-1) so we don't see it again.
                    recommender.add_feedback(movie['id'], -1)
                    break
                elif user_input in ['y', 'yes']:
                    recommender.add_feedback(movie['id'], 1)
                    break
                elif user_input in ['n', 'no']:
                    recommender.add_feedback(movie['id'], 0)
                    break
                else:
                    print("Invalid input. Please type y, n, s, or q.")
                    
        except KeyboardInterrupt:
            print("\nSaving feedback and exiting. Goodbye!")
            break

if __name__ == "__main__":
    main()

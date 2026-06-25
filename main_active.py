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
            valid_feedback = recommender.feedback_df[recommender.feedback_df['liked'] != -1] if hasattr(recommender, 'feedback_df') and not recommender.feedback_df.empty else []
            
            if recommender.is_trained:
                print("🧠 ML MODEL ACTIVE: Showing highest probability matches")
            else:
                if len(valid_feedback) > 0 and len(valid_feedback['liked'].unique()) < 2:
                    print("🎲 COLD START: The AI needs at least one 'Yes' AND one 'No' vote to learn!")
                else:
                    print("🎲 COLD START: Rate some movies to train the model!")
            print("="*60)
            
            # Seed Movie Prompt (only if NO feedback at all)
            if len(recommender.feedback_df) == 0:
                print("\nWould you like to start with a specific movie? (y/n)")
                ans = input("> ").strip().lower()
                if ans == 'y':
                    query = input("\nEnter movie title to search: ").strip()
                    results = recommender.search_movies(query)
                    
                    if not results:
                        print(f"\nNo local matches found for '{query}'. Starting with random movie.")
                        input("Press Enter to continue...")
                    else:
                        print(f"\nFound {len(results)} matches:")
                        for i, r in enumerate(results[:10], 1):
                            print(f"  {i}. {r['title']} ({r['year']})")
                            
                        choice = input("\nEnter the number of the movie (or press Enter to skip): ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(results[:10]):
                            selected = results[int(choice)-1]
                            print(f"\nAwesome! Seeded model with: {selected['title']}.")
                            recommender.add_feedback(selected['id'], 1)
                            continue # Restart loop to re-evaluate state
            
            # Get the next movie to show
            movie = recommender.get_next_movie()
            
            if not movie:
                print("You have rated every single movie in the dataset! Wow!")
                break
                
            # Display Movie
            print(f"\n🎥 TITLE  : {movie['title']}")
            print(f"🎭 GENRES : {movie['genres']}")
            if movie['is_ml']:
                if movie.get('is_exploration'):
                    print(f"🎲 ML WILDCARD EXPLORATION : {movie['probability']}% (Testing boundaries!)")
                else:
                    print(f"🎯 ML PREDICTED MATCH : {movie['probability']}%")
            print(f"\n📖 OVERVIEW:\n{movie['overview']}")
            print("\n" + "-"*60)
            
            # Get Feedback
            while True:
                user_input = input("Do you like this kind of movie? (y=Yes / n=No / b=Bad Execution / u=Unseen / r=Reset / q=Quit): ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    print("\nSaving feedback and exiting. Goodbye!")
                    sys.exit(0)
                elif user_input in ['r', 'reset', 'clear']:
                    print("\nScrubbing your feedback history. Neural network reset!")
                    recommender.clear_feedback()
                    break
                elif user_input in ['b', 'bad']:
                    print("  -> Understood! The movie is bad, but the genre is good. Logging as a YES for the neural network.")
                    recommender.add_feedback(movie['id'], 1)
                    break
                elif user_input in ['u', 'unseen', 's', 'skip']:
                    # User hasn't seen it, we ignore it for training by logging a -1
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

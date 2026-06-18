import sys
import os
from recommender_profile import ProfileRecommender

def print_menu(profile_length):
    print("\n" + "="*50)
    print(f"USER PROFILE: {profile_length} movies liked.")
    print("="*50)
    print("Commands:")
    print("  [movie name] : Add a movie to your profile")
    print("  recs [num]   : Generate [num] recommendations (default 5)")
    print("  clear        : Wipe your profile clear and start over")
    print("  quit         : Exit the application")
    print("="*50)

def main():
    print("Welcome to the TMDB User Profile Recommender!")
    print("---------------------------------------------")
    
    if not os.path.exists("movies_dataset.csv"):
        print("Dataset not found. Please run 'python fetch_data.py' first.")
        sys.exit(1)
        
    print("Loading Machine Learning model and dataset...")
    try:
        recommender = ProfileRecommender()
    except Exception as e:
        print(f"Error loading recommender: {e}")
        sys.exit(1)
        
    print(f"Model loaded successfully with {len(recommender.df)} movies.\n")
    
    if recommender.history:
        print(f"Loaded existing profile with {len(recommender.history)} movies:")
        for m in recommender.history:
            print(f" - {m}")
    
    while True:
        try:
            print_menu(len(recommender.history))
            user_input = input("Command > ").strip()
            
            if not user_input:
                continue
                
            command = user_input.lower()
                
            if command in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            elif command == 'clear':
                recommender.clear_profile()
                print("Profile cleared.")
                
            elif command.startswith('recs'):
                # Allow user to specify number of results, e.g. "recs 10"
                parts = command.split()
                top_n = 5
                if len(parts) > 1 and parts[1].isdigit():
                    top_n = int(parts[1])
                    
                results = recommender.get_profile_recommendations(top_n=top_n)
                
                if isinstance(results, str):
                    print(f"\n-> {results}")
                else:
                    print(f"\nBased on your unique profile, here are {top_n} matches:")
                    for i, rec in enumerate(results, 1):
                        print(f"  {i}. {rec['title']} (Final: {rec['final_score']} | Sim: {rec['similarity_score']} | Rating: {rec['vote_average']}/10) - {rec['genres']}")
                        
            else:
                # Treat input as a movie title
                print(f"Adding '{user_input}' to profile...")
                result_msg = recommender.add_to_profile(user_input)
                print(f"-> {result_msg}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()

import sys
import os
from recommender_profile import ProfileRecommender

def print_menu(profile_length):
    print("\n" + "="*50)
    print(f"USER PROFILE: {profile_length} movies liked.")
    print("="*50)
    print("Commands:")
    print("  [movie name] : Add a movie to your profile (Offline-first)")
    print("  fetch [name] : Add a movie, but bypass offline and force TMDB Search")
    print("  recs [num]   : Generate [num] recommendations (default 5)")
    print("  remove [name]: Remove a movie (or franchise) from your profile")
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
            year = m.get('year', 'Unknown')
            print(f" - {m['title']} ({year})")
    
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
                # Allow user to specify number of results and optional genre filters, e.g. "recs 10 action comedy"
                parts = command.split()
                top_n = 5
                filter_genres = []
                
                if len(parts) > 1:
                    if parts[1].isdigit():
                        top_n = int(parts[1])
                        filter_genres = parts[2:]
                    else:
                        top_n = 5
                        filter_genres = parts[1:]
                    
                results = recommender.get_profile_recommendations(top_n=top_n, filter_genres=filter_genres)
                
                if isinstance(results, str):
                    print(f"\n-> {results}")
                else:
                    print(f"\nBased on your unique profile, here are {top_n} matches:")
                    for i, rec in enumerate(results, 1):
                        print(f"  {i}. {rec['title']} (Final: {rec['final_score']} | Sim: {rec['similarity_score']} | Rating: {rec['vote_average']}/10) - {rec['genres']}")
                        
            elif command.startswith('fetch '):
                movie_name = user_input[6:].strip()
                print(f"Fetching '{movie_name}' from TMDB...")
                result_msg = recommender.add_to_profile_online(movie_name)
                print(f"-> {result_msg}")
                
            elif command.startswith('remove '):
                movie_name = user_input[7:].strip()
                print(f"Removing '{movie_name}' from profile...")
                result_msg = recommender.remove_from_profile(movie_name)
                print(f"-> {result_msg}")
                        
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

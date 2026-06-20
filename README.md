# Local Machine Learning Movie Recommender

This project is a local, offline Machine Learning recommendation engine that analyzes your movie history and predicts what you will want to watch next using TF-IDF Vectorization and Cosine Similarity.

## Core Scripts

- `fetch_data.py`: Connects to the live TMDB API to download a snapshot of the global Top 4,000 most popular and top-rated movies into a local offline matrix (`movies_dataset.csv`).
- `main_profile.py`: The interactive CLI that allows you to add movies to your profile, auto-detect franchises, and generate weighted recommendations.
- `recommender_profile.py`: The brain of the operation. Handles the natural language processing, TF-IDF vector math, and pooling logic to rank your recommendations.

---

## Recent Architectural Upgrades

The ML engine recently underwent a massive overhaul to solve several inherent flaws in mathematical text analysis and global API queries.

### 1. Max Pooling vs. Average Pooling
**The Problem:** When adding an massive 26-movie franchise (like James Bond) to a user's profile, calculating the profile using an "Average Pooling" vector diluted highly specific, rare plot keywords (like "MI6" or "espionage") because they were averaged out across 26 massive overviews.
**The Solution:** Switched the mathematical algorithm to **Max Pooling** (`liked_vectors.max(axis=0)`). The engine now preserves the *maximum strength* of every unique keyword it finds across your history. This ensures highly specific plot points are permanently locked into your profile at full strength.

### 2. Genre Weighting Boost
**The Problem:** Because an overview paragraph has 100+ words, while a genre string only has 3 words (`Action, Adventure, Thriller`), the mathematical frequency of the genre tags was completely overpowered by random generic words in the overview text (like "man", "world", "save").
**The Solution:** The engine now artificially duplicates genre tags (repeating them 2x) before creating the TF-IDF matrix. This physically forces the math to prioritize genre matching without completely ignoring the plot overviews.

### 3. Review Score Multiplier
**The Problem:** The engine only cared about text similarity. A terrible, universally panned sequel (e.g. *Terminator 3*) would naturally outrank a critically acclaimed masterpiece (e.g. *Terminator 2*) simply because its plot summary shared more identical words.
**The Solution:** The database now natively extracts TMDB's 1-10 `vote_average` score. The engine calculates the similarity score, and then applies a mathematical modifier: `Final Score = Similarity Score * (Vote Average / 10)`. This allows critically acclaimed films to retain 90%+ of their score, while awful films lose upwards of 60% of their score. 

### 4. The Foreign Film Handicap
**The Problem:** TMDB's global `/movie/popular` endpoint returns the most popular movies *globally*. Because the Chinese and Indian domestic markets are massive, their domestic hits (*Baaghi 3*, *The Legend of Hei*) naturally rank in the global Top 4000. Because their overviews were translated to English, the ML engine enthusiastically recommended them alongside Western blockbusters.
**The Solution:** The database now extracts the `original_language` tag. The ML engine checks this tag and applies a brutal **50% mathematical penalty** to any film that is not natively English (`en`). Foreign films can still appear in your recommendations, but only if they have flawless similarity and extreme critical acclaim to offset the handicap.

### 5. Franchise Discovery Previews
**The Problem:** When adding a movie (e.g., *The Godfather*), the engine would detect the franchise and blindly ask: `Would you like to add the entire franchise to your profile? (y/n)`. Users had no idea if the franchise contained 3 films or 27 films.
**The Solution:** The engine now intercepts the prompt, dynamically fetches the franchise collection from TMDB in real-time, and prints a bulleted preview list of every single movie (and its release year) contained within that collection *before* asking the user to confirm the addition.

### 6. Database Expansion
**The Problem:** The local offline matrix was hard-capped at 1,000 movies. Older, critically acclaimed films (like the original *John Wick*) had naturally fallen out of the global Top 1000 trending list, meaning they didn't exist in the local dataset and literally could not be recommended.
**The Solution:** The offline matrix limit was massively expanded to over 4,000 unique movies, ensuring that classic masterpieces and older franchise entries are permanently preserved in the local TF-IDF calculations.

### 7. Dynamic Genre Filtering
**The Problem:** Because Max Pooling creates a massive, blended profile of everything a user loves (e.g., Action, Romance, Horror), asking for general recommendations often resulted in an overwhelming mix of genres that didn't fit a specific mood.
**The Solution:** The `recs` command was updated to accept optional genre filters (e.g., `recs 5 Action Comedy`). The engine still relies on the user's entire profile to understand their pacing and textual preferences, but strictly filters the final output to only include movies that match the requested genres.

### 8. The Ingestion Firewall
**The Problem:** Because TMDB is a crowdsourced database, global queries often return movies with completely blank English overviews, or obscure regional films with entirely non-Latin titles (e.g., Chinese characters). These movies broke the terminal UI and were literally invisible to the text-based TF-IDF engine.
**The Solution:** An aggressive `is_valid_movie` firewall was bolted onto `fetch_data.py`. It intercepts every single API request and drops any movie that lacks an overview or contains non-Latin characters in its title *before* it can enter the offline dataset. 

### 9. Active Learning Neural Wipe
**The Problem:** The `main_active.py` script continuously trained its Logistic Regression model on the user's historical `y/n` feedback, but there was no way to reset the model if the user wanted to start a fresh training session.
**The Solution:** A Hard Reset (`r`) option was added to the Active Learning CLI. Triggering it instantly obliterates the `user_feedback.csv` file, wipes the model from memory, and throws the engine back into "Cold Start" randomized mode.

### 10. TMDB Community Keywords Integration
**The Problem:** The "Vocabulary Mismatch Problem." Because the TF-IDF engine relies strictly on the literal words present in a movie's plot overview, it mathematically failed to connect movies that were semantically identical but descriptively different (e.g., *The Batman* is described as a "dark detective crime thriller," while *Spiderman* is described as a "superhero saving the world with magic/powers"). 
**The Solution:** The pipeline was heavily upgraded to actively scrape **TMDB Community Keywords** (e.g., `superhero`, `based on comic`, `spoof`) for every movie. These human-assigned tags are injected directly into the ML feature space with a **2x Mathematical Multiplier**, granting both ML engines immediate semantic awareness and successfully bridging massive vocabulary gaps between related genres.

### 11. Execution vs Taste Split
**The Problem:** The Active Learning neural network only learns from textual features (genres, keywords, plot). If the UI asked "Do you like this?" and the user voted `n` because the movie was poorly executed (e.g., 2/10 rating), the model incorrectly assumed the user hated the *genre*.
**The Solution:** A `b = Bad Execution (But I like the genre)` option was added to the quiz. This mathematically records a positive vote (`1`) for the textual features, correctly teaching the model that the user's *taste* aligns with the genre, regardless of the individual film's quality.

### 12. Discovery Temperature
**The Problem:** Because the Profile Recommendation engine uses a deterministic mathematical matrix, asking for recommendations for a static profile (e.g., `recs horror`) yielded the exact same Top 5 films every single time.
**The Solution:** A "Discovery Temperature" was injected into the final scoring equation in `recommender_profile.py`. It applies a randomized mathematical penalty (between `0.85` and `1.0`) to every candidate film's score. This gently shuffles the mathematical rankings, allowing close-tie films to bubble to the surface and ensuring completely fresh recommendations on every search.

### 13. Epsilon-Greedy Exploration
**The Problem:** The Active Learning neural network operated with 100% "Greedy Exploitation," mathematically meaning it *always* returned the #1 highest probability movie. If a user exclusively voted "yes" for one genre, the engine became trapped in a 99% probability echo chamber, refusing to ever test other genres.
**The Solution:** A classic **Epsilon-Greedy Algorithm** was implemented. The engine rolls a virtual 100-sided die: 80% of the time it performs normal Exploitation, but 20% of the time it enters **Exploration Mode**. In Exploration mode, it explicitly targets a "Wildcard" movie hovering between 30%-70% probability. This systematically tests the user on unfamiliar genres to constantly challenge and widen the decision boundary.


### 14. Online Search Override
**The Problem:** Because the engine prioritized local, offline database searches to save API calls, it would aggressively auto-match the first exact title it found locally. This created a "Remake Trap"—if the local database happened to contain a 2009 remake of a film, but the user wanted the 1980 original, the engine blindly auto-selected the remake and blocked the user from fetching the original.
**The Solution:** A dedicated `fetch [movie name]` command was added to the Profile Recommender. This explicitly overrides the local database priority, bypassing local matches entirely, and punches straight through to the TMDB API to present a full disambiguation menu of all online results.

### 15. Dynamic CSV Alignment
**The Problem:** When appending newly fetched movies into the offline CSV, the dictionary keys were being injected out-of-order due to the recent Keywords upgrade. Because pandas appends data row-by-row based on dictionary key order, keyword strings were accidentally written into the `vote_average` column, crashing the ML engine when it attempted to run math on the string data.
**The Solution:** A hard-coded alignment firewall was added to `fetch_data.py`'s append logic. Before any dynamic fetch is written to disk, the script forcibly drops incompatible columns (like `release_date`) and re-indexes the dictionary keys to perfectly match the 7-column CSV schema, preventing data shifting.
### 16. Remake Collisions
**The Problem:** Previously, the engine saved your history as a list of text titles (`["Hellraiser"]`). If you added the 2022 *Hellraiser*, you could never add the 1987 *Hellraiser* because the script thought you already had it!
**The Solution:** Your history is now tracked entirely by Unique TMDB IDs, rather than text titles. You can now add infinite movies with the exact same name to your profile without the engine locking you out. When the script booted up, it automatically scanned your existing text-based history, found the correct IDs, and safely migrated your entire profile to the new JSON architecture!
### 17. Pipeline ID Handoff Refactor
**The Problem:** Even after migrating `user_history.json` to an ID-based schema, the TMDB Search API was returning the title *string* back to the recommender. The recommender would then search the offline dataframe for that string to find its ID, accidentally grabbing the wrong ID if multiple movies shared the same name.
**The Solution:** The `fetch_data.py` pipeline was refactored to pass the exact TMDB ID securely into the `recommender_profile.py` engine alongside the title, completely bypassing the fatal string-based lookup.

### 18. Release Year Backfill & Disambiguation
**The Problem:** After adding support for multiple movies with the same name, the CLI simply displayed `- Hellraiser` twice with no contextual data, making it impossible to tell which one was the original and which was the remake.
**The Solution:** An automated migration script queried the TMDB API to fetch and backfill the release year into `user_history.json` for all existing user movies. The core `fetch` and display pipelines were updated to natively append and present the `(year)` alongside titles to eliminate UI confusion.

### 19. Surgical Remove Command with Franchise Purging
**The Problem:** The only way to remove a bad recommendation was to use the `clear` command to wipe the entire profile.
**The Solution:** A dedicated `remove [name]` command was implemented. It features a local disambiguation menu if multiple matches are found in the user's profile. Crucially, it mirrors the "add" workflow: if a user elects to remove a movie that belongs to a franchise, the script detects it and prompts the user to purge the *entire* franchise from their history in one keystroke.
### 20. Profile Inspection Command
**The Problem:** To check the current state of their history, users previously had to exit the CLI interface and manually inspect `user_history.json` or reboot the application entirely to trigger the startup printout.
**The Solution:** A `profile` command was added to the main CLI menu. This command instantly prints out the user's entire history array directly within the terminal, fully formatted with release dates, allowing for rapid verification of additions or removals without breaking workflow.

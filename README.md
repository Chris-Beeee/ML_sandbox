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

---

## Known ML Limitations & Future Roadmap

### The Vocabulary Mismatch Problem
Because the current engine uses TF-IDF (Term Frequency - Inverse Document Frequency), it relies entirely on **exact textual keyword matching** within the TMDB plot summaries and genres. It lacks true semantic awareness.
- **The Symptom:** If a user adds the *Scream* franchise to their profile, the engine will confidently recommend other actual teen slashers (like *Bodies Bodies Bodies*) over direct slapstick parodies (like *Scary Movie*). This is because actual slashers share the dark, violent vocabulary of *Scream* (e.g., "killer", "deadly", "game"), whereas a parody's plot summary uses completely different semantic vocabulary (e.g., "spoof", "hilarious", "laughs").
- **Future Solutions:** To overcome this, the engine could be upgraded to use **Collaborative Filtering** (recommending based on human behavior trends rather than text), or it could integrate TMDB's community **Keywords** matrix (which explicitly tags both movies with `spoof` or `teen-slasher` to bridge the semantic gap).

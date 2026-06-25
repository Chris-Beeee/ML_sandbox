
# Hybrid Machine Learning Movie Recommender (CLI Experiment)

**A personal experiment applying Software QA practices to Machine Learning.**

This is a hybrid (offline-first + TMDB API fallback) recommendation engine built with TF-IDF Vectorization and Cosine Similarity. It includes two modes:

- **Profile-based recommendations** — Build a persistent user profile and get weighted suggestions.
- **'Tinder for movies' (Active Learning)** — Interactive session with Epsilon-Greedy exploration to avoid echo chambers.

**Goal of the project**: Test how QA techniques (edge-case hunting, data validation, risk identification, iterative refinement, bias mitigation, and regression testing) apply when building and evolving an ML system from scratch.

**Tech Stack**: Python, scikit-learn, pandas, TMDB API, Pytest

### Key Features
- Offline dataset (4,000+ movies) for speed and privacy, with seamless live API fallback
- Max Pooling + Genre Boosting + Review Score weighting
- Franchise auto-detection with preview
- Epsilon-Greedy active learning (exploration vs exploitation)
- Robust handling of real-world data messiness (remakes, foreign films, vocabulary mismatch, missing data)
- Parameterised and randomised Pytest suites for validation

### Quick Start

1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your TMDB API key (optional — works with offline data only)
4. Run one of the mains:
   - `python main_profile.py` → Profile mode
   - `python main_active.py` → Active learning / Tinder-style mode

**Note**: A pre-built `movies_dataset.csv` is included so you can try it immediately without an API key.

---

### QA Challenges & Solutions Applied

This project was deliberately used as a vehicle to apply structured QA thinking to ML development. Below are the major issues I identified and resolved during rapid iteration:

#### Data Quality & Ingestion
- **Ingestion Firewall** — Dropped invalid entries (blank overviews, non-Latin titles) before they entered the dataset.
- **Dynamic CSV Alignment** — Fixed column shifting when appending new fields (e.g., keywords).
- **Database Expansion** — Increased from 1k to 4k+ movies to retain older classics.
- **Release Year Backfill & Disambiguation** — Improved handling of remakes with identical titles.

#### Recommendation Logic & Mathematical Issues
- **Max Pooling vs Average Pooling** — Prevented dilution of rare keywords when adding large franchises (e.g., James Bond).
- **Genre Weighting Boost** — Duplicated genre tags to stop them being overpowered by plot text.
- **Review Score Multiplier** — Penalised poorly rated films even if text similarity was high.
- **Foreign Film Handicap** — Applied language-based penalty to reduce irrelevant global-popularity bias.
- **TMDB Community Keywords Integration** — Bridged vocabulary mismatch (e.g., "dark detective" vs "superhero").
- **Discovery Temperature** — Added controlled randomness to avoid identical recommendations every time.

#### User Experience & Active Learning
- **Franchise Discovery Previews** — Show full list + years before bulk-adding.
- **Epsilon-Greedy Exploration** — Prevented echo chambers by forcing occasional wildcards (20% exploration rate).
- **Execution vs Taste Split** — Added option to separate "bad execution" from "dislike genre".
- **Hard Reset** — Full model wipe for fresh cold-start sessions.
- **Online Search Override** — `fetch [title]` command to bypass local matches for remakes.

(Additional detailed entries and regression fixes are in the commit history.)

### Testing Approach
- Chose movies because of strong domain knowledge → rapid oracle-based testing.
- Added parameterised + randomised Pytest suites for validation.
- Next step: Design a testing strategy usable by non-domain experts.

---

**Status**: Experimental / Rapid prototype (started ~1 week ago). The focus has been on applying QA rigour rather than building a polished consumer product.

Feel free to explore the code and tests. Feedback welcome!

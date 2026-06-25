# Hybrid Machine Learning Movie Recommender (CLI Experiment)

**A personal experiment applying Software QA practices to Machine Learning.**

This is a hybrid (offline-first + TMDB API fallback) recommendation engine built with TF-IDF Vectorization and Cosine Similarity. It includes two modes:

- **Profile-based recommendations** — Build a persistent user profile and get weighted suggestions.
- **'Tinder for movies' (Active Learning)** — Interactive session with Epsilon-Greedy exploration to avoid echo chambers.

**Goal of the project:** Test how QA techniques (edge-case hunting, data validation, risk identification, iterative refinement, bias mitigation, and regression testing) apply when building and evolving an ML system from scratch.

**Tech Stack:** Python, scikit-learn, pandas, TMDB API, Pytest

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
3. Copy `.env.example` to `.env` and add your TMDB API key (optional — works fully offline)
4. Run one of the mains:
   - `python main_profile.py` → Profile mode
   - `python main_active.py` → Active learning / Tinder-style mode

**Note:** A pre-built `movies_dataset.csv` is included so you can try it immediately without an API key.

### Running the Tests

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Run all tests
pytest

# Run with verbose output + coverage
pytest -v --cov=.

# Run specific test files
pytest tests/test_data_validation.py -v
pytest tests/test_recommendation.py -v
pytest tests/test_active_learning.py -v

# Run with random ordering (great for spotting flaky edge cases)
pytest --randomly-seed=42

Key Test Suites:test_data_validation.py — Data quality & ingestion checks
test_recommendation.py — Core similarity logic, edge cases & bias checks
test_active_learning.py — Epsilon-Greedy and user profile behaviour

All tests run fully offline using the included movies_dataset.csv.




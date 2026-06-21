import pytest
import os
import json
import pandas as pd
from unittest.mock import patch
from recommender_profile import ProfileRecommender

@pytest.fixture
def mock_dataset(tmp_path):
    # Create a minimal dataset for testing
    df = pd.DataFrame([
        {"id": 1, "title": "The Matrix", "genres": "Action, Sci-Fi", "overview": "Hackers learn the truth.", "keywords": "simulation"},
        {"id": 2, "title": "The Matrix Reloaded", "genres": "Action, Sci-Fi", "overview": "Neo returns.", "keywords": "simulation"},
        {"id": 3, "title": "Inception", "genres": "Action, Sci-Fi", "overview": "Dream within a dream.", "keywords": "dreams"},
        {"id": 4, "title": "Toy Story", "genres": "Animation, Family", "overview": "Toys come alive.", "keywords": "toys"},
        {"id": 5, "title": "The Omen", "genres": "Horror", "overview": "Scary child.", "keywords": "demon"}
    ])
    
    dataset_path = tmp_path / "mock_dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    return str(dataset_path)

@pytest.fixture
def recommender(mock_dataset, tmp_path):
    history_file = str(tmp_path / "test_history.json")
    ignore_file = str(tmp_path / "test_ignore.json")
    
    # Write empty json arrays initially
    with open(history_file, 'w') as f:
        json.dump([], f)
    with open(ignore_file, 'w') as f:
        json.dump([], f)
        
    return ProfileRecommender(dataset_path=mock_dataset, history_file=history_file, ignore_file=ignore_file)

@patch('builtins.input', side_effect=['1']) # Automatically select the first match in menus
def test_add_to_profile(mock_input, recommender):
    # It should automatically add exact single matches without input
    result = recommender.add_to_profile("The Matrix")
    
    assert len(recommender.history) == 1
    assert recommender.history[0]["title"] == "The Matrix"
    assert "Added" in result

@patch('builtins.input', side_effect=['1'])
def test_add_to_ignore_list(mock_input, recommender):
    result = recommender.add_to_profile("Toy Story", target="ignore")
    
    assert len(recommender.ignore_list) == 1
    assert len(recommender.history) == 0
    assert recommender.ignore_list[0]["title"] == "Toy Story"
    assert "Ignored" in result
    
def test_recommendation_filtering(recommender):
    # Directly manipulate the profile
    recommender.history = [{"id": 1, "title": "The Matrix"}]
    
    # We should get The Matrix Reloaded and Inception as recommendations because of Sci-Fi
    recs = recommender.get_profile_recommendations(top_n=2)
    assert len(recs) > 0
    rec_titles = [r['title'] for r in recs]
    
    assert "The Matrix Reloaded" in rec_titles
    assert "The Matrix" not in rec_titles # Should filter itself
    
    # Now add Reloaded to ignore list
    recommender.ignore_list = [{"id": 2, "title": "The Matrix Reloaded"}]
    recs2 = recommender.get_profile_recommendations(top_n=2)
    rec2_titles = [r['title'] for r in recs2]
    
    assert "The Matrix Reloaded" not in rec2_titles # Should be ignored!
    
def test_genre_filter_empty_results(recommender):
    recommender.history = [{"id": 1, "title": "The Matrix"}]
    
    # A typo or non-existent genre should return empty, not crash
    recs = recommender.get_profile_recommendations(top_n=5, filter_genres=["invalid_genre_123"])
    assert len(recs) == 0

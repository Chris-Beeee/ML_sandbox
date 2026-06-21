import pytest
from unittest.mock import patch, MagicMock
import fetch_data
import os

@patch('fetch_data.requests.get')
@patch('builtins.input', side_effect=['1']) # Automatically select the first match in search
def test_search_and_append_movie(mock_input, mock_get, tmp_path, mocker):
    # Mock token
    mocker.patch('fetch_data.get_access_token', return_value="fake_token")
    
    # Mock TMDB Search Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "id": 555,
                "title": "Mock Movie Title",
                "overview": "A fake movie.",
                "genre_ids": [28],
                "vote_average": 8.0,
                "original_language": "en",
                "release_date": "2025-01-01"
            }
        ]
    }
    
    # Mock the keywords fetch which is a secondary API call
    mock_keywords_response = MagicMock()
    mock_keywords_response.status_code = 200
    mock_keywords_response.json.return_value = {"keywords": [{"name": "fake_keyword"}]}
    
    # Configure mock to return the correct response based on the URL it was called with
    def side_effect(url, *args, **kwargs):
        if "search/movie" in url:
            return mock_response
        elif "keywords" in url:
            return mock_keywords_response
        return MagicMock(status_code=404)
        
    mock_get.side_effect = side_effect
    
    # Point the dataset to a temporary path so we don't mess up the real one
    test_csv = tmp_path / "test_dataset.csv"
    mocker.patch('fetch_data.DATASET_FILE', str(test_csv))
    
    # Test
    result = fetch_data.search_and_append_movie("Mock Movie Title")
    
    # Validate it successfully processed the mock API response and returned the tuple
    assert len(result) == 1
    assert result[0][0] == 555
    assert result[0][1] == "Mock Movie Title"
    
    # Validate it saved to the mock dataset
    assert os.path.exists(test_csv)

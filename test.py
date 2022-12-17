import pytest
from app import analytics_emo


def test_analytics():
    assert analytics_emo("This is a good review")[0]["label"] == "LABEL_3"


if __name__ == "__main__":
    pytest.main(["-s", "test.py"])
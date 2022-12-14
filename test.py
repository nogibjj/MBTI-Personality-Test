import pytest


def test_a():
    assert 100 == 100


if __name__ == "__main__":
    pytest.main(["-s", "test.py"])

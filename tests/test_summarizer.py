# test_summarizer.py
import sys
from pathlib import Path

# Append the scripts directory to sys.path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from summarizer import extract_text_from_html, is_valid_arxiv_id, count_words


def test_extract_text_from_html():
    html_content = "<html><body><p>Test paragraph.</p></body></html>"
    text = extract_text_from_html(html_content)
    assert text == "Test paragraph."


def test_count_words():
    text = "This is a test string with eight words."
    assert count_words(text) == 8


def test_is_valid_arxiv_id():
    valid_id = "2312.12321v1"  # replace with a real valid ID for the test
    is_valid, _ = is_valid_arxiv_id(valid_id)
    assert is_valid

    invalid_id = "xxx"  # replace with a real invalid ID for the test
    is_valid, _ = is_valid_arxiv_id(invalid_id)
    assert not is_valid


# More tests can be added as needed

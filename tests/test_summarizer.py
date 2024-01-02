# test_summarizer.py
import sys
from pathlib import Path

# Append the scripts directory to sys.path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from summarizer import (
    extract_text_from_html,
    is_valid_arxiv_id,
    count_words,
    get_url_content,
)


def test_extract_text_from_html():
    html_content = """
        <html>
            <head>
                <style>
                    .ltx_page_content {
                        background-color: #f0f0f0;
                        padding: 20px;
                    }

                    .ltx_document {
                        background-color: #ffffff;
                        margin: 10px;
                        padding: 15px;
                        border: 1px solid #ddd;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    }

                    .ltx_section {
                        color: #333333;
                        font-family: Arial, sans-serif;
                        font-size: 16px;
                        margin: 5px 0;
                    }
                </style>
            </head>
            <body>
                <div class="ltx_page_content">
                    <div class="ltx_document">
                        <section class="ltx_section">Test paragraph.</section>
                    </div>
                </div>
            </body>
        </html>
        """
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


def test_valid_arxiv_html():
    valid_id = "https://browse.arxiv.org/html/2312.16171v1"  # replace with a real valid ID for the test
    response = get_url_content(valid_id)
    assert response is not None

    invalid_id = "https://browse.arxiv.org/html/2312.17581v1"  # replace with a real invalid ID for the test
    response = get_url_content(invalid_id)
    assert not (response is None)


# More tests can be added as needed

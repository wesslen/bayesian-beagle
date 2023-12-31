import typer
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import arxiv
import json
from pathlib import Path

app = typer.Typer()


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    client = arxiv.Client()
    search_by_id = arxiv.Search(id_list=[arxiv_id])
    try:
        first_result = next(client.results(search_by_id))
        return True, first_result
    except StopIteration:
        return False, None


def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def count_words(text: str) -> int:
    words = text.split()
    return len(words)


def summarize_text(text: str) -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to summarize academic \
                    articles. Output the summary as Markdown with Headings \
                    for different sections. You may also output code snippets \
                    following Markdown conventions.",
            },
            {
                "role": "user",
                "content": f"Please summarize the following text:\n\n{text}",
            },
        ],
    )

    return response.choices[0].message.content


def tldr_title(text: str) -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to summarize academic \
                    article abstracts. "
            },
            {
                "role": "user",
                "content": f"Write a tl;dr in fewer than 20 words for \
                    the following text:\n\n{text}",
            },
        ],
    )

    return response.choices[0].message.content


@app.command()
def summarize(input_jsonl: str, output_file_path: str = "data/output.jsonl"):
    """
    Summarizes texts from Arxiv HTML pages listed in a JSONL file using OpenAI's Chat API.

    Args:
    input_jsonl (str): Path to the input JSONL file.
    output_file_path (str): Output JSONL file path.
    """
    output_path = Path(output_file_path)
    with open(input_jsonl, "r") as file:
        for line in file:
            data = json.loads(line)
            arxiv_id = data["id"]
            categories = data["categories"]

            valid, first_result = is_valid_arxiv_id(arxiv_id)
            if not valid:
                typer.echo(f"Invalid Arxiv ID: {arxiv_id}")
                continue

            url = f"https://browse.arxiv.org/html/{arxiv_id}"
            response = requests.get(url)
            html_content = response.text

            word_count = count_words(html_content)
            if word_count > 15000:
                typer.echo(
                    f"Warning: HTML content for {arxiv_id} exceeds 15,000 tokens. Skipping."
                )
                continue

            text = extract_text_from_html(html_content)
            summary = summarize_text(text)
            tldr = tldr_title(first_result.summary)

            output_data = {
                "id": arxiv_id,
                "text": summary,
                "meta": {
                    "url": url,
                    "title": first_result.title,
                    "subtitle": tldr,
                    "categories": categories,
                    "publish_date": first_result.published.strftime("%Y-%m-%d"),
                },
            }

            with output_path.open("a") as output_file:
                output_file.write(json.dumps(output_data) + "\n")

            typer.echo(f"Summary for {arxiv_id} written to {output_path}")


if __name__ == "__main__":
    app()

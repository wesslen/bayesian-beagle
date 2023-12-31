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
    except:
        return False, None


def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def truncate_string(text, token_threshold):
    """
    Truncate a string after a specified number of word tokens.

    :param text: The input string.
    :param token_threshold: The maximum number of word tokens allowed.
    :return: A truncated version of the string if it exceeds the token threshold.
    """

    # Split the text into words (tokens)
    tokens = text.split()

    # Check if the number of tokens is greater than the threshold
    if len(tokens) > token_threshold:
        # Truncate the list of tokens and join back into a string
        truncated_text = ' '.join(tokens[:token_threshold])
        return truncated_text
    else:
        # If the text is within the limit, return it as is
        return text

def remove_double_quotes(input_string):
    return input_string.replace('"', '')

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

    # TODO: Generalize this to allow different models and prompts. This should be appended to each record. 
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
    # TODO: check if output_path exists, if so, read it in. check it includes "id" key. raise error if not.

    with open(input_jsonl, "r") as file:
        for line in file:
            data = json.loads(line)

            # TODO: if output_path exists, check "id" versus data. if found, skip with continue.
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
                    f"Warning: HTML content for {arxiv_id} exceeds 15,000 tokens. Truncating."
                )
                html_content = truncate_string(html_content)

            text = extract_text_from_html(html_content)
            summary = summarize_text(text)
            tldr = tldr_title(first_result.summary)

            output_data = {
                "id": arxiv_id,
                "text": summary,
                "meta": {
                    "url": url,
                    "title": remove_double_quotes(first_result.title),
                    "subtitle": remove_double_quotes(tldr),
                    "categories": categories,
                    "publish_date": first_result.published.strftime("%Y-%m-%d"),
                    "truncated": True if word_count > 15000 else False,
                },
            }

            with output_path.open("a") as output_file:
                output_file.write(json.dumps(output_data) + "\n")

            typer.echo(f"Summary for {arxiv_id} written to {output_path}")


if __name__ == "__main__":
    app()

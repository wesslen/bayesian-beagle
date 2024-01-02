import typer
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import arxiv
import json
from pathlib import Path
import logging
from strip_tags import strip_tags


class OpenAIAssistant:
    def __init__(self, model="gpt-3.5-turbo-1106"):
        self.client = OpenAI()
        self.model = model
        self.cache = {}

    def process_text(self, text: str, task: str) -> str:
        # Check if the request is already in the cache
        if (text, task) in self.cache:
            return self.cache[(text, task)]

        if task == "summarize":
            system_message = "You are a helpful assistant to summarize academic articles \
                              Output the summary as markdown with headings for different \
                              sections. Add in bolding for key terminology. Add in bullets \
                              to summarize sections. Start with three major takeways on the \
                              most important findings of the paper. End with a section critiquing \
                              the paper and raising any potential problems with the paper."
            user_message = f"Summarize the following text in 300 or fewer words:\n\n{text}"

        elif task == "tldr":
            system_message = "You are a helpful assistant to summarize academic \
                              article abstracts."
            user_message = f"Write a tl;dr in fewer than 20 words for \
                             the following text:\n\n{text}"

        else:
            raise ValueError("Invalid task specified")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

        result = response.choices[0].message.content
        # Cache the result
        self.cache[(text, task)] = result
        return result


app = typer.Typer()
# Instantiate the OpenAIAssistant
MODEL = "gpt-3.5-turbo-1106"
assistant = OpenAIAssistant(model=MODEL)

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    client = arxiv.Client()
    search_by_id = arxiv.Search(id_list=[arxiv_id])
    try:
        first_result = next(client.results(search_by_id))
        return True, first_result
    except BaseException:
        return False, None


def get_url_content(url: str):
    """
    Fetch the content from the provided URL.

    :param url: The URL to fetch the content from.
    :return: The content of the response if successful, None otherwise.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content.decode("utf-8")
        else:
            logger.error(
                f"Bad response from {url}, status code: {response.status_code}"
            )
            return None
    except requests.RequestException as e:
        logger.error(f"An error occurred while fetching {url}: {e}")
        return None


def extract_text_from_html(html_content: str) -> str:
    stripped = strip_tags(
        html_content,
        [".ltx_page_content"],
        removes=[
            ".ltx_authors",
            ".ltx_bibliography",
            ".package-alerts",
            ".section",
            ".ltx_table",
            ".ltx_listing",
            ".ltx_picture",
        ],
    )

    return stripped


def remove_double_quotes(input_string):
    """
    Removes all double quotes from the input string.

    Args:
    input_string (str): The string from which to remove double quotes.

    Returns:
    str: The input string with all double quotes removed.
    """
    return input_string.replace('"', "")


def count_words(text: str) -> int:
    words = text.split()
    return len(words)


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
        truncated_text = " ".join(tokens[:token_threshold])
        return truncated_text
    else:
        # If the text is within the limit, return it as is
        return text


def extract_first_png_image(html_content):
    """
    Extract the first .png image URL from a given HTML content.

    :param html_content: A string containing HTML content.
    :return: URL of the first .png image or None if no .png image is found.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")

        for img in soup.find_all("img"):
            if img["src"].endswith(".png"):
                return img["src"]

        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


@app.command()
def summarize(
    input_jsonl: str,
    output_file_path: str = "data/output.jsonl",
    force_generate_all: bool = False,
):
    """
    Summarizes texts from Arxiv HTML pages listed in a JSONL file using OpenAI's Chat API.

    Args:
    input_jsonl (str): Path to the input JSONL file.
    output_file_path (str): Output JSONL file path.
    """
    output_path = Path(output_file_path)

    # Check if output_path exists and read processed IDs
    existing_ids = set()
    if output_path.is_file():
        with output_path.open("r") as output_file:
            for line in output_file:
                try:
                    existing_data = json.loads(line)
                    existing_ids.add(existing_data["id"])
                except json.JSONDecodeError as e:
                    logging.error("Error reading output file: %s", e)

    with open(input_jsonl, "r") as file:
        for line in file:
            data = json.loads(line)
            arxiv_id = data["id"]
            categories = data["categories"]

            # Skip if already processed
            if force_generate_all is False:
                if arxiv_id in existing_ids:
                    logging.info(
                        f"Arxiv ID {arxiv_id} already processed. Skipping."
                    )
                    continue

            valid, first_result = is_valid_arxiv_id(arxiv_id)
            if not valid:
                logging.info(f"Invalid Arxiv ID: {arxiv_id}")
                continue

            url = f"https://browse.arxiv.org/html/{arxiv_id}"
            html_content = get_url_content(url)
            if html_content is None:
                logging.info(f"HTML output not available for {arxiv_id}")
                continue

            text = extract_text_from_html(html_content)

            # get first image
            try:
                png_url = extract_first_png_image(html_content)
                if png_url is None:
                    logger.warning("No .png image found in the HTML content.")
                else:
                    png_url = f"{url}/{png_url}"
                    print(f"Found .png image URL: {png_url}")

            except Exception as e:
                logger.error(f"Failed to extract .png image: {e}")

            # count words, if longer than 15,000 then truncate
            word_count = count_words(text)
            if word_count > 15000:
                logging.info(
                    f"Warning: HTML content for {arxiv_id} exceeds 15,000 tokens. Truncating."
                )
                text = truncate_string(text, token_threshold=15000)

            logging.info(f"{word_count} word counts")
            summary = assistant.process_text(text, "summarize")
            tldr = assistant.process_text(first_result.summary, "tldr")

            output_data = {
                "id": arxiv_id,
                "text": summary,
                "meta": {
                    "url": url,
                    "title": remove_double_quotes(first_result.title),
                    "subtitle": remove_double_quotes(tldr),
                    "categories": categories,
                    "publish_date": first_result.published.strftime(
                        "%Y-%m-%d"
                    ),
                    "model": MODEL,
                    "image": png_url,
                    "word_count": word_count,
                    "is_truncated": True if word_count > 15000 else False,
                },
            }

            with output_path.open("a") as output_file:
                output_file.write(json.dumps(output_data) + "\n")

            logging.info(f"Summary for {arxiv_id} written to {output_path}")


if __name__ == "__main__":
    app()

import os
import arxiv
import json
import logging
from pathlib import Path
import typer
import requests
from typing import List, Tuple
from bs4 import BeautifulSoup
from strip_tags import strip_tags
from langchain_community.document_loaders import TextLoader

from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate, jinja2_formatter
from langchain.docstore.document import Document
from langchain_community.document_loaders import ArxivLoader

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
app = typer.Typer()


def get_url(arxiv_id: str):
    return f"https://browse.arxiv.org/html/{arxiv_id}"

def preprocess_arxiv(arxiv_id: str) -> Tuple[List[Document],str,str]:
    url = get_url(arxiv_id)
    html_content = get_url_content(url)
    arxiv_docs = ArxivLoader(query=arxiv_id, load_max_docs=1).load()
    if html_content:
        abstract = get_abstract_section(url)
        cleaned_html = clean_html_section(html_content)
        docs = html_to_docs(cleaned_html)

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
        docs[0].metadata['Title'] = arxiv_docs[0].metadata['Title']
        docs[0].metadata['Authors'] = arxiv_docs[0].metadata['Authors']
        docs[0].metadata['Published'] = arxiv_docs[0].metadata['Published']
        docs[0].metadata['Summary'] = abstract
        docs[0].metadata['png_url'] = png_url
        docs[0].metadata['extraction'] = "HTML"

        return docs
    elif html_content is None:
        logging.info(f"HTML output not available for {arxiv_id}")
        logger.info(f"Using PDF and ArxivLoader instead for {arxiv_id}")
        arxiv_docs[0].metadata['png_url'] = None
        arxiv_docs[0].metadata['extraction'] = "PDF"
        return arxiv_docs

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
                f"Bad response from {url} status code: {response.status_code}"
            )
            return None
    except requests.RequestException as e:
        logger.error(f"An error occurred while fetching {url}: {e}")
        return None

def clean_html_section(html_content: str):
    stripped = strip_tags(
        html_content,
        [".ltx_page_content"],
        removes=[
            ".ltx_authors",
            ".ltx_bibliography",
            ".package-alerts",
            ".section",  # remove license,arxiv number header
            ".ltx_table",  # remove tables
            ".ltx_tabular",
            ".ltx_listing",
            ".ltx_picture",
            ".ltx_Math",
            ".ltx_equation",  # remove math
            ".ltx_theorem",
        ],
    )
    return stripped

def get_abstract_section(html_content: str) -> str:
    stripped = strip_tags(
        html_content,
        [".ltx_abstract"],
        removes=[
            ".ltx_title_abstract",
        ],
    )
    return stripped.replace("\n"," ")

def html_to_docs(html_content: str):
    filename = "/tmp/input.txt"

    with open(filename, "w") as file:
        file.write(html_content)
        logger.info(f"Write file {filename}")

    loader = TextLoader(filename)
    docs = loader.load()

    # Optionally delete the temporary file after use
    os.remove(filename)

    return docs

def is_valid_arxiv_id(arxiv_id: str):
    client = arxiv.Client()
    search_by_id = arxiv.Search(id_list=[arxiv_id])
    try:
        first_result = next(client.results(search_by_id))
        return True, first_result
    except BaseException:
        return False, None

def remove_double_quotes(input_string) -> str:
    """
    Removes all double quotes from the input string.

    Args:
    input_string (str): The string from which to remove double quotes.

    Returns:
    str: The input string with all double quotes removed.
    """
    return input_string.replace('"', "")

def read_file_as_string(file_name) -> str:
    try:
        with open(file_name, 'r') as file:
            # Reading the file content
            content = file.read()
            return content.replace("\n","")
    except FileNotFoundError:
        return "File not found."

def extract_first_png_image(html_content):
    """
    Extract the first .png image URL from the HTML content, searching through each
    <section class="ltx_section"> until a .png image is found.

    :param html_content: A string containing HTML content.
    :return: URL of the first .png image found in the sections, or None if no .png image is found.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")

        # Iterate through each section with class 'ltx_section'
        for section in soup.find_all("section", class_="ltx_section"):
            # Find the first img with .png in the current section
            img = section.find("img", src=lambda x: x and x.endswith(".png"))
            if img:
                return img["src"]

        return None
    except Exception as e:
        logger.error(f"An error occurred fetching the first image: {e}")
        raise

class OpenAIAssistant:
    def __init__(self, model="gpt-3.5-turbo-1106", temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.cache = {}  # Initialize cache

        # Set up Jinja2 environment
        self.template_env = Environment(loader=FileSystemLoader('templates/'))

    def count_token(self, text: str) -> int:
        token_count = len(text) // 4  # Rough estimate of token count
        return token_count

    def truncate_string(self, text, token_threshold):
        """
        Truncate a string after a specified number of characters based on token approximation.

        :param text: The input string.
        :param token_threshold: The maximum number of tokens allowed.
        :return: A truncated version of the string if it exceeds the character limit based on token approximation.
        """

        # Calculate the character limit (approximation based on 4 characters per token)
        char_limit = token_threshold * 4

        # Check if the number of characters is greater than the limit
        if len(text) > char_limit:
            # Truncate the text to the character limit
            truncated_text = text[:char_limit]
            return truncated_text
        else:
            # If the text is within the limit, return it as is
            return text

    def get_summary(self, arxiv_id, prompt_name) -> str:
        # Load the template based on prompt_name
        # template = self.template_env.get_template(f"{prompt_name}.jinja2")

        # jinja_prompt = jinja2_formatter(template)
        # prompt = PromptTemplate.from_template(jinja_prompt)

        docs = preprocess_arxiv(arxiv_id)
    
        if not docs:
            logger.error("Failed to preprocess the document.")

        if prompt_name == "summarization":
            system_message = read_file_as_string(f"templates/{prompt_name}.txt")
            # count words, if longer than 15,000 then truncate
            word_count = self.count_token(docs[0].page_content)
            logging.info(f"Raw {word_count} word counts")
            THRESHOLD = 13500
            if word_count > THRESHOLD:
                logging.info(
                    f"Warning: HTML content for {arxiv_id} exceeds {THRESHOLD} tokens. Truncating."
                )
                text = self.truncate_string(docs[0].page_content, token_threshold=THRESHOLD)
                docs[0].page_content = text
                logging.info(f"Truncated: {THRESHOLD} word counts")

            # Check if the request is already in the cache
            if (docs[0].metadata["Title"], prompt_name) in self.cache:
                return self.cache[(docs[0].metadata["Title"], prompt_name)]
            
            # Set metadata
            docs[0].metadata["word_count"] = word_count
            docs[0].metadata["is_truncated"] = True if word_count > THRESHOLD else False
            
        elif prompt_name == "tldr":
            system_message = read_file_as_string(f"templates/{prompt_name}.txt")

            tldr_docs = [Document(page_content=docs[0].metadata["Summary"])]
            # overrwrite if tldr
            docs = tldr_docs
            
        else:
            raise ValueError("Invalid prompt specified")

        prompt = PromptTemplate.from_template(system_message)
        # Define LLM chain with the rendered prompt
        llm = ChatOpenAI(temperature=0, model_name=self.model)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        # Run the chain, using invoke instead of run
        result = stuff_chain.invoke(docs)

        # Cache
        if prompt_name == "summarization":
            self.cache[(docs[0].metadata["Title"], prompt_name)] = result

        return result

MODEL = "gpt-3.5-turbo-1106"
TEMPERATURE = 0
assistant = OpenAIAssistant(model=MODEL,temperature=TEMPERATURE)

@app.command()
def summarize(
    input_jsonl: str,
    output_file_path: str = "data/output.jsonl",
    force_generate_all: bool = typer.Option(
        False, "-f", "--force-generate-all"
    ),
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
            try: 
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logging.error("Error reading input file: %s", e)

            arxiv_id = data["id"]
            categories = data["categories"]

            # Skip if already processed
            if force_generate_all is False:
                if arxiv_id in existing_ids:
                    logging.info(
                        f"Arxiv ID {arxiv_id} already processed. Skipping."
                    )
                    continue

            # Get summary using the 'summarization' template
            summary = assistant.get_summary(arxiv_id, 'summarization')
            if summary is None:
                logger.error(
                    f"Skipping record {arxiv_id} due to an error in text summarization."
                )
                continue  # Skip current record

            # # run tldr
            tldr_response = assistant.get_summary(arxiv_id, 'tldr')
            if tldr_response is None:
                logger.error(
                    f"Skipping record {arxiv_id} due to an error in creating tldr."
                )

            output_data = {
                "id": arxiv_id,
                "text": summary['output_text'],
                "meta": {
                    "links": {
                        "pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "html": f"https://browse.arxiv.org/html/{arxiv_id}",
                        "abs": f"https://arxiv.org/abs/{arxiv_id}",
                    },
                    "authors": summary['input_documents'][0].metadata["Authors"],
                    "title": remove_double_quotes(summary['input_documents'][0].metadata["Title"]),
                    "subtitle": remove_double_quotes(summary['input_documents'][0].metadata["Title"]) if tldr_response is None else remove_double_quotes(tldr_response['output_text']),
                    "categories": categories,
                    "publish_date": summary['input_documents'][0].metadata["Published"],
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "image": summary['input_documents'][0].metadata["png_url"], 
                    "word_count": summary['input_documents'][0].metadata["word_count"],
                    "extraction": summary['input_documents'][0].metadata["extraction"],
                    "is_truncated": summary['input_documents'][0].metadata["is_truncated"],
                },
            }

            with output_path.open("a") as output_file:
                output_file.write(json.dumps(output_data) + "\n")

            logging.info(f"Summary for {arxiv_id} written to {output_path}")


if __name__ == "__main__":
    app()
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import arxiv
import fitz  # PyMuPDF
import requests
import tiktoken
import typer
from bs4 import BeautifulSoup
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import (
    ArxivLoader,
    PDFMinerPDFasHTMLLoader,
    TextLoader,
)
from langchain_core.prompts import PromptTemplate
from langchain_fireworks.chat_models import ChatFireworks
from PIL import Image
from strip_tags import strip_tags

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
app = typer.Typer()


def get_url(arxiv_id: str, type: str):
    if type == "html":
        return f"https://browse.arxiv.org/html/{arxiv_id}"
    elif type == "pdf":
        return f"https://arxiv.org/pdf/{arxiv_id}"


def count_token(text: str) -> int:
    token_count = len(text) // 4  # Rough estimate of token count
    return token_count


def extract_png_image(html_content: str, base_url: str) -> str:
    """Extracts the first PNG image URL from HTML content."""
    try:
        png_url = extract_first_png_image(html_content)
        if png_url is None:
            logger.warning("No .png image found in the HTML content.")
            return None
        else:
            return f"{base_url}/{png_url}"
    except Exception as e:
        logger.error(f"Failed to extract .png image: {e}")
        return None


def extract_first_image_from_arxiv_paper(
    id, output_dir="img", output_format="png", min_width=100, min_height=100
) -> str:
    # Construct the URL for the Arxiv PDF
    pdf_url = f"https://arxiv.org/pdf/{id}.pdf"

    # Create the output directory (including subdirectory for the Arxiv ID) if it does not exist
    output_path = os.path.join(output_dir, id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download the PDF
    response = requests.get(pdf_url)
    if response.status_code != 200:
        logger.info("Failed to download the PDF to get the image")
        return None

    # Save the PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(response.content)
        temp_pdf_path = temp_pdf.name

    # Open the PDF with fitz
    pdf_file = fitz.open(temp_pdf_path)

    # Process the PDF
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)

        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
            img = image_list[0]
            xref = img[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))

            if image.width >= min_width and image.height >= min_height:
                image_index = 1
                image_path = os.path.join(
                    output_path, f"image_{image_index}.{output_format}"
                )
                image.save(image_path, format=output_format.upper())
                pdf_file.close()
                os.remove(temp_pdf_path)  # Delete the temporary PDF file
                return image_path
            else:
                logger.info(
                    f"[-] Skipping image on page {page_index} due to its small size."
                )
        else:
            logger.info(f"[!] No images found on page {page_index}")

    pdf_file.close()
    os.remove(temp_pdf_path)  # Delete the temporary PDF file
    return None


def preprocess_arxiv(arxiv_id: str) -> List[Document]:
    """Preprocesses the Arxiv document identified by the arxiv_id."""
    url = get_url(arxiv_id, "html")
    html_content = get_url_content(url)
    arxiv_documents = ArxivLoader(query=arxiv_id, load_max_docs=1).load()

    if html_content:
        abstract = get_abstract_section(html_content)
        cleaned_html = clean_html_section(html_content)
        documents = html_to_docs(cleaned_html)
        png_url = extract_png_image(html_content, url)

        metadata = {
            "Title": arxiv_documents[0].metadata["Title"],
            "Authors": arxiv_documents[0].metadata["Authors"],
            "Published": arxiv_documents[0].metadata["Published"],
            "Summary": abstract,
            "png_url": png_url,
            "extraction": "HTML",
            "word_counts": count_token(cleaned_html),
        }

        documents[0].metadata.update(metadata)
        return documents

    else:
        logger.info(
            f"HTML output not available for {arxiv_id}. Using PDF and ArxivLoader instead."
        )
        pdf_url = get_url(arxiv_id, "pdf")
        PDFMinerPDFasHTMLLoader(pdf_url)
        arxiv_documents[0].metadata.update(
            {
                "png_url": extract_first_image_from_arxiv_paper(arxiv_id),
                "extraction": "PDF",
            }
        )
        return arxiv_documents


def get_url_content(url: str) -> Optional[str]:
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
            logger.error(f"Bad response from {url} status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        logger.error(f"An error occurred while fetching {url}: {e}")
        return None


def clean_html_section(html_content: str) -> Optional[str]:
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
    return stripped.replace("\n", " ")


def get_sections(html_content: str):
    stripped = strip_tags(
        html_content,
        [".ltx_section"],
        removes=[
            ".ltx_authors",
            ".ltx_bibliography",
            ".package-alerts",
            ".section",  # remove license;arxiv number header
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


def read_prompt_as_string(file_name) -> str:
    try:
        with open(file_name, "r") as file:
            # Reading the file content
            content = file.read()
            return content.replace("\n", "")
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


def get_tiktoken_count(text, encoding):
    token_integers = encoding.encode(
        text, disallowed_special=(encoding.special_tokens_set - {"<|endoftext|>"})
    )
    return len(token_integers)


class FireworksAIAssistant:
    def __init__(
        self,
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.3,
        threshold=16000,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.threshold = threshold
        self.cache = {}  # Initialize cache

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

    def get_map_reduce(self, split_docs):
        llm = ChatFireworks(temperature=self.temperature, model_name=self.model, max_tokens=200)

        map_template = read_prompt_as_string("templates/map_summarization.txt")
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Reduce
        reduce_template = read_prompt_as_string("templates/reduce_summarization.txt")
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        # Run chain
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Add metadata
            metadata={
                "is_truncated": True,
                "extraction": "PDF",
                "Authors": "Authors",
                "Published": "YYYY-MM-DD",
                "word_count": 99999,
                "png_url": None,
            },
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=True,
        )

        results = map_reduce_chain.invoke(split_docs)

        return results

    def get_summary(self, arxiv_id, prompt_name) -> str:
        docs = preprocess_arxiv(arxiv_id)

        if not docs:
            logger.error("Failed to preprocess the document.")

        if prompt_name == "summarization":
            # Load the template based on prompt_name
            system_message = read_prompt_as_string(f"templates/{prompt_name}.txt")
            # count words, if longer than 30,000 then truncate
            word_count = get_tiktoken_count(docs[0].page_content, encoding)
            logging.info(f"Raw {word_count} word counts")
            if word_count > self.threshold:
                logging.info(
                    f"Warning: HTML content for {arxiv_id} exceeds {THRESHOLD} tokens. Running Map-Reduce summarization."
                )

                text_splitter = TokenTextSplitter(
                    chunk_size=5000,
                    chunk_overlap=100,
                    disallowed_special=(
                        encoding.special_tokens_set - {"<|endoftext|>"}
                    ),
                )

                split_docs = text_splitter.split_documents(docs)
                split_docs = [doc for doc in split_docs if len(doc.page_content) > 100]

                results = self.get_map_reduce(split_docs)

                results["input_documents"][0].metadata["word_count"] = word_count
                results["input_documents"][0].metadata["is_truncated"] = (
                    True if word_count > THRESHOLD else False
                )

                return results

            # Check if the request is already in the cache
            if (docs[0].metadata["Title"], prompt_name) in self.cache:
                return self.cache[(docs[0].metadata["Title"], prompt_name)]

            # Set metadata
            docs[0].metadata["word_count"] = word_count
            docs[0].metadata["is_truncated"] = True if word_count > THRESHOLD else False

        elif prompt_name == "tldr":
            system_message = read_prompt_as_string(f"templates/{prompt_name}.txt")
            tldr_docs = [Document(page_content=docs[0].metadata["Summary"])]
            # overrwrite if tldr
            docs = tldr_docs

        else:
            raise ValueError("Invalid prompt specified")

        prompt = PromptTemplate.from_template(system_message)
        # Define LLM chain with the rendered prompt
        llm = ChatFireworks(temperature=self.temperature, model_name=self.model, max_tokens=500)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )

        # Run the chain, using invoke instead of run
        result = stuff_chain.invoke(docs)

        # Cache
        if prompt_name == "summarization":
            self.cache[(docs[0].metadata["Title"], prompt_name)] = result

        return result


MODEL = "accounts/fireworks/models/mixtral-8x7b-instruct"
TEMPERATURE = 0.1
THRESHOLD = 27500
assistant = FireworksAIAssistant(
    model=MODEL, temperature=TEMPERATURE, threshold=THRESHOLD
)
encoding = tiktoken.get_encoding("cl100k_base")


@app.command()
def summarize(
    input_jsonl: str,
    output_file_path: str = "data/output.jsonl",
    force_generate_all: bool = typer.Option(False, "-f", "--force-generate-all"),
):
    """
    Summarizes texts from Arxiv HTML pages listed in a JSONL file using Firework AI's Chat API.

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
                    logging.info(f"Arxiv ID {arxiv_id} already processed. Skipping.")
                    continue

            # Get summary using the 'summarization' template
            summary = assistant.get_summary(arxiv_id, "summarization")
            if summary is None:
                logger.error(
                    f"Skipping record {arxiv_id} due to an error in text summarization."
                )
                continue  # Skip current record

            # if summary["output_text"][0:9] == "I'm sorry":
            #     continue # skip if

            # # run tldr
            tldr_response = assistant.get_summary(arxiv_id, "tldr")
            if tldr_response is None:
                logger.error(
                    f"Skipping record {arxiv_id} due to an error in creating tldr."
                )

            output_data = {
                "id": arxiv_id,
                "text": summary["output_text"],
                "meta": {
                    "links": {
                        "pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "html": f"https://browse.arxiv.org/html/{arxiv_id}",
                        "abs": f"https://arxiv.org/abs/{arxiv_id}",
                    },
                    "authors": summary["input_documents"][0].metadata["Authors"],
                    "title": remove_double_quotes(
                        summary["input_documents"][0].metadata["Title"]
                    ),
                    "subtitle": remove_double_quotes(
                        summary["input_documents"][0].metadata["Title"]
                    )
                    if tldr_response is None
                    else remove_double_quotes(tldr_response["output_text"]),
                    "categories": categories,
                    "publish_date": summary["input_documents"][0].metadata["Published"],
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "image": summary["input_documents"][0].metadata["png_url"],
                    "word_count": summary["input_documents"][0].metadata["word_count"],
                    "extraction": summary["input_documents"][0].metadata["extraction"],
                    "is_truncated": summary["input_documents"][0].metadata[
                        "is_truncated"
                    ],
                },
            }

            with output_path.open("a") as output_file:
                output_file.write(json.dumps(output_data) + "\n")

            logging.info(f"Summary for {arxiv_id} written to {output_path}")


if __name__ == "__main__":
    app()

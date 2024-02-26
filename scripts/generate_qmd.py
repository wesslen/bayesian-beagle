import json
import re
from datetime import datetime
from pathlib import Path

import typer
from jinja2 import BaseLoader, Environment

app = typer.Typer()

# Jinja2 template for the QMD file
QMD_TEMPLATE = """
---
title: "{{ example['meta']['title'] }}"
id: "{{ example['id'] }}"
description: "{{ example['meta']['subtitle'] }}"
author: {{ authors }}
date: "{{ example['meta']['publish_date'] }}"
image: "{{ image }}"
categories: {{ example['meta']['categories'] }}
format:
  html:
    code-overflow: wrap
---

![]({{ example['meta']['image'] }})

{{ example['text'] }}

## Appendix

|          |          |
|----------|----------|
| Model     | {{ example['meta']['model'] }}       |
| Date Generated     | {{ timestamp.strftime('%Y-%m-%d') }}       |
| Abstract | [{{ example['meta']['links']['abs'] }}]({{ example['meta']['links']['abs'] }})        |
| HTML     | [{{ example['meta']['links']['html'] }}]({{ example['meta']['links']['html'] }})       |
| Truncated       | {{ example['meta']['is_truncated'] }}       |
| Word Count       | {{ example['meta']['word_count'] }}       |
"""


def remove_unicode_accents(input_str):
    input_str = input_str.replace("$\infty$", "Infinity")
    return input_str.replace("\\'", "")


def remove_patterns(text):
    # Regex pattern: \ followed by any characters except \ or { or }, followed by {}
    pattern = r"\\[^\\{}]*\{.*?\}"
    # Replace found patterns with an empty string
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def remove_accents(input_str):
    # Mapping of accented characters to their unaccented counterparts
    accents_mapping = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "ñ": "n",
        "Ñ": "N",
        # Add more mappings as needed
    }

    # Replace each accented character with its unaccented counterpart
    return "".join(accents_mapping.get(char, char) for char in input_str)


def get_image_path(example):
    image_path = example.get("meta", {}).get("image", None)
    
    if image_path is None:
        image = "../../../bayesian-beagle.png"
    elif "img/" in image_path:
        image = f"../../{image_path}"
    else:
        image = image_path
    
    return image

def convert_to_folder_name(title):
    """
    Convert a given string to a folder name format by replacing spaces, slashes,
    question marks, colons, commas, and hyphens with underscores.

    Parameters:
    title (str): The string to be converted into folder name format.

    Returns:
    str: The converted folder name.
    """

    # Characters to be removed from the string
    special_chars = ":{}[],&*#?|><=%@`/"

    # Remove special characters and replace spaces with underscores
    sanitized = "".join(char for char in title if char not in special_chars)

    sanitized = remove_accents(sanitized)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.replace("-", "_")
    sanitized = sanitized.replace("'", "")

    return sanitized


def create_qmd_file(example, output_folder, force_generate_all=False):
    """
    Create a .qmd file from a JSON dictionary
    """
    title = remove_unicode_accents(example["meta"]["title"])
    folder_name = convert_to_folder_name(title)
    example["meta"]["title"] = title
    example["meta"]["subtitle"] = remove_unicode_accents(example["meta"]["subtitle"])
    example["meta"]["subtitle"] = remove_patterns(example["meta"]["subtitle"])
    current_date = example["meta"]["publish_date"]
    example["meta"]["image"] = get_image_path(example)
    file_name = f"{current_date}-{folder_name}.qmd"
    folder_path = Path(output_folder) / folder_name
    file_path = folder_path / file_name
    authors = example["meta"]["authors"]

    # Check if the file already exists
    if not force_generate_all and file_path.is_file():
        print(f"File already exists: {file_path}")
        return

    env = Environment(loader=BaseLoader())
    template = env.from_string(QMD_TEMPLATE)

    # Render the template with data
    rendered_qmd = template.render(
        example=example,
        current_date=current_date,
        timestamp=datetime.now(),
        image=example["meta"]["image"],
        authors=authors,
    )

    # Create output sub-folder
    folder_path.mkdir(parents=True, exist_ok=True)

    # Write the rendered content to a file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(rendered_qmd)

    print(f"File saved: {file_path}")


@app.command()
def generate_qmd(
    input_jsonl: str,
    output_folder: str,
    force_generate_all: bool = typer.Option(False, "-f", "--force-generate-all"),
):
    """
    Generate QMD files from a JSONL file
    """
    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            example = json.loads(line)
            create_qmd_file(example, output_folder, force_generate_all)


if __name__ == "__main__":
    app()

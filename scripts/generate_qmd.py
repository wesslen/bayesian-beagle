import json
import typer
from jinja2 import Environment, BaseLoader
from pathlib import Path
from datetime import datetime

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


def convert_to_folder_name(title):
    """
    Convert a given string to a folder name format by replacing spaces, slashes,
    question marks, colons, commas, and hyphens with underscores.

    Parameters:
    title (str): The string to be converted into folder name format.

    Returns:
    str: The converted folder name.
    """
    # Characters to be replaced
    replace_chars = " /?:,-"

    # Create a translation table for replacing characters
    trans_table = str.maketrans(replace_chars, "_" * len(replace_chars))

    # Replace specified characters with underscores
    folder_name = title.translate(trans_table)

    return folder_name


def create_qmd_file(example, output_folder, force_generate_all=False):
    """
    Create a .qmd file from a JSON dictionary
    """
    title = example["meta"]["title"]
    folder_name = convert_to_folder_name(title)
    current_date = example["meta"]["publish_date"]
    image = (
        "../../../bayesian-beagle.png"
        if example["meta"]["image"] is None
        else example["meta"]["image"]
    )
    file_name = f"{current_date}-{folder_name}.qmd"
    folder_path = Path(output_folder) / folder_name
    file_path = folder_path / file_name
    authors = ", ".join(example["meta"]["authors"])

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
        image=image,
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
    force_generate_all: bool = typer.Option(
        False, "-f", "--force-generate-all"
    ),
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

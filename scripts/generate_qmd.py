import json
import typer
from jinja2 import Environment, BaseLoader
from pathlib import Path

app = typer.Typer()

# Jinja2 template for the QMD file
QMD_TEMPLATE = """
---
title: "{{ example['meta']['title'] }}"
subtitle: "{{ example['meta']['subtitle'] }}"
author: "{{ example['meta']['model'] }}"
date: "{{ example['meta']['publish_date'] }}"
link: "{{ example['meta']['url'] }}"
categories: {{ example['meta']['categories'] }}
format:
  html:
    code-overflow: wrap
---

{{ example['text'] }}

### Appendix

**Link**: [{{ example['meta']['url'] }}]({{ example['meta']['url'] }})
<br>
**Truncated**: {{ example['meta']['is_truncated'] }}
<br>
**Word Count**: {{ example['meta']['word_count'] }}
"""


def create_qmd_file(example, output_folder):
    """
    Create a .qmd file from a JSON dictionary
    """
    title = example["meta"]["title"]
    folder_name = title.replace(" ", "_")
    current_date = example["meta"]["publish_date"]
    file_name = f"{current_date}-{folder_name}.qmd"
    folder_path = Path(output_folder) / folder_name
    file_path = folder_path / file_name

    # Check if the file already exists
    if file_path.is_file():
        print(f"File already exists: {file_path}")
        return

    env = Environment(loader=BaseLoader())
    template = env.from_string(QMD_TEMPLATE)

    # Render the template with data
    rendered_qmd = template.render(example=example, current_date=current_date)

    # Create output sub-folder
    folder_path.mkdir(parents=True, exist_ok=True)

    # Write the rendered content to a file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(rendered_qmd)

    print(f"File saved: {file_path}")


@app.command()
def generate_qmd(input_jsonl: str, output_folder: str):
    """
    Generate QMD files from a JSONL file
    """
    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            example = json.loads(line)
            create_qmd_file(example, output_folder)


if __name__ == "__main__":
    app()

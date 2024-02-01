import json
import logging
from pathlib import Path

import srsly
import typer

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
app = typer.Typer()


def remove_records_from_file(file_path, arxiv_ids):
    updated_records = []
    ids_found = set()

    with open(file_path, "r") as file:
        for line in file:
            try:
                data = json.loads(line)
                if data["id"] not in arxiv_ids:
                    updated_records.append(data)
                else:
                    ids_found.add(data["id"])
            except json.JSONDecodeError as e:
                logging.error("Error reading file %s: %s", file_path, e)

    srsly.write_jsonl(file_path, updated_records)
    return ids_found


@app.command()
def remove_data(arxiv_ids: str, option: str):
    arxiv_id_list = arxiv_ids.split(",")
    input_path = Path("data/input.jsonl")
    output_path = Path("data/output.jsonl")

    if option not in ["both", "input", "output"]:
        logging.info("Invalid option. Please choose 'both', 'input', or 'output'.")
        return

    ids_found_in_input = set()
    ids_found_in_output = set()

    if option in ["both", "input"]:
        ids_found_in_input = remove_records_from_file(input_path, arxiv_id_list)

    if option in ["both", "output"]:
        ids_found_in_output = remove_records_from_file(output_path, arxiv_id_list)

    all_ids_found = ids_found_in_input.union(ids_found_in_output)
    removed_ids = ", ".join(all_ids_found)

    if all_ids_found:
        logging.info(f"Removed records with IDs: {removed_ids}")
    else:
        logging.info("No records found for the given IDs")


if __name__ == "__main__":
    app()

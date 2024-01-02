# Append the scripts directory to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "scripts"))

import pytest
from generate_qmd import convert_to_folder_name, create_qmd_file

EXAMPLE_JSON = {
    "meta": {
        "title": "Example Title",
        "subtitle": "Example Subtitle",
        "model": "Example Model",
        "publish_date": "2022-01-01",
        "links": {
            "abs":"https://arxiv.org/abs/2401.00741v1",
            "html": "https://browse.arxiv.org/html/2401.00741v1",
            "pdf": "https://arxiv.org/pdf/2401.00741v1.pdf",
        },
        "categories": ["category1", "category2"],
        "authors": ["author1", "author2"],
        "is_truncated": True,
        "image": None,
        "word_count": 1000,
    },
    "text": "Example text",
}


@pytest.fixture(scope="module")
def output_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("output")


def test_convert_to_folder_name():
    folder_name = convert_to_folder_name("Example Title")
    assert folder_name == "Example_Title"

    folder_name = convert_to_folder_name("Title with spaces")
    assert folder_name == "Title_with_spaces"

    folder_name = convert_to_folder_name("Title/with/slashes")
    assert folder_name == "Title_with_slashes"

    folder_name = convert_to_folder_name("Title?with?question?marks")
    assert folder_name == "Title_with_question_marks"

    folder_name = convert_to_folder_name("Title:with:colons")
    assert folder_name == "Title_with_colons"

    folder_name = convert_to_folder_name("Title,with,commas")
    assert folder_name == "Title_with_commas"

    folder_name = convert_to_folder_name("Title-with-hyphens")
    assert folder_name == "Title_with_hyphens"


def test_create_qmd_file(output_folder):
    create_qmd_file(EXAMPLE_JSON, output_folder)

    current_date = EXAMPLE_JSON["meta"]["publish_date"]
    folder_name = convert_to_folder_name(EXAMPLE_JSON["meta"]["title"])
    file_name = f"{current_date}-{folder_name}.qmd"
    file_path = Path(output_folder) / folder_name / file_name

    assert file_path.is_file()


def test_create_qmd_file_force_generate_all(output_folder):
    create_qmd_file(EXAMPLE_JSON, output_folder, force_generate_all=True)


#     current_date = EXAMPLE_JSON["meta"]["publish_date"]
#     folder_name = convert_to_folder_name(EXAMPLE_JSON["meta"]["title"])
#     file_name = f"{current_date}-{folder_name}.qmd"
#     file_path = Path(output_folder) / folder_name / file_name

#     generate_qmd(file_path, output_folder, force_generate_all=True)


# def test_create_qmd_file_existing_file(output_folder):
#     create_qmd_file(EXAMPLE_JSON, output_folder)

#     # Modify the example JSON
#     EXAMPLE_JSON["meta"]["publish_date"] = str(datetime.now().date())

#     current_date = EXAMPLE_JSON["meta"]["publish_date"]
#     folder_name = convert_to_folder_name(EXAMPLE_JSON["meta"]["title"])
#     file_name = f"{current_date}-{folder_name}.qmd"
#     file_path = Path(output_folder) / folder_name / file_name

#     create_qmd_file(EXAMPLE_JSON, output_folder)

# assert not file_path.is_file()

# def test_generate_qmd(output_folder):
#     input_jsonl = "examples.jsonl"
#     create_qmd_file(EXAMPLE_JSON, output_folder)

#     runner = typer.CliRunner()
#     result = runner.invoke(app, ["generate_qmd", input_jsonl, output_folder])

#     assert result.exit_code == 0

#     current_date = EXAMPLE_JSON["meta"]["publish_date"]
#     folder_name = convert_to_folder_name(EXAMPLE_JSON["meta"]["title"])
#     file_name = f"{current_date}-{folder_name}.qmd"
#     file_path = Path(output_folder) / folder_name / file_name

#     assert file_path.is_file()

# Bayesian beagle blog 🐶

Welcome to the Bayesian beagle blog! This project is a unique intersection of machine learning and scientific communication, providing a platform where readers can quickly get insights from the latest research papers hosted on [ArXiv](https://arxiv.org/). Utilizing state-of-the-art Large Language Models (LLMs), our system generates concise, comprehensible summaries of complex research articles, covering a wide array of disciplines.

Our blog is built using [Quarto](https://quarto.org/), an open-source scientific and technical publishing system designed for creating beautiful, data-driven content. It is then published with [Netlify](https://app.netlify.com/).

```mermaid
graph LR
    A["Download weekly Arxiv articles"] --> B["Predict and Filter LLM topic"]
    B --> C["Summarize short docs"]
    B --> D["Summarize by Map-Reduce long docs"]
    C --> E["Update website with summaries weekly"]
    D --> E
```

[![Netlify Status](https://api.netlify.com/api/v1/badges/7b28658b-5d30-42ac-a70e-a0a0deedf114/deploy-status)](https://app.netlify.com/sites/bayesian-beagle/deploys)

## Features

- **Curated ArXiv Articles**: A handpicked selection of the most intriguing and high-impact research papers from various fields on ArXiv.
- **Automated Summaries**: Each article is accompanied by a summary automatically generated by a sophisticated Large Language Model tailored for scientific content, utilizing Arxiv's new [HTML (beta) formatting](https://info.arxiv.org/about/accessible_HTML.html).
- **Regular Updates**: Our collection is updated regularly via GitHub actions to include new research findings and innovations.
- **LLM-research**: Coverage focuses on LLM-related research.

## How It Works

1. **Article Selection**: We curate a list of ArXiv articles based on recency, impact, and relevance to a diverse audience.
2. **Summary Generation**: LLMs are employed to read and understand the selected articles and provide a human-readable summary.
3. **Blog Publication**: These summaries are formatted and published as blog posts on our Quarto-powered platform.

## Usage

The blog is live at **[https://bayesian-beagle.netlify.app/](https://bayesian-beagle.netlify.app/)**

Navigate to the blog using the provided link and enjoy the latest research summaries. If you're interested in how the blog is generated or want to suggest improvements, feel free to check the repository or open an issue.

## Installation and Setup

To clone and run this project locally, you'll need [Git](https://git-scm.com/downloads), Quarto, and the necessary Python packages installed on your computer. From your command line:

```bash
# Clone this repository
git clone https://github.com/wesslen/bayesian-beagle.git

# Go into the repository
cd bayesian-beagle

# Create venv
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies for summary
pip install -r requirements-summarizer.txt

# Install dependencies for build
pip install -r requirements-build.txt

# Install dependencies for langchain
pip install -r requirements-langchain.txt

# Curate arxiv ids in data/input.jsonl, ensure they have HTML renderings

# Generate summaries
python scripts/summarizer.py data/input.jsonl

# Create quarto posts of summaries
python scripts/generate_qmd.py data/output.jsonl posts

# Build the Quarto blog
quarto render
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- ArXiv for making scientific articles openly accessible to all.
- [Vincent Warmerdam](https://koaning.io/) for his [Arxiv-Frontpage project](https://github.com/koaning/arxiv-frontpage), which I [extended](https://github.com/wesslen/arxiv-frontpage) for custom LLM labels and models
- [Posit](https://posit.co/) for their outstanding publishing tool, Quarto.
- [Simon Willison](https://github.com/simonw)'s helpful [`strip-tags`](https://github.com/simonw/strip-tags) library

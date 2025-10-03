# Semantic Search Engine for Medium Articles

##  Project Overview

This project builds, evaluates, and compares two distinct search engines for a corpus of Medium articles:

1.  A **classical, keyword-based search engine** using the TF-IDF algorithm.
2.  A **modern, semantic search engine** using a state-of-the-art Sentence Transformer model.

For a complete walkthrough of the methodology, experimental results, and a detailed analysis, please see the full project report: **Semantic_keywords_report.pdf**.

##  How to Reproduce the Results

The entire project is self-contained within the `Semantic_keywords.ipynb` Jupyter Notebook and is designed to be run in a Google Colab environment.

1.  **Open in Colab:** Upload and open the `Semantic_keywords.ipynb` notebook in Google Colab.
2.  **Run All Cells:** Execute the cells sequentially from top to bottom by selecting "Runtime" > "Run all".

The notebook will handle all necessary package installations, data downloading, model training, and final evaluations automatically. The output will replicate the tables and results presented in the project report.

##  Core Technologies Used

*   **Python:** The primary programming language.
*   **Scikit-learn:** For the TF-IDF baseline model and data splitting.
*   **Sentence-Transformers:** For building the core semantic search engine.
*   **KeyBERT:** For advanced, thematic keyword extraction.
*   **Pandas:** For data manipulation and preprocessing.
*   **Kaggle Hub:** For programmatic dataset access.
# Contextual Sentiment Analysis: The Homonyms Problem

##  Project Overview

This project investigates the challenge of contextual sentiment analysis, focusing on sentences where a single word can convey opposite meanings. It follows the requirements of the "Homonyms Problem in the text" assignment.

The core of the project is a comparative analysis between:

1. A **baseline model** using classical, static GloVe word embeddings.
2. Several **advanced models** using state-of-the-art, contextual Transformer architectures (BERT and RoBERTa).

The analysis proves the superiority of contextual models and demonstrates that their limitations can be probed and understood through targeted experimentation.

For a complete walkthrough of the methodology, results, and conclusions, please see the full project report: **Homonyms_Problem_Report.pdf**.

##  How to Reproduce the Results

The entire project is contained within the homonyms_problem.ipynb Jupyter Notebook.

1. **Environment:** This notebook is designed to be run in a Google Colab environment.
2. **Execution:** Simply open homonyms_problem.ipynb in Colab and run the cells sequentially from top to bottom ("Runtime" > "Run all").
3. **Dependencies:** The notebook will handle all necessary package installations (wget, transformers, etc.) automatically.

The notebook will download the required GloVe vectors, load the pre-trained models from Hugging Face, and reproduce all the analytical tables presented in the final report.

##  Core Technologies Used

- **Python**
- **Pandas** & **NumPy**
- **Scikit-learn** (for the Logistic Regression baseline)
- **PyTorch**
- **Hugging Face Transformers** (for BERT and RoBERTa models)
- **GloVe** (for static word embeddings)
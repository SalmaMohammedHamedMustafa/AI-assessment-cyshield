### Introduction

**The Problem:** Users expect to find information not just by matching keywords, but by expressing their intent in natural language. Traditional search methods, like TF-IDF, excel at finding documents with exact keyword overlap but often fail to understand the user's true goal, especially for conceptual or nuanced queries. This project addresses the challenge of building a truly "smart" search engine that can comprehend semantic meaning.

**Project Goal & Accomplishment:** The primary goal of this project was to develop, compare, and evaluate two fundamentally different search pipelines for a large corpus of Medium articles:

1. A **classical, keyword-based search engine** using the TF-IDF algorithm.
2. A **modern, semantic search engine** leveraging a state-of-the-art deep learning Sentence Transformer model.

The objective was to determine which approach provides a more effective and useful search experience. To accomplish this, a complete machine learning pipeline was constructed, including robust data preprocessing, methodical model tuning using a dedicated validation set, and a final, unbiased head-to-head comparison on an unseen test set. This project successfully demonstrates the tangible trade-offs between the two methods and provides a clear verdict on the superior approach.

### Data Description

**1. Original Dataset**

The project utilizes the **"Medium Articles" dataset** sourced from Kaggle (hsankesara/medium-articles). The original raw dataset contained **337 instances** (articles) and featured columns such as title, text, claps, reading_time, and url. For the purpose of building a search engine, this project focused exclusively on the text column, which contains the full content of each article.

**2. Data Preprocessing Pipeline**

To ensure the quality and consistency of the data used for modeling, a comprehensive preprocessing pipeline was implemented. This pipeline performed the following key steps:

- **Duplicate Removal:** Identified and removed **107 duplicate articles**.
- **Text Cleaning:** Standardized the text by removing HTML tags, URLs, and special characters, followed by lowercasing and normalizing whitespace.
- **Content Filtering:** Filtered out articles that were too short (under 50 words) or excessively long (top 1% quantile), resulting in the removal of **4 articles**.

After this rigorous process, a final, clean dataset of **226 high-quality articles** was prepared for the modeling phase, representing a **67.1%** retention rate from the original data.

**3. Data Splitting**

To facilitate a methodologically sound machine learning workflow, the preprocessed dataset was divided into three distinct, non-overlapping sets:

- **Training Set:** **158 articles (70%)** - Used exclusively for training the final models.
- **Validation Set:** **34 articles (15%)** - Used for hyperparameter tuning and model selection.
- **Test Set:** **34 articles (15%)** - A completely held-out set used only for the final, unbiased evaluation of the champion models.

The split was performed using a **stratified** approach based on article word count (categorized into quartiles: short, medium, long, very_long). As shown in the verification table below, this ensured that the distribution of article lengths was consistent across all three sets, preventing any potential biases in the evaluation.

| Word Quartile | Train Set % | Validation Set % | Test Set % |
| ------------- | ----------- | ---------------- | ---------- |
| short         | 24.68       | 26.47            | 26.47      |
| medium        | 25.32       | 23.53            | 23.53      |
| long          | 24.68       | 26.47            | 23.53      |
| very_long     | 25.32       | 23.53            | 26.47      |

------

### Baseline Experiment: TF-IDF Search

**1. Goal**

The goal of the baseline experiment was to build and optimize a traditional keyword-based search engine using the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm. This model serves as a robust benchmark against which the more advanced deep learning model can be compared. The primary objective was to determine the optimal hyperparameters for the TfidfVectorizer to ensure the baseline was as strong as possible.

**2. Experimental Steps & Results**

The optimization process was conducted iteratively using the **validation set** and a set of general-purpose validation queries.

**Round 1: Broad Hyperparameter Search**

A broad search was initially performed across 72 different parameter combinations, focusing on max_features, min_df, max_df, and ngram_range. The top-performing models from this round consistently converged on a vocabulary size of **1,000 features**.

- **Top Result (Round 1):**
  - **Score:** 0.1501
  - **Parameters:** {'max_features': 1000, 'min_df': 2, 'ngram_range': (1, 1)}

**Round 2: Refined Hyperparameter Search**

While the first round provided a good starting point, it was hypothesized that a smaller, more focused vocabulary might yield better performance, especially with the inclusion of bigrams (ngram_range=(1, 2)), which can capture more specific concepts. A second, more granular search was conducted over 30 combinations, focusing on max_features values at and below 1000.

This refined search yielded a significant improvement in the validation score and identified a clear "sweet spot" for the vocabulary size.

- **Top 5 Results (Round 2):**

| Rank  | Validation Score | max_features | min_df | ngram_range |
| ----- | ---------------- | ------------ | ------ | ----------- |
| **1** | **0.1684**       | **50**       | **3**  | **(1, 2)**  |
| 2     | 0.1684           | 50           | 4      | (1, 2)      |
| 3     | 0.1684           | 50           | 5      | (1, 2)      |
| 4     | 0.1347           | 5            | 3      | (1, 2)      |
| 5     | 0.1347           | 5            | 4      | (1, 2)      |

The results clearly show that performance peaked with a vocabulary of **50 features** and then began to decline as the feature count increased, indicating that larger vocabularies were introducing noise.

**3. Conclusion**

The iterative hyperparameter tuning process was highly successful. It revealed that a smaller, more curated vocabulary that includes bigrams provides the best performance for a general-purpose keyword search on this dataset.

The optimal configuration for the baseline TF-IDF model was determined to be:

- **max_features**: 50
- **min_df**: 3
- **ngram_range**: (1, 2)
- **max_df**: 0.9

This optimized model, trained on the full training set with these parameters, serves as the final, robust baseline for the comparative evaluation.

------

### Other Experiments: Semantic Search Engine

**1. Goal**

The goal of this experiment was to move beyond keyword-matching and build a superior search engine using modern deep learning techniques. The objective was to leverage pre-trained Sentence Transformer models to create a system capable of understanding the *semantic meaning* of the articles and user queries. The experiment was designed to identify the most effective pre-trained model for this specific task and dataset.

**2. Experimental Steps & Results**

The experiment was conducted using the **validation set** to ensure an unbiased selection process.

**Steps:**

1. **Candidate Selection:** Three popular and high-performing Sentence Transformer models were chosen as candidates, each with different strengths:
   - all-MiniLM-L6-v2: A fast and efficient model, well-balanced for performance and resource usage.
   - all-mpnet-base-v2: A larger, higher-quality model known for generating excellent general-purpose embeddings.
   - msmarco-distilbert-base-v4: A model specifically fine-tuned on a massive question-answering dataset, making it a strong candidate for information retrieval.
2. **Evaluation:** Each model was tasked with encoding the validation documents and processing the general-purpose validation_queries. The average cosine similarity score for the top 5 results for each query was used as the performance metric.
3. **Model Selection:** The model with the highest average validation score was selected as the champion for the final semantic search engine.

**Results:**

The evaluation produced a clear winner. The performance of each model on the validation set is summarized in the table below:

| Model Name                 | Average Validation Score | Notes                              |
| -------------------------- | ------------------------ | ---------------------------------- |
| **all-MiniLM-L6-v2**       | **0.2936**               | **Best Performance**               |
| all-mpnet-base-v2          | 0.2925                   | Close second, slightly lower score |
| msmarco-distilbert-base-v4 | 0.2506                   | Lower performance on this task     |

**3. Conclusion**

The experiment successfully identified **all-MiniLM-L6-v2** as the optimal model for this semantic search task. Despite all-mpnet-base-v2 being a larger model, the more compact all-MiniLM-L6-v2 achieved a slightly higher score, offering the best balance of performance and efficiency.

Following its selection, the final Semantic Search engine was built by:

1. Encoding all **158 articles** from the training set using the all-MiniLM-L6-v2 model, resulting in an embedding matrix of shape (158, 384).
2. Integrating the KeyBERT library to provide a mechanism for extracting thematic keywords from search results, thereby adding a layer of explainability to the deep learning model.

This optimized semantic engine is now ready for the final head-to-head comparison against the TF-IDF baseline.

------

### Overall Conclusion

After a comprehensive process of data preparation, iterative tuning, and model selection, the optimized TF-IDF (baseline) and Semantic Search (deep learning) models were subjected to a final, definitive evaluation on an unseen test set. This final comparison provides a clear and multi-faceted answer to the central question: which method is best?

**1. Summary of Findings**

The head-to-head evaluation revealed a stark contrast in the capabilities and performance profiles of the two models.

- **Relevance and Quality of Results:** The Semantic Search model demonstrated a profound and consistently superior ability to understand user intent.
  - On conceptual queries like *"how to handle burnout at work"* and *"the role of creativity in technology,"* the Semantic model provided contextually relevant articles. In stark contrast, the TF-IDF model failed completely, returning scores of 0.0000 as it could not find any keyword matches in its limited vocabulary.
  - Even on specific queries, the TF-IDF model often fixated on a single keyword out of context (e.g., finding articles about "code" for a query about "clean code").
  - The only scenario where the TF-IDF model performed comparably was on queries with strong, unambiguous keyword overlap, such as *"explaining machine learning to a beginner."*
- **Speed and Performance:** The trade-off for the semantic model's superior intelligence is a significant difference in computational cost.
  - **TF-IDF Model:** Averaged **86.35 ms** per query.
  - **Semantic Model:** Averaged **1439.15 ms** per query.
  - This makes the traditional TF-IDF model approximately **16.7 times faster** than the deep learning-based semantic model in this evaluation.

**2. Final Verdict**

While the TF-IDF model offers a significant advantage in search latency, its performance is brittle and unreliable. It is only effective for a narrow range of queries and fails completely when a user's search terms fall outside its limited, keyword-based worldview. This makes it unsuitable for a modern, user-centric application where understanding natural language is paramount.

The **Semantic Search engine is unequivocally the best model**. Its ability to grasp the meaning and intent behind a query—even in the absence of direct keyword matches—represents a transformational improvement in search quality. It provides a far more robust, intelligent, and useful experience for the end-user.

For a real-world application, the significant increase in relevance provided by the Semantic Search model would justify the investment in the necessary computational resources to manage its higher latency. Therefore, the deep learning approach is the clear and recommended choice.
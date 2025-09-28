![SentHive Logo](misc/picture/SentHive.jpg)

# SentHive
**Aggregate. Summarize. Understand.**

![Python](https://img.shields.io/badge/python-3.6.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview
SentHive is a lightweight pipeline for sentiment labeling, aggregation, and summarization.  
It helps you quickly analyze large-scale chat/text datasets and generate concise, AI-driven summaries.

---

## Features
- Label massive text datasets with sentiment (positive, negative, neutral)
- Aggregate text data according to custom labels
- Generate concise summaries using AI
- Support text cleaning with `Regularization_removal.py`  
- Organize and store multiple projects under `Demand_nanlysis_data`

## Directory Structure
```angular2html
## Directory Structure
SentHive/
├─ Demand_analysis_data/
│  ├─ Project1/
│  │  └─ Dataset/xxx.csv
│  ├─ Project2/
│  │  └─ Dataset/xxx.csv
├─ Sentiment_Analysis_Automation/
│  ├─ utils/
│  │  ├─ Labeler.py
│  │  ├─ SentimentStats.py
│  │  ├─ keyword_cluster.py
│  │  ├─ ai_summary.py
│  │  ├─ pipeline.py
│  │  └─ logger_utils.py
│  ├─ script/
│  │  ├─ xlsx_to_csv_and_drop_col.py
│  │  ├─ Regularized_matching.py
│  │  └─ Regularization_removal.py
│  ├─ model/
│  │  ├─ sentence_transformer_model/
│  │  └─ sentiment_model/
│  └─ vocabulary/
│     ├─ daily_expressions.txt
│     └─ hit_stopword.txt
├─ misc/picture/SentHive.jpg
└─ requirements.txt
```

## Directory Explanation

### Demand_analysis_data
  Use this folder to store multiple project directories.
  For example, if you work on Task1, create a `Project1/` folder and put all related datasets inside it.
  Save any modifications, additions, or processed files directly in the same folder.
  When you start a new task, create another folder such as `Project2/` or `Project3/`.
  You can specify the active project in the YAML configuration file.
  
### utils

- **Labeler.py**
  Use your own sentiment labeling model(s) to tag text.
  - You need to configure your own model(s) inside this file (support multiple models).
  - Follow the provided format (details left for later documentation).
  
- **SentimentStats.py**
  After labeling, this script processes the text and groups it by the specified label column.
  - It counts the number of texts per label.
  - It also computes the number of **unique users** (non-repeated `player_id`) per label.
  
- **keyword_cluster.py**
  Perform text cleaning and similarity-based clustering.
  - Groups similar text together by semantic similarity.
  - Produces a dictionary: **{key phrase -> number of users}**
  - Converts the final dictionary into a DataFrame for later steps.
  
- **ai_summary.py**
  Generate concise summaries using a large language model (LLM).
  - Takes the DataFrame from `keyword_cluster.py`.
  - Concatenates rows into limited-length text chunks.
  - Sends them to your configured LLM for summarization.
  - You must configure your own LLM here

- **pipeline.py**
  The **main entry point** of the program.
  - Input: a CSV/text file.
  - Processing: runs through labeling -> statistics -> clustering -> summarization.
  - Output: the final summarized insights.
  
- **logger_utils.py**
  Records the execution time for each step in the pipeline, helping monitor performance.
  - Provides utility functions or decorators to automatically log function runtimes.
  - Useful for pipeline stages like labeling, statistics, clustering, and summarization.
  
### Script
The `script/` folder contains handy utilities for preprocessing datasets:

- **xlsx_to_csv_and_drop_col.py**
  Convert `.xlsx` files into `.csv` format and drop specified columns.
  
- **Regularization_removal.py**
  Remove rows containing specified keywords from text using regex matching.
  
- **Regularization_matching.py**
  Extract rows containing specified keywords from text using regex matching.
  
### model
This folder stores pre_trained or fine-tuned models required for the pipeline.

- **sentence_transformer_model/**
  Contains the sentence embedding model used for text vectorization.
  
- **sentiment_model/**
  Contains the sentiment classification model used for text labeling

### vocabulary
This folder contains auxiliary text resources for data cleaning and preprocessing.

- **daily_expressions.txt**
  A list of daily expressions used to normalize or filter conversational text.
  
- **hit_stopword.txt**
  The HIT stopword list (Harbin Institute of Technology), used to remove common stopwords during text cleaning.
  

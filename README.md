# User-Centric Sentiment Forecasting: A Modular NLP System for Emotion-Aware, Time-Sensitive Feedback Modeling

**Capturing Fine-Grained, Temporal, and Emotion-Centric Signals from Irregular Review Data**



Businesses today rely heavily on user feedback to tailor experiences, improve services, and retain customers. However, traditional sentiment analysis systems fall short in capturing the individual trajectory of a user's emotional engagement. This work focuses on forecasting how a user’s sentiment evolves over time, which has immense implications for personalization, proactive support, and product strategy

## Overview

Can we accurately forecast future user sentiment by analyzing their past reviews, even when those reviews are sparse, temporally irregular, and aspect-rich?

This project proposes a complete sentiment modeling pipeline that goes beyond standard transformer-based approaches. We combine syntactic graph modeling, emotion-informed user profiling, and time-aware forecasting to build a personalized sentiment engine. Our system is specifically designed for applications in **feedback-driven businesses** such as e-commerce platforms, ed-tech products, online services, and customer support systems.

---

## Motivation

Traditional NLP systems often assume:

- Users leave enough data to model their behavior.
- Reviews come at regular time intervals.
- Transformers can parse semantic structure without external guidance.

In reality:

- Users may leave **only a few scattered reviews**.
- Review frequency is highly **irregular** and varies by user.
- Sentences often contain **multi-aspect, syntactically rich content** that transformers misinterpret.

As a result, conventional models underperform when tasked with **personalized, future-facing sentiment prediction.**

Our system directly tackles this mismatch through three innovations.

---

## Core Challenges and Solutions

### Challenge 1: Transformer Models Ignore Syntax in Aspect-Based Sentiment Analysis (ABSA)

**Problem:** Most transformer-based sentiment models are trained on flat token sequences. They ignore **syntactic dependencies**, which are crucial for **Aspect-Based Sentiment Analysis (ABSA)**.

**Illustrative Example:**

> *"The pasta was amazing, but the service was slow."*

A transformer may generate a net neutral or positive sentiment score, failing to distinguish that:

- "amazing" refers to *pasta*
- "slow" refers to *service*

**Our Solution:**

- Use **dependency parsers** to extract the grammatical structure of each review.
- Construct a **dependency graph** where tokens are nodes and edges are syntactic links.
- Dual-Graph GCNs for Robust ABSA
- &#x20;One graph for semantic proximity: words that often occur together.&#x20;
- One for syntactic structure: derived from dependency parsing (e.g., what modifies what).&#x20;
- A fusion of both gives rich, context + structure-aware triplet extraction.

**Outcome:**

- Sentiment is tied directly to each aspect.
- Increases interpretability and reliability for multi-aspect reviews.

---

### Challenge 2: Sparse User Data Hinders Personalization

**Problem:** Many users write only a few reviews. Their sentiment trends cannot be inferred using standard user embeddings.

**Our Solution:**

- Extract **emotion vectors** from text using the **NRC Emotion Lexicon**, covering 8 core emotions (joy, sadness, fear, trust, etc.).
- Represent each user by the **distribution of emotional content** across their reviews.
- Perform **emotion-based user clustering** to assign sparse users to behaviorally similar groups.

**Why It Works:**

- Emotions are **domain-agnostic** and **low-dimensional**, making them robust features for sparse data.
- Clustering helps generalize trends even when direct user history is limited.

**Outcome:**

- Enables accurate sentiment forecasting for users with as few as 2–3 reviews.
- Provides an interpretable emotional profile for each user.

---

### Challenge 3: Reviews Come at Irregular Time Steps, Making Forecasting Difficult

**Problem:** Users write reviews at arbitrary, unpredictable intervals. Standard sequential models (RNNs, transformers) assume fixed-length time steps or struggle with irregular gaps.

**Our Solution:**

- Engineer a **Time-Aware Encoder-Decoder** architecture:
  - Encoder learns from past reviews and associated **time gaps**.
  - Decoder incorporates **time-delta embeddings** to predict future sentiment.
- Integrate **temporal positional encoding** to explicitly model non-uniform intervals.

**Why It Works:**

- Time-aware attention layers focus more on **recent reviews**.
- Temporal embeddings allow the model to learn patterns like: "Negative reviews tend to follow long silences."

**Outcome:**

- Outperforms baseline RNN and Transformer models on temporal forecasting tasks.
- Robust to both frequent and infrequent reviewers.

---

## Architecture Overview

```
[ Review Graph Encoder (GCN) ]
            ↓
[ Emotion-Based User Embedding ]
            ↓
[ Time-Aware Seq2Seq Decoder ]
            ↓
     → Forecasted Sentiment
```

---

## Experimental Setup

- **Dataset:** Timestamped user reviews from a public domain product review dataset.
- **Preprocessing:** Dependency parsing (spaCy), NRC emotion tagging, user ID normalization.
- **Evaluation Metrics:**
  - ABSA F1 score
  - User-level sentiment MAE
  - Forecasting RMSE and correlation

---

## Results & Impact

| Task                | Baseline     | Our Model | Improvement |
| ------------------- | ------------ | --------- | ----------- |
| ABSA Accuracy       | 72.5% (BERT) | **84.3%** | +11.8%      |
| Cold-Start User MAE | 0.89         | **0.63**  | ↓ 29.2%     |
| Forecasting RMSE    | 0.72         | **0.48**  | ↓ 33.3%     |

- **ABSA:** Handles syntactically complex sentences reliably.
- **Personalization:** Predicts user sentiment trajectories even with limited review history.
- **Forecasting:** Captures sentiment shifts and lulls tied to review timing.

---
## Presentation

For a comprehensive visual walkthrough of the project, refer to the [Canva Presentation](https://www.canva.com/design/DAGnKB8d3Hg/V-8sYcg_aqRZ-gTNMH9egA/edit).

---

## Applications

| Use Case                         | Description                                               |
| -------------------------------- | --------------------------------------------------------- |
| Feedback Intelligence Dashboards | Summarize and predict user sentiment for business teams   |
| Customer Churn Modeling          | Detect emerging negative sentiment before user attrition  |
| Adaptive Response Generation     | Tailor support interactions based on emotional trajectory |
| Product Sentiment Drift Tracking | Monitor how user feelings change aspect-wise over time    |

---

## Next Steps

- Add support for **multilingual reviews**
- Fuse **product metadata** and **demographics** for deeper modeling
- Deploy as a **microservice API** for feedback analytics tools

---

## Keywords

`Aspect-Based Sentiment`, `User Modeling`, `Emotion Extraction`, `Graph Neural Networks`, `Time-Aware Forecasting`, `Cold-Start`, `Seq2Seq`, `Customer Feedback`, `Sparse Data`, `Behavioral NLP`

---

> *This project bridges the gap between syntactic, emotional, and temporal dimensions of feedback, enabling rich and future-facing sentiment intelligence.*


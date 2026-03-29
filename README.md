# NueroEngagement Intelligence Platfrom
Designing and measuring a content engagement metric end-to-end – combining Reddit behavioral signals at scale with Meta's TRIBE v2 neural predictions.

## Overview
NueroEngagement Intelligence Platform is an end-to-end product data science project that answers one question:
>> What drives content engagement – and can predicted brain responses explain variance that behavioral signals alone cannot?

The project is structured in four layers, each mapping to a core Data Scientists competency:
<img width="677" height="289" alt="image" src="https://github.com/user-attachments/assets/33a29c0d-baee-4956-b9db-e0c84d274be5" />

## 🎯 The Research Question
Standard engagement mertics (views, likes, shares) measure volume. They don't capture quality – whether content genuinely resolnated, prompted reflection, or drove depth of engagement.

This project proposes the **Neural Engagement Score (NES)**: a composite behavioral metric designed to proxy engagement quality. It then asks:
>> Does predicted cortical activation from Meta AI Research's TRIBE v2 foundation model explain variance in NES that behavioral signals alone cannot account for?

This question sits at the intersection of product analytics, experimentation, and computational neuroscience.

## 🗂️ Project Structure
<img width="368" height="678" alt="image" src="https://github.com/user-attachments/assets/7c1049d0-fe28-40f3-86d3-c0d4474634a1" />

## 📐 Layer 1 — Behavioral Data Pipeline
### Data Scource
**Pushshift Reddit Dataset** – publicly available via Acadmemic Torrents. No API required. No terms restrictions for research use

Subreddit focus: r/technology, r/gadgets, r/apple – content-feed products analogous to Meta's social surfaces.

### DuckDB Schema 
CREATE TABLE posts (
  post_id VARCHAR PRIMARY KEY,
  subreddit VARCHAR,
  title VARCHAR,
  selftext VARCHAR,
  score INTEGER,
  num_comments INTEGER,
  upvote_ratio FLOAT,
  url VARCHAR,
  is_video BOOLEAN,
  created_utc BIGINT,
  hours_since_post FLOAT
);

CREATE TABLE nes_scores (
post_id VARCHAR PRIMARY KEY,
nes FLOAT,
engagement_tier VARCHAR,
week DATE,
computed_at TIMESTAMP DEFAULT current_timestamp
);

### Core SQL Queries
#### Engagement Funnel
-- engagement_funnel.sql
SELECT
  engagement_tier,
  COUNT(*) AS post_count,
  AVG(score) AS avg_score,
  AVG(num_comments) AS avg_comments,
  AVG(upvote_ratio) AS avg_upvote_ratio,
  ROUND(COUNT(*) * 100.0
    / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM nes_scores
GROUP BY engagement_tier
ORDER BY avg_score DESC;

#### Weekly Retention / Trend:
-- retention_weekly.sql
SELECT
  weekm
  AVG(nes) as mean_nes,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nes) AS median_nes,
  COUNT(*) AS post_volume,
  AVG(nes) - LAG(AVG(nes)) OVER (ORDER BY week) AS week_over_week_delta
FROM nes_scores
GROUP BY week
ORDER BY week;

#### NES Distribution Validation:
-- nes_validation.sql
SELECT
  ROUND(nes, 1) AS nes_bucket,
  COUNT(*) AS frequency,
  AVG(score) AS avg_behavioral_Score
FROM nes_scores
GROUP BY nes_bucket
ORDER BY nes_bucket;

## 📏 Layer 2 — Neural Engagement Score (NES)
NES is an original composite metric designed to capture **quality** of engagement, not just volume. It weights four behavioral signals:
def neural_engagement_score(row: dict) -> float:
    """
    NES: Composite engagement quality metric.

    Components:
      quality     — upvote ratio       (are reactions positive?)
      depth       — comment density    (are people discussing?)
      velocity    — score per hour     (did it engage fast?)
      controversy — penalty flag       (divisive content scores lower)

    Weights determined via sensitivity analysis on held-out validation set.

    Args:
        row (dict): Single post record with behavioral fields.

    Returns:
        float: NES in range [0, 1]
    """
    quality     = row['upvote_ratio']
    depth       = min(row['num_comments'] / 100, 1.0)
    velocity    = min(row['score'] / max(row['hours_since_post'], 1), 1.0)
    controversy = 1.0 if row['upvote_ratio'] < 0.50 else 0.0

    nes = (
        0.40 * quality
      + 0.35 * depth
      + 0.25 * velocity
      - 0.10 * controversy
    )
    return round(max(nes, 0.0), 4)


### Metric Validation Checklist
<img width="658" height="230" alt="image" src="https://github.com/user-attachments/assets/c59a6542-786d-4a03-8be6-e7773a6ca40a" />


## 🧪 Layer 3 — A/B Experiment Framework
### Experiment Design
Research Question:
  Does video content drive higher NES than text-only posts
  in tenchology subreddits, controlling for post age and subreddit?

Unit of randomization: post_id (post-level)
Control : Text-only posts (n ≈ 10,000)
Treatement: Posts with video/media (n ≈ 10,000)

Primary metric: Neural Engagement Score (NES)
Guardrail metric: upvote_ratio (must not drop > 2%)
Min detectable effect: 5% relative lift in NES
Statistical power: 80%
Significance level: α = 0.05

Primary test: Welch's t-test (unequal variance assumed)
Fallback test: Mann-Whitney U (is NES non-normal)
Effect size: Cohen's d

### Example Output
════════════════════════════════════════════════
         A/B EXPERIMENT RESULTS
════════════════════════════════════════════════
  Control   (text-only)  — Mean NES : 0.412
  Treatment (video)      — Mean NES : 0.487

  Absolute lift    :  +0.075
  Relative lift    :  +18.2%
  p-value          :  0.0014  ✅ Significant
  Cohen's d        :  0.43    (Medium effect)

  Guardrail check  :  upvote_ratio Δ = +0.8%  ✅ Safe

  Recommendation   :  Video content drives meaningfully
                      higher engagement quality. Prioritize
                      video surfaces in content ranking.
════════════════════════════════════════════════

## 🔬 Layer 4 — TRIBE v2 AI Layer
## What is TRIBE v2?
TRIBE v2 (Meta AI Research, 2026) is a multimodal brain encoding modek that predicts fMRI cortical responses to naturalistic stimuli. It combines LLaMA 3.2 (text), V-JEPA2(video), and Wav2Vec-BERT (audio) into a unified Transformer that maps inputs onto the fsaverage5 cortical surface (~20,000 vertices).

>> This project uses TRIBE v2 for non-commercial research purposes under CC BY-NC 4.0.

### The Hypothesis
Behavioral signals (upvotes, comments) measure revealed preference. Brain encoding models measure predicted neural response. The question is:
>> Does predicted cortical activation (CES) expakin variance in NES that behavioral signals alone cannot?

### Implementation
# tribe_layer.py
from tribev2 import TribeModel
from scipy.stats import spearmanr

def run_tribe_correlation(video_posts_df, model_cache="./cache"):
    """
    For posts with video content, compare predicted cortical activation
    (CES from TRIBE v2) against behavioral NES scores.

    Returns Spearman correlation + interpretation.
    """
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=model_cache
    )
    ces_scores, nes_scores = [], []

    for _, row in video_posts_df.iterrows():
        df_events = model.get_events_dataframe(
            video_path=row['local_video_path']
        )
        preds, _ = model.predict(events=df_events)
        ces = float(preds.mean())      # Cortical Engagement Score
        ces_scores.append(ces)
        nes_scores.append(row['nes'])

    r, p = spearmanr(ces_scores, nes_scores)
    return {
        "spearman_r":      round(r, 4),
        "p_value":         round(p, 4),
        "n":               len(ces_scores),
        "significant":     p < 0.05,
        "interpretation": (
            "TRIBE v2 adds explanatory power beyond behavioral signals."
            if p < 0.05 else
            "Neural activation does not significantly predict NES. "
            "Behavioral signals are sufficient at this scale."
        )
    }

### Honest Result Framing
<img width="677" height="277" alt="image" src="https://github.com/user-attachments/assets/ff726614-2cf1-4952-9e3f-39e46e9ec23b" />

## ⚙️ Installation
1. Clone the repo
bashgit clone https://github.com/YOUR_USERNAME/neuro-engagement-intelligence.git
cd neuro-engagement-intelligence

2. Create a virtual environment
bashpython -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

3. Install dependencies
bashpip install -r requirements.txt

4. (Optional) Authenticate with HuggingFace for TRIBE v2
Required only for Layer 4. Needs LLaMA 3.2 access (gated model):
bashhuggingface-cli login
Create a read access token at huggingface.co/settings/tokens and paste when prompted.

## 🚀 Quick Start
Run the full pipeline
bash# Step 1: Ingest Reddit data → DuckDB
python pipeline/ingest.py --input data/raw/ --db data/processed/reddit.duckdb

### Step 2: Compute NES for all posts
python pipeline/metric.py --db data/processed/reddit.duckdb

### Step 3: Run A/B experiment
python pipeline/experiment.py --db data/processed/reddit.duckdb

### Step 4: (Optional) Run TRIBE v2 correlation layer
python pipeline/tribe_layer.py --db data/processed/reddit.duckdb

## Launch the Streamlit dashboard
streamlit run dashboard/app.py

## 📊 Dashboard Pages
<img width="579" height="241" alt="image" src="https://github.com/user-attachments/assets/1457cb4d-8405-489a-b3a8-4118dc78a1c5" />

## 🧰 Tech Stack
<img width="527" height="431" alt="image" src="https://github.com/user-attachments/assets/d40e4e6f-dd4c-43cd-8d0b-7672dc1002aa" />

## ✅ Running Tests
pytest tests/ -v

Tests cover:
- NES return a float in [0,1]
- NES controversy penalty applies correctly ay threshold
- Experiment returns all required keys (winner, p_value, cohen_d, guardrail_safe)
- Winner direction is consistent with NES delta
- TRIBE v2 correlation output is well-formed with expected keys

## 🗺️ Roadmap
✅ Pushshift ingestion piepeline -> DuckDB
✅ NES metric definition ->  validation suite
✅ A/B experiment framework (Welch's t-test + Cohen's d + guardrails)
✅ TRIBE v2 CES -> NES correlation layer
✅ Streamlit dashboard (5 pages)
✅ Github Actions CI
✅ NES weight optimization via grid search
✅ Multi-subreddit generalization test
✅ Temporal holdout validation (train Q1, test Q2)
✅ Alognauts 2025 benchamrk integration

## Install dependencies
pip install duckdb zstandard
```

For the torrent download, install **qBittorrent** (GUI) from [qbittorrent.org](https://www.qbittorrent.org/download) — free, no account needed.

---

### Step 2 — Download only the 3 subreddits you need

Go to this URL:
**https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4**

Click **Download Torrent** → open in qBittorrent → when it asks which files to download, **search for and select only:**
```
technology_submissions.zst
gadgets_submissions.zst
apple_submissions.zst

## 📄 License & Attribution
This project is for non-commercial, research, and portfolio use only.
TRIBE v2 component built on facebookresearch/tribev2 by Meta AI Research, licensed under CC BY-NC 4.0.
Reddit data sourced from Pushshift under academic research terms.

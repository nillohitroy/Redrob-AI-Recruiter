# Redrob AI Candidate Ranker - Team TGB

This repository contains the source code, pre-computed artifacts, and deployment configurations for Team TGB's submission to the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. 

This system is engineered as a **Two-Stage Cascade Ranker** designed to strictly adhere to the Stage 3 reproduction constraints: **≤5 minutes wall-clock time, ≤16GB RAM, and 100% CPU-only execution with zero external network API calls.**

---

## Architecture Overview

To balance latency, compute constraints, and reasoning quality, the pipeline is split into offline pre-computation and a highly optimized runtime engine.

### 1. Offline Pre-Computation (Phase 1)
* **Semantic Embedding:** All 100,000 candidate profiles were vectorized offline using `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in a FAISS index.
* **Behavioral Scoring:** We pre-calculated a composite behavioral multiplier based on recruiter response rates, recent activity, and honeypot disqualification flags.

### 2. The Runtime Engine (Phase 2 & 3)
When `rank.py` is executed, it runs entirely locally on CPU:
1. **Semantic Recall (FAISS):** The Job Description is vectorized, and we instantly query the FAISS index to retrieve the top 2,000 mathematically similar candidates.
2. **Behavioral & Hard Logic Precision:** We apply the behavioral multipliers to the semantic scores. We then parse the JSON data *only* for the top 2,000 candidates to apply hard JD disqualifiers
3. **Safe Reasoning Generation:** We isolate the Top 100 candidates. Using a highly quantized local LLM (`Qwen1.5-0.5B-Chat-GGUF` running via `llama.cpp`), we extract specific factual data (title, experience, skills) and generate a grounded justification for each candidate.

---

## Repository Structure

### Core Execution Files
* `rank.py` — The primary headless CLI script that executes the end-to-end pipeline.
* `app.py` — A Streamlit UI wrapper for the Hugging Face Sandbox deployment.
* `requirements.txt` — Dependency list, including the pre-compiled CPU wheel for `llama-cpp-python`.
* `.streamlit/config.toml` — Sandbox configuration to bypass default upload limits and XSRF blocks.

### Pre-Computed Artifacts (Requires Git LFS)
* `candidate_index.faiss` — The offline vector index containing 100,000 candidate embeddings.
* `candidate_metadata.csv` — Pre-calculated behavioral multipliers and honeypot flags.

### Dataset & Hackathon Context Files
* `candidates.jsonl` — The primary 100K candidate dataset (must be unzipped).
* `sample_candidates.json` — A 50-candidate subset for rapid sandbox testing.
* `candidate_schema.json` — JSON schema definitions for the candidate records.
* `job_description.docx` — The target role requirements used to tune the semantic query.
* `redrob_signals_doc.docx` — Reference for the 23 behavioral signals used in our multiplier logic.
* `submission_spec.docx` — The hackathon constraints and evaluation criteria.
* `validate_submission.py` — The official script to verify CSV format compliance.
* `sample_submission.csv` — Formatting reference provided by the organizers.

---

## Local Execution (Stage 3 Code Reproduction)

The runtime pipeline is designed to run locally on a standard CPU machine within the 5-minute constraint.

### 1. Prerequisites & Setup
Ensure you have Python 3.10+ installed. Because the FAISS index is >100MB, you must use Git LFS to clone the repository properly.

```bash
# Clone the repository with LFS enabled
git clone https://github.com/nillohitroy/Redrob-AI-Recruiter.git
cd Redrob-AI-Recruiter

# Pull the large FAISS index file
git lfs pull

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Ranker Engine
Ensure the unzipped candidates.jsonl file is located in your working directory. Run the exact command mandated by the submission spec:

```bash
cd src
python rank.py --candidates ../dataset/candidates.jsonl --out ./submission.csv
```
## Validation
Before uploading to the hackathon portal, verify the output perfectly matches the required schema using the official validator tool.

```bash
python validate_submission.py submission.csv
```

## Sandbox Environment
To prove small-sample reproducibility under strict CPU constraints, a live Sandbox is deployed via Hugging Face Spaces.

Demo Link: [Hugging Space Environment](https://huggingface.co/spaces/nillohitroy/redrob-ai-ranker)

### How to use the sandbox
1. Navigate to the Sandbox URL.

2. Provide a sample dataset. You can provide this in two ways:

    a. Upload File Tab: Standard file drag-and-drop.

    b. Paste Text Tab: A fallback mechanism to paste JSONL data directly.

3. Click Run Ranker Engine.

The system will execute the exact same rank.py logic and provide a downloadable submission.csv

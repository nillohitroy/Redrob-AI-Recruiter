import argparse
import pandas as pd
import numpy as np
import faiss
import json
import time
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

def run_pipeline(candidates_path, output_path):
    start_time = time.time()
    print("Initiating Redrob AI Ranker Pipeline...")

    # --- 1. LOAD ARTIFACTS ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    INDEX_PATH = os.path.join(base_dir, 'candidate_index.faiss')
    META_PATH = os.path.join(base_dir, 'candidate_metadata.csv')
    
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Pre-computed FAISS index or metadata CSV not found. Run Phase 1 first.")

    index = faiss.read_index(INDEX_PATH)
    metadata_df = pd.read_csv(META_PATH)

    # --- 2. EXTRACT UPLOADED CANDIDATES FIRST ---
    candidate_details = {}
    print("Parsing candidate file...")
    
    def extract_cand(cand):
        c_id = cand.get('candidate_id')
        if not c_id: return
        history = cand.get('career_history', [])
        profile = cand.get('profile', {})
        skills = [s.get('name', '') for s in cand.get('skills', []) if isinstance(s, dict)]
        
        # Safely extract experience to prevent type errors
        try:
            exp = float(profile.get('years_of_experience', 0))
        except (ValueError, TypeError):
            exp = 0.0
            
        candidate_details[c_id] = {
            "companies": [h.get('company', '').lower() for h in history if h.get('company')],
            "title": profile.get('current_title', 'Engineer') or 'Engineer',
            "experience": exp,
            "skills": ", ".join(skills[:3]) if skills else "relevant technical tools"
        }

    try:
        with open(candidates_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                for cand in data:
                    extract_cand(cand)
    except json.JSONDecodeError:
        with open(candidates_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    cand = json.loads(line)
                    extract_cand(cand)
                except json.JSONDecodeError:
                    pass

    uploaded_ids = set(candidate_details.keys())
    if not uploaded_ids:
        raise ValueError("No valid candidates found in the input file.")

    # --- 3. VECTORIZE JOB DESCRIPTION ---
    jd_text = """
    Senior AI Engineer. Deep technical depth in modern ML systems: embeddings, retrieval, ranking, LLMs, fine-tuning. 
    Production experience with embeddings-based retrieval systems (sentence-transformers, BGE).
    Production experience with vector databases or hybrid search infrastructure.
    Strong Python. Evaluation frameworks for ranking systems.
    """
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_vector = embed_model.encode([jd_text]).astype('float32')
    faiss.normalize_L2(jd_vector)

    # --- 4. SEMANTIC RECALL (Filtered by Uploaded File) ---
    RECALL_K = len(metadata_df)
    distances, indices = index.search(jd_vector, RECALL_K)
    
    recall_df = metadata_df.iloc[indices[0]].copy()
    recall_df['semantic_score'] = distances[0]
    
    # Filter immediately to only the candidates in the uploaded file
    recall_df = recall_df[recall_df['candidate_id'].isin(uploaded_ids)].copy()

    # Apply Behavioral Math and drop honeypots
    recall_df['composite_score'] = recall_df['semantic_score'] * recall_df['behavioral_multiplier']
    recall_df = recall_df[recall_df['is_honeypot_flag'] == 0]

    # --- 5. HARD DISQUALIFICATIONS ---
    CONSULTANCIES = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
    RESEARCH_TERMS = {'lab', 'university', 'research', 'academic', 'institute'}

    def is_disqualified(c_id):
        companies = candidate_details.get(c_id, {}).get('companies', [])
        if not companies: return False
        if all(any(cons in comp for cons in CONSULTANCIES) for comp in companies): return True
        if all(any(res in comp for res in RESEARCH_TERMS) for comp in companies): return True
        return False

    # Apply disqualifications to the ENTIRE valid pool before cutting off the top N
    recall_df['is_disqualified'] = recall_df['candidate_id'].apply(is_disqualified)
    final_pool = recall_df[recall_df['is_disqualified'] == False].copy()

    # Sort and handle deterministic tie-breaking
    final_pool = final_pool.sort_values(by=['composite_score', 'candidate_id'], ascending=[False, True])
    
    # Isolate top 100 (If the dataset has less than 100, it simply returns what it has)
    top_100 = final_pool.head(100).copy()

    # Normalize scores 0.70 to 0.99
    max_score, min_score = top_100['composite_score'].max(), top_100['composite_score'].min()
    if max_score > min_score:
        top_100['composite_score'] = ((top_100['composite_score'] - min_score) / (max_score - min_score)) * 0.29 + 0.70
    elif max_score == min_score and not top_100.empty:
        top_100['composite_score'] = 0.85
    
    top_100['rank'] = range(1, len(top_100) + 1)

    # --- 6. REASONING GENERATION ---
    print("Loading LLM for Reasoning Generation...")
    model_path = hf_hub_download(repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF", filename="qwen1_5-0_5b-chat-q4_k_m.gguf")
    llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)

    reasonings = []
    for _, row in top_100.iterrows():
        c_id = row['candidate_id']
        facts = candidate_details[c_id]
        
        # Pre-fill Prompting
        prompt = f"""<|im_start|>system
You are a technical recruiter. Write exactly one professional sentence explaining why the candidate is a fit. Use ONLY the provided facts. Do not hallucinate.
<|im_end|>
<|im_start|>user
Title: {facts['title']}
Experience: {facts['experience']} years
Skills: {facts['skills']}
<|im_end|>
<|im_start|>assistant
This candidate is a strong fit because they have"""

        # REMOVED '.' from stop tokens to allow float experience numbers (e.g. 4.5)
        # Increased max_tokens to 45 so it does not cut off mid-sentence
        output = llm(prompt, max_tokens=45, temperature=0.1, stop=["<|im_end|>", "\n"])
        raw_text = output['choices'][0]['text'].strip()
        
        # Safety fallback
        if len(raw_text) > 10:
            final_reasoning = f"This candidate is a strong fit because they have {raw_text}"
            if not final_reasoning.endswith('.'):
                final_reasoning += '.'
        else:
            final_reasoning = f"This candidate is a strong fit because they have {facts['experience']} years of experience as a {facts['title']} with key skills in {facts['skills']}."
            
        reasonings.append(final_reasoning)

    top_100['reasoning'] = reasonings

    # --- 7. FORMAT & EXPORT ---
    submission_df = top_100[['candidate_id', 'rank', 'composite_score', 'reasoning']].copy()
    submission_df.rename(columns={'composite_score': 'score'}, inplace=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"Pipeline complete in {elapsed:.2f} seconds. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl file")
    parser.add_argument("--out", required=True, help="Path to output submission.csv file")
    
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)

import yaml
import os
from openai import OpenAI
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# Load API key from .env or environment
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_resume(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def embed_resume_sections(resume: List[Dict]) -> Dict[str, List[float]]:
    embeddings = {}
    for section in resume:
        if section["type"] == "experience":
            combined_text = section.get("summary", "")
            for contrib in section.get("contributions", []):
                combined_text += " " + contrib["description"]
            embeddings[section["id"]] = get_embedding(combined_text)
    return embeddings

def rank_by_embedding(resume: List[Dict], job_description: str) -> List[Dict]:
    job_embedding = get_embedding(job_description)
    resume_embeddings = embed_resume_sections(resume)

    scores = []
    for section_id, embedding in resume_embeddings.items():
        score = cosine_similarity([embedding], [job_embedding])[0][0]
        for section in resume:
            if section["id"] == section_id:
                scores.append((score, section))
                break

    scores.sort(reverse=True, key=lambda x: x[0])
    return [section for score, section in scores]

def generate_tailored_resume(
    resume,
    job_description,
    top_n=3,
    use_embeddings=False,
    ordering="relevancy"
):
    """
    Generate a tailored resume with flexible ordering.
    ordering options: 'relevancy', 'chronological', 'hybrid'
    use_embeddings=False will use offline keyword scoring.
    """

    # --- Step 1: Rank by chosen scoring method ---
    if use_embeddings:
        ranked_experience = rank_by_embedding(resume, job_description)
    else:
        from resume_scoring_engine import rank_resume_sections
        ranked_experience = rank_resume_sections(resume, job_description)

    # --- Step 2: Apply ordering preference ---
    if ordering == "relevancy":
        ordered_experience = ranked_experience

    elif ordering == "chronological":
        def parse_start_year(date_str):
            parts = date_str.split("–")[0].strip()
            try:
                return int(parts.split()[-1])
            except:
                return 0
        ordered_experience = sorted(
            [sec for sec in resume if sec["type"] == "experience"],
            key=lambda s: parse_start_year(s.get("dates", "")),
            reverse=True
        )

    elif ordering == "hybrid":
        top_relevant = ranked_experience[:top_n]
        remaining = [sec for sec in resume if sec["type"] == "experience" and sec not in top_relevant]

        def parse_start_year(date_str):
            parts = date_str.split("–")[0].strip()
            try:
                return int(parts.split()[-1])
            except:
                return 0
        remaining_sorted = sorted(remaining, key=lambda s: parse_start_year(s.get("dates", "")), reverse=True)

        ordered_experience = top_relevant + remaining_sorted

    else:
        ordered_experience = ranked_experience  # default fallback

    # --- Step 3: Build final tailored resume ---
    tailored = [sec for sec in resume if sec["type"] in ["summary", "skills"]]
    tailored.extend(ordered_experience)
    return tailored


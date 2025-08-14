import yaml
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings cache file
CACHE_FILE = "embeddings_cache.json"


# =====================
# YAML LOAD
# =====================
def load_resume(file_path):
    """Load the YAML resume as a list of sections."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            resume_sections = yaml.safe_load(f)
        if not isinstance(resume_sections, list):
            raise ValueError("Resume YAML must be a list of sections")
        return resume_sections
    except Exception as e:
        print(f"Error loading resume YAML: {e}")
        return []


# =====================
# TEXT BUILDING
# =====================
def _safe_list(x):
    return x if isinstance(x, (list, tuple)) else []

def _safe_dict_list(x):
    """Ensure we only iterate dict-like entries; coerce non-dicts to empty."""
    lst = _safe_list(x)
    return [e for e in lst if isinstance(e, dict)]

def section_text(section):
    """
    Build a searchable string from a section, including tags and skills_used.
    Nil-safe for missing/None fields (entries, contributions, tags, summary).
    """
    if not isinstance(section, dict):
        return ""

    text_parts = []

    # Summary
    summary = section.get("summary")
    if isinstance(summary, str) and summary.strip():
        text_parts.append(summary.strip())

    # Tags
    tags = _safe_list(section.get("tags"))
    text_parts.extend(str(t).strip() for t in tags if t)

    # Experience -> contributions + skills_used
    if section.get("type") == "experience":
        for contrib in _safe_dict_list(section.get("contributions")):
            desc = contrib.get("description")
            if isinstance(desc, str) and desc.strip():
                text_parts.append(desc.strip())
            skills_used = _safe_list(contrib.get("skills_used"))
            text_parts.extend(str(s).strip() for s in skills_used if s)
    else:
        # Other sections -> flatten entries values
        for entry in _safe_dict_list(section.get("entries")):
            for value in entry.values():
                if value is None:
                    continue
                text_parts.append(str(value).strip())

    # Join safely
    return " ".join([t for t in text_parts if t])


# =====================
# CACHE MANAGEMENT
# =====================
def _load_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading embeddings cache: {e}")
    return {}


def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Error saving embeddings cache: {e}")


def clear_embeddings_cache():
    """Delete the embeddings cache file."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
    except Exception as e:
        print(f"Error clearing embeddings cache: {e}")


# =====================
# OPENAI EMBEDDINGS
# =====================
def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI with caching."""
    try:
        from openai_utils import get_embedding as _get_embedding
        cache = _load_cache()
        key = f"{model}:{text.strip()}"
        if key in cache:
            return cache[key]
        emb = _get_embedding(text, model)
        cache[key] = emb
        _save_cache(cache)
        return emb
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


def rank_by_embedding(resume, job_description, model="text-embedding-3-small"):
    """Rank sections by semantic similarity using OpenAI embeddings."""
    try:
        job_embedding = get_embedding(job_description, model=model)
        section_scores = []
        for sec in resume:
            emb = get_embedding(section_text(sec), model=model)
            score = cosine_similarity([job_embedding], [emb])[0][0]
            section_scores.append((sec, score))

        section_scores.sort(key=lambda x: x[1], reverse=True)
        scores_dict = {id(sec): score for sec, score in section_scores}
        ranked_sections = [sec for sec, _ in section_scores]
        return ranked_sections, scores_dict
    except Exception as e:
        print(f"Error ranking by embedding: {e}")
        return resume, {}


# =====================
# TF-IDF RANKING
# =====================
def rank_resume_sections(resume, job_description):
    """Rank sections by keyword-based TF-IDF similarity."""
    corpus = [job_description] + [section_text(sec) for sec in resume]
    tfidf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    ranked_sections = [resume[i] for i in ranked_indices]
    scores_dict = {id(resume[i]): float(cosine_similarities[i]) for i in range(len(resume))}
    return ranked_sections, scores_dict


# =====================
# MAIN GENERATOR
# =====================
def generate_tailored_resume(
    resume,
    job_description,
    top_n=None,
    use_embeddings=False,
    ordering="relevancy",
    embedding_model="text-embedding-3-small"
):
    """
    Generate a tailored resume based on ordering preference.
    Returns: (list of sections, scores_dict)
    """
    if ordering == "chronological":
        return [sec for sec in resume if sec["type"] != "header"], {}

    if use_embeddings:
        ranked_sections, scores = rank_by_embedding(resume, job_description, model=embedding_model)
    else:
        ranked_sections, scores = rank_resume_sections(resume, job_description)

    if ordering == "relevancy":
        return (ranked_sections[:top_n] if top_n else ranked_sections), scores

    if ordering == "hybrid":
        top_relevant = ranked_sections[:top_n]
        remaining = [sec for sec in resume if sec not in top_relevant and sec["type"] != "header"]
        return top_relevant + remaining, scores

    return resume, scores


# =====================
# IMPACT BULLET GENERATION
# =====================
def enhance_experience_with_impact(
    resume,
    job_description,
    use_gpt=False,
    model="gpt-3.5-turbo",
    mark_generated=True,
    bullets_per_role=1
):
    """
    Append impact bullets to each experience section.
    Adds metadata: impact: True, source: generated
    """
    if not use_gpt or bullets_per_role < 1:
        return resume

    try:
        from openai_utils import chat_completion
        for section in resume:
            if section.get("type") != "experience":
                continue
            existing = "\n".join("- " + c.get("description", "") for c in section.get("contributions", []))
            base_prompt = f"""Job Description:
{job_description}

Role: {section.get('title','')} at {section.get('company','')}
Summary: {section.get('summary','')}
Contributions:
{existing}

Generate ONE new bullet point (max 250 characters) that is:
- Relevant to the job description
- Results/impact-focused (numbers or outcomes if possible)
- Not redundant with existing bullets
Return only the bullet sentence, no leading dash.
"""
            for _ in range(bullets_per_role):
                try:
                    messages = [
                        {"role": "system", "content": "You are an expert resume writer optimizing resumes for specific job descriptions."},
                        {"role": "user", "content": base_prompt.strip()}
                    ]
                    new_bullet, resp = chat_completion(messages, model=model, max_tokens=200, temperature=0.5, context="resume_semantic_scoring_engine:impact_bullet")
                    if not new_bullet:
                        continue
                    contrib = {
                        "description": new_bullet,
                        "skills_used": [],
                        "impact": True if mark_generated else False,
                        "source": "generated" if mark_generated else None,
                    }
                    section.setdefault("contributions", []).append(contrib)
                except Exception as e:
                    print(f"Error generating impact bullet: {e}")
                    continue
        return resume
    except Exception as e:
        print(f"Error enhancing experience with impact: {e}")
        return resume

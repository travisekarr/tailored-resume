import yaml
import os
import json
import re
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

    # Tags (support dict with hard/soft)
    tags_obj = section.get("tags")
    tags_list = []
    if isinstance(tags_obj, dict):
        tags_list.extend([t for t in (tags_obj.get("hard") or []) if t])
        tags_list.extend([t for t in (tags_obj.get("soft") or []) if t])
    else:
        tags_list = _safe_list(tags_obj)
    text_parts.extend(str(t).strip() for t in tags_list if t)

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
    try:
        corpus = [job_description] + [section_text(sec) for sec in resume]
        tfidf = TfidfVectorizer().fit_transform(corpus)
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        ranked_indices = cosine_similarities.argsort()[::-1]
        ranked_sections = [resume[i] for i in ranked_indices]
        scores_dict = {id(resume[i]): float(cosine_similarities[i]) for i in range(len(resume))}
        return ranked_sections, scores_dict
    except Exception as e:
        print(f"Error in rank_resume_sections: {e}")
        return resume, {}


# =====================
# MAIN GENERATOR
# =====================
def generate_tailored_resume(
    resume,
    job_description,
    top_n=None,
    use_embeddings=False,
    ordering="relevancy",
    embedding_model="text-embedding-3-small",
    role_filter=True,
):
    """
    Generate a tailored resume based on ordering preference.
    Returns: (list of sections, scores_dict)
    """
    if ordering == "chronological":
        out_sections = [sec for sec in resume if sec["type"] != "header"]
        scores = {}
        # Optionally filter contributions by relevancy role
        if role_filter:
            _apply_role_filter(out_sections, job_description)
        return out_sections, scores

    if use_embeddings:
        ranked_sections, scores = rank_by_embedding(resume, job_description, model=embedding_model)
    else:
        ranked_sections, scores = rank_resume_sections(resume, job_description)

    if ordering == "relevancy":
        out_sections = (ranked_sections[:top_n] if top_n else ranked_sections)
        if role_filter:
            _apply_role_filter(out_sections, job_description)
        return out_sections, scores

    if ordering == "hybrid":
        top_relevant = ranked_sections[:top_n]
        remaining = [sec for sec in resume if sec not in top_relevant and sec["type"] != "header"]
        out_sections = top_relevant + remaining
        if role_filter:
            _apply_role_filter(out_sections, job_description)
        return out_sections, scores

    out_sections = [sec for sec in resume if sec.get("type") != "header"]
    if role_filter:
        _apply_role_filter(out_sections, job_description)
    return out_sections, scores


# =====================
# ROLE-BASED CONTRIBUTION FILTERING
# =====================
def _classify_role(job_description: str) -> str:
    """Return 'developer', 'manager', or 'both' based on JD signals."""
    if not isinstance(job_description, str) or not job_description.strip():
        return "both"
    text = job_description.lower()
    # Token-like scan
    import re as _re
    tokens = _re.findall(r"[a-zA-Z0-9\-\./#]+", text)

    mgr_terms = {
        "manager","management","managing","people","direct reports","head","director","vp",
        "leadership","stakeholder","stakeholders","roadmap","hiring","coaching","mentoring",
        "performance","budget","budgets","program","portfolio","strategy","strategic",
        "org","organizational","operational","operations","lead manager","engineering manager",
        "sr manager","senior manager","team management","line manager","staffing",
    }
    dev_terms = {
        "developer","engineer","software engineer","senior engineer","principal engineer",
        "individual contributor","ic","hands-on","coding","programming","build","implement",
        "api","microservices","architecture","architect","design","system design","sde","swe",
    }
    # Count hits (substring in tokens and raw text for multi-word phrases)
    def count_terms(terms):
        c = 0
        for t in terms:
            if " " in t:
                if t in text:
                    c += 2  # weight phrases higher
            else:
                c += sum(1 for tok in tokens if tok == t)
        return c

    mgr_score = count_terms(mgr_terms)
    dev_score = count_terms(dev_terms)

    # Heuristic thresholds
    if mgr_score >= dev_score + 2:
        return "manager"
    if dev_score >= mgr_score + 2:
        return "developer"
    return "both"


def _apply_role_filter(sections: list, job_description: str):
    role = _classify_role(job_description)
    if role == "both":
        return
    for sec in sections:
        if sec.get("type") != "experience":
            continue
        contribs = sec.get("contributions") or []
        def _ok(c):
            rel = str((c or {}).get("relevancy", "both")).strip().lower()
            return rel in ("both", role)
        filtered = [c for c in contribs if _ok(c)]
        if filtered:
            sec["contributions"] = filtered
        # If filtering drops all, keep originals to avoid empty sections


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


def generate_tailored_summary(
    resume,
    job_description,
    use_gpt: bool = False,
    model: str = "gpt-3.5-turbo",
    use_embeddings: bool = False,
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """
    Build a concise tailored summary string:
      - Opening line from header title/years
      - Top 3–4 matched skills (JD ∩ resume tags/skills)
      - Notable achievement picked from resume (overlap or embeddings)
      - Optional GPT paraphrase that preserves facts via placeholders
    """
    try:
        header = next((sec for sec in resume if sec.get("type") == "header"), {}) or {}
        title = header.get("title", "Experienced Professional")
        years_exp = header.get("years_experience")
        opening_line = f"{title} with {years_exp}+ years of proven expertise" if years_exp else f"{title} with proven expertise"

        # Build JD keywords (simple tokens)
        jd_tokens = {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", job_description or "")}

        # Flatten resume tags/skills with common variants
        def _flatten_tags(tags):
            if isinstance(tags, dict):
                vals = []
                vals.extend(tags.get("hard", []) or [])
                vals.extend(tags.get("soft", []) or [])
                return [str(t).strip().lower() for t in vals if t]
            elif isinstance(tags, list):
                return [str(t).strip().lower() for t in tags if t]
            return []

        def _tag_variants(t: str) -> set[str]:
            v = {t}
            v.add(t.replace("_", "/")); v.add(t.replace("/", "_"))
            v.add(t.replace("-", "_")); v.add(t.replace("_", "-"))
            v.add(t.replace("_", " "))
            v.add(t.replace(".", "")); v.add(t.replace(".", " "))
            return {x for x in v if x}

        all_resume_terms = set()
        for section in resume or []:
            tags = section.get("tags")
            for t in _flatten_tags(tags):
                all_resume_terms.update(_tag_variants(t))
            if section.get("type") == "experience":
                for c in (section.get("contributions") or []):
                    for s in (c.get("skills_used") or []):
                        all_resume_terms.update(_tag_variants(str(s).strip().lower()))

        matched_skills = sorted(list(jd_tokens & all_resume_terms))
        top_skills_str = ", ".join(matched_skills[:4]) if matched_skills else None

        # Achievement: embeddings (if on) or overlap from resume_utils
        try:
            from resume_utils import pick_best_achievement_overlap as _ru_pick_best_achievement_overlap
        except Exception:
            _ru_pick_best_achievement_overlap = None

        def _pick_best_achievement_embeddings() -> str | None:
            try:
                job_vec = get_embedding(job_description or "", model=embedding_model)
                if not job_vec:
                    return None
                cands = []
                for sec in resume or []:
                    if sec.get("type") == "experience":
                        for c in (sec.get("contributions") or []):
                            d = (c or {}).get("description")
                            if isinstance(d, str) and d.strip():
                                cands.append(d.strip())
                summ = next((sec for sec in resume or [] if sec.get("type") == "summary"), {})
                for a in (summ.get("achievements") or []):
                    if isinstance(a, str) and a.strip():
                        cands.append(a.strip())
                seen, uniq = set(), []
                for c in cands:
                    k = c.lower()
                    if k not in seen:
                        uniq.append(c); seen.add(k)
                best, best_score = None, -1.0
                for cand in uniq:
                    vec = get_embedding(cand, model=embedding_model)
                    if not vec:
                        continue
                    score = cosine_similarity([job_vec], [vec])[0][0]
                    if score > best_score:
                        best, best_score = cand, float(score)
                return best
            except Exception:
                return None

        achievement = None
        if use_embeddings:
            achievement = _pick_best_achievement_embeddings()
        if not achievement and callable(_ru_pick_best_achievement_overlap):
            achievement = _ru_pick_best_achievement_overlap(resume, job_description)

        parts = [opening_line]
        if top_skills_str:
            parts.append(f"specializing in {top_skills_str}")
        if achievement:
            parts.append(f"Notable achievement: {achievement}")
        offline = ". ".join(parts) + "."
        if not use_gpt:
            return offline

        # Placeholder-protected paraphrase
        outline = []
        outline.append("[TITLE] with [YEARS]+ years of proven expertise" if years_exp else "[TITLE] with proven expertise")
        if top_skills_str:
            outline.append("specializing in [SKILLS]")
        if achievement:
            outline.append("Notable achievement: [ACHIEVEMENT]")
        outline_text = ". ".join(outline) + "."
        try:
            from openai_utils import chat_completion
            sys = "You rewrite text concisely without adding facts. Preserve placeholders exactly."
            prompt = (
                "Rewrite the following into a polished 2–4 sentence professional summary (<=300 characters),\n"
                "keeping the placeholders [TITLE], [YEARS], [SKILLS], and [ACHIEVEMENT] EXACTLY AS WRITTEN.\n"
                "Do NOT introduce any new facts, numbers, or achievements.\n\n"
                f"Outline:\n{outline_text}"
            )
            msg = [
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ]
            text, _resp = chat_completion(msg, model=model, max_tokens=180, temperature=0.2, context="sem_engine:tailored_summary")
            templated = (text or outline_text).strip()
            subs = {
                "[TITLE]": title,
                "[YEARS]": str(years_exp) if years_exp is not None else "",
                "[SKILLS]": top_skills_str or "",
                "[ACHIEVEMENT]": achievement or "",
            }
            for k, v in subs.items():
                templated = templated.replace(k, v)
            return re.sub(r"\s{2,}", " ", templated).strip(" .") + "."
        except Exception:
            return offline
    except Exception:
        return ""
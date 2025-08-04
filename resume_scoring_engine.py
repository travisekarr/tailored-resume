import yaml
from typing import List, Dict

def load_resume(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_keywords(text: str) -> List[str]:
    return [word.lower().strip(".,;:()") for word in text.split() if len(word) > 2]

def score_section(section: Dict, job_keywords: List[str]) -> float:
    section_keywords = set()
    for contrib in section.get("contributions", []):
        section_keywords.update(extract_keywords(contrib["description"]))
        section_keywords.update([s.lower() for s in contrib.get("skills_used", [])])
    section_keywords.update([t.lower() for t in section.get("tags", [])])
    match_count = len(set(job_keywords) & section_keywords)
    return match_count / (len(section_keywords) + 1e-5)

def rank_resume_sections(resume: List[Dict], job_description: str) -> List[Dict]:
    """
    Rank only experience sections by keyword match.
    Header, summary, skills, and bottom sections are excluded from reordering.
    """
    # Separate sections
    header = next((sec for sec in resume if sec["type"] == "header"), None)
    fixed_top = [sec for sec in resume if sec["type"] in ["summary", "skills"] or sec["id"].startswith("tech_")]
    experience_sections = [sec for sec in resume if sec["type"] == "experience"]
    fixed_bottom = [sec for sec in resume if sec["type"] in ["education", "certifications", "projects", "awards"]]

    # Rank experiences
    job_keywords = extract_keywords(job_description)
    scored = []
    for section in experience_sections:
        score = score_section(section, job_keywords)
        scored.append((score, section))

    scored.sort(reverse=True, key=lambda x: x[0])
    ranked_experience = [section for score, section in scored]

    # Assemble resume
    tailored = []
    if header:
        tailored.append(header)

    tailored.extend(fixed_top)
    tailored.extend(ranked_experience)
    tailored.extend(fixed_bottom)

    return tailored

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
    job_keywords = extract_keywords(job_description)
    scored = []
    for section in resume:
        if section["type"] == "experience":
            score = score_section(section, job_keywords)
            scored.append((score, section))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [section for score, section in scored]

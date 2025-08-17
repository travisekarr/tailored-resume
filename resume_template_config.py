import yaml
import os

def load_resume_templates(cfg_path="resume_templates.yaml"):
    if not os.path.exists(cfg_path):
        return []
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    templates = data.get("resume_templates", [])
    return templates

def get_default_template_id(templates):
    for t in templates:
        if t.get("default"):
            return t["id"]
    return templates[0]["id"] if templates else None

def get_template_path_by_id(templates, template_id):
    for t in templates:
        if t["id"] == template_id:
            return t["path"]
    return None

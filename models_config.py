# models_config.py
import os, yaml

def load_models_cfg(path: str = "models.yaml") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # Build reverse index: id -> (endpoint, display, pricing)
        index = {}
        for endpoint, arr in (cfg.get("models") or {}).items():
            for m in (arr or []):
                index[m["id"]] = {
                    "endpoint": endpoint,
                    "display": m.get("display", m["id"]),
                    "pricing": (m.get("pricing") or {}),
                    "default": bool(m.get("default")),
                }
        cfg["_index"] = index
        return cfg
    except Exception as e:
        print(f"Error loading models config: {e}")
        return {}

def ui_choices(cfg: dict, group: str) -> list[tuple[str, str]]:
    g = (cfg.get("ui", {}).get("groups", {}) or {}).get(group) or {}
    ids = g.get("allow") or []
    idx = cfg["_index"]
    return [(idx[i]["display"], i) for i in ids if i in idx]

def ui_default(cfg: dict, group: str) -> str | None:
    g = (cfg.get("ui", {}).get("groups", {}) or {}).get(group) or {}
    default_id = g.get("default")
    if default_id:
        return default_id
    # fallback to first with default:true in that endpoint
    endpoint = g.get("endpoint")
    for m in (cfg.get("models", {}).get(endpoint, []) or []):
        if m.get("default"):
            return m["id"]
    return None

def model_display(cfg: dict, model_id: str) -> str:
    return cfg["_index"].get(model_id, {}).get("display", model_id)

def model_pricing(cfg: dict, model_id: str) -> dict:
    return cfg["_index"].get(model_id, {}).get("pricing", {}) or {}

def model_endpoint(cfg: dict, model_id: str) -> str | None:
    return cfg["_index"].get(model_id, {}).get("endpoint")

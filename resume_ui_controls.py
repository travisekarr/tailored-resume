"""
Shared UI controls for storage configuration and resume tailoring.
"""

from __future__ import annotations
import os
import re
from io import BytesIO
import base64
import streamlit as st

def _ensure_writable_dir(path: str) -> tuple[bool, str, str]:
    """
    Ensure path exists and is writable.
    Returns: (ok, resolved_dir, error_msg)
    """
    resolved = os.path.abspath(path or "")
    try:
        os.makedirs(resolved, exist_ok=True)
        test_path = os.path.join(resolved, ".write_test.tmp")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True, resolved, ""
    except Exception as e:
        return False, resolved, str(e)

def render_storage_controls(*, defaults: dict, key_prefix: str = "gen") -> dict:
    """
    Render DB path + resume save folder controls and validate/save.
    Returns:
      {
        "db_path": str,
        "resume_save_dir": str,   # resolved absolute dir
        "dir_ok": bool,
        "dir_error": str,
      }
    """
    db_default = (defaults or {}).get("db", "jobs.db")
    save_default = (defaults or {}).get("resume_save_dir", "kept_resumes")

    db_path = st.text_input("Database file", value=db_default, key=f"{key_prefix}_db_path")
    resume_save_dir_in = st.text_input("Resume save folder", value=save_default, key=f"{key_prefix}_save_dir")

    ok, resolved_dir, err = _ensure_writable_dir(resume_save_dir_in)
    if ok:
        st.caption(f"Save directory: {resolved_dir}")
    else:
        st.error(f"Cannot create or write to folder: {resolved_dir}\n\n{err}")

    return {
        "db_path": db_path,
        "resume_save_dir": resolved_dir,
        "dir_ok": ok,
        "dir_error": err,
    }

# Model config helpers for shared selectors
try:
	# Lazy import; keep this optional so other parts of the app can still run if models_config changes
	from models_config import load_models_cfg, ui_choices, ui_default  # type: ignore
	_MODEL_CFG = load_models_cfg()
except Exception:
	_MODEL_CFG = None
	def ui_choices(cfg, group): return [("gpt-3.5-turbo", "gpt-3.5-turbo")]
	def ui_default(cfg, group): return "gpt-3.5-turbo"

def _model_selectbox(label: str, group: str, *, key: str, disabled: bool = False):
	group_map = {"chat": "rephrasing", "embeddings": "embeddings", "summary": "summary", "rephrasing": "rephrasing"}
	actual_group = group_map.get(group, group)
	choices = ui_choices(_MODEL_CFG, actual_group)
	default_id = ui_default(_MODEL_CFG, actual_group)
	ids = [id for _, id in choices]
	labels = {id: display for display, id in choices}
	def _fmt(x): return labels.get(x, x)
	try:
		default_idx = ids.index(default_id) if default_id in ids else 0
	except Exception:
		default_idx = 0
	return st.selectbox(label, ids, index=default_idx, key=key, disabled=disabled, format_func=_fmt)

def render_resume_sidebar_controls(*, key_prefix: str = "app") -> dict:
	"""
	Render shared sidebar controls for resume generation.
	Returns:
	  {
	    "summary_mode": str,
	    "strict_mode": bool,
	    "selected_model": str,
	    "add_impact": bool,
	    "bullets_per_role": int,
	    "show_generated": bool,
	    "use_embeddings": bool,
	    "embedding_model": str,
	  }
	"""
	st.markdown("### Usage & Models")
	summary_mode = st.radio(
		"Summary Mode",
		("Offline (free)", "GPT-powered (API cost)"),
		index=0,
		key=f"{key_prefix}_rad_summary_mode",
	)
	strict_mode = (summary_mode == "GPT-powered (API cost)") and st.checkbox(
		"Strict factual mode (no new claims)",
		value=True,
		help="Model may only rephrase supplied facts. No new achievements or metrics.",
		key=f"{key_prefix}_chk_strict_mode",
	)
	selected_model = _model_selectbox(
		"GPT Model (pricing per 1K tokens)",
		group="rephrasing",
		key=f"{key_prefix}_sel_gpt_model",
		disabled=False,
	)

	st.markdown("---")
	st.markdown("### Experience Enhancements")
	add_impact = st.checkbox("Add tailored impact statements (per role)", value=False, key=f"{key_prefix}_chk_add_impact")
	bullets_per_role = 1
	if add_impact:
		bullets_per_role = st.slider("Impact bullets per role", 1, 3, 1, key=f"{key_prefix}_sld_bullets_per_role")
	show_generated = st.checkbox("Show generated impact statements", value=True, key=f"{key_prefix}_chk_show_generated")

	st.markdown("---")
	st.markdown("### Matching Options")
	use_embeddings = st.checkbox("Use semantic matching (OpenAI embeddings)", value=False, key=f"{key_prefix}_chk_use_embeddings")
	embedding_model = _model_selectbox(
		"Embeddings model (pricing per 1M tokens)",
		group="embeddings",
		key=f"{key_prefix}_sel_embed_model",
		disabled=not use_embeddings,
	)

	return {
		"summary_mode": summary_mode,
		"strict_mode": bool(strict_mode),
		"selected_model": selected_model,
		"add_impact": bool(add_impact),
		"bullets_per_role": int(bullets_per_role),
		"show_generated": bool(show_generated),
		"use_embeddings": bool(use_embeddings),
		"embedding_model": embedding_model,
	}

def _strip_html(html: str) -> str:
	"""Plain-text from HTML."""
	return re.sub(r"<[^>]+>", "", html or "")

def select_preview_mode(*, key_prefix: str = "prev", default: str = "Formatted (HTML)") -> str:
	"""
	Shared preview mode selector.
	Returns one of: "Formatted (HTML)", "Plain Text", "PDF Preview"
	"""
	modes = ("Formatted (HTML)", "Plain Text", "PDF Preview")
	try:
		default_idx = modes.index(default)
	except Exception:
		default_idx = 0
	return st.radio("Preview Mode", modes, index=default_idx, key=f"{key_prefix}_preview_mode")

def show_preview_and_download(
	*, html: str, base_name: str, mode: str, key_prefix: str = "prev", display: bool = True,
	db_path: str | None = None, job_id: str | None = None
) -> dict:
	"""
	Render the preview (optional) and a matching download button for the selected mode.
	Returns: { "download_label", "data", "mime", "name", "pdf_bytes", "clicked" }
	"""
	out = {"download_label": None, "data": None, "mime": None, "name": None, "pdf_bytes": None, "clicked": False}

	if not html:
		st.warning("No content to preview.")
		return out

	if mode == "Formatted (HTML)":
		if display:
			st.components.v1.html(html, height=800, scrolling=True)
		out.update({
			"download_label": "📥 Download Resume as HTML",
			"data": html,
			"mime": "text/html",
			"name": f"{base_name}.html",
		})
	elif mode == "Plain Text":
		plain = _strip_html(html)
		if display:
			st.text_area("Plain Text Resume", plain, height=800, key=f"{key_prefix}_plain_text")
		out.update({
			"download_label": "📥 Download Resume as Plain Text",
			"data": plain,
			"mime": "text/plain",
			"name": f"{base_name}.txt",
		})
	elif mode == "PDF Preview":
		pdf_bytes = None
		try:
			import pdfkit
			pdf_bytes = pdfkit.from_string(html, False)
			if display:
				b64 = base64.b64encode(pdf_bytes).decode("utf-8")
				st.markdown(
					f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800" type="application/pdf"></iframe>',
					unsafe_allow_html=True
				)
		except Exception as e:
			if display:
				st.info("pdfkit is not available. Install with `pip install pdfkit` (and wkhtmltopdf).")
		out.update({
			"download_label": "📥 Download Resume as PDF",
			"data": pdf_bytes,
			"mime": "application/pdf",
			"name": f"{base_name}.pdf",
			"pdf_bytes": pdf_bytes,
		})

	if out["data"] is not None:
		clicked = st.download_button(
			out["download_label"],
			data=out["data"],
			file_name=out["name"],
			mime=out["mime"],
			key=f"{key_prefix}_download",
		)
		out["clicked"] = bool(clicked)
		# If downloaded and we have job context, mark as saved here
		if out["clicked"] and db_path and job_id:
			try:
				# Prefer set_job_resume if available
				from job_store import set_job_resume as _set_job_resume
				from datetime import datetime, timezone
				now_iso = datetime.now(timezone.utc).isoformat()
				try:
					_set_job_resume(db_path=db_path, job_id=job_id, saved=True, saved_at=now_iso, resume_path="")
				except TypeError:
					try:
						_set_job_resume(db_path=db_path, job_key=job_id, saved=True, saved_at=now_iso, resume_path="")
					except TypeError:
						# Legacy minimal signature
						_set_job_resume(db_path, job_id, "")
			except Exception:
				# Fallback: set a status so the row reflects being saved
				try:
					from job_store import set_job_status as _set_job_status
					_set_job_status(db_path, job_id, status="resume_saved")
				except Exception:
					# Best-effort; ignore if job_store doesn't support these fields
					pass
	return out

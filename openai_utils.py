# openai_utils.py
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from job_store import log_api_usage
import traceback

_client = OpenAI()

# Embedding API

def get_embedding(text, model="text-embedding-3-small"):
    try:
        resp = _client.embeddings.create(input=[text], model=model)
        emb = resp.data[0].embedding
        # Log usage
        usage = getattr(resp, "usage", None)
        if usage:
            it = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
            ot = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
            tt = getattr(usage, "total_tokens", None) or (it + ot)
            rid = getattr(resp, "id", "") or getattr(resp, "request_id", "")
            log_api_usage(
                db_path=None,
                endpoint="embeddings",
                model=model,
                input_tokens=int(it or 0),
                output_tokens=int(ot or 0),
                total_tokens=int(tt or 0),
                request_id=rid,
                context="openai_utils:get_embedding",
                meta={"text": text}
            )
        return emb
    except Exception as e:
        print(f"[openai_utils] Error in get_embedding: {e}\n{traceback.format_exc()}")
        return []

# Chat completion API

def chat_completion(messages, model="gpt-3.5-turbo", max_tokens=256, temperature=0.2, context="openai_utils:chat_completion"):
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            it = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
            ot = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
            tt = getattr(usage, "total_tokens", None) or (it + ot)
            rid = getattr(resp, "id", "") or getattr(resp, "request_id", "")
            log_api_usage(
                db_path=None,
                endpoint="chat.completions",
                model=model,
                input_tokens=int(it or 0),
                output_tokens=int(ot or 0),
                total_tokens=int(tt or 0),
                request_id=rid,
                context=context,
                meta={"messages": messages}
            )
        return resp.choices[0].message.content.strip(), resp
    except Exception as e:
        err_msg = str(e)
        if ("model_not_found" in err_msg or "does not exist" in err_msg or "invalid_request_error" in err_msg) and model != "gpt-4o":
            print(f"[openai_utils] Model '{model}' not found, retrying with fallback 'gpt-4o'.")
            try:
                resp = _client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                usage = getattr(resp, "usage", None)
                if usage:
                    it = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
                    ot = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
                    tt = getattr(usage, "total_tokens", None) or (it + ot)
                    rid = getattr(resp, "id", "") or getattr(resp, "request_id", "")
                    log_api_usage(
                        db_path=None,
                        endpoint="chat.completions",
                        model="gpt-4o",
                        input_tokens=int(it or 0),
                        output_tokens=int(ot or 0),
                        total_tokens=int(tt or 0),
                        request_id=rid,
                        context=context,
                        meta={"messages": messages}
                    )
                return resp.choices[0].message.content.strip(), resp
            except Exception as e2:
                print(f"[openai_utils] Error in chat_completion fallback: {e2}\n{traceback.format_exc()}")
                return "", None
        print(f"[openai_utils] Error in chat_completion: {e}\n{traceback.format_exc()}")
        return "", None

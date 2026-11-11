"""extract_json and system prompt shared by the COGNITION direct baselines (from cog_direct.py)."""
import re, json


def extract_json(text):
    """verbatim from the artifact's cognition/run_eval.py"""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I|re.M)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, (dict, list)) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None
    return None


SYSTEM = ("You are a vision agent. Output ONLY a JSON object that strictly matches the given schema. "
          "If the schema contains optional field 'reasoning', you MUST fill it following the detailed cognitive tracing format provided.")

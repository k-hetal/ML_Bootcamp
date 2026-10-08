"""Pipeline: Whisper STT -> LLM #1 refinement -> LLM #2 record extraction (+ grounding checks); LLM #3 answers questions."""
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import groq
from dotenv import load_dotenv
from groq import Groq

from prompts import ANALYSIS_SYSTEM, ASK_SYSTEM, REFINE_SYSTEM, TRANSLATE_SYSTEM
from schemas import (STATEMENT_TYPES, UNSPECIFIED, Answer, Evidence, MeetingRecord, Statement, Topic, fmt_time)

load_dotenv()

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # Groq free-tier file limit
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
# Preference lists: the first model your key can actually access is used (override via .env).
REFINE_PREFS = ["llama-3.1-8b-instant", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "qwen/qwen3-32b"]
ANALYSIS_PREFS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.1-8b-instant"]
ASK_PREFS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.1-8b-instant"]

Status = Optional[Callable[[int, str], None]]


class PipelineError(Exception):
    """Error whose message is safe to show to the user."""


# ----------------------------------------------------------------- client / models
_client: Optional[Groq] = None


def get_client() -> Groq:
    global _client
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise PipelineError("GROQ_API_KEY is not set. Put it in a .env file next to app.py (see .env.example).")
    if _client is None:
        _client = Groq(api_key=key, max_retries=2, timeout=180)
    return _client


def list_models() -> List[str]:
    try:
        return sorted(m.id for m in get_client().models.list().data)
    except Exception as e:
        raise _explain(e, "Listing models")


def resolve_model(env_name: str, prefs: List[str], avoid: Optional[str] = None) -> str:
    forced = os.getenv(env_name)
    if forced:
        return forced
    try:
        avail = set(list_models())
    except PipelineError:
        raise
    cands = [p for p in prefs if p in avail]
    if not cands:
        raise PipelineError(
            f"None of the preferred models {prefs} are available to your key. "
            f"Available: {sorted(avail)}. Set {env_name} in .env to one of them."
        )
    return next((p for p in cands if p != avoid), cands[0])


def _explain(e: Exception, what: str) -> PipelineError:
    if isinstance(e, PipelineError):
        return e
    if isinstance(e, groq.AuthenticationError):
        msg = "Groq rejected the API key (401). Check GROQ_API_KEY."
    elif isinstance(e, groq.NotFoundError):
        msg = ("Groq says the model does not exist or your key has no access (404). Check the model-access/limits "
               "settings of your Groq organisation, or set REFINE_MODEL / ANALYSIS_MODEL in .env. "
               f"Details: {e}")
    elif isinstance(e, groq.RateLimitError):
        msg = "Groq rate limit reached (429). Wait a minute and try again."
    elif isinstance(e, groq.APIConnectionError):
        msg = "Could not reach Groq. Check your internet connection."
    elif isinstance(e, groq.BadRequestError):
        msg = f"Groq rejected the request (400): {e}"
    else:
        msg = str(e)
    return PipelineError(f"{what} failed: {msg}")


def _chat(model: str, system: str, user: str, *, json_mode=False, max_tokens=4096) -> str:
    kw = dict(model=model, temperature=0.1, max_tokens=max_tokens,
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    try:
        r = get_client().chat.completions.create(**kw)
    except groq.BadRequestError:
        if not json_mode:
            raise
        kw.pop("response_format")  # model without JSON mode: rely on prompt + validation retries
        r = get_client().chat.completions.create(**kw)
    text = r.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ----------------------------------------------------------------- stage 1
CHUNK_SEC = 300  # long recordings are transcribed in 5-minute pieces so Whisper cannot silently stop early


def _ffmpeg() -> Optional[str]:
    """System ffmpeg if present, otherwise the binary bundled with the imageio-ffmpeg pip package."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _duration(path: str) -> Optional[float]:
    ff = _ffmpeg()
    if ff:
        p = subprocess.run([ff, "-i", path], capture_output=True, text=True, errors="ignore")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if path.lower().endswith(".wav"):
        try:
            import wave
            with wave.open(path) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            pass
    return None


@dataclass
class Prepared:
    chunks: list              # [(path, offset_sec, duration_sec|None)]
    duration: Optional[float]
    work: str = ""            # temp folder


def _prepare(path: str, work: str, warnings: List[str]) -> Prepared:
    size = os.path.getsize(path)
    if size == 0:
        raise PipelineError("The audio file is empty. Upload a valid recording.")
    dur = _duration(path)
    ff = _ffmpeg()
    if not ff:
        if size > MAX_UPLOAD_BYTES:
            raise PipelineError(f"File is {size/1e6:.1f} MB; the limit is 25 MB. Install ffmpeg (auto-convert) or upload a smaller file.")
        warnings.append("ffmpeg was not found, so the recording is sent in one piece and its length cannot be verified. "
                        "Run `pip install imageio-ffmpeg` to avoid truncated transcripts.")
        return Prepared([(path, 0.0, dur)], dur, work)
    full = os.path.join(work, "full.wav")
    p = subprocess.run([ff, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", full], capture_output=True)
    if p.returncode != 0 or not os.path.exists(full) or os.path.getsize(full) < 100:
        raise PipelineError("The audio file could not be read (it may be corrupt or not real audio).")
    dur = _duration(full) or dur
    if dur and dur > CHUNK_SEC + 30:
        subprocess.run([ff, "-y", "-i", full, "-f", "segment", "-segment_time", str(CHUNK_SEC), "-c", "copy",
                        os.path.join(work, "chunk_%03d.wav")], capture_output=True)
        files = sorted(f for f in os.listdir(work) if f.startswith("chunk_"))
        if files:
            return Prepared([(os.path.join(work, f), i * float(CHUNK_SEC), _duration(os.path.join(work, f)))
                             for i, f in enumerate(files)], dur, work)
    return Prepared([(full, 0.0, dur)], dur, work)


def _g(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


LANG_CODES = {"russian": "ru", "hindi": "hi", "spanish": "es", "french": "fr", "german": "de", "portuguese": "pt",
              "italian": "it", "arabic": "ar", "chinese": "zh", "japanese": "ja", "korean": "ko", "turkish": "tr",
              "ukrainian": "uk", "polish": "pl", "dutch": "nl", "bengali": "bn", "indonesian": "id"}
# Typical Whisper hallucinations on silence, hold music or noise.
_HALLU = re.compile(r"thank(?:s| you)(?: so much)? for (?:watching|listening)|subscribe|like and share|"
                    r"follow me on|find me on (?:twitter|instagram)|link in the description|amara\.org|"
                    r"subtitles? by|translated by|продолжение следует|субтитры", re.I)


def is_english(lang: Optional[str]) -> bool:
    return (lang or "en").lower() in ("en", "english")


def _clean_segments(segs: List[dict]) -> List[dict]:
    """Drop hallucinated, silent and looping segments using Whisper's own confidence signals."""
    out, last = [], ""
    for s in segs:
        t = s["raw"]
        key = re.sub(r"[^\w]+", " ", t.lower()).strip()
        if _HALLU.search(t) and len(t.split()) <= 25:
            continue
        if s.get("no_speech", 0) > 0.6 and s.get("logprob", 0) < -0.8:
            continue
        if s.get("compression", 0) > 2.4 or s.get("logprob", 0) < -1.6:
            continue
        if key and key == last:          # exact repeat of the previous line = decoding loop
            continue
        last = key
        out.append(s)
    return out


def _stt_once(path: str, glossary: str, temperature: float, use_prompt: bool, language: Optional[str] = "en"):
    """Returns (segments, detected_language). language=None lets Whisper auto-detect."""
    with open(path, "rb") as f:
        data = f.read()
    kw = dict(file=(os.path.basename(path), data), model=STT_MODEL, temperature=temperature,
              response_format="verbose_json", timestamp_granularities=["segment"])
    if language:
        kw["language"] = language
    if glossary.strip() and use_prompt and (language in (None, "en")):
        kw["prompt"] = "Terms: " + glossary.strip()[:600]  # biases Whisper toward domain vocabulary
    res = get_client().audio.transcriptions.create(**kw)
    out = []
    for s in (_g(res, "segments") or []):
        t = (_g(s, "text", "") or "").strip()
        if t:
            out.append({"start": float(_g(s, "start", 0) or 0), "end": float(_g(s, "end", 0) or 0), "raw": t,
                        "no_speech": float(_g(s, "no_speech_prob", 0) or 0),
                        "logprob": float(_g(s, "avg_logprob", 0) or 0),
                        "compression": float(_g(s, "compression_ratio", 0) or 0)})
    if not out:
        t = (res if isinstance(res, str) else _g(res, "text", "") or "").strip()
        if t:
            out = [{"start": 0.0, "end": 0.0, "raw": t}]
    return out, (str(_g(res, "language", "") or "").lower() or None)


def stage_1_transcribe(prep: Prepared, glossary: str, warnings: List[str], info: Optional[dict] = None) -> List[dict]:
    """Timestamped segments [{"id","start","end","raw"}]. Language is auto-detected on the first piece (never forced to
    English), hallucinations are filtered, and a piece is retried if Whisper stopped before its end."""
    info = info if info is not None else {}
    segs, lang_arg = [], None
    for path, off, cdur in prep.chunks:
        try:
            part, det = _stt_once(path, glossary, 0.0, True, lang_arg)
            if "lang" not in info and det:
                info["lang"] = det
                lang_arg = "en" if is_english(det) else LANG_CODES.get(det)
            last = max((s["end"] for s in part), default=0.0)
            if cdur and cdur > 45 and cdur - last > 20 and last < cdur * 0.9:  # transcript ends well before the audio does
                retry, _ = _stt_once(path, glossary, 0.2, False, lang_arg)
                if max((s["end"] for s in retry), default=0.0) > last:
                    part = retry
            ff = _ffmpeg()
            for n in range(3):  # tail recovery: re-transcribe only the missing end of the audio
                last = max((s["end"] for s in part), default=0.0)
                if not (ff and cdur and cdur > 45 and cdur - last > 20):
                    break
                cut = max(last - 1.0, 0.0)
                tail = os.path.join(prep.work, f"tail_{os.path.basename(path)}_{n}.wav")
                subprocess.run([ff, "-y", "-ss", f"{cut:.2f}", "-i", path, "-ac", "1", "-ar", "16000", tail], capture_output=True)
                if not os.path.exists(tail):
                    break
                tp, _ = _stt_once(tail, glossary, 0.0, True, lang_arg)
                more = [dict(s, start=s["start"] + cut, end=s["end"] + cut) for s in tp]
                more = [dict(s, start=max(s["start"], last)) for s in more if s["end"] > last + 0.5]
                if not more:
                    break
                part += more
        except groq.BadRequestError as e:
            raise PipelineError(f"Groq could not decode this audio (unsupported or unreadable file). Details: {e}")
        except Exception as e:
            raise _explain(e, "Speech-to-text")
        before = len(part)
        part = _clean_segments(part)
        if len(part) < before:
            warnings.append(f"Removed {before - len(part)} low-confidence or hallucinated line(s) from the speech-to-text output.")
        for s in part:
            s["start"] += off
            s["end"] += off
        segs += part
    if not segs:
        raise PipelineError("No speech was detected in the recording.")
    for i, s in enumerate(segs):
        s["id"] = i
    end = max(s["end"] for s in segs)
    if prep.duration and end > 0 and prep.duration - end > 15:
        warnings.append(f"Transcript covers {fmt_time(end)} of a {fmt_time(prep.duration)} recording. The last "
                        f"{fmt_time(prep.duration - end)} produced no recognisable speech (silence, noise or very quiet audio). "
                        "Check the end of the recording.")
    return segs


# ----------------------------------------------------------------- stage 2
_NEG = re.compile(r"\b(?:not|no|never|cannot|can't|won't|don't|didn't|isn't|aren't|wasn't|shouldn't|couldn't|without)\b", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.*\S)\s*$")


def _key(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def _safe_refinement(raw: str, new: str) -> Optional[str]:
    """Return reason string if the refinement looks unsafe, else None."""
    if not new:
        return "empty output"
    ratio = len(new) / max(len(raw), 1)
    if not 0.75 <= ratio <= 1.45:
        return f"length changed too much ({ratio:.0%})"
    if Counter(_NUM.findall(raw)) - Counter(_NUM.findall(new)):
        return "numbers were altered"
    if len(_NEG.findall(new)) < len(_NEG.findall(raw)):
        return "negations were removed"
    a, b = _key(raw), _key(new)  # letters/digits only: ignores punctuation, casing and spacing fixes
    if a and difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() < 0.8:
        return "wording was rewritten"
    return None


def stage_2_refine(segs: List[dict], model: str, glossary: str = "", warnings: Optional[List[str]] = None,
                   batch: int = 14) -> List[dict]:
    """Adds a 'refined' field to every segment. Unsafe lines fall back to raw text."""
    warnings = warnings if warnings is not None else []
    system = REFINE_SYSTEM + (f"\n\nDomain glossary (preferred spellings): {glossary.strip()}" if glossary.strip() else "")
    rejected = 0
    for i in range(0, len(segs), batch):
        part = segs[i:i + batch]
        body = "\n".join(f"[{k}] {s['raw']}" for k, s in enumerate(part))
        try:
            out = _chat(model, system, f"<transcript>\n{body}\n</transcript>", max_tokens=4096)
        except Exception as e:
            raise _explain(e, "Transcript refinement")
        got = {int(m.group(1)): m.group(2) for ln in out.splitlines() if (m := _LINE.match(ln))}
        for k, s in enumerate(part):
            new = re.sub(r"</?transcript>", "", got.get(k, "")).strip()
            bad = _safe_refinement(s["raw"], new)
            if bad:
                rejected += 1
                new = s["raw"]
            s["refined"] = new
    if rejected:
        warnings.append(f"{rejected} of {len(segs)} transcript lines failed the meaning-safety check; raw wording kept for those lines.")
    return segs


def stage_2_translate(segs: List[dict], model: str, lang: str, glossary: str = "", warnings: Optional[List[str]] = None,
                      batch: int = 12) -> List[dict]:
    """LLM #1 for non-English audio: faithful translation + clean-up into English, stored in 'refined'."""
    warnings = warnings if warnings is not None else []
    system = TRANSLATE_SYSTEM.format(lang=lang) + (f"\n\nGlossary (preferred English terms): {glossary.strip()}" if glossary.strip() else "")
    bad_n = 0
    for i in range(0, len(segs), batch):
        part = segs[i:i + batch]
        ctx = segs[max(0, i - 2):i]
        pre = "".join(f"(context, do not output) {c['raw']}\n" for c in ctx)
        body = "\n".join(f"[{k}] {s['raw']}" for k, s in enumerate(part))
        try:
            out = _chat(model, system, f"{pre}<transcript>\n{body}\n</transcript>", max_tokens=4096)
        except Exception as e:
            raise _explain(e, "Transcript translation")
        got = {int(m.group(1)): m.group(2) for ln in out.splitlines() if (m := _LINE.match(ln))}
        for k, s in enumerate(part):
            new = re.sub(r"</?transcript>", "", got.get(k, "")).strip()
            if not new or Counter(_NUM.findall(s["raw"])) - Counter(_NUM.findall(new)):
                bad_n += 1
                new = new or s["raw"]
            s["refined"] = new
    if bad_n:
        warnings.append(f"{bad_n} translated line(s) were missing or changed a number; please check them.")
    return segs



# ----------------------------------------------------------------- stage 3
def _extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.I)
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if a != -1 and b > a else text


_HEDGE = re.compile(r"\b(could|might|maybe|perhaps|should we|what if|possibly|may|i think)\b", re.I)
_CONFIRM = re.compile(r"\b(agree[ds]?|decid(?:e|ed)|let'?s|we will|we'll|approved?|go(?:ing)? with|confirmed?|settled|yes|sounds good)\b", re.I)
_COMMIT = re.compile(r"\b(i'?ll|i will|we'?ll|we will|will|going to|let me|i can take|i'?m going to)\b", re.I)


def _words(s: str) -> List[str]:
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if len(w) > 2 or w.isdigit()]


def _tx(segs: List[dict]) -> str:
    return "\n".join(f"[S{s['id']} | {fmt_time(s['start'])}] {s.get('refined') or s['raw']}" for s in segs)


def locate(quote: str, segs: List[dict], thr: float = 0.6):
    """Find the 1-3 consecutive lines that best contain a quote -> (first_idx, last_idx) or None."""
    qw = _words(quote)
    if not qw:
        return None
    best, best_score = None, 0.0
    for i in range(len(segs)):
        for w in (1, 2, 3):
            win = segs[i:i + w]
            vocab = set(_words(" ".join(f"{s['raw']} {s.get('refined', '')}" for s in win)))
            score = sum(x in vocab for x in qw) / len(qw) - 0.01 * (w - 1)
            if score > best_score:
                best, best_score = (i, i + len(win) - 1), score
    return best if best and best_score >= thr else None


def _span(segs, loc):
    return segs[loc[0]]["start"], segs[loc[1]]["end"]


def ground_record(rec: MeetingRecord, segs: List[dict]) -> List[str]:
    """Drop/blank anything the transcript does not support and attach timestamps. Mutates rec."""
    vocab = set(w for s in segs for w in _words(f"{s['raw']} {s.get('refined', '')}"))
    warns: List[str] = []

    decisions, seen = [], set()
    for d in rec.decisions:
        if d.decision.lower() in seen:
            continue
        seen.add(d.decision.lower())
        loc = locate(d.evidence, segs)
        if not loc:
            warns.append(f"Dropped decision without transcript evidence: {d.decision!r}")
        elif _HEDGE.search(d.evidence) and not _CONFIRM.search(d.evidence):
            warns.append(f"Dropped decision that looks like an unconfirmed proposal: {d.decision!r}")
        else:
            d.start, d.end = _span(segs, loc)
            decisions.append(d)
    rec.decisions = decisions

    items, seen = [], set()
    for t in rec.action_items:
        if t.task.lower() in seen:
            continue
        seen.add(t.task.lower())
        loc = locate(t.evidence, segs)
        if not loc:
            warns.append(f"Dropped task without transcript evidence: {t.task!r}")
            continue
        t.start, t.end = _span(segs, loc)
        if t.owner != UNSPECIFIED and not any(w in vocab for w in _words(t.owner)):
            warns.append(f"Owner {t.owner!r} not found in transcript; set to UNSPECIFIED.")
            t.owner = UNSPECIFIED
        if t.deadline != UNSPECIFIED and not any(w in vocab for w in _words(t.deadline)):
            warns.append(f"Deadline {t.deadline!r} not found in transcript; set to UNSPECIFIED.")
            t.deadline = UNSPECIFIED
        if t.status == "confirmed" and _HEDGE.search(t.evidence) and not (_COMMIT.search(t.evidence) or _CONFIRM.search(t.evidence)):
            t.status, t.review_note = "proposed", t.review_note or "Evidence is hedged, so it was downgraded to a proposal."
        if t.status != "confirmed" and not t.review_note:
            t.review_note = "Not clearly agreed in the recording; please verify."
        items.append(t)
    rec.action_items = items

    n, stm = len(segs), []
    for s in rec.statements:
        if s.segment is None or not 0 <= s.segment < n:
            loc = locate(s.text, segs, 0.7)
            s.segment = loc[0] if loc else None
        if s.segment is None:
            continue
        s.start = segs[s.segment]["start"]
        stm.append(s)
    rec.statements = stm

    topics = []
    for tp in sorted(rec.topics, key=lambda x: x.start_segment):
        a, b = max(0, min(tp.start_segment, n - 1)), max(0, min(tp.end_segment, n - 1))
        if b < a:
            a, b = b, a
        tp.start_segment, tp.end_segment, tp.start, tp.end = a, b, segs[a]["start"], segs[b]["end"]
        topics.append(tp)
    rec.topics = topics
    return warns


def stage_3_generate_record(segs: List[dict], model: str) -> MeetingRecord:
    base = f"<transcript>\n{_tx(segs)}\n</transcript>\n\nReturn the JSON object now."
    user, last = base, None
    for _ in range(3):
        try:
            content = _chat(model, ANALYSIS_SYSTEM, user, json_mode=True, max_tokens=6000)
        except Exception as e:
            raise _explain(e, "Meeting analysis")
        try:
            return MeetingRecord.model_validate(json.loads(_extract_json(content)))
        except ValueError as e:  # JSONDecodeError and pydantic ValidationError
            last = e
            user = base + f"\n\nYour previous output was invalid ({str(e)[:300]}). Return ONLY the corrected JSON object."
    raise PipelineError(f"The analysis model did not return a valid record after 3 attempts: {last}")


# ----------------------------------------------------------------- LLM #3: ask your meeting
def _select_context(question: str, segs: List[dict], budget: int = 24000) -> List[dict]:
    if len(_tx(segs)) <= budget:
        return segs
    qw = set(_words(question))
    scored = sorted(range(len(segs)), key=lambda i: -len(qw & set(_words(segs[i].get("refined") or segs[i]["raw"]))))
    keep, size = set(), 0
    for i in scored:
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(segs) and j not in keep:
                keep.add(j)
                size += len(segs[j].get("refined") or segs[j]["raw"]) + 20
        if size > budget:
            break
    return [segs[i] for i in sorted(keep)]


def ask_meeting(question: str, segs: List[dict], model: Optional[str] = None) -> Answer:
    question = question.strip()
    if not question:
        raise PipelineError("Type a question first.")
    model = model or resolve_model("ASK_MODEL", ASK_PREFS)
    ctx = _select_context(question, segs)
    by_id = {s["id"]: s for s in segs}
    user = f"<transcript>\n{_tx(ctx)}\n</transcript>\n\nQuestion: {question}\n\nReturn the JSON object now."
    try:
        out = _chat(model, ASK_SYSTEM, user, json_mode=True, max_tokens=1200)
        data = json.loads(_extract_json(out))
    except ValueError:
        raise PipelineError("The Q&A model returned an unreadable answer. Please try again.")
    except Exception as e:
        raise _explain(e, "Answering the question")
    ev = []
    for e in data.get("evidence") or []:
        try:
            s = by_id.get(int(e.get("segment")))
        except (TypeError, ValueError, AttributeError):
            continue
        if s and _words(str(e.get("quote", ""))) and locate(str(e["quote"]), [s], 0.6):
            ev.append(Evidence(quote=str(e["quote"]).strip(), start=s["start"], end=s["end"]))
    answer = str(data.get("answer", "")).strip() or "No answer was produced."
    return Answer(question=question, answer=answer, evidence=ev, grounded=bool(ev))


# ----------------------------------------------------------------- orchestration
def full_text(segs: List[dict], field_: str, stamps: bool = False) -> str:
    return "\n".join((f"[{fmt_time(s['start'])}] " if stamps else "") + (s.get(field_) or s["raw"]) for s in segs)


@dataclass
class PipelineResult:
    segments: List[dict]
    record: MeetingRecord
    warnings: List[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    audio_duration: Optional[float] = None

    @property
    def raw_transcript(self) -> str:
        return " ".join(s["raw"] for s in self.segments)

    @property
    def refined_transcript(self) -> str:
        return " ".join(s.get("refined") or s["raw"] for s in self.segments)

    @property
    def transcript_end(self) -> float:
        return max((s["end"] for s in self.segments), default=0.0)

    @property
    def duration(self) -> float:
        return self.audio_duration or self.transcript_end

    def to_dict(self) -> dict:
        return {"models": self.models, "raw_transcript": self.raw_transcript,
                "refined_transcript": self.refined_transcript, "segments": self.segments,
                "audio_duration": self.audio_duration,
                "meeting_record": self.record.model_dump(), "warnings": self.warnings, "timings_sec": self.timings}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        r, T = self.record, fmt_time
        md = ["# Meeting Record — VerbaTrack", "", "## Summary", r.summary, "", "## Timeline"]
        md += [f"- {T(t.start)}–{T(t.end)}  {t.title}" for t in r.topics] or ["- None"]
        md += ["", "## Discussion Minutes"] + ([f"- {m}" for m in r.minutes] or ["- None"])
        md += ["", "## Key Decisions"]
        md += [f"- **{d.decision}**  \n  Evidence: \"{d.evidence}\"  \n  Timestamp: {T(d.start)}" for d in r.decisions] or ["- No confirmed decisions."]
        md += ["", "## Action Items"]
        for t in r.action_items:
            md += [f"- **{t.task}** — {t.label}  \n  Owner: {t.owner} | Deadline: {t.deadline}  \n  Evidence: \"{t.evidence}\" ({T(t.start)})"
                   + (f"  \n  Review note: {t.review_note}" if t.review_note else "")]
        if not r.action_items:
            md += ["- No action items."]
        md += ["", "## Statement Classification"]
        md += [f"- [{T(s.start)}] **{s.type}**: {s.text}" for s in r.statements] or ["- None"]
        md += ["", f"_Models: STT={self.models.get('stt')}, refinement={self.models.get('refinement')}, "
                   f"analysis={self.models.get('analysis')}, Q&A={self.models.get('ask')}_"]
        return "\n".join(md)


def run_full_pipeline(audio_path: str, glossary: str = "", on_status: Status = None) -> PipelineResult:
    say = on_status or (lambda s, m: None)
    t, timings, warnings = time.time(), {}, []

    analysis_model = resolve_model("ANALYSIS_MODEL", ANALYSIS_PREFS)
    refine_model = resolve_model("REFINE_MODEL", REFINE_PREFS, avoid=analysis_model)
    ask_model = resolve_model("ASK_MODEL", ASK_PREFS, avoid=analysis_model)
    if refine_model == analysis_model:
        warnings.append("Only one chat model is available, so the LLM stages share the same model.")

    with tempfile.TemporaryDirectory() as work:
        say(1, f"Transcribing with {STT_MODEL}…")
        prep = _prepare(audio_path, work, warnings)
        info: dict = {}
        segs = stage_1_transcribe(prep, glossary, warnings, info)
        timings["stt"], t = round(time.time() - t, 1), time.time()

    say(2, f"Refining transcript with {refine_model}…")
    lang = info.get("lang")
    if is_english(lang):
        stage_2_refine(segs, refine_model, glossary, warnings)
    else:
        warnings.append(f"Detected spoken language: {lang}. Stage 1 transcribed it natively; LLM #1 translated it to English.")
        stage_2_translate(segs, analysis_model, lang, glossary, warnings)
    timings["refinement"], t = round(time.time() - t, 1), time.time()

    say(3, f"Extracting minutes, decisions and tasks with {analysis_model}…")
    record = stage_3_generate_record(segs, analysis_model)
    warnings += ground_record(record, segs)
    timings["analysis"] = round(time.time() - t, 1)

    return PipelineResult(segs, record, warnings, {"stt": STT_MODEL, "refinement": refine_model, "analysis": analysis_model,
                                                   "ask": ask_model},
                          timings, prep.duration)


def process_upload(data: bytes, filename: str, glossary: str = "", on_status: Status = None) -> PipelineResult:
    """Validate an uploaded file (bytes), run the pipeline, always clean up the temp file."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise PipelineError(f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}.")
    if not data:
        raise PipelineError("The uploaded file is empty. Upload a valid recording.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return run_full_pipeline(path, glossary, on_status)
    finally:
        if os.path.exists(path):
            os.remove(path)

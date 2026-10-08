import difflib
import html
import math
import re

import pandas as pd
import streamlit as st

import pipeline as pl
from wer import wer as compute_wer
from schemas import fmt_time as T

st.set_page_config(page_title="VerbaTrack", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', system-ui, sans-serif !important; }
.stApp { background: #f6f7f9; color: #1f2933; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
h1 { font-size: 1.7rem !important; font-weight: 600 !important; letter-spacing: -0.01em; color: #17202a; padding-bottom: .2rem; }
h2, h3 { font-weight: 600 !important; color: #17202a; letter-spacing: -0.005em; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e3e7ec; }
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid #dde2e8; }
.stTabs [data-baseweb="tab"] { padding: .55rem 0; height: auto; font-weight: 500; color: #5b6770; }
.stTabs [aria-selected="true"] { color: #1d4ed8; }
.stTabs [data-baseweb="tab-highlight"] { background: #1d4ed8; }
.stButton > button, .stDownloadButton > button { border-radius: 6px; border: 1px solid #cfd6de; background: #fff; color: #1f2933; font-weight: 500; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #1d4ed8; color: #1d4ed8; }
.stButton > button[kind="primary"] { background: #1d4ed8; border-color: #1d4ed8; color: #fff; }
.stButton > button[kind="primary"]:hover { background: #1e40af; color: #fff; }
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: .4rem 0 1.4rem; }
.kpi { border: 1px solid #e3e7ec; border-radius: 8px; padding: 12px 16px; background: #fff; border-left: 4px solid var(--c); }
.kpi b { display: block; font-size: 1.55rem; font-weight: 600; line-height: 1.2; }
.kpi span { font-size: .78rem; color: #5b6770; }
.card { background: #fff; border: 1px solid #e3e7ec; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; }
.card .t { font-weight: 600; margin-bottom: 4px; }
.card .q { color: #5b6770; font-size: .88rem; font-style: italic; }
.card .ts { font-size: .78rem; color: #1d4ed8; font-weight: 500; }
.pill { display: inline-block; padding: 1px 9px; border-radius: 99px; font-size: .74rem; font-weight: 500; border: 1px solid; }
.p-ok { color: #166534; background: #ecfdf3; border-color: #bbe5cb; }
.p-prop { color: #92400e; background: #fff7e6; border-color: #f2d9a6; }
.p-rev { color: #9f1239; background: #fff1f3; border-color: #f4c2cb; }
.p-info { color: #1e40af; background: #eef3ff; border-color: #c7d6fb; }
.p-mute { color: #475569; background: #f1f5f9; border-color: #d5dde6; }
.bar { display: flex; height: 34px; border-radius: 6px; overflow: hidden; border: 1px solid #d5dde6; margin: 6px 0 14px; }
.bar div { font-size: .72rem; color: #fff; display: flex; align-items: center; padding: 0 8px; white-space: nowrap; overflow: hidden; border-right: 2px solid #fff; }
table.cmp { width: 100%; border-collapse: collapse; background: #fff; font-size: .92rem; }
table.cmp th { text-align: left; font-weight: 600; padding: 8px 10px; border-bottom: 2px solid #dde2e8; background: #fafbfc; }
table.cmp td { vertical-align: top; padding: 7px 10px; border-bottom: 1px solid #edf0f3; line-height: 1.55; }
table.cmp td.ts { width: 58px; color: #64748b; font-variant-numeric: tabular-nums; white-space: nowrap; }
del.d { background: #fde2e2; text-decoration: none; border-radius: 3px; padding: 0 2px; }
ins.i { background: #d9f3e3; text-decoration: none; border-radius: 3px; padding: 0 2px; }
u.f { text-decoration: none; border-bottom: 2px dotted #94a3b8; }
[class*="st-key-kpi_"] button { height: 4.1rem; justify-content: flex-start; padding: 0 16px; font-size: 1.02rem; font-weight: 600; background: #fff; border-left: 4px solid var(--c, #64748b); white-space: pre-line; text-align: left; }
[class*="st-key-kpi_"] button[kind="primary"] { background: #eef3ff; color: #1d4ed8; border-color: #1d4ed8; border-left-color: #1d4ed8; }
.st-key-kpi_dur { --c: #64748b; } .st-key-kpi_dec { --c: #16a34a; } .st-key-kpi_act { --c: #d97706; } .st-key-kpi_rev { --c: #dc2626; } .st-key-kpi_top { --c: #1d4ed8; }
.spk { border-left: 4px solid var(--c); background: #fff; border-radius: 0 8px 8px 0; padding: 8px 14px; margin: 6px 0; border-top: 1px solid #e3e7ec; border-right: 1px solid #e3e7ec; border-bottom: 1px solid #e3e7ec; }
.spk b { font-size: .85rem; } .spk small { color: #64748b; margin-left: 8px; }
div[data-testid="stSidebar"] h2 { font-size: 1rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.title("VerbaTrack")

for k, v in (("result", None), ("error", None), ("audio", None), ("seek", None), ("qa", []), ("topic", 0), ("seek_n", 0), ("kpi", None)):
    st.session_state.setdefault(k, v)


def seek(start, end=None):
    st.session_state.seek_n += 1
    st.session_state.seek = (float(start or 0), float(end) if end else None)


def play_btn(label, start, end, key):
    st.button(label, key=key, on_click=seek, args=(start, end), use_container_width=True)


def render_decisions(prefix):
    if not res.record.decisions:
        st.info("No explicit decisions were reached.")
    for i, d in enumerate(res.record.decisions):
        c1, c2 = st.columns([8, 1.2])
        c1.markdown(f"<div class='card'><div class='t'>{esc(d.decision)}</div><div class='q'>“{esc(d.evidence)}”</div>"
                    f"<div class='ts'>{T(d.start)}</div></div>", unsafe_allow_html=True)
        with c2:
            play_btn("▶ Play", d.start, d.end, f"{prefix}d{i}")


def render_actions(items, prefix, empty="No actionable tasks were assigned."):
    if not items:
        st.info(empty)
    cls = {"confirmed": "p-ok", "proposed": "p-prop", "needs_review": "p-rev"}
    for i, t in enumerate(items):
        c1, c2 = st.columns([8, 1.2])
        miss = t.label.startswith("Confirmed,")
        note = f"<div class='q' style='margin-top:4px'>Review: {esc(t.review_note)}</div>" if t.review_note else ""
        c1.markdown(
            f"<div class='card'><div class='t'>{esc(t.task)}</div>"
            f"{pill(t.label, 'p-info' if miss else cls[t.status])} "
            f"{pill('Owner: ' + t.owner, 'p-mute')} {pill('Deadline: ' + t.deadline, 'p-mute')}"
            f"<div class='q' style='margin-top:6px'>“{esc(t.evidence)}”</div>{note}<div class='ts'>{T(t.start)}</div></div>",
            unsafe_allow_html=True)
        with c2:
            play_btn("▶ Play", t.start, t.end, f"{prefix}t{i}")


def render_wer(prefix, res=None):
    """WER calculator: paste a human reference and the transcript to score."""
    st.markdown("Paste the **reference** (the correct, human-checked transcript) and the **transcript to score**. "
                "Case, punctuation and `[mm:ss]` stamps are ignored.")
    rk, hk = f"{prefix}_ref", f"{prefix}_hyp"
    if res is not None:
        b1, b2, _ = st.columns([1.4, 1.4, 4])
        b1.button("Fill with refined transcript", key=f"{prefix}_fr", on_click=lambda: st.session_state.update({hk: res.refined_transcript}))
        b2.button("Fill with raw transcript", key=f"{prefix}_fw", on_click=lambda: st.session_state.update({hk: res.raw_transcript}))
    c1, c2 = st.columns(2)
    ref = c1.text_area("Reference transcript (ground truth)", key=rk, height=220)
    hyp = c2.text_area("Transcript to score", key=hk, height=220)
    fill = st.checkbox("Ignore filler words (um, uh, hmm)", key=f"{prefix}_fill")
    if not (ref.strip() and hyp.strip()):
        return
    try:
        r = compute_wer(ref, hyp, fill)
    except ValueError as e:
        st.error(str(e)); return
    m = st.columns(6)
    m[0].metric("WER", f"{r.wer:.1%}"); m[1].metric("Accuracy", f"{r.accuracy:.1%}"); m[2].metric("Reference words", r.n_ref)
    m[3].metric("Substitutions", r.subs); m[4].metric("Deletions", r.dels); m[5].metric("Insertions", r.ins)
    st.caption(f"WER = (S + D + I) / N = ({r.subs} + {r.dels} + {r.ins}) / {r.n_ref}. It can exceed 100% when there are many insertions.")
    if res is not None:
        a, b = compute_wer(ref, res.raw_transcript, fill), compute_wer(ref, res.refined_transcript, fill)
        st.markdown(f"<table class='cmp'><tr><th>Against this reference</th><th>WER</th><th>S</th><th>D</th><th>I</th></tr>"
                    f"<tr><td>Raw transcript (Whisper)</td><td>{a.wer:.2%}</td><td>{a.subs}</td><td>{a.dels}</td><td>{a.ins}</td></tr>"
                    f"<tr><td>Refined transcript (LLM #1)</td><td>{b.wer:.2%}</td><td>{b.subs}</td><td>{b.dels}</td><td>{b.ins}</td></tr></table>",
                    unsafe_allow_html=True)
    with st.expander("Word-by-word alignment"):
        out = []
        for op, rw, hw in r.ops:
            out.append(esc(rw) if op == "H" else f"<del class='d'>{esc(rw)}</del>" if op == "D" else f"<ins class='i'>{esc(hw)}</ins>" if op == "I"
                       else f"<del class='d'>{esc(rw)}</del><ins class='i'>{esc(hw)}</ins>")
        st.caption("Red = reference word missed or replaced, green = word added or replacing it.")
        st.markdown("<div style='line-height:2;background:#fff;border:1px solid #e3e7ec;border-radius:8px;padding:12px'>" + " ".join(out) + "</div>",
                    unsafe_allow_html=True)


def esc(s):
    return html.escape(str(s))


def diff_cols(raw: str, ref: str):
    """Word diff: strong colours for changed words, dotted underline for punctuation/case-only edits."""
    a, b = raw.split(), ref.split()
    ka, kb = [pl._key(w) for w in a], [pl._key(w) for w in b]
    ra, rb, words, fmt = [], [], 0, 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
        if tag == "equal":
            for x, y in zip(a[i1:i2], b[j1:j2]):
                if x == y:
                    ra.append(esc(x)); rb.append(esc(y))
                else:
                    fmt += 1; ra.append(esc(x)); rb.append(f"<u class='f'>{esc(y)}</u>")
        else:
            words += max(i2 - i1, j2 - j1)
            if i2 > i1:
                ra.append(f"<del class='d'>{esc(' '.join(a[i1:i2]))}</del>")
            if j2 > j1:
                rb.append(f"<ins class='i'>{esc(' '.join(b[j1:j2]))}</ins>")
    return " ".join(ra), " ".join(rb), words, fmt


def pill(text, cls):
    return f"<span class='pill {cls}'>{esc(text)}</span>"


# ------------------------------------------------------------------ sidebar
res = st.session_state.result
with st.sidebar:
    st.header("Recording")
    up = st.file_uploader("English meeting audio", type=[e.strip(".") for e in sorted(pl.ALLOWED_EXT)])
    glossary = st.text_area("Domain glossary (optional)", placeholder="e.g. Kubernetes, OKR, Priya Raman",
                            help="Comma-separated terms. Helps Whisper and the refinement model spell domain words correctly.")
    go = st.button("Process meeting", type="primary", disabled=up is None, use_container_width=True)

    audio_bytes = st.session_state.audio if res is not None else (up.getvalue() if up is not None else None)
    if audio_bytes:
        sk = st.session_state.seek
        kw = {}
        if sk and res is not None:
            kw = dict(start_time=int(sk[0]), end_time=int(math.ceil(sk[1])) + 1 if sk[1] else None)
            st.caption(f"Playing from {T(sk[0])}")
        try:
            st.audio(audio_bytes, autoplay=st.session_state.seek_n > 0 and res is not None, **kw)
        except TypeError:  # older Streamlit without autoplay
            st.audio(audio_bytes, **kw)

    with st.expander("Connection check"):
        if st.button("Test Groq & list models"):
            try:
                st.write("Accessible models:", pl.list_models())
                a = pl.resolve_model("ANALYSIS_MODEL", pl.ANALYSIS_PREFS)
                r = pl.resolve_model("REFINE_MODEL", pl.REFINE_PREFS, avoid=a)
                q = pl.resolve_model("ASK_MODEL", pl.ASK_PREFS, avoid=a)
                st.success(f"Refinement: {r} | analysis: {a} | Q&A: {q}")
            except pl.PipelineError as e:
                st.error(str(e))

# ------------------------------------------------------------------ run
if go and up is not None:
    st.session_state.update(result=None, error=None, qa=[], seek=None, seek_n=0, topic=0, kpi=None)
    with st.status("Processing meeting…", expanded=True) as status:
        try:
            out = pl.process_upload(up.getvalue(), up.name, glossary, on_status=lambda s, m: st.write(f"**Stage {s}/3** — {m}"))
            st.session_state.result, st.session_state.audio = out, up.getvalue()
            status.update(label="Pipeline completed", state="complete", expanded=False)
        except pl.PipelineError as e:
            st.session_state.error = str(e)
            status.update(label="Pipeline failed", state="error")
        except Exception as e:  # unexpected
            st.session_state.error = f"Unexpected error: {e}"
            status.update(label="Pipeline failed", state="error")
    if st.session_state.result is not None:
        st.rerun()  # refresh sidebar player with stored audio

if st.session_state.error:
    st.error(st.session_state.error)

res = st.session_state.result
if res is None:
    if not st.session_state.error:
        st.info("Upload a recording in the sidebar and click **Process meeting**.")
    with st.expander("WER calculator (no recording needed)"):
        render_wer("e")
    st.stop()

rec, segs = res.record, res.segments
review = sum(t.status != "confirmed" or t.owner == pl.UNSPECIFIED or t.deadline == pl.UNSPECIFIED for t in rec.action_items)
kpis = [("dur", "Duration", T(res.duration)), ("dec", "Decisions", len(rec.decisions)), ("act", "Action items", len(rec.action_items)),
        ("rev", "Need review", review), ("top", "Topics", len(rec.topics))]


def toggle_kpi(k):
    st.session_state.kpi = None if st.session_state.kpi == k else k


for col, (k, label, v) in zip(st.columns(5), kpis):
    col.button(f"{v}   {label}", key=f"kpi_{k}", use_container_width=True, on_click=toggle_kpi, args=(k,),
               type="primary" if st.session_state.kpi == k else "secondary")

k = st.session_state.kpi
if k:
    with st.container(border=True):
        h1, h2 = st.columns([8, 1])
        h1.markdown(f"#### {dict((a, b) for a, b, _ in kpis)[k]}")
        h2.button("Close", key="kpi_close", on_click=toggle_kpi, args=(k,), use_container_width=True)
        if k == "dur":
            words = len(res.raw_transcript.split())
            cov = res.transcript_end / res.duration if res.duration else 1
            c = st.columns(4)
            c[0].metric("Recording length", T(res.duration))
            c[1].metric("Transcript reaches", T(res.transcript_end), f"{cov:.0%} of audio", delta_color="normal" if cov > .95 else "inverse")
            c[2].metric("Words", words)
            c[3].metric("Pace", f"{words / max(res.duration / 60, .1):.0f} wpm")
            if cov <= .95:
                st.warning("The transcript ends before the audio does. See the validation notes below.")
            st.write("**Models:** " + " · ".join(f"{a}: `{b}`" for a, b in res.models.items() if b))
            st.write("**Time taken:** " + " · ".join(f"{a} {b}s" for a, b in res.timings.items()))
        elif k == "dec":
            render_decisions("k")
        elif k == "act":
            render_actions(rec.action_items, "k")
        elif k == "rev":
            flagged = [t for t in rec.action_items if t.status != "confirmed" or t.owner == pl.UNSPECIFIED or t.deadline == pl.UNSPECIFIED]
            render_actions(flagged, "kr", "Nothing needs review: every task is confirmed with an owner and a deadline.")
        elif k == "top":
            for i, t in enumerate(rec.topics):
                c1, c2 = st.columns([8, 1.6])
                c1.markdown(f"<div class='card'><div class='t'>{esc(t.title)}</div><div class='ts'>{T(t.start)} – {T(t.end)}</div></div>", unsafe_allow_html=True)
                c2.button("Open in timeline", key=f"kt{i}", use_container_width=True,
                          on_click=lambda i=i, t=t: (st.session_state.update(topic=i), seek(t.start, t.end)))
            if not rec.topics:
                st.info("No topics were detected.")

if res.warnings:
    with st.expander(f"{len(res.warnings)} validation note(s)"):
        for w in res.warnings:
            st.write("-", w)

tabs = st.tabs(["Meeting record", "Timeline", "Transcript", "Ask your meeting", "Exports", "Evaluate (WER)", "Pipeline"])

# ---- Meeting record: decisions first, action items on the line below
with tabs[0]:
    st.subheader("Summary")
    st.write(rec.summary)

    st.subheader("Key decisions")
    render_decisions("m")

    st.subheader("Action items")
    render_actions(rec.action_items, "m")

    st.subheader("Discussion minutes")
    for p in rec.minutes:
        st.markdown(f"- {p}")

    st.subheader("Statement classification")
    st.caption("Proposals and questions are kept separate from real decisions and commitments.")
    if rec.statements:
        sc = {"Decision": "p-ok", "Proposal": "p-prop", "Action": "p-info", "Request": "p-mute", "Unresolved": "p-rev"}
        rows = "".join(f"<tr><td class='ts'>{T(s.start)}</td><td>{esc(s.text)}</td><td>{pill(s.type, sc[s.type])}</td></tr>" for s in rec.statements)
        st.markdown(f"<table class='cmp'><tr><th>Time</th><th>Statement</th><th>Type</th></tr>{rows}</table>", unsafe_allow_html=True)
    else:
        st.info("No classified statements.")

# ---- Timeline
with tabs[1]:
    if not rec.topics or res.duration <= 0:
        st.info("No topic timeline could be built for this recording.")
    else:
        palette = ["#1d4ed8", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1", "#4d7c0f", "#475569"]
        dur = res.duration
        bar = "".join(f"<div style='flex:{max(t.end - t.start, dur * .03):.1f};background:{palette[i % 8]}' title='{esc(t.title)}'>{esc(t.title)}</div>"
                      for i, t in enumerate(rec.topics))
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:.8rem;color:#64748b'><span>00:00</span><span>{T(dur)}</span></div>"
                    f"<div class='bar'>{bar}</div>", unsafe_allow_html=True)
        cols = st.columns(len(rec.topics))
        for i, t in enumerate(rec.topics):
            cols[i].button(f"{t.title}\n{T(t.start)}–{T(t.end)}", key=f"tp{i}", use_container_width=True,
                           type="primary" if st.session_state.topic == i else "secondary",
                           on_click=lambda i=i, t=t: (st.session_state.update(topic=i), seek(t.start, t.end)))
        cur = rec.topics[min(st.session_state.topic, len(rec.topics) - 1)]
        st.subheader(f"{cur.title}  ·  {T(cur.start)}–{T(cur.end)}")
        for s in segs[cur.start_segment:cur.end_segment + 1]:
            st.markdown(f"<div style='margin:4px 0'><span class='ts' style='color:#1d4ed8;font-size:.8rem;margin-right:10px'>{T(s['start'])}</span>"
                        f"{esc(s.get('refined') or s['raw'])}</div>", unsafe_allow_html=True)

# ---- Transcript comparison
with tabs[2]:
    view = st.radio("View", ["Side by side with highlights", "Plain text"], horizontal=True, label_visibility="collapsed")
    if view.startswith("Side"):
        rows, tw, tf = [], 0, 0
        for s in segs:
            a, b, w, f = diff_cols(s["raw"], s.get("refined") or s["raw"])
            tw, tf = tw + w, tf + f
            rows.append(f"<tr><td class='ts'>{T(s['start'])}</td><td>{a}</td><td>{b}</td></tr>")
        st.caption(f"{tw} word(s) corrected, {tf} punctuation/capitalisation edit(s). "
                   "Red = removed from raw, green = added by refinement, dotted underline = punctuation or casing only.")
        st.markdown(f"<table class='cmp'><tr><th>Time</th><th>Raw transcript (Whisper)</th><th>Refined transcript (LLM #1)</th></tr>{''.join(rows)}</table>",
                    unsafe_allow_html=True)
    else:
        a, b = st.columns(2)
        a.markdown("#### Raw")
        a.text_area("raw", pl.full_text(segs, "raw", True), height=480, label_visibility="collapsed")
        b.markdown("#### Refined")
        b.text_area("refined", pl.full_text(segs, "refined", True), height=480, label_visibility="collapsed")

# ---- Ask your meeting (LLM #3)
with tabs[3]:
    st.caption("Answers come only from the transcript and link to the exact moment in the recording.")
    examples = ["Why did we reject the first approach?", "What is still pending?", "Was a deadline actually agreed upon?"]
    ec = st.columns(3)
    pending = None
    for i, q in enumerate(examples):
        if ec[i].button(q, key=f"ex{i}", use_container_width=True):
            pending = q
    with st.form("ask", clear_on_submit=True):
        q = st.text_input("Your question", placeholder="Ask anything about this meeting")
        if st.form_submit_button("Ask", type="primary"):
            pending = q or pending
    if pending:
        with st.spinner("Searching the transcript…"):
            try:
                st.session_state.qa.insert(0, pl.ask_meeting(pending, segs, res.models.get("ask")))
            except pl.PipelineError as e:
                st.error(str(e))
    for n, a in enumerate(st.session_state.qa):
        st.markdown(f"<div class='card'><div class='t'>{esc(a.question)}</div>{esc(a.answer)}</div>", unsafe_allow_html=True)
        if not a.grounded:
            st.caption("No supporting passage was found in the transcript.")
        for k, e in enumerate(a.evidence):
            c1, c2 = st.columns([8, 1.2])
            c1.markdown(f"<div class='q'>{T(e.start)} — “{esc(e.quote)}”</div>", unsafe_allow_html=True)
            with c2:
                play_btn("▶ Play", e.start, e.end, f"qa{n}_{k}")

# ---- Exports
with tabs[4]:
    st.subheader("Download outputs")
    c = st.columns(4)
    c[0].download_button("Record (JSON)", res.to_json(), "meeting_record.json", "application/json", use_container_width=True)
    c[1].download_button("Record (Markdown)", res.to_markdown(), "meeting_record.md", "text/markdown", use_container_width=True)
    c[2].download_button("Raw transcript", pl.full_text(segs, "raw", True), "raw_transcript.txt", use_container_width=True)
    c[3].download_button("Refined transcript", pl.full_text(segs, "refined", True), "refined_transcript.txt", use_container_width=True)
    with st.expander("Preview Markdown"):
        st.markdown(res.to_markdown())

# ---- Evaluate
with tabs[5]:
    render_wer("w", res)

# ---- Pipeline
with tabs[6]:
    st.json({"models": res.models, "timings_sec": res.timings})

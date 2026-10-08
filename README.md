
# VerbaTrack

VerbaTrack is a meeting transcription and analysis tool. Upload a recorded meeting and it generates a raw transcript, a refined transcript, meeting minutes, decisions, and action items. The results can be reviewed in the browser and downloaded for later use.

## Running it

You need Python 3.10 or newer and a Groq API key.

```
pip install -r requirements.txt
cp .env.example .env
```

On Windows use `copy` instead of `cp`. Open `.env` and paste your key after `GROQ_API_KEY=`. Then:

```
streamlit run app.py
```

The page opens at http://localhost:8501. Pick an audio file in the sidebar, optionally list any names or jargon in the glossary box, and press Process meeting.

There is also a small REST version if you want to script it:

```
uvicorn api:app --reload
```

`POST /process` takes an audio file and returns everything as JSON.
 `POST /ask` takes a question plus the `segments` list from `/process`.
 Interactive docs are at http://127.0.0.1:8000/docs.

## What happens to a recording

1. Whisper (`whisper-large-v3` on Groq) transcribes the audio and returns timestamped segments. Long files are cut into 5 minute pieces first, because a single long request sometimes stops early. If a piece ends well before its audio does, it is retried, and if that fails only the missing tail is transcribed again.
2. A first language model fixes punctuation, capitalisation, grammar and misheard terms, a few lines at a time. Any line where a number disappears, a "not" disappears, or the wording changes too much is thrown away and the raw line is kept.
3. A second language model reads the refined transcript and writes the summary, minutes, decisions, action items, a classification of key statements and a list of topics with time ranges.
4. Before anything is shown, the code checks the result against the transcript. A decision or task has to quote the recording, and the quote has to actually be found there. Owners and deadlines that never appear in the transcript are replaced with UNSPECIFIED. Hedged statements ("maybe we should...") do not become decisions.
5. A third model answers questions in the Ask tab, using only the transcript, and points to the lines it relied on.

Which model plays which role is decided at start-up from the models your key can use. In the demo run these were `whisper-large-v3`, `openai/gpt-oss-20b` for refinement and `openai/gpt-oss-120b` for analysis and questions. You can force a choice with `REFINE_MODEL`, `ANALYSIS_MODEL`, `ASK_MODEL` or `STT_MODEL` in `.env`.

## What you get

The Meeting record tab has the summary, decisions, action items and minutes. Every decision and task shows the quote it came from, a timestamp and a Play button that jumps the audio player in the sidebar to that spot. Tasks are labelled as confirmed, proposed, or needing review, and a missing owner or deadline is written as UNSPECIFIED rather than guessed.

The other tabs are a topic timeline, the raw and refined transcripts side by side with changes highlighted, the Ask tab, and the downloads. The five cards at the top (duration, decisions, action items, need review, topics) can be clicked for more detail.

Downloads: the full record as JSON, the same record as Markdown, and the raw and refined transcripts as text files with timestamps. The Markdown and JSON contain the same decisions and tasks.

## Errors

Unsupported file types, empty files, files that cannot be decoded, recordings with no speech, a missing or rejected API key, an unavailable model and rate limits each produce a message in the page instead of a crash. If the transcript ends before the audio does, a note says by how much.

## Measuring accuracy (WER)

Word error rate is the standard way to score a speech transcript. You need a reference transcript, meaning the words that were really said, written or corrected by a person. Paste it into the Evaluate (WER) tab, then paste the transcript you want to score next to it (or press "Fill with refined transcript" / "Fill with raw transcript" to take it from the current run). The tab also works without a recording, from an expander on the start page.

```
WER = (substitutions + deletions + insertions) / words in the reference
```

The words are lined up with a standard edit-distance alignment (`wer.py`). Before comparing, both texts are lowercased and stripped of punctuation and `[mm:ss]` stamps, and numbers like 1,000 are written as 1000. There is a checkbox to ignore fillers such as "um" and "uh". The tab shows WER, accuracy (1 minus WER), the three error counts, a table with the raw and the refined transcript scored against the same reference, and a colour-coded word-by-word alignment.

Two things to keep in mind. WER is only an accuracy figure if the reference is independent of the system. If you paste the raw Whisper text as the "reference" and the refined text as the transcript, the number only tells you how much the refinement step changed, not how correct either one is. And WER treats every word equally, so a dropped "not" costs the same as a typo. That is why the decisions and tasks are also checked by hand against the recording.


## Files

```
app.py            the Streamlit interface
pipeline.py       transcription, refinement, analysis, checks, Q&A
prompts.py        every instruction given to the language models
schemas.py        the structure of the meeting record
wer.py            word error rate calculation
api.py            optional REST wrapper
```

## Known limits

- English only.
- No speaker labels. The transcript says what was said, not who said it, so task owners only appear when the speaker names them.
- Groq accepts about 25 MB per request, which is why audio is converted and split. Free-tier rate limits can slow down or interrupt a very long meeting; just run it again.
- Whisper can miss very quiet speech. The app reports it when part of the recording produced no text.
- The analysis model sees the whole transcript at once, so meetings much longer than a couple of hours may need to be split.

## If something goes wrong

`404 model_not_found` means your key cannot use the model the app picked. Open "Connection check" in the sidebar to see which models you have, then set the matching variable in `.env`.

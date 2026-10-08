"""Word error rate: word-level edit distance between a human reference and a hypothesis transcript."""
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "hmm", "mm", "mhm", "ah"}


def normalize(text: str, drop_fillers: bool = False) -> List[str]:
    """Lowercase, drop [mm:ss] stamps and punctuation, keep apostrophes inside words and digits."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", " ", text)           # [03:21] timestamps
    text = re.sub(r"[-–—/]", " ", text)                                  # state-of-the-art -> state of the art
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)                        # 1,000 -> 1000
    text = re.sub(r"[^a-z0-9'\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)                      # keep decimal points only
    words = [w.strip("'") for w in text.split()]
    words = [w for w in words if w]
    return [w for w in words if w not in FILLERS] if drop_fillers else words


@dataclass
class WERResult:
    wer: float
    hits: int
    subs: int
    dels: int
    ins: int
    n_ref: int
    ops: List[Tuple[str, str, str]]   # (op, ref_word, hyp_word) with op in H, S, D, I

    @property
    def accuracy(self) -> float:
        return 1 - self.wer


def wer(reference: str, hypothesis: str, drop_fillers: bool = False) -> WERResult:
    r, h = normalize(reference, drop_fillers), normalize(hypothesis, drop_fillers)
    if not r:
        raise ValueError("The reference transcript has no words after normalisation.")
    n, m = len(r), len(h)
    prev = list(range(m + 1))
    back = [[0] * (m + 1) for _ in range(n + 1)]       # 0 hit/sub (diag), 1 delete (up), 2 insert (left)
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        back[i][0] = 1
        for j in range(1, m + 1):
            d = prev[j - 1] + (r[i - 1] != h[j - 1])
            u, l = prev[j] + 1, cur[j - 1] + 1
            cur[j] = min(d, u, l)
            back[i][j] = 0 if cur[j] == d else (1 if cur[j] == u else 2)
        prev = cur
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        b = back[i][j]
        if b == 0:
            ops.append(("H" if r[i - 1] == h[j - 1] else "S", r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif b == 1:
            ops.append(("D", r[i - 1], "")); i -= 1
        else:
            ops.append(("I", "", h[j - 1])); j -= 1
    ops.reverse()
    c = {k: sum(1 for o in ops if o[0] == k) for k in "HSDI"}
    return WERResult((c["S"] + c["D"] + c["I"]) / n, c["H"], c["S"], c["D"], c["I"], n, ops)

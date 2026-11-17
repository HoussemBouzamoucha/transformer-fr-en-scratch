"""Compare two joint BPE vocab files side by side in a standalone HTML page.

Default: before vs after splitting punctuation off words. Rebuild the "before" file with
    python scripts/train_own_bpe.py --no-pretokenize --out logs/bpe_inspect/own_bpe_8k_multi30k_nopretok.json
then
    python scripts/compare_bpe.py

Colours as in visualize_bpe.py: purple = French only, blue = English only,
green = shared, grey = special / unused.
"""
import argparse
import html
import random
import sys
import webbrowser
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--before", default="logs/bpe_inspect/own_bpe_8k_multi30k_nopretok.json")
parser.add_argument("--after", default="data/vocab/own_bpe_8k_multi30k.json")
parser.add_argument("--before-name", default="Before: punctuation attached")
parser.add_argument("--after-name", default="After: punctuation split off")
parser.add_argument("--data-dir", default="data/raw/multi30k")
parser.add_argument("--split", default="test")
parser.add_argument("--n", type=int, default=100, help="first / random / last N sentence pairs")
parser.add_argument("--top", type=int, default=150, help="how many tokens to list as removed / added")
parser.add_argument("--min-count", type=int, default=1)
parser.add_argument("--threshold", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="logs/bpe_inspect/bpe_compare.html")
parser.add_argument("--no-open", action="store_true")
args = parser.parse_args()

sys.stdout.reconfigure(encoding="utf-8")
KINDS = ["fr", "en", "both", "none"]
LABEL = {"fr": "French only", "en": "English only", "both": "Shared", "none": "Special / unused"}

data_dir = Path(args.data_dir)
fr_lines = [l.rstrip("\n") for l in open(data_dir / f"{args.split}.fr", encoding="utf-8")]
en_lines = [l.rstrip("\n") for l in open(data_dir / f"{args.split}.en", encoding="utf-8")]
assert len(fr_lines) == len(en_lines), "FR and EN files must have the same number of lines"
pairs = list(zip(fr_lines, en_lines))


class Side:
    def __init__(self, path, name):
        self.name = name
        self.bpe = BPETokenizer().load(path)
        if not self.bpe.lang_counts:
            sys.exit(f"{path} has no per-language counts; retrain it with scripts/train_own_bpe.py")
        self.fr = self.bpe.lang_counts.get("fr", {})
        self.en = self.bpe.lang_counts.get("en", {})
        self.cls = {t: self._kind(t) for t in self.bpe.vocab}
        self.vocab_totals = Counter(self.cls.values())

        # Encode the whole split once; reuse for stats and sentence rows.
        self.enc = [(self.bpe.encode(f, lang="fr", threshold=args.threshold),
                     self.bpe.encode(e, lang="en", threshold=args.threshold)) for f, e in pairs]
        self.sent_totals = {"fr": Counter(), "en": Counter()}
        self.used = Counter()
        for f_ids, e_ids in self.enc:
            for lang, ids in (("fr", f_ids), ("en", e_ids)):
                for t in ids:
                    tok = self.bpe.id_to_token[t]
                    self.sent_totals[lang][self.cls[tok]] += 1
                    self.used[tok] += 1

    def _kind(self, tok):
        f = self.fr.get(tok, 0) >= args.min_count
        e = self.en.get(tok, 0) >= args.min_count
        return "both" if f and e else "fr" if f else "en" if e else "none"

    def tok_per_sent(self, lang):
        i = 0 if lang == "fr" else 1
        return sum(len(x[i]) for x in self.enc) / len(self.enc)

    def shared_pct(self, lang):
        c = self.sent_totals[lang]
        return 100 * c["both"] / (sum(c.values()) or 1)

    def chip(self, tok):
        b = self.bpe
        cls = self.cls.get(tok, "none")
        end = tok.endswith(b.EOW)
        text = tok[: -len(b.EOW)] if end else tok
        tip = (f"id {b.vocab.get(tok, '?')} · {LABEL[cls]} · fr {self.fr.get(tok, 0)} · "
               f"en {self.en.get(tok, 0)}" + (" · ends a word" if end else ""))
        mark = '<span class="eow">␣</span>' if end else ""
        return (f'<span class="tok {cls}" title="{html.escape(tip)}">{html.escape(text)}{mark}'
                f'<span class="id">{b.vocab.get(tok, "")}</span></span>')

    def chips(self, ids):
        return "".join(self.chip(self.bpe.id_to_token[t]) for t in ids)


A = Side(args.before, args.before_name)
B = Side(args.after, args.after_name)


# ------------------------------------------------------------------ summary
def fmt_delta(a, b, unit="", digits=0, good=None):
    d = b - a
    if abs(d) < 10 ** -digits / 2:
        return '<span class="d">—</span>'
    sign = "+" if d > 0 else "−"
    cls = "" if good is None else ("up" if (d > 0) == good else "down")
    return f'<span class="d {cls}">{sign}{abs(d):,.{digits}f}{unit}</span>'


rows = []
for k in KINDS:
    rows.append((f'<span class="dot {k}"></span>Vocab: {LABEL[k].lower()}',
                 A.vocab_totals[k], B.vocab_totals[k], 0, "", {"both": True, "none": False}.get(k)))
for lang in ("fr", "en"):
    rows.append((f"{lang.upper()} tokens per sentence", A.tok_per_sent(lang), B.tok_per_sent(lang), 2, "", False))
for lang in ("fr", "en"):
    rows.append((f"{lang.upper()} sentence tokens that are shared", A.shared_pct(lang), B.shared_pct(lang), 1, "%", True))
rows.append((f"Distinct tokens used on {args.split}", len(A.used), len(B.used), 0, "", None))

table = "".join(
    f"<tr><td>{label}</td><td>{a:,.{dg}f}{u}</td><td>{b:,.{dg}f}{u}</td>"
    f"<td>{fmt_delta(a, b, u, dg, good)}</td></tr>"
    for label, a, b, dg, u, good in rows)


def bar(counts):
    total = sum(counts.values()) or 1
    return '<div class="bar">' + "".join(
        f'<span class="seg {k}" style="width:{100 * counts[k] / total:.2f}%" '
        f'title="{LABEL[k]}: {counts[k]:,} ({100 * counts[k] / total:.1f}%)"></span>'
        for k in KINDS if counts[k]) + "</div>"


bars = ""
for title, get in [("Vocabulary entries", lambda s: s.vocab_totals),
                   (f"Tokens in {args.split} French sentences", lambda s: s.sent_totals["fr"]),
                   (f"Tokens in {args.split} English sentences", lambda s: s.sent_totals["en"])]:
    bars += (f'<div class="barrow"><div class="bt">{title}</div>'
             f'<div><span class="bl">Before</span>{bar(get(A))}</div>'
             f'<div><span class="bl">After</span>{bar(get(B))}</div></div>')


# ------------------------------------------------------- vocab differences
def top_unique(side, other):
    only = [t for t in side.bpe.vocab if t not in other.bpe.vocab and t not in side.bpe.special_tokens]
    only.sort(key=lambda t: -(side.fr.get(t, 0) + side.en.get(t, 0)))
    return only, "".join(side.chip(t) for t in only[: args.top])


removed, removed_html = top_unique(A, B)
added, added_html = top_unique(B, A)
punct_removed = sum(any(not c.isalnum() for c in t.replace(A.bpe.EOW, "")) for t in removed)


# ---------------------------------------------------------------- sentences
rng = random.Random(args.seed)
idx = list(range(len(pairs)))
samples = [("First", idx[: args.n]), ("Random", sorted(rng.sample(idx, min(args.n, len(idx))))),
           ("Last", idx[-args.n:])]
sent_html = []
for title, picked in samples:
    blocks = []
    for i in picked:
        rows_html = ""
        for li, lang in enumerate(("fr", "en")):
            a_ids, b_ids = A.enc[i][li], B.enc[i][li]
            rows_html += (
                f'<div class="srow"><span class="lang">{lang}</span><span class="tag">before</span>'
                f'<span class="n">{len(a_ids)}</span><div class="toks">{A.chips(a_ids)}</div></div>'
                f'<div class="srow"><span class="lang"></span><span class="tag">after</span>'
                f'<span class="n">{len(b_ids)}</span><div class="toks">{B.chips(b_ids)}</div></div>')
        blocks.append(f'<div class="pair"><span class="line">#{i}</span>{rows_html}</div>')
    sent_html.append(f"<h3>{title} {len(picked)}</h3>{''.join(blocks)}")


page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BPE before and after</title>
<style>
:root {{
  --bg:#fafafa; --fg:#1d1d1f; --muted:#6e6e73; --card:#fff; --line:#e3e3e8;
  --fr:#7c3aed; --fr-bg:#f1eafe; --en:#2563eb; --en-bg:#e6efff;
  --both:#15803d; --both-bg:#e3f6e9; --none:#8a8a8e; --none-bg:#efeff1;
  --up:#15803d; --down:#c2410c;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg:#141416; --fg:#ececf0; --muted:#9a9aa2; --card:#1d1d21; --line:#2c2c32;
  --fr:#b794ff; --fr-bg:#2a2140; --en:#7aa7ff; --en-bg:#1a2640;
  --both:#5fd38a; --both-bg:#173123; --none:#8a8a92; --none-bg:#26262b;
  --up:#5fd38a; --down:#ff9a62;
}} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px }}
h1 {{ font-size:22px; margin:0 0 4px }} h2 {{ font-size:18px; margin:36px 0 8px }}
h3 {{ font-size:14px; color:var(--muted); margin:20px 0 8px; font-weight:600 }}
.sub,.note {{ color:var(--muted); font-size:13px; margin:4px 0 }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 }}
.legend button {{ border:1px solid var(--line); background:var(--card); color:var(--fg);
  border-radius:999px; padding:4px 12px; font:inherit; font-size:13px; cursor:pointer }}
.legend button.off {{ opacity:.35 }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:baseline }}
.dot.fr {{ background:var(--fr) }} .dot.en {{ background:var(--en) }}
.dot.both {{ background:var(--both) }} .dot.none {{ background:var(--none) }}
.tablewrap {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line);
  border-radius:10px; overflow:hidden; font-size:14px }}
th,td {{ padding:7px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap }}
th:first-child,td:first-child {{ text-align:left }}
th {{ font-size:12px; color:var(--muted); font-weight:600 }}
tr:last-child td {{ border-bottom:0 }}
td {{ font-variant-numeric:tabular-nums }}
.d {{ color:var(--muted) }} .d.up {{ color:var(--up) }} .d.down {{ color:var(--down) }}
.barrow {{ margin:14px 0 }} .bt {{ font-size:13px; color:var(--muted); margin-bottom:4px }}
.barrow > div:not(.bt) {{ display:flex; align-items:center; gap:8px; margin:3px 0 }}
.bl {{ width:48px; font-size:12px; color:var(--muted) }}
.bar {{ flex:1; display:flex; height:12px; border-radius:6px; overflow:hidden; background:var(--line) }}
.seg.fr {{ background:var(--fr) }} .seg.en {{ background:var(--en) }}
.seg.both {{ background:var(--both) }} .seg.none {{ background:var(--none) }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px }}
@media (max-width:760px) {{ .cols {{ grid-template-columns:1fr }} }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px }}
.panel h3 {{ margin:0 0 4px }}
.grid {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:8px }}
.tok {{ display:inline-flex; align-items:baseline; gap:3px; border-radius:5px; padding:1px 6px;
  font:13px/1.6 ui-monospace,Consolas,monospace; white-space:pre; cursor:default }}
.tok .id {{ font-size:10px; opacity:.55 }} .tok .eow {{ opacity:.5 }}
.tok.fr {{ color:var(--fr); background:var(--fr-bg) }}
.tok.en {{ color:var(--en); background:var(--en-bg) }}
.tok.both {{ color:var(--both); background:var(--both-bg) }}
.tok.none {{ color:var(--none); background:var(--none-bg) }}
body.hide-ids .tok .id {{ display:none }}
.hide-fr .tok.fr, .hide-en .tok.en, .hide-both .tok.both, .hide-none .tok.none {{ opacity:.12 }}
.pair {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:8px 10px; margin-bottom:8px; position:relative }}
.line {{ position:absolute; right:10px; top:6px; font-size:11px; color:var(--muted) }}
.srow {{ display:grid; grid-template-columns:24px 46px 26px 1fr; align-items:start; gap:4px; margin:2px 0 }}
.srow + .srow:nth-child(odd) {{ margin-top:8px }}
.lang {{ font-size:11px; font-weight:600; text-transform:uppercase; color:var(--muted); padding-top:3px }}
.tag {{ font-size:11px; color:var(--muted); padding-top:3px }}
.n {{ font-size:11px; color:var(--muted); text-align:right; padding-top:3px; font-variant-numeric:tabular-nums }}
.toks {{ display:flex; flex-wrap:wrap; gap:3px; padding-right:36px }}
</style></head>
<body class="hide-ids">
<main>
<h1>Joint BPE: before vs after</h1>
<p class="sub"><b>Before</b> = {html.escape(A.name)} ({html.escape(args.before)})<br>
<b>After</b> = {html.escape(B.name)} ({html.escape(args.after)})<br>
Sentences from {html.escape(args.data_dir)}/{html.escape(args.split)}, encoded with threshold {args.threshold}.
A token belongs to a language if it appears there ≥ {args.min_count}×. Hover a token for its id and counts.</p>
<div class="legend">
  <button data-k="fr"><span class="dot fr"></span>French only</button>
  <button data-k="en"><span class="dot en"></span>English only</button>
  <button data-k="both"><span class="dot both"></span>Shared</button>
  <button data-k="none"><span class="dot none"></span>Special / unused</button>
  <button id="ids">Show ids</button>
</div>
<p class="note">␣ = token ends a word (&lt;/w&gt;). ⁐ = join mark: no space on that side of the punctuation.</p>

<h2>Summary</h2>
<div class="tablewrap"><table>
<tr><th>Metric</th><th>Before</th><th>After</th><th>Change</th></tr>{table}
</table></div>
{bars}

<h2>Vocabulary differences</h2>
<p class="note">{len(removed):,} entries exist only before ({punct_removed:,} of them contain punctuation);
{len(added):,} entries exist only after. Showing the {args.top} most frequent of each.</p>
<div class="cols">
  <div class="panel"><h3>Only before — freed up</h3><div class="grid">{removed_html}</div></div>
  <div class="panel"><h3>Only after — new</h3><div class="grid">{added_html}</div></div>
</div>

<h2>Sentences</h2>
{''.join(sent_html)}
</main>
<script>
document.querySelectorAll('.legend button[data-k]').forEach(b => b.onclick = () => {{
  b.classList.toggle('off'); document.body.classList.toggle('hide-' + b.dataset.k);
}});
const ids = document.getElementById('ids');
ids.onclick = () => {{
  const hidden = document.body.classList.toggle('hide-ids');
  ids.textContent = hidden ? 'Show ids' : 'Hide ids';
}};
</script>
</body></html>
"""

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(page, encoding="utf-8")

print(f"{'':34}{'before':>10}{'after':>10}")
for label, a, b, dg, u, _ in rows:
    label = label.split("</span>")[-1]
    print(f"{label:34}{a:>10,.{dg}f}{b:>10,.{dg}f}")
print(f"only-before entries: {len(removed)} ({punct_removed} with punctuation); only-after: {len(added)}")
print(f"Wrote {out.resolve()}")
if not args.no_open:
    webbrowser.open(out.resolve().as_uri())

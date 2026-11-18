"""Colour the joint BPE vocabulary by language and write a standalone HTML page.

    purple = French-only token     blue = English-only token
    green  = shared (seen in both) grey = special / unused

    python scripts/visualize_bpe.py
    python scripts/visualize_bpe.py --bpe data/vocab/own_bpe_8k_wmt.json --data-dir data/raw/wmt14 --min-count 5

A token counts as "in" a language when it appears there at least --min-count
times, so a single stray occurrence doesn't turn a token green.
"""
import argparse
import html
import random
import sys
import webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--bpe", default="data/vocab/own_bpe_8k_multi30k.json")
parser.add_argument("--data-dir", default="data/raw/multi30k")
parser.add_argument("--split", default="test")
parser.add_argument("--n", type=int, default=100, help="first / random / last N")
parser.add_argument("--min-count", type=int, default=1)
parser.add_argument("--threshold", type=int, default=5, help="vocab threshold when encoding sentences")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="logs/bpe_inspect/bpe_languages.html")
parser.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
args = parser.parse_args()

sys.stdout.reconfigure(encoding="utf-8")
rng = random.Random(args.seed)
bpe = BPETokenizer().load(args.bpe)
if not bpe.lang_counts:
    sys.exit("This vocab file has no per-language counts. Retrain it with scripts/train_own_bpe.py.")

fr_counts = bpe.lang_counts.get("fr", {})
en_counts = bpe.lang_counts.get("en", {})
LABEL = {"fr": "French only", "en": "English only", "both": "Shared", "none": "Special / unused"}


def lang_of(tok):
    in_fr = fr_counts.get(tok, 0) >= args.min_count
    in_en = en_counts.get(tok, 0) >= args.min_count
    if in_fr and in_en:
        return "both"
    return "fr" if in_fr else "en" if in_en else "none"


vocab = sorted(bpe.vocab.items(), key=lambda kv: kv[1])  # (token, id) by id
classes = {tok: lang_of(tok) for tok, _ in vocab}
totals = {k: sum(c == k for c in classes.values()) for k in LABEL}


def chip(tok, tid):
    cls = classes.get(tok, "none")
    word_end = tok.endswith(bpe.EOW)
    text = tok[: -len(bpe.EOW)] if word_end else tok
    tip = (f"id {tid} · {LABEL[cls]} · fr {fr_counts.get(tok, 0)} · en {en_counts.get(tok, 0)}"
           + (" · ends a word" if word_end else ""))
    end = '<span class="eow">␣</span>' if word_end else ""
    return (f'<span class="tok {cls}" title="{html.escape(tip)}">'
            f'{html.escape(text)}{end}<span class="id">{tid}</span></span>')


def sample(items, n):
    idx = list(range(len(items)))
    rand = sorted(rng.sample(idx, min(n, len(idx))))
    return [("First", idx[:n]), ("Random", rand), ("Last", idx[-n:])]


# ---- vocab sections
vocab_html = []
for title, picked in sample(vocab, args.n):
    chips = "".join(chip(*vocab[i]) for i in picked)
    vocab_html.append(f'<h3>{title} {len(picked)}</h3><div class="grid">{chips}</div>')

# ---- sentence sections
data_dir = Path(args.data_dir)
fr = [l.rstrip("\n") for l in open(data_dir / f"{args.split}.fr", encoding="utf-8")]
en = [l.rstrip("\n") for l in open(data_dir / f"{args.split}.en", encoding="utf-8")]
assert len(fr) == len(en), "FR and EN files must have the same number of lines"
pairs = list(zip(fr, en))

sent_html, sent_tok_stats = [], {"fr": dict.fromkeys(LABEL, 0), "en": dict.fromkeys(LABEL, 0)}
for title, picked in sample(pairs, args.n):
    rows = []
    for i in picked:
        cells = []
        for lang, text in zip(("fr", "en"), pairs[i]):
            ids = bpe.encode(text, lang=lang, threshold=args.threshold)
            for t in ids:
                sent_tok_stats[lang][classes.get(bpe.id_to_token[t], "none")] += 1
            chips = "".join(chip(bpe.id_to_token[t], t) for t in ids)
            cells.append(f'<div class="sent"><span class="lang">{lang}</span>{chips}</div>')
        rows.append(f'<div class="pair"><span class="line">#{i}</span>{"".join(cells)}</div>')
    sent_html.append(f'<h3>{title} {len(picked)}</h3>{"".join(rows)}')


def share_bar(counts):
    total = sum(counts.values()) or 1
    segs = "".join(f'<span class="seg {k}" style="width:{100 * v / total:.2f}%" '
                   f'title="{LABEL[k]}: {v} ({100 * v / total:.1f}%)"></span>'
                   for k, v in counts.items() if v)
    return f'<div class="bar">{segs}</div>'


def pct(d, k):
    return f"{100 * d[k] / (sum(d.values()) or 1):.1f}%"


stats_html = f"""
<div class="stats">
  <div class="card"><div class="k">Vocabulary</div><div class="v">{len(vocab):,}</div></div>
  <div class="card fr"><div class="k">French only</div><div class="v">{totals['fr']:,}</div></div>
  <div class="card en"><div class="k">English only</div><div class="v">{totals['en']:,}</div></div>
  <div class="card both"><div class="k">Shared</div><div class="v">{totals['both']:,}</div></div>
  <div class="card none"><div class="k">Special / unused</div><div class="v">{totals['none']:,}</div></div>
</div>
<p class="note">Vocabulary entries by language</p>{share_bar(totals)}
<p class="note">Tokens in the sampled French sentences — {pct(sent_tok_stats['fr'], 'both')} are shared</p>
{share_bar(sent_tok_stats['fr'])}
<p class="note">Tokens in the sampled English sentences — {pct(sent_tok_stats['en'], 'both')} are shared</p>
{share_bar(sent_tok_stats['en'])}
"""

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Joint BPE languages</title>
<style>
:root {{
  --bg:#fafafa; --fg:#1d1d1f; --muted:#6e6e73; --card:#fff; --line:#e3e3e8;
  --fr:#7c3aed; --fr-bg:#f1eafe; --en:#2563eb; --en-bg:#e6efff;
  --both:#15803d; --both-bg:#e3f6e9; --none:#8a8a8e; --none-bg:#efeff1;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg:#141416; --fg:#ececf0; --muted:#9a9aa2; --card:#1d1d21; --line:#2c2c32;
  --fr:#b794ff; --fr-bg:#2a2140; --en:#7aa7ff; --en-bg:#1a2640;
  --both:#5fd38a; --both-bg:#173123; --none:#8a8a92; --none-bg:#26262b;
}} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px }}
h1 {{ font-size:22px; margin:0 0 4px }} h2 {{ font-size:18px; margin:32px 0 8px }}
h3 {{ font-size:14px; color:var(--muted); margin:20px 0 8px; font-weight:600 }}
.sub,.note {{ color:var(--muted); font-size:13px; margin:4px 0 }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 }}
.legend button {{ border:1px solid var(--line); background:var(--card); color:var(--fg);
  border-radius:999px; padding:4px 12px; font:inherit; font-size:13px; cursor:pointer }}
.legend button.off {{ opacity:.35 }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; margin:16px 0 }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px }}
.card .k {{ font-size:12px; color:var(--muted) }} .card .v {{ font-size:22px; font-weight:600 }}
.card.fr .v {{ color:var(--fr) }} .card.en .v {{ color:var(--en) }}
.card.both .v {{ color:var(--both) }} .card.none .v {{ color:var(--none) }}
.bar {{ display:flex; height:12px; border-radius:6px; overflow:hidden; background:var(--line); margin-bottom:8px }}
.seg.fr {{ background:var(--fr) }} .seg.en {{ background:var(--en) }}
.seg.both {{ background:var(--both) }} .seg.none {{ background:var(--none) }}
.grid {{ display:flex; flex-wrap:wrap; gap:4px }}
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
  padding:8px 10px; margin-bottom:6px; position:relative }}
.line {{ position:absolute; right:10px; top:6px; font-size:11px; color:var(--muted) }}
.sent {{ display:flex; flex-wrap:wrap; gap:3px; align-items:center; margin:2px 0; padding-right:40px }}
.lang {{ font-size:11px; font-weight:600; text-transform:uppercase; color:var(--muted); width:22px }}
</style></head>
<body class="hide-ids">
<main>
<h1>Joint BPE: tokens by language</h1>
<p class="sub">{html.escape(args.bpe)} · a token belongs to a language if it appears there ≥ {args.min_count}×
 · sentences from {html.escape(args.split)} encoded with threshold {args.threshold} · hover a token for its id and counts</p>
<div class="legend">
  <button data-k="fr"><span class="dot" style="background:var(--fr)"></span>French only</button>
  <button data-k="en"><span class="dot" style="background:var(--en)"></span>English only</button>
  <button data-k="both"><span class="dot" style="background:var(--both)"></span>Shared</button>
  <button data-k="none"><span class="dot" style="background:var(--none)"></span>Special / unused</button>
  <button id="ids">Show ids</button>
</div>
{stats_html}
<h2>Vocabulary</h2>
<p class="note">␣ marks a token that ends a word (&lt;/w&gt;).</p>
{''.join(vocab_html)}
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

print(f"Vocab {len(vocab)}: French only {totals['fr']}, English only {totals['en']}, "
      f"shared {totals['both']}, special/unused {totals['none']}")
for lang in ("fr", "en"):
    print(f"  shared tokens in sampled {lang} sentences: {pct(sent_tok_stats[lang], 'both')}")
print(f"Wrote {out.resolve()}")
if not args.no_open:
    webbrowser.open(out.resolve().as_uri())

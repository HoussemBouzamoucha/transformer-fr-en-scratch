import random
from pathlib import Path


def sample_pairs(data_dir, sample_size=100_000, seed=42):
    """Reservoir-sample aligned (fr, en) training lines in one pass.

    Same data_dir/sample_size/seed -> same lines, so every tokenizer we compare
    is trained on exactly the same text. Returns ({"fr": [...], "en": [...]}, total_pairs).
    """
    data_dir = Path(data_dir)
    rng = random.Random(seed)
    sample, n = [], 0
    with open(data_dir / "train.fr", encoding="utf-8") as f, open(data_dir / "train.en", encoding="utf-8") as e:
        for n, (fr, en) in enumerate(zip(f, e), 1):
            if len(sample) < sample_size:
                sample.append((fr.strip(), en.strip()))
            else:
                j = rng.randrange(n)
                if j < sample_size:
                    sample[j] = (fr.strip(), en.strip())
    return {"fr": [p[0] for p in sample], "en": [p[1] for p in sample]}, n

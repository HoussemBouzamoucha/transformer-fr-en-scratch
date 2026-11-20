"""Paper-faithful pieces: Post-LN, build_model switches, beam search, token
batching, checkpoint averaging, and that old (Pre-LN) configs rebuild as before."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.data.dataset import TokenBatchSampler
from src.inference.beam_search import beam_search, length_penalty
from src.inference.greedy import greedy_decode, strip_after_eos
from src.model.layers import EncoderLayer
from src.model.transformer import build_model
from src.training.checkpoints import average_checkpoints, save_step_checkpoint, step_checkpoints

PAD, BOS, EOS, VOCAB = 0, 1, 2, 40
SMALL = dict(d_model=32, n_heads=2, n_encoder_layers=2, n_decoder_layers=2, d_ff=64,
             dropout=0.0, max_seq_len=64)


def small_model(**overrides):
    torch.manual_seed(0)
    return build_model({**SMALL, **overrides}, VOCAB, PAD).eval()


def random_src(batch=3, length=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    body = torch.randint(3, VOCAB, (batch, length), generator=g)
    src = torch.cat([torch.full((batch, 1), BOS), body, torch.full((batch, 1), EOS)], dim=1)
    src[0, -3:] = torch.tensor([EOS, PAD, PAD])  # one shorter, padded sentence
    return src


# ---- Post-LN / Pre-LN -------------------------------------------------------

def test_post_ln_output_is_normalized():
    """Post-LN ends every sublayer with LayerNorm, so outputs are standardized."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32) * 7 + 3
    out = EncoderLayer(32, 2, 64, dropout=0.0, norm_position="post")(x)
    assert torch.allclose(out.mean(-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(2, 5), atol=1e-3)
    pre = EncoderLayer(32, 2, 64, dropout=0.0, norm_position="pre")(x)
    assert pre.std(-1).mean() > 2  # residual stream keeps the input's scale


def test_post_ln_model_has_no_final_norm():
    post = small_model(norm_position="post").state_dict()
    pre = small_model(norm_position="pre").state_dict()
    assert "encoder.norm.weight" not in post and "decoder.norm.weight" not in post
    assert "encoder.norm.weight" in pre and "decoder.norm.weight" in pre
    assert "encoder.layers.0.norm1.weight" in post  # same per-layer names in both


def test_old_config_rebuilds_pre_ln():
    """Configs written before norm_position existed must give the old Pre-LN model,
    with the parameter names old checkpoints were saved with."""
    legacy = small_model()
    assert legacy.encoder.layers[0].pre and legacy.decoder.layers[0].pre
    keys = set(legacy.state_dict())
    for k in ["encoder.norm.weight", "decoder.norm.bias", "encoder.layers.1.norm2.weight",
              "decoder.layers.0.norm3.weight", "output_projection.weight"]:
        assert k in keys


def test_build_model_rejects_unknown_switch():
    with pytest.raises(ValueError, match="norm_position"):
        small_model(norm_position="middle")
    with pytest.raises(ValueError, match="pos_encoding"):
        small_model(pos_encoding="rope")  # until RoPE is implemented


def test_post_ln_forward_backward():
    model = small_model(norm_position="post").train()
    src, tgt = random_src(), random_src(seed=2)
    model(src, tgt).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name


# ---- Beam search ------------------------------------------------------------

def test_beam_1_matches_greedy():
    """With one beam and no length penalty, beam search is greedy decoding."""
    for norm in ("post", "pre"):
        model = small_model(norm_position=norm)
        src = random_src()
        greedy = [strip_after_eos(ids, EOS) for ids in greedy_decode(model, src, BOS, EOS, max_len=20).tolist()]
        beam = beam_search(model, src, BOS, EOS, beam_size=1, alpha=0.0, extra_len=50, max_len=20)
        assert beam == greedy


def test_beam_respects_length_limit():
    model = small_model()
    src = random_src(length=5)
    out = beam_search(model, src, BOS, EOS, beam_size=4, alpha=0.6, extra_len=3)
    real_len = ((src != PAD).sum(1) - 2).tolist()
    for ids, n in zip(out, real_len):
        assert len(ids) <= n + 3
        assert EOS not in ids  # </s> is stripped (an untrained model may still emit <s>)


def test_beam_beats_or_ties_greedy_score():
    """Beam 4 should find a hypothesis at least as good (by its own objective) as greedy."""
    model = small_model()
    src = random_src()

    def norm_score(ids, b, alpha):
        seq = torch.tensor([[BOS] + ids + [EOS]])
        memory, mask = model.encode(src[b:b + 1])
        logp = torch.log_softmax(model.output_projection(model.decode(seq[:, :-1], memory, mask)), -1)
        total = logp[0, torch.arange(len(ids) + 1), seq[0, 1:]].sum().item()
        return total / length_penalty(len(ids) + 1, alpha)

    greedy = [strip_after_eos(i, EOS) for i in greedy_decode(model, src, BOS, EOS, max_len=12).tolist()]
    beam = beam_search(model, src, BOS, EOS, beam_size=4, alpha=0.6, max_len=12)
    for b in range(src.size(0)):
        if len(greedy[b]) < 12:  # greedy ended with </s>, so both are complete hypotheses
            assert norm_score(beam[b], b, 0.6) >= norm_score(greedy[b], b, 0.6) - 1e-4


def test_beam_restricts_output_ids():
    model = small_model()
    allowed = list(range(3, 15))
    out = beam_search(model, random_src(), BOS, EOS, beam_size=4, max_len=10, allowed_ids=allowed)
    assert all(t in allowed for ids in out for t in ids)


# ---- Token batching -----------------------------------------------------------

def test_token_batches_cover_every_pair_once_and_fit():
    g = torch.Generator().manual_seed(0)
    lengths = [(int(a), int(b)) for a, b in torch.randint(3, 40, (500, 2), generator=g)]
    sampler = TokenBatchSampler(lengths, max_tokens=300, seed=1)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(500))
    for b in batches:
        assert len(b) * max(max(lengths[i]) for i in b) <= 300
    assert len(batches) == len(sampler)


def test_token_batches_reshuffle_per_epoch_deterministically():
    lengths = [(5 + i % 20, 6 + i % 17) for i in range(200)]
    s = TokenBatchSampler(lengths, max_tokens=120, seed=3)
    s.set_epoch(1); a = list(s)
    s.set_epoch(1); b = list(s)
    s.set_epoch(2); c = list(s)
    assert a == b and a != c


def test_token_batch_too_small_errors():
    with pytest.raises(ValueError):
        TokenBatchSampler([(10, 50)], max_tokens=20)


# ---- Checkpoint averaging -----------------------------------------------------

def test_average_checkpoints(tmp_path):
    states = [{"w": torch.full((2,), float(v)), "count": torch.tensor(v)} for v in (1, 2, 6)]
    for step, st in zip((100, 200, 300), states):
        save_step_checkpoint(tmp_path, step, {"model_state": st, "step": step}, keep_last=2)
    paths = step_checkpoints(tmp_path)
    assert [p.name for p in paths] == ["step_0000200.pt", "step_0000300.pt"]  # oldest deleted
    avg = average_checkpoints(paths)
    assert torch.allclose(avg["model_state"]["w"], torch.full((2,), 4.0))
    assert avg["model_state"]["count"].item() == 2  # non-float buffers come from the first
    assert avg["step"] == 300 and avg["averaged_from"] == ["step_0000200.pt", "step_0000300.pt"]


def test_ffn_dropout_switch():
    def ffn_p(model):
        return model.encoder.layers[0].ff.net[2].p
    assert ffn_p(small_model(dropout=0.1)) == 0.1                   # old models: same as dropout
    assert ffn_p(small_model(dropout=0.1, ffn_dropout=0.0)) == 0.0  # paper
    paper_inference = build_model({**SMALL, "dropout": 0.1, "ffn_dropout": 0.3}, VOCAB, PAD, dropout=0.0)
    assert ffn_p(paper_inference) == 0.0                             # inference turns every dropout off

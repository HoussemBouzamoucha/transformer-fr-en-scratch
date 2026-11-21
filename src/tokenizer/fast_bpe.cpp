// Multithreaded C++ encoder for the joint BPE in bpe.py.
//
// Mirrors BPETokenizer.tokenize()/encode() exactly (tests/test_fast_bpe.py checks
// it token for token): whitespace split -> _pieces() (punctuation, language-aware
// apostrophes, join marks) -> _segment() (merges by rank) -> _split() (vocab
// threshold) -> ids. Of normalize(), NFC stays in Python (one call per batch); the
// character map (_MAP) and control-character removal (_CONTROL) are done here, with
// both tables read from normalize.py. Collapsing spaces doesn't change the split.
//
// Everything the Python side knows is handed over as plain arrays, so this file
// has no dependencies: strings are interned to ids, and the character classes
// (Python's \w, \s, str.isspace, _wordchar) come as a 0x110000-entry table built
// from Python itself, so Unicode rules can't drift between the two.
//
// Built on first use by fast_bpe.py (g++/clang++), called through ctypes.

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

using u32 = uint32_t;
using i32 = int32_t;
using i64 = int64_t;
using ustr = std::u32string;

namespace {

constexpr u32 N_CP = 0x110000;
constexpr uint8_t RE_W = 1;    // matches re's \w
constexpr uint8_t RE_S = 2;    // matches re's \s
constexpr uint8_t SPACE = 4;   // str.isspace() (what str.split() splits on)
constexpr uint8_t WORDC = 8;   // bpe._wordchar(): isalnum() or "_"
constexpr uint8_t CTRL = 16;   // removed by normalize() (_CONTROL)

struct UHash {
    size_t operator()(const ustr& s) const noexcept {
        uint64_t h = 1469598103934665603ULL;  // FNV-1a
        for (char32_t c : s) { h ^= c; h *= 1099511628211ULL; }
        return static_cast<size_t>(h);
    }
};

using Cache = std::unordered_map<ustr, std::vector<i32>, UHash>;

struct Model {
    std::vector<uint8_t> cls;              // N_CP character classes
    std::unordered_map<u32, i32> char_id;  // one-character string -> string id
    std::unordered_map<u32, std::vector<u32>> char_map;  // normalize._MAP
    std::vector<i32> vocab_id;             // string id -> token id, -1 = not in vocab
    std::vector<i32> split_a, split_b;     // string id -> the pair that first produced it
    std::unordered_map<uint64_t, i32> rank;  // (a, b) -> merge rank (last one wins, like a dict)
    std::vector<i32> merge_a, merge_b, merge_out;
    i32 eow = -1, unk = 0, pretok = 2;
    u32 glue = 0;

    std::mutex mu;  // one encode call at a time per model (caches are per model)
    std::unordered_map<i64, std::vector<std::unique_ptr<Cache>>> caches;  // (lang, mask) -> per thread

    uint8_t c(u32 cp) const { return cp < N_CP ? cls[cp] : 0; }
};

struct Result {
    std::vector<i32> ids;
    std::vector<i64> offs;  // n_sentences + 1
};

inline uint64_t pair_key(i32 a, i32 b) {
    return (static_cast<uint64_t>(static_cast<u32>(a)) << 32) | static_cast<u32>(b);
}

void decode_utf8(const unsigned char* s, size_t n, std::vector<u32>& out) {
    out.clear();
    size_t i = 0;
    while (i < n) {
        unsigned char b = s[i];
        if (b < 0x80) {
            out.push_back(b); i += 1;
        } else if ((b >> 5) == 0x6 && i + 1 < n) {
            out.push_back((u32(b & 0x1F) << 6) | (s[i + 1] & 0x3F)); i += 2;
        } else if ((b >> 4) == 0xE && i + 2 < n) {
            out.push_back((u32(b & 0x0F) << 12) | (u32(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F)); i += 3;
        } else if ((b >> 3) == 0x1E && i + 3 < n) {
            out.push_back((u32(b & 0x07) << 18) | (u32(s[i + 1] & 0x3F) << 12) |
                          (u32(s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F));
            i += 4;
        } else {  // never produced by Python's encoder
            out.push_back(0xFFFD); i += 1;
        }
    }
}

// bpe._is_word
inline bool is_word(const Model& m, const ustr& p) {
    return !p.empty() && (m.c(p.front()) & WORDC) && (m.c(p.back()) & WORDC);
}

// BPETokenizer._pieces. lang: 1 = fr, 2 = en, 0 = anything else.
void pieces(const Model& m, const u32* w, size_t n, int lang, std::vector<ustr>& out) {
    out.clear();
    if (m.pretok == 0) { out.emplace_back(w, w + n); return; }

    ustr s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if (w[i] != m.glue) s.push_back(w[i]);

    std::vector<ustr> parts;  // re.findall(r"\w+|[^\w\s]")
    for (size_t i = 0; i < s.size();) {
        uint8_t k = m.c(s[i]);
        if (k & RE_W) {
            size_t j = i + 1;
            while (j < s.size() && (m.c(s[j]) & RE_W)) ++j;
            parts.emplace_back(s, i, j - i);
            i = j;
        } else {
            if (!(k & RE_S)) parts.emplace_back(1, s[i]);
            ++i;
        }
    }
    if (parts.size() <= 1) { out = std::move(parts); return; }

    if (lang == 1 || lang == 2) {
        std::vector<ustr> merged;
        size_t i = 0;
        while (i < parts.size()) {
            const ustr& p = parts[i];
            if (p.size() == 1 && p[0] == U'\'' && !merged.empty() && is_word(m, merged.back()) &&
                i + 1 < parts.size() && is_word(m, parts[i + 1])) {
                if (lang == 1) {         // French elision: "l'" "herbe"
                    merged.back().push_back(U'\'');
                    i += 1;
                } else {                 // English clitic: "man" "'s"
                    ustr q(1, U'\'');
                    q += parts[i + 1];
                    merged.push_back(std::move(q));
                    i += 2;
                }
                continue;
            }
            merged.push_back(p);
            i += 1;
        }
        parts = std::move(merged);
    }

    out = parts;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        if (!(m.c(parts[i + 1][0]) & WORDC)) out[i + 1].insert(out[i + 1].begin(), m.glue);
        else out[i].push_back(m.glue);
    }
}

// BPETokenizer._segment: apply the lowest-rank merge everywhere, repeat.
void segment(const Model& m, const ustr& piece, std::vector<i32>& syms, std::vector<i32>& tmp) {
    syms.clear();
    for (char32_t ch : piece) {
        auto it = m.char_id.find(ch);
        syms.push_back(it == m.char_id.end() ? -1 : it->second);  // -1: unseen character
    }
    syms.push_back(m.eow);
    while (syms.size() > 1) {
        i32 best = INT_MAX;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            if (syms[i] < 0 || syms[i + 1] < 0) continue;
            auto it = m.rank.find(pair_key(syms[i], syms[i + 1]));
            if (it != m.rank.end() && it->second < best) best = it->second;
        }
        if (best == INT_MAX) break;
        const i32 a = m.merge_a[best], b = m.merge_b[best], o = m.merge_out[best];
        tmp.clear();
        for (size_t i = 0; i < syms.size();) {
            if (i + 1 < syms.size() && syms[i] == a && syms[i + 1] == b) { tmp.push_back(o); i += 2; }
            else tmp.push_back(syms[i++]);
        }
        syms.swap(tmp);
    }
}

// BPETokenizer._split, then vocab.get(token, unk).
void split_emit(const Model& m, i32 t, const uint8_t* allowed, std::vector<i32>& out) {
    if (t < 0) { out.push_back(m.unk); return; }
    if (m.split_a[t] < 0 || (m.vocab_id[t] >= 0 && (!allowed || allowed[t]))) {
        out.push_back(m.vocab_id[t] >= 0 ? m.vocab_id[t] : m.unk);
        return;
    }
    split_emit(m, m.split_a[t], allowed, out);
    split_emit(m, m.split_b[t], allowed, out);
}

struct Worker {
    std::vector<ustr> ps;
    std::vector<i32> syms, tmp, word_ids;
    std::vector<u32> cps, norm;
};

// normalize() after NFC: text.translate(_MAP), then _CONTROL.sub("", text).
void apply_map(const Model& m, const std::vector<u32>& in, std::vector<u32>& out) {
    out.clear();
    for (u32 c : in) {
        auto it = m.char_map.find(c);
        if (it == m.char_map.end()) {
            if (!(m.c(c) & CTRL)) out.push_back(c);
        } else {
            for (u32 d : it->second)
                if (!(m.c(d) & CTRL)) out.push_back(d);
        }
    }
}

void encode_word(const Model& m, const u32* w, size_t n, int lang, const uint8_t* allowed,
                 Cache& cache, size_t cache_cap, Worker& wk, std::vector<i32>& out) {
    ustr key(w, w + n);
    auto it = cache.find(key);
    if (it != cache.end()) {
        out.insert(out.end(), it->second.begin(), it->second.end());
        return;
    }
    wk.word_ids.clear();
    pieces(m, w, n, lang, wk.ps);
    for (const ustr& p : wk.ps) {
        segment(m, p, wk.syms, wk.tmp);
        for (i32 t : wk.syms) split_emit(m, t, allowed, wk.word_ids);
    }
    out.insert(out.end(), wk.word_ids.begin(), wk.word_ids.end());
    if (cache.size() < cache_cap) cache.emplace(std::move(key), wk.word_ids);
}

void encode_sentence(const Model& m, const u32* s, size_t n, int lang, const uint8_t* allowed,
                     Cache& cache, size_t cache_cap, Worker& wk, std::vector<i32>& out) {
    size_t i = 0;
    while (i < n) {
        while (i < n && (m.c(s[i]) & SPACE)) ++i;
        size_t j = i;
        while (j < n && !(m.c(s[j]) & SPACE)) ++j;
        if (j > i) encode_word(m, s + i, j - i, lang, allowed, cache, cache_cap, wk, out);
        i = j;
    }
}

}  // namespace

// strings: n_str interned strings as code points, string k = cps[offs[k] : offs[k+1]].
// merges: 3 * n_merges string ids (a, b, a+b), in rank order.
API void* bpe_new(const u32* cps, const i64* offs, i32 n_str, const i32* vocab_id,
                  const i32* split_a, const i32* split_b, const i32* merges, i32 n_merges,
                  const uint8_t* cls, i32 eow, i32 unk, u32 glue, i32 pretok,
                  const u32* map_from, const i64* map_offs, const u32* map_cps, i32 n_map) {
    auto* m = new Model();
    for (i32 k = 0; k < n_map; ++k)
        m->char_map[map_from[k]].assign(map_cps + map_offs[k], map_cps + map_offs[k + 1]);
    m->cls.assign(cls, cls + N_CP);
    m->vocab_id.assign(vocab_id, vocab_id + n_str);
    m->split_a.assign(split_a, split_a + n_str);
    m->split_b.assign(split_b, split_b + n_str);
    for (i32 k = 0; k < n_str; ++k)
        if (offs[k + 1] - offs[k] == 1) m->char_id[cps[offs[k]]] = k;
    m->merge_a.resize(n_merges);
    m->merge_b.resize(n_merges);
    m->merge_out.resize(n_merges);
    m->rank.reserve(static_cast<size_t>(n_merges) * 2);
    for (i32 r = 0; r < n_merges; ++r) {
        m->merge_a[r] = merges[3 * r];
        m->merge_b[r] = merges[3 * r + 1];
        m->merge_out[r] = merges[3 * r + 2];
        m->rank[pair_key(m->merge_a[r], m->merge_b[r])] = r;
    }
    m->eow = eow;
    m->unk = unk;
    m->glue = glue;
    m->pretok = pretok;
    return m;
}

API void bpe_free(void* h) { delete static_cast<Model*>(h); }

// Encode sentences given as one UTF-8 buffer separated by '\n' (no '\n' inside a
// sentence). allowed: one byte per string id (threshold split) or null.
// normalize: apply _MAP/_CONTROL (the text must already be NFC).
// cache_key identifies (lang, allowed) so the per-thread word caches can be kept
// across calls.
API void* bpe_encode(void* h, const char* buf, i64 n_bytes, i32 lang, const uint8_t* allowed,
                     i64 cache_key, i32 normalize, i32 n_threads) {
    Model& m = *static_cast<Model*>(h);
    std::lock_guard<std::mutex> lock(m.mu);

    std::vector<i64> starts{0};
    for (const char* p = buf; (p = static_cast<const char*>(std::memchr(p, '\n', buf + n_bytes - p)));
         ++p)
        starts.push_back(p - buf + 1);
    const size_t n_sent = starts.size();
    starts.push_back(n_bytes + 1);  // sentence k = buf[starts[k] : starts[k+1] - 1]

    constexpr size_t CHUNK = 256;
    const size_t n_chunks = (n_sent + CHUNK - 1) / CHUNK;
    size_t T = static_cast<size_t>(std::max<i32>(1, n_threads));
    T = std::max<size_t>(1, std::min(T, n_chunks));

    auto& caches = m.caches[cache_key];
    while (caches.size() < T) caches.emplace_back(new Cache());
    const size_t cache_cap = std::max<size_t>(100000, 2000000 / caches.size());

    std::vector<std::vector<i32>> chunk_ids(n_chunks);
    std::vector<std::vector<i64>> chunk_lens(n_chunks);
    std::atomic<size_t> next{0};
    auto work = [&](size_t t) {
        Worker wk;
        Cache& cache = *caches[t];
        for (;;) {
            size_t c = next.fetch_add(1);
            if (c >= n_chunks) break;
            auto& ids = chunk_ids[c];
            auto& lens = chunk_lens[c];
            const size_t end = std::min(n_sent, (c + 1) * CHUNK);
            for (size_t k = c * CHUNK; k < end; ++k) {
                const i64 a = starts[k], b = starts[k + 1] - 1;
                decode_utf8(reinterpret_cast<const unsigned char*>(buf) + a, static_cast<size_t>(b - a), wk.cps);
                const size_t before = ids.size();
                const std::vector<u32>* text = &wk.cps;
                if (normalize) { apply_map(m, wk.cps, wk.norm); text = &wk.norm; }
                encode_sentence(m, text->data(), text->size(), lang, allowed, cache, cache_cap, wk, ids);
                lens.push_back(static_cast<i64>(ids.size() - before));
            }
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 1; t < T; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& th : pool) th.join();

    auto* r = new Result();
    size_t total = 0;
    for (auto& v : chunk_ids) total += v.size();
    r->ids.reserve(total);
    r->offs.reserve(n_sent + 1);
    r->offs.push_back(0);
    for (size_t c = 0; c < n_chunks; ++c) {
        r->ids.insert(r->ids.end(), chunk_ids[c].begin(), chunk_ids[c].end());
        for (i64 len : chunk_lens[c]) r->offs.push_back(r->offs.back() + len);
    }
    return r;
}

API i64 res_n_sent(void* r) { return static_cast<i64>(static_cast<Result*>(r)->offs.size()) - 1; }
API i64 res_n_ids(void* r) { return static_cast<i64>(static_cast<Result*>(r)->ids.size()); }
API void res_copy(void* r, i32* ids, i64* offs) {
    auto* res = static_cast<Result*>(r);
    if (!res->ids.empty()) std::memcpy(ids, res->ids.data(), res->ids.size() * sizeof(i32));
    std::memcpy(offs, res->offs.data(), res->offs.size() * sizeof(i64));
}
API void res_free(void* r) { delete static_cast<Result*>(r); }


// =========================================================================
// Training: BPETokenizer.train() in C++ (fast_bpe.py FastTrainer drives it).
//
//   tr_count_text   _piece_counts(): pre-tokenize + count, multithreaded; the
//                   per-thread blocks are merged in sentence order, so pieces
//                   keep Python's first-seen order (it ends up in the saved JSON)
//   tr_merge        the merge loop, single-threaded (each merge depends on the
//                   last): incremental pair counts + max-heap with lazy deletion,
//                   ties broken like Python's heap of (-count, (a, b)): highest
//                   count, then the smallest (a, b) by code points
//   tr_fit          _fit_vocab() + _count_tokens(), multithreaded; each prefix is
//                   segmented once and reused across the fixpoint passes
//   tr_count_vocab  _count_tokens() with every merge (classic BPE, no fitting)
// Symbols (characters, </w>, merge results) are interned strings; Python reads
// them back with tr_syms. tests/test_fast_bpe.py checks the saved vocab file is
// byte-identical to the one the Python train() writes.
// =========================================================================
namespace {

struct OrderedCounts {  // collections.Counter: insertion-ordered string counts
    std::vector<ustr> keys;
    std::vector<i64> vals;
    std::unordered_map<ustr, i32, UHash> index;
    void add(const ustr& k, i64 v) {
        auto it = index.find(k);
        if (it == index.end()) {
            index.emplace(k, static_cast<i32>(keys.size()));
            keys.push_back(k);
            vals.push_back(v);
        } else {
            vals[it->second] += v;
        }
    }
};

using TokCounts = std::vector<std::pair<i32, i64>>;  // first-seen order

struct Fitted {
    std::vector<char> kept;         // per symbol
    std::vector<TokCounts> counts;  // per fit slot
};

struct Trainer {
    const Model* pre = nullptr;  // character classes, _MAP, glue, pretokenize
    std::vector<OrderedCounts> slots;
    std::vector<ustr> sym;
    std::unordered_map<ustr, i32, UHash> sym_id;
    std::vector<i32> split_a, split_b;  // first pair that made each symbol (split_of)
    std::vector<i32> m_a, m_b, m_out;
    std::vector<i64> m_count;
    i32 eow = -1;
    std::vector<u32> char_cp;
    std::vector<i64> char_n;
    i64 fit_k = 0;
    std::vector<i32> kept_ids;
    std::vector<TokCounts> counts;

    i32 intern(const ustr& s) {
        auto it = sym_id.find(s);
        if (it != sym_id.end()) return it->second;
        const i32 id = static_cast<i32>(sym.size());
        sym.push_back(s);
        sym_id.emplace(s, id);
        split_a.push_back(-1);
        split_b.push_back(-1);
        return id;
    }
};

std::vector<i64> line_starts(const char* buf, i64 n_bytes) {
    std::vector<i64> starts{0};
    for (const char* p = buf; (p = static_cast<const char*>(std::memchr(p, '\n', buf + n_bytes - p))); ++p)
        starts.push_back(p - buf + 1);
    starts.push_back(n_bytes + 1);  // line k = buf[starts[k] : starts[k+1] - 1]
    return starts;
}

template <class F>
void parallel_ranges(size_t n, size_t T, F fn) {  // fn(begin, end, thread)
    T = std::max<size_t>(1, std::min(T, n));
    std::vector<std::thread> pool;
    for (size_t t = 1; t < T; ++t) pool.emplace_back(fn, n * t / T, n * (t + 1) / T, t);
    fn(0, n / T, 0);
    for (auto& th : pool) th.join();
}

struct HeapEntry { i64 c; i32 a, b; };

}  // namespace

API void* tr_new(void* pre) {
    auto* tr = new Trainer();
    tr->pre = static_cast<Model*>(pre);
    return tr;
}

API void tr_free(void* h) { delete static_cast<Trainer*>(h); }

API i32 tr_intern(void* h, const u32* cps, i64 n) {
    return static_cast<Trainer*>(h)->intern(ustr(cps, cps + n));
}

// _piece_counts(lines, lang) for slot `slot`: lines joined by '\n', NFC already
// applied when normalize (the rest of normalize() is done here, as in encoding).
API void tr_count_text(void* h, i32 slot, const char* buf, i64 n_bytes, i32 lang, i32 normalize,
                       i32 n_threads) {
    Trainer& tr = *static_cast<Trainer*>(h);
    const Model& m = *tr.pre;
    if (static_cast<size_t>(slot) >= tr.slots.size()) tr.slots.resize(slot + 1);
    OrderedCounts& global = tr.slots[slot];
    const std::vector<i64> starts = line_starts(buf, n_bytes);
    const size_t n_sent = starts.size() - 1;

    constexpr size_t CHUNK = 16384;
    const size_t n_chunks = (n_sent + CHUNK - 1) / CHUNK;
    const size_t T = static_cast<size_t>(std::max<i32>(1, n_threads));
    for (size_t round = 0; round < n_chunks; round += T) {  // T chunks in parallel, merged in order
        const size_t nr = std::min(T, n_chunks - round);
        std::vector<OrderedCounts> local(nr);
        auto work = [&](size_t j) {
            Worker wk;
            const size_t c = round + j, end = std::min(n_sent, (c + 1) * CHUNK);
            for (size_t k = c * CHUNK; k < end; ++k) {
                decode_utf8(reinterpret_cast<const unsigned char*>(buf) + starts[k],
                            static_cast<size_t>(starts[k + 1] - 1 - starts[k]), wk.cps);
                const std::vector<u32>* text = &wk.cps;
                if (normalize) { apply_map(m, wk.cps, wk.norm); text = &wk.norm; }
                const u32* s = text->data();
                const size_t n = text->size();
                for (size_t i = 0; i < n;) {
                    while (i < n && (m.c(s[i]) & SPACE)) ++i;
                    size_t e = i;
                    while (e < n && !(m.c(s[e]) & SPACE)) ++e;
                    if (e > i) {
                        pieces(m, s + i, e - i, lang, wk.ps);
                        for (const ustr& p : wk.ps) local[j].add(p, 1);
                    }
                    i = e;
                }
            }
        };
        std::vector<std::thread> pool;
        for (size_t j = 1; j < nr; ++j) pool.emplace_back(work, j);
        work(0);
        for (auto& th : pool) th.join();
        for (const OrderedCounts& oc : local)
            for (size_t i = 0; i < oc.keys.size(); ++i) global.add(oc.keys[i], oc.vals[i]);
    }
}

// Character counts over every slot (the joint corpus), for the rare-character set.
API i64 tr_char_counts(void* h) {
    Trainer& tr = *static_cast<Trainer*>(h);
    std::unordered_map<u32, i64> cnt;
    for (const OrderedCounts& oc : tr.slots)
        for (size_t i = 0; i < oc.keys.size(); ++i)
            for (char32_t ch : oc.keys[i]) cnt[ch] += oc.vals[i];
    tr.char_cp.clear();
    tr.char_n.clear();
    for (auto& kv : cnt) { tr.char_cp.push_back(kv.first); tr.char_n.push_back(kv.second); }
    return static_cast<i64>(tr.char_cp.size());
}

API void tr_chars(void* h, u32* cps, i64* counts) {
    Trainer& tr = *static_cast<Trainer*>(h);
    std::copy(tr.char_cp.begin(), tr.char_cp.end(), cps);
    std::copy(tr.char_n.begin(), tr.char_n.end(), counts);
}

// The merge loop of train(). rare: characters that never merge. eow: id of "</w>".
// Stops after n_new_max new tokens or when the best pair is seen < 2 times.
API i64 tr_merge(void* h, const u32* rare_cps, i32 n_rare, i32 eow, i64 n_new_max) {
    Trainer& tr = *static_cast<Trainer*>(h);
    tr.eow = eow;
    std::unordered_map<ustr, i64, UHash> joint;
    for (const OrderedCounts& oc : tr.slots)
        for (size_t i = 0; i < oc.keys.size(); ++i) joint[oc.keys[i]] += oc.vals[i];

    std::unordered_map<u32, i32> char_sym;
    auto char_id = [&](u32 c) {
        auto it = char_sym.find(c);
        if (it != char_sym.end()) return it->second;
        const i32 id = tr.intern(ustr(1, c));
        char_sym.emplace(c, id);
        return id;
    };
    std::vector<std::vector<i32>> words;
    std::vector<i64> freq;
    words.reserve(joint.size());
    freq.reserve(joint.size());
    for (auto& kv : joint) {
        std::vector<i32> w;
        w.reserve(kv.first.size() + 1);
        for (char32_t c : kv.first) w.push_back(char_id(c));
        w.push_back(eow);
        words.push_back(std::move(w));
        freq.push_back(kv.second);
    }
    joint.clear();
    std::vector<char> rare_sym;
    for (i32 k = 0; k < n_rare; ++k) {
        auto it = char_sym.find(rare_cps[k]);
        if (it == char_sym.end()) continue;
        if (rare_sym.size() <= static_cast<size_t>(it->second)) rare_sym.resize(it->second + 1, 0);
        rare_sym[it->second] = 1;
    }
    auto ok = [&](i32 x, i32 y) {  // pairs_of(): no pair touches a rare character
        return !((static_cast<size_t>(x) < rare_sym.size() && rare_sym[x]) ||
                 (static_cast<size_t>(y) < rare_sym.size() && rare_sym[y]));
    };

    std::unordered_map<uint64_t, i64> pc;
    std::unordered_map<uint64_t, std::vector<i32>> where;
    for (size_t wi = 0; wi < words.size(); ++wi) {
        const auto& w = words[wi];
        for (size_t i = 0; i + 1 < w.size(); ++i) {
            if (!ok(w[i], w[i + 1])) continue;
            const uint64_t k = pair_key(w[i], w[i + 1]);
            pc[k] += freq[wi];
            auto& v = where[k];
            if (v.empty() || v.back() != static_cast<i32>(wi)) v.push_back(static_cast<i32>(wi));
        }
    }

    const std::vector<ustr>& S = tr.sym;
    auto less = [&S](const HeapEntry& x, const HeapEntry& y) {  // max-heap order
        if (x.c != y.c) return x.c < y.c;
        const int ca = S[x.a].compare(S[y.a]);
        if (ca != 0) return ca > 0;
        return S[x.b].compare(S[y.b]) > 0;
    };
    std::vector<HeapEntry> heap;
    heap.reserve(pc.size());
    for (auto& kv : pc)
        heap.push_back({kv.second, static_cast<i32>(kv.first >> 32), static_cast<i32>(kv.first & 0xFFFFFFFFu)});
    std::make_heap(heap.begin(), heap.end(), less);

    std::vector<i64> stamp(words.size(), -1);
    std::vector<uint64_t> changed;
    std::vector<i32> nw;
    i64 n_new = 0;
    while (n_new < n_new_max && !heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), less);
        const HeapEntry e = heap.back();
        heap.pop_back();
        const uint64_t key = pair_key(e.a, e.b);
        auto pit = pc.find(key);
        const i64 count = pit == pc.end() ? 0 : pit->second;
        if (e.c != count) continue;  // stale
        if (count < 2) break;

        const i32 a = e.a, b = e.b;
        const ustr merged = S[a] + S[b];
        const i32 o = tr.intern(merged);
        const i64 step = static_cast<i64>(tr.m_a.size());
        tr.m_a.push_back(a);
        tr.m_b.push_back(b);
        tr.m_out.push_back(o);
        tr.m_count.push_back(count);
        if (tr.split_a[o] < 0) {  // first time this string is made (split_of.setdefault)
            tr.split_a[o] = a;
            tr.split_b[o] = b;
            ++n_new;
        }

        std::vector<i32> list;
        auto wit = where.find(key);
        if (wit != where.end()) { list = std::move(wit->second); where.erase(wit); }
        changed.clear();
        for (i32 wi : list) {
            if (stamp[wi] == step) continue;
            stamp[wi] = step;
            auto& w = words[wi];
            bool has = false;  // a word without the pair would net to zero: skip it
            for (size_t i = 0; i + 1 < w.size(); ++i)
                if (w[i] == a && w[i + 1] == b) { has = true; break; }
            if (!has) continue;
            const i64 f = freq[wi];
            for (size_t i = 0; i + 1 < w.size(); ++i) {
                if (!ok(w[i], w[i + 1])) continue;
                const uint64_t k = pair_key(w[i], w[i + 1]);
                pc[k] -= f;
                changed.push_back(k);
            }
            nw.clear();
            for (size_t i = 0; i < w.size();) {
                if (i + 1 < w.size() && w[i] == a && w[i + 1] == b) { nw.push_back(o); i += 2; }
                else nw.push_back(w[i++]);
            }
            for (size_t i = 0; i + 1 < nw.size(); ++i) {
                if (!ok(nw[i], nw[i + 1])) continue;
                const uint64_t k = pair_key(nw[i], nw[i + 1]);
                pc[k] += f;
                changed.push_back(k);
                if (nw[i] == o || nw[i + 1] == o) {  // other pairs were already in the word
                    auto& v = where[k];
                    if (v.empty() || v.back() != wi) v.push_back(wi);
                }
            }
            w.swap(nw);
        }
        pc.erase(key);
        std::sort(changed.begin(), changed.end());
        changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
        for (uint64_t k : changed) {
            auto it = pc.find(k);
            const i64 c = it == pc.end() ? 0 : it->second;
            if (c > 0) {
                heap.push_back({c, static_cast<i32>(k >> 32), static_cast<i32>(k & 0xFFFFFFFFu)});
                std::push_heap(heap.begin(), heap.end(), less);
            } else if (it != pc.end()) {
                pc.erase(it);
            }
        }
    }
    return static_cast<i64>(tr.m_a.size());
}

API void tr_merges(void* h, i32* a, i32* b, i32* out, i64* counts) {
    Trainer& tr = *static_cast<Trainer*>(h);
    std::copy(tr.m_a.begin(), tr.m_a.end(), a);
    std::copy(tr.m_b.begin(), tr.m_b.end(), b);
    std::copy(tr.m_out.begin(), tr.m_out.end(), out);
    std::copy(tr.m_count.begin(), tr.m_count.end(), counts);
}

namespace {

struct FitState {
    Trainer& tr;
    std::vector<i32> slots;
    size_t T;
    std::unordered_map<uint64_t, std::vector<i32>> ranks;  // pair -> every rank it has
    std::vector<std::vector<std::vector<i32>>> chars;      // slot -> piece -> symbols
    std::vector<std::vector<std::vector<i32>>> segs;       // slot -> piece -> segmentation

    FitState(Trainer& t, const i32* s, i32 n, i32 n_threads) : tr(t), slots(s, s + n) {
        T = static_cast<size_t>(std::max<i32>(1, n_threads));
        for (size_t r = 0; r < tr.m_a.size(); ++r)
            ranks[pair_key(tr.m_a[r], tr.m_b[r])].push_back(static_cast<i32>(r));
        for (i32 sl : slots) {
            const OrderedCounts& oc = tr.slots[sl];
            std::vector<std::vector<i32>> cs(oc.keys.size());
            for (size_t i = 0; i < oc.keys.size(); ++i) {
                for (char32_t c : oc.keys[i]) cs[i].push_back(tr.sym_id.at(ustr(1, c)));
                cs[i].push_back(tr.eow);
            }
            chars.push_back(std::move(cs));
            segs.emplace_back(oc.keys.size());
        }
    }

    // _segment(piece, ranks) with ranks = {pair: r for r, pair in enumerate(merges[:k])}
    // (so a pair merged twice uses its LAST rank below k, like the dict).
    void segment(const std::vector<i32>& in, i64 k, std::vector<i32>& syms, std::vector<i32>& tmp) const {
        syms = in;
        while (syms.size() > 1) {
            i32 best = INT_MAX;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                auto it = ranks.find(pair_key(syms[i], syms[i + 1]));
                if (it == ranks.end() || it->second[0] >= k) continue;
                const auto& v = it->second;
                const i32 r = v.size() == 1 ? v[0]
                                            : *(std::lower_bound(v.begin(), v.end(), static_cast<i32>(k)) - 1);
                if (r < best) best = r;
            }
            if (best == INT_MAX) break;
            const i32 a = tr.m_a[best], b = tr.m_b[best], o = tr.m_out[best];
            tmp.clear();
            for (size_t i = 0; i < syms.size();) {
                if (i + 1 < syms.size() && syms[i] == a && syms[i + 1] == b) { tmp.push_back(o); i += 2; }
                else tmp.push_back(syms[i++]);
            }
            syms.swap(tmp);
        }
    }

    void segment_all(i64 k) {
        for (size_t s = 0; s < slots.size(); ++s) {
            auto& cs = chars[s];
            auto& out = segs[s];
            parallel_ranges(cs.size(), T, [&](size_t b, size_t e, size_t) {
                std::vector<i32> tmp;
                for (size_t i = b; i < e; ++i) segment(cs[i], k, out[i], tmp);
            });
        }
    }

    // _count_tokens over the current segmentations. vocab: per symbol;
    // allowed: per slot, per symbol, or empty for no threshold.
    std::vector<TokCounts> count(const std::vector<char>& vocab,
                                 const std::vector<std::vector<char>>& allowed) const {
        const size_t n_sym = tr.sym.size();
        std::vector<TokCounts> out(slots.size());
        for (size_t s = 0; s < slots.size(); ++s) {
            const OrderedCounts& oc = tr.slots[slots[s]];
            const char* al = allowed.empty() ? nullptr : allowed[s].data();
            const auto& sg = segs[s];
            const size_t n = sg.size();
            const size_t TT = std::max<size_t>(1, std::min(T, n));
            std::vector<std::vector<i64>> cnt(TT, std::vector<i64>(n_sym, 0));
            std::vector<std::vector<uint64_t>> first(TT, std::vector<uint64_t>(n_sym, UINT64_MAX));
            parallel_ranges(n, TT, [&](size_t b, size_t e, size_t t) {
                auto& c = cnt[t];
                auto& fs = first[t];
                std::vector<i32> stack;
                for (size_t i = b; i < e; ++i) {
                    const i64 f = oc.vals[i];
                    uint64_t pos = 0;
                    for (i32 tok : sg[i]) {  // _split, depth first, left part first
                        stack.assign(1, tok);
                        while (!stack.empty()) {
                            const i32 x = stack.back();
                            stack.pop_back();
                            if (tr.split_a[x] < 0 || (vocab[x] && (!al || al[x]))) {
                                if (vocab[x]) {
                                    c[x] += f;
                                    const uint64_t key = (static_cast<uint64_t>(i) << 24) |
                                                         std::min<uint64_t>(pos++, 0xFFFFFF);
                                    if (key < fs[x]) fs[x] = key;
                                }
                            } else {
                                stack.push_back(tr.split_b[x]);
                                stack.push_back(tr.split_a[x]);
                            }
                        }
                    }
                }
            });
            std::vector<std::pair<uint64_t, i32>> order;
            for (size_t x = 0; x < n_sym; ++x) {
                uint64_t fst = UINT64_MAX;
                for (size_t t = 0; t < TT; ++t) fst = std::min(fst, first[t][x]);
                if (fst != UINT64_MAX) order.push_back({fst, static_cast<i32>(x)});
            }
            std::sort(order.begin(), order.end());
            for (auto& pr : order) {
                i64 total = 0;
                for (size_t t = 0; t < TT; ++t) total += cnt[t][pr.second];
                out[s].push_back({pr.second, total});
            }
        }
        return out;
    }
};

}  // namespace

// _fit_vocab(lang_pieces of `slots`, vocab_size, threshold); n_fixed = specials +
// base characters; base: their symbol ids (always in the vocab). Returns k; the
// kept tokens and per-slot counts are read with tr_kept / tr_counts.
API i64 tr_fit(void* h, const i32* slots, i32 n_slots, const i32* base, i32 n_base, i64 vocab_size,
               i64 n_fixed, i64 threshold, i32 n_threads) {
    Trainer& tr = *static_cast<Trainer*>(h);
    FitState st(tr, slots, n_slots, n_threads);
    const size_t n_sym = tr.sym.size();
    std::vector<char> base_mask(n_sym, 0);
    for (i32 i = 0; i < n_base; ++i) base_mask[base[i]] = 1;
    std::unordered_map<i64, Fitted> cache;

    auto used = [&](i64 k) -> const Fitted& {
        auto hit = cache.find(k);
        if (hit != cache.end()) return hit->second;
        st.segment_all(k);
        std::vector<char> kept(n_sym, 0);
        for (i64 r = 0; r < k; ++r) kept[tr.m_out[r]] = 1;
        for (;;) {
            std::vector<char> vocab(n_sym);
            for (size_t x = 0; x < n_sym; ++x) vocab[x] = base_mask[x] | kept[x];
            std::vector<TokCounts> counts = st.count(vocab, {});
            std::vector<char> emitted(n_sym, 0);
            if (threshold > 0) {
                std::vector<std::vector<char>> allowed(st.slots.size(), std::vector<char>(n_sym, 0));
                for (size_t s = 0; s < counts.size(); ++s)
                    for (auto& tc : counts[s])
                        if (tc.second >= threshold) allowed[s][tc.first] = 1;
                for (auto& tcs : st.count(vocab, allowed))
                    for (auto& tc : tcs) emitted[tc.first] = 1;
            } else {
                for (auto& tcs : counts)
                    for (auto& tc : tcs) emitted[tc.first] = 1;
            }
            bool subset = true;
            for (size_t x = 0; x < n_sym; ++x)
                if (kept[x] && !emitted[x]) { subset = false; kept[x] = 0; }
            if (subset) return cache.emplace(k, Fitted{std::move(kept), std::move(counts)}).first->second;
        }
    };
    auto size_of = [](const Fitted& f) {
        i64 n = 0;
        for (char c : f.kept) n += c;
        return n;
    };

    i64 lo = 0, hi = static_cast<i64>(tr.m_a.size()), k;
    if (n_fixed + size_of(used(hi)) <= vocab_size) {
        k = hi;
    } else {
        while (hi - lo > 1) {  // invariant: used(lo) fits, used(hi) doesn't
            const i64 mid = (lo + hi) / 2;
            if (n_fixed + size_of(used(mid)) <= vocab_size) lo = mid;
            else hi = mid;
        }
        k = lo;
    }
    const Fitted& f = used(k);
    tr.fit_k = k;
    tr.kept_ids.clear();
    for (size_t x = 0; x < f.kept.size(); ++x)
        if (f.kept[x]) tr.kept_ids.push_back(static_cast<i32>(x));
    tr.counts = f.counts;
    return k;
}

// _count_tokens(lang_pieces of `slots`, vocab, every merge): classic BPE's counts.
API void tr_count_vocab(void* h, const i32* slots, i32 n_slots, const i32* vocab_ids, i32 n_vocab,
                        i32 n_threads) {
    Trainer& tr = *static_cast<Trainer*>(h);
    FitState st(tr, slots, n_slots, n_threads);
    std::vector<char> vocab(tr.sym.size(), 0);
    for (i32 i = 0; i < n_vocab; ++i) vocab[vocab_ids[i]] = 1;
    st.segment_all(static_cast<i64>(tr.m_a.size()));
    tr.counts = st.count(vocab, {});
    tr.kept_ids.clear();
}

API i64 tr_n_kept(void* h) { return static_cast<i64>(static_cast<Trainer*>(h)->kept_ids.size()); }
API void tr_kept(void* h, i32* out) {
    auto& v = static_cast<Trainer*>(h)->kept_ids;
    std::copy(v.begin(), v.end(), out);
}
API i64 tr_n_counts(void* h, i32 s) { return static_cast<i64>(static_cast<Trainer*>(h)->counts[s].size()); }
API void tr_counts(void* h, i32 s, i32* ids, i64* vals) {
    auto& v = static_cast<Trainer*>(h)->counts[s];
    for (size_t i = 0; i < v.size(); ++i) { ids[i] = v[i].first; vals[i] = v[i].second; }
}
API i64 tr_n_syms(void* h) { return static_cast<i64>(static_cast<Trainer*>(h)->sym.size()); }
API i64 tr_syms_len(void* h) {
    i64 n = 0;
    for (auto& s : static_cast<Trainer*>(h)->sym) n += static_cast<i64>(s.size());
    return n;
}
API void tr_syms(void* h, u32* cps, i64* offs) {
    auto& S = static_cast<Trainer*>(h)->sym;
    offs[0] = 0;
    for (size_t i = 0; i < S.size(); ++i) {
        std::copy(S[i].begin(), S[i].end(), cps + offs[i]);
        offs[i + 1] = offs[i] + static_cast<i64>(S[i].size());
    }
}

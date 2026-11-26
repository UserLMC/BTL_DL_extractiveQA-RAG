"""
nli_judge.py – Chấm ảo giác tự động bằng NLI (TV3, phần nhóm tự cài).

Thuật toán (mục 7 kế hoạch):
  1. Câu trả lời -> bỏ thẻ trích dẫn [n] -> tách thành câu h_1..h_m.
     Câu từ chối thì KHÔNG chấm NLI (tính riêng ở chỉ số từ chối).
  2. Với mỗi h_j: p_j = P(entailment | premise = ngữ cảnh, hypothesis = h_j).
     Ngữ cảnh 3–5 chunk có thể vượt 512 token -> 2 cách gộp:
        agg="max"    : chấm từng chunk riêng, p_j = max_c P(entail | chunk_c, h_j)
        agg="concat" : nối các chunk thành 1 premise (bị cắt nếu quá dài)
        agg="both"   : p_j = max(max, concat) - cứu các câu cần GHÉP nhiều Điều (qtype "multi"),
                       mà "max" từng chunk riêng sẽ đánh trượt oan
  3. h_j được coi là có căn cứ nếu p_j >= tau.
  4. Faithfulness(answer) = (#câu có căn cứ) / m.  Answer "unsupported" nếu faithfulness < 1.

Backend:
  - "hf"      : cross-encoder NLI đa ngôn ngữ (mặc định mDeBERTa-v3 fine-tune XNLI/multilingual-NLI)
  - "lexical" : xấp xỉ offline không cần GPU (độ phủ âm tiết + kiểm tra con số + phủ định).
                Chỉ dùng để kiểm thử / baseline, KHÔNG phải NLI thật.

Ngoài ra: calibrate() chọn (tau, agg) trên tập dev có nhãn người, fleiss_kappa(),
agreement_with_humans() cho câu hỏi nghiên cứu (4).

Ví dụ:
  python -m rag.hallucination.nli_judge judge --gen outputs/generations.jsonl \
      --chunks data/chunks.jsonl --out outputs/nli.jsonl --backend hf --tau 0.5 --agg max
  python -m rag.hallucination.nli_judge calibrate --dev tests/data/nli_dev.jsonl --backend hf
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rag.generation.prompts import is_refusal, strip_citations

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


# ---------------------------------------------------------------------------
# Tách câu tiếng Việt
# ---------------------------------------------------------------------------
_ABBR = ("TS.", "ThS.", "PGS.", "GS.", "Tp.", "TP.", "v.v.", "Đ.", "K.")


def split_sentences(text: str) -> List[str]:
    """
    Tách câu theo . ! ? ; và xuống dòng/gạch đầu dòng.
    Không tách ở số thập phân (2.5, 3,2), viết tắt (TS., ThS.), hay số hiệu văn bản (3626/QĐ-ĐHQGHN).
    """
    # bỏ trích dẫn theo từng dòng để GIỮ ký tự xuống dòng (gạch đầu dòng là ranh giới câu)
    t = "\n".join(strip_citations(line) for line in unicodedata.normalize("NFC", text or "").splitlines())
    for i, ab in enumerate(_ABBR):
        # (?<!\w): "K." / "Đ." chỉ là viết tắt khi đứng riêng, không phải đuôi của "HK." hay "NĐ."
        t = re.sub(r"(?<!\w)" + re.escape(ab), f"\x00{i}\x00", t)
    parts = re.split(r"(?<=[.!?;])\s+(?=[^\s\d])|\n+\s*(?:[-•*]|\d+[.)])?\s*", t)
    out = []
    for p in parts:
        for i, ab in enumerate(_ABBR):
            p = p.replace(f"\x00{i}\x00", ab)
        p = p.strip(" -•*\t")
        if len(re.findall(r"\w+", p)) >= 2:   # bỏ mảnh vụn 0–1 từ
            out.append(p)
    return out


QUESTION_MODES = ("none", "short", "always")
SHORT_WORDS = 8


def build_hypothesis(question: Optional[str], sentence: str, mode: str = "short") -> str:
    """
    Giả thuyết đưa vào NLI. Câu trả lời cụt ("đạt từ loại Tốt trở lên") không có chủ thể, nên NLI
    không biết nó nói về điều gì và chấm sai. Ghép câu hỏi vào để câu có đủ ngữ cảnh:
      none   : chỉ câu trả lời (hành vi cũ)
      short  : chỉ ghép câu hỏi khi câu trả lời ngắn (< SHORT_WORDS từ); mặc định
      always : luôn ghép
    """
    if mode not in QUESTION_MODES:
        raise ValueError(f"mode phải thuộc {QUESTION_MODES}")
    q = (question or "").strip()
    if mode == "none" or not q:
        return sentence
    if mode == "short" and len(re.findall(r"\w+", sentence)) >= SHORT_WORDS:
        return sentence
    return f"{q.rstrip('?').strip()}? {sentence}"


# ---------------------------------------------------------------------------
# Backend NLI
# ---------------------------------------------------------------------------
class HFNLI:
    def __init__(self, model_id: str = DEFAULT_NLI_MODEL, device: Optional[str] = None,
                 max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        ent = [i for name, i in labels.items() if name.startswith("entail")]
        if not ent:
            raise ValueError(f"Không tìm thấy nhãn entailment trong {labels}")
        self.entail_idx = ent[0]

    def entail_probs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        out: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                probs = self.model(**enc).logits.softmax(-1)[:, self.entail_idx]
            out.extend(probs.tolist())
        return out


def _tokens(s: str) -> List[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())


_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_STOP = set("là của và các có được cho trong theo với những này đó thì khi nào gì ra vào "
            "một bị đã sẽ cần phải để từ tại về như".split())
_NEG = {"không", "chưa", "chẳng", "cấm"}


class LexicalNLI:
    """
    Xấp xỉ entailment bằng luật – chỉ để chạy offline/kiểm thử.
      p = độ phủ âm tiết nội dung của hypothesis trong premise
      * 0.1 nếu hypothesis có con số không xuất hiện trong premise (sai số liệu)
      * 0.3 nếu hypothesis có từ phủ định mà premise không có (hoặc ngược lại ở cùng ngữ cảnh)
    """

    model_id = "lexical-approx"

    def entail_probs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        out = []
        for premise, hyp in pairs:
            ptoks, htoks = _tokens(premise), _tokens(hyp)
            pset = set(ptoks)
            content = [t for t in htoks if t not in _STOP and not t.isdigit()]
            cover = sum(t in pset for t in content) / len(content) if content else 0.0
            pnums = {n.replace(",", ".") for n in _NUM_RE.findall(premise)}
            hnums = {n.replace(",", ".") for n in _NUM_RE.findall(hyp)}
            if hnums - pnums:
                cover *= 0.1
            hneg, pneg = _NEG & set(htoks), _NEG & pset
            if hneg - pneg:
                cover *= 0.3
            out.append(round(cover, 4))
        return out


def make_nli(name: str, model_id: str = DEFAULT_NLI_MODEL):
    if name == "hf":
        return HFNLI(model_id)
    if name == "lexical":
        return LexicalNLI()
    raise ValueError(f"backend không hỗ trợ: {name}")


# ---------------------------------------------------------------------------
# Chấm
# ---------------------------------------------------------------------------
def sentence_scores(nli, hyps: List[str], chunk_texts: List[str], agg: str = "max") -> List[float]:
    if not hyps:
        return []
    if not chunk_texts:
        return [0.0] * len(hyps)          # closed-book: không có căn cứ trong ngữ cảnh
    if agg == "concat":
        premise = "\n".join(chunk_texts)
        return nli.entail_probs([(premise, h) for h in hyps])
    if agg == "max":
        pairs = [(c, h) for h in hyps for c in chunk_texts]
        probs = nli.entail_probs(pairs)
        n = len(chunk_texts)
        return [max(probs[i * n:(i + 1) * n]) for i in range(len(hyps))]
    if agg == "both":
        a = sentence_scores(nli, hyps, chunk_texts, "max")
        b = sentence_scores(nli, hyps, chunk_texts, "concat")
        return [max(x, y) for x, y in zip(a, b)]
    raise ValueError("agg phải là 'max', 'concat' hoặc 'both'")


def judge_answer(nli, answer: str, chunk_texts: List[str], tau: float = 0.5, agg: str = "max",
                 question: Optional[str] = None, question_mode: str = "short") -> Dict:
    if is_refusal(answer):
        return {"refusal": True, "sentences": [], "faithfulness": None, "supported": None}
    sentences = split_sentences(answer)
    hyps = [build_hypothesis(question, h, question_mode) for h in sentences]
    probs = sentence_scores(nli, hyps, chunk_texts, agg)
    sents = [{"text": h, "p_entail": round(p, 4), "entailed": p >= tau} for h, p in zip(sentences, probs)]
    faith = sum(s["entailed"] for s in sents) / len(sents) if sents else 0.0
    return {"refusal": False, "sentences": sents, "faithfulness": round(faith, 4),
            "supported": faith >= 1.0 - 1e-9}


def judge_file(nli, gens: List[Dict], chunk_index: Dict[str, Dict], tau: float, agg: str,
               gold_map: Optional[Dict[str, List[str]]] = None,
               gold_configs: Sequence[str] = ("S0",),
               questions: Optional[Dict[str, str]] = None, question_mode: str = "short") -> List[Dict]:
    """
    Chấm từng dòng sinh. Với cấu hình KHÔNG có ngữ cảnh (S0 closed-book) mọi câu đều "không có căn cứ"
    nếu premise rỗng, nên so S0 với S1..S4 bằng faithfulness là vô nghĩa. Khi truyền gold_map
    ({qid: [gold_chunk_id]}), các cấu hình trong gold_configs được chấm với CHUNK GOLD của câu đó,
    nghĩa là đo "câu trả lời từ trí nhớ của LLM có khớp quy chế không".
      premise_source = "retrieved" | "gold" | "none"
    Câu ngoài kho không có chunk gold -> premise rỗng -> mọi câu không từ chối đều bị coi là không có
    căn cứ (đúng ý nghĩa: đáng lẽ phải từ chối).
    """
    out = []
    for g in gens:
        ids, source = list(g.get("chunk_ids", [])), "retrieved"
        if not ids and gold_map is not None and g["config"] in gold_configs:
            ids, source = list(gold_map.get(str(g["qid"]), [])), "gold"
        if not ids:
            source = "none"
        texts = [chunk_index[c]["text"] for c in ids]
        out.append({"qid": g["qid"], "config": g["config"], "tau": tau, "agg": agg,
                    "nli_model": getattr(nli, "model_id", "?"), "premise_source": source,
                    "question_mode": question_mode if questions else "none",
                    **judge_answer(nli, g["answer"], texts, tau, agg,
                                   question=(questions or {}).get(str(g["qid"])), question_mode=question_mode)})
    return out


def summarize_by_config(judged: List[Dict]) -> Dict[str, Dict]:
    agg = defaultdict(list)
    for j in judged:
        agg[j["config"]].append(j)
    res = {}
    for cfg, items in sorted(agg.items()):
        ans = [x for x in items if not x["refusal"]]
        res[cfg] = {
            "n": len(items),
            "refusal_rate": round(100 * (len(items) - len(ans)) / len(items), 2),
            "premise_source": dict(Counter(x.get("premise_source", "retrieved") for x in items)),
            "faithfulness_mean": round(100 * sum(x["faithfulness"] for x in ans) / len(ans), 2) if ans else None,
            "unsupported_rate": round(100 * sum(not x["supported"] for x in ans) / len(ans), 2) if ans else None,
        }
    return res


# ---------------------------------------------------------------------------
# Hiệu chỉnh ngưỡng trên tập dev có nhãn người
# dev jsonl: {"id", "premise_chunks": [...], "hypothesis", "label": 1 (có căn cứ) | 0}
# ---------------------------------------------------------------------------
def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    n = len(y_true)
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp else 0.0      # lớp "có căn cứ"
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn else 0.0      # lớp "ảo giác" – quan trọng hơn
    return {"acc": (tp + tn) / n if n else 0.0, "macro_f1": (f1_pos + f1_neg) / 2,
            "f1_hallucination": f1_neg, "kappa": cohen_kappa(list(y_true), list(y_pred))}


def calibrate(nli, dev: List[Dict], taus: Iterable[float], aggs: Iterable[str] = ("max", "concat", "both"),
              select_by: str = "macro_f1") -> Tuple[Dict, List[Dict]]:
    """Trả về (cấu hình tốt nhất, toàn bộ bảng quét)."""
    table = []
    y = [int(d["label"]) for d in dev]
    for agg in aggs:
        probs = [sentence_scores(nli, [d["hypothesis"]], d["premise_chunks"], agg)[0] for d in dev]
        for tau in taus:
            pred = [int(p >= tau) for p in probs]
            table.append({"agg": agg, "tau": round(tau, 3), **{k: round(v, 4) for k, v in binary_metrics(y, pred).items()}})
    # phá hòa: ưu tiên F1 lớp ảo giác, rồi tau gần 0.5 (ít quá khớp tập dev nhỏ)
    best = max(table, key=lambda r: (r[select_by], r["f1_hallucination"], -abs(r["tau"] - 0.5)))
    return best, table


def evaluate_fixed(nli, data: List[Dict], tau: float, agg: str) -> Dict[str, float]:
    """
    Đánh giá một cặp (tau, agg) ĐÃ CHỐT trên một tập có nhãn người KHÁC tập đã dùng để chọn tau.
    Chọn tau và báo độ khớp trên cùng một tập sẽ làm kết quả bị thổi phồng; dùng hàm này cho tập báo cáo.
    """
    y = [int(d["label"]) for d in data]
    probs = [sentence_scores(nli, [d["hypothesis"]], d["premise_chunks"], agg)[0] for d in data]
    pred = [int(p >= tau) for p in probs]
    return {"n": len(y), **{k: round(v, 4) for k, v in binary_metrics(y, pred).items()}}


# ---------------------------------------------------------------------------
# Đồng thuận: Cohen (2 bên), Fleiss (≥3 người), NLI vs người
# ---------------------------------------------------------------------------
def cohen_kappa(a: Sequence, b: Sequence) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence]) -> float:
    """
    ratings[i] = danh sách nhãn của các người chấm cho item i (mỗi item cùng số người chấm n).
    kappa = (P_bar - P_e) / (1 - P_e)
    """
    N = len(ratings)
    if N == 0:
        return 0.0
    n = len(ratings[0])
    if n < 2 or any(len(r) != n for r in ratings):
        raise ValueError("Mỗi item phải có cùng số người chấm, tối thiểu 2")
    cats = sorted({c for r in ratings for c in r}, key=str)
    counts = [[Counter(r)[c] for c in cats] for r in ratings]
    P_i = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]
    P_bar = sum(P_i) / N
    p_j = [sum(row[j] for row in counts) / (N * n) for j in range(len(cats))]
    P_e = sum(p * p for p in p_j)
    return 1.0 if P_e == 1 else (P_bar - P_e) / (1 - P_e)


# Nhãn người (mục 7): 1 đúng-có căn cứ, 2 đúng-không căn cứ, 3 mâu thuẫn, 4 bịa, 5/6 từ chối
HUMAN_TO_SUPPORTED = {1: 1, 2: 0, 3: 0, 4: 0}


def agreement_with_humans(judged: List[Dict], human: Dict[Tuple[str, str], int]) -> Dict:
    """human: {(qid, config): nhãn 1..6}. Bỏ các câu từ chối (nhãn 5/6 hoặc NLI thấy từ chối)."""
    y_h, y_n = [], []
    for j in judged:
        lab = human.get((str(j["qid"]), j["config"]))
        if lab not in HUMAN_TO_SUPPORTED or j["refusal"]:
            continue
        y_h.append(HUMAN_TO_SUPPORTED[lab])
        y_n.append(int(j["supported"]))
    m = binary_metrics(y_h, y_n) if y_h else {}
    return {"n": len(y_h), **{k: round(v, 4) for k, v in m.items()}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="NLI judge cho câu trả lời RAG")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--gen", required=True)
    j.add_argument("--chunks", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--test", help="file câu hỏi (inputs/test.jsonl hoặc data/rag_test/*.jsonl); "
                   "dùng để ghép câu hỏi vào giả thuyết và chấm S0 với chunk gold")
    j.add_argument("--question-mode", choices=list(QUESTION_MODES), default="short",
                   help="ghép câu hỏi vào giả thuyết: none | short (mặc định) | always")
    j.add_argument("--wandb-project")
    c = sub.add_parser("calibrate")
    c.add_argument("--dev", required=True)
    e = sub.add_parser("evaluate", help="đánh giá (tau, agg) đã chốt trên tập có nhãn khác tập calibrate")
    e.add_argument("--data", required=True)
    e.add_argument("--tau", type=float, required=True)
    e.add_argument("--agg", choices=["max", "concat", "both"], default="max")
    for p in (j, c, e):
        p.add_argument("--backend", choices=["hf", "lexical"], default="hf")
        p.add_argument("--model", default=DEFAULT_NLI_MODEL)
    j.add_argument("--tau", type=float, default=0.5)
    j.add_argument("--agg", choices=["max", "concat", "both"], default="max")
    a = ap.parse_args(argv)

    nli = make_nli(a.backend, a.model)
    if a.cmd == "calibrate":
        best, table = calibrate(nli, _read_jsonl(a.dev), [i / 20 for i in range(1, 20)])
        for r in table:
            print(r)
        print("BEST:", best)
        return
    if a.cmd == "evaluate":
        print(json.dumps(evaluate_fixed(nli, _read_jsonl(a.data), a.tau, a.agg), ensure_ascii=False, indent=2))
        return
    chunk_index = {(ch.get("chunk_id") or ch["id"]): ch for ch in _read_jsonl(a.chunks)}  # TV2 dùng khóa "id"
    tests = _read_jsonl(a.test) if a.test else []
    _qid = lambda t: str(t.get("qid") or t["id"])            # inputs/test.jsonl dùng qid, repo dùng id
    gold_map = ({_qid(t): list(t.get("gold_chunk_ids") or t.get("gold_chunks") or []) for t in tests}
                if tests else None)
    questions = {_qid(t): t["question"] for t in tests} or None
    judged = judge_file(nli, _read_jsonl(a.gen), chunk_index, a.tau, a.agg, gold_map=gold_map,
                        questions=questions, question_mode=a.question_mode)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = summarize_by_config(judged)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.wandb_project:
        from rag.generation.generate import log_wandb
        log_wandb(a.wandb_project, "tv3-nli-judge",
                  {"tau": a.tau, "agg": a.agg, "nli_model": getattr(nli, "model_id", a.model)}, res)


if __name__ == "__main__":
    main()

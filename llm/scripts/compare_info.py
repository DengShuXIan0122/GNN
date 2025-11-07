#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare two .info files (JSONL, one sample per line) produced by GNN retrieval.
Reports summary deltas on Top-K hit, EM, and candidate statistics, plus examples
where coverage improves or regresses.

Usage:
  python llm/scripts/compare_info.py --a "PATH/基test.info" --b "PATH/test.info" --k "1,3,5,10" --examples 5
"""
import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Any, Set


def parse_record(line: str) -> Tuple[str, Set[str], List[Tuple[str, float]], Any]:
    data = json.loads(line)
    q = str(data.get("question", "")).strip()
    answers = set(map(str, data.get("answers", [])))
    cand_raw = data.get("cand", []) or []
    cand: List[Tuple[str, float]] = []
    for c in cand_raw:
        # cand item could be [mid, score] or a bare id; be defensive
        if isinstance(c, list) and len(c) >= 2:
            mid = str(c[0])
            try:
                score = float(c[1])
            except Exception:
                score = float("nan")
            cand.append((mid, score))
        elif isinstance(c, list) and len(c) == 1:
            cand.append((str(c[0]), float("nan")))
        else:
            cand.append((str(c), float("nan")))
    em = data.get("em", None)
    return q, answers, cand, em


def load_info(path: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                q, answers, cand, em = parse_record(line)
            except Exception as e:
                # skip bad lines
                # print(f"[WARN] Failed to parse line {ln} in {path}: {e}")
                continue
            if q:
                records[q] = {
                    "answers": answers,
                    "cand": cand,
                    "em": em,
                }
    return records


def topk_hit(answers: Set[str], cand: List[Tuple[str, float]], k: int) -> int:
    mids = [m for m, _ in cand[:k]]
    return 1 if answers.intersection(mids) else 0


def stats(records: Dict[str, Dict[str, Any]], k_list: List[int]) -> Dict[str, Any]:
    qs = list(records.keys())
    n = len(qs)
    if n == 0:
        return {
            "n": 0,
            "avg_cand_len": 0.0,
            "avg_top1_score": float("nan"),
            "em_rate": float("nan"),
            "topk": {k: float("nan") for k in k_list},
        }
    cand_lens = []
    top1_scores = []
    ems = []
    topk_rates = {k: 0 for k in k_list}
    for q in qs:
        r = records[q]
        answers = r["answers"]
        cand = r["cand"]
        cand_lens.append(len(cand))
        if cand:
            s = cand[0][1]
            top1_scores.append(s if isinstance(s, float) and not math.isnan(s) else float("nan"))
        em = r.get("em")
        if isinstance(em, (int, float)):
            ems.append(float(em))
        for k in k_list:
            topk_rates[k] += topk_hit(answers, cand, k)
    avg_cand_len = sum(cand_lens) / n if n > 0 else 0.0
    # filter out nan for average
    valid_scores = [s for s in top1_scores if isinstance(s, float) and not math.isnan(s)]
    avg_top1_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else float("nan")
    em_rate = (sum(ems) / len(ems)) if ems else float("nan")
    return {
        "n": n,
        "avg_cand_len": avg_cand_len,
        "avg_top1_score": avg_top1_score,
        "em_rate": em_rate,
        "topk": {k: topk_rates[k] / n for k in k_list},
    }


def compare(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]], k_list: List[int]) -> Dict[str, Any]:
    common_qs = sorted(set(a.keys()) & set(b.keys()))
    m = len(common_qs)
    improvements = {k: [] for k in k_list}
    regressions = {k: [] for k in k_list}
    for q in common_qs:
        ra, rb = a[q], b[q]
        for k in k_list:
            ha = topk_hit(ra["answers"], ra["cand"], k)
            hb = topk_hit(rb["answers"], rb["cand"], k)
            if ha == 0 and hb == 1:
                improvements[k].append(q)
            elif ha == 1 and hb == 0:
                regressions[k].append(q)
    return {
        "common_n": m,
        "improvements": improvements,
        "regressions": regressions,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare two .info files (baseline vs improved)")
    ap.add_argument("--a", required=True, help="Path to baseline .info (e.g., 基test.info)")
    ap.add_argument("--b", required=True, help="Path to improved .info (e.g., test.info)")
    ap.add_argument("--k", default="1,3,5,10", help="Comma-separated K list for Top-K hit rates")
    ap.add_argument("--examples", type=int, default=5, help="Number of example questions to show for improvements/regressions")
    args = ap.parse_args()

    k_list = [int(x) for x in args.k.split(",") if x.strip().isdigit()]

    if not os.path.isfile(args.a):
        print(f"[ERROR] File A not found: {args.a}")
        return 1
    if not os.path.isfile(args.b):
        print(f"[ERROR] File B not found: {args.b}")
        return 1

    A = load_info(args.a)
    B = load_info(args.b)

    # Restrict stats to common questions to make a fair comparison
    common_qs = sorted(set(A.keys()) & set(B.keys()))
    A_common = {q: A[q] for q in common_qs}
    B_common = {q: B[q] for q in common_qs}

    sa = stats(A_common, k_list)
    sb = stats(B_common, k_list)
    diff = compare(A_common, B_common, k_list)

    print("==== Compare .info files ====")
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"Common questions: {diff['common_n']}")

    print("\n-- Candidate stats --")
    print(f"Avg cand len: A={sa['avg_cand_len']:.3f} | B={sb['avg_cand_len']:.3f} | Δ={sb['avg_cand_len']-sa['avg_cand_len']:.3f}")
    a_top1 = sa['avg_top1_score']
    b_top1 = sb['avg_top1_score']
    a_top1_str = f"{a_top1:.6f}" if isinstance(a_top1, float) and not math.isnan(a_top1) else "NaN"
    b_top1_str = f"{b_top1:.6f}" if isinstance(b_top1, float) and not math.isnan(b_top1) else "NaN"
    delta_top1 = (b_top1 - a_top1) if all(isinstance(x, float) and not math.isnan(x) for x in [a_top1, b_top1]) else float("nan")
    delta_top1_str = f"{delta_top1:.6f}" if isinstance(delta_top1, float) and not math.isnan(delta_top1) else "NaN"
    print(f"Avg top1 score: A={a_top1_str} | B={b_top1_str} | Δ={delta_top1_str}")

    print("\n-- Top-K hit rates --")
    for k in k_list:
        ra = sa['topk'][k]
        rb = sb['topk'][k]
        delta = rb - ra
        print(f"Top{k} hit: A={ra:.4f} | B={rb:.4f} | Δ={delta:.4f}")

    a_em = sa['em_rate']
    b_em = sb['em_rate']
    a_em_str = f"{a_em:.4f}" if isinstance(a_em, float) and not math.isnan(a_em) else "NaN"
    b_em_str = f"{b_em:.4f}" if isinstance(b_em, float) and not math.isnan(b_em) else "NaN"
    delta_em = (b_em - a_em) if all(isinstance(x, float) and not math.isnan(x) for x in [a_em, b_em]) else float("nan")
    delta_em_str = f"{delta_em:.4f}" if isinstance(delta_em, float) and not math.isnan(delta_em) else "NaN"
    print("\n-- EM (from .info if present) --")
    print(f"EM: A={a_em_str} | B={b_em_str} | Δ={delta_em_str}")

    # Examples
    for k in k_list:
        imp = diff['improvements'][k]
        reg = diff['regressions'][k]
        print(f"\n== Examples (Top{k}) ==")
        print(f"Improvements ({len(imp)} total):")
        for q in imp[:args.examples]:
            ans = sorted(A_common[q]['answers'])
            cand_b = [m for m, _ in B_common[q]['cand'][:k]]
            cand_a = [m for m, _ in A_common[q]['cand'][:k]]
            print(f"  [IMP] Q='{q[:80]}' | ans={ans} | A_top{k}={cand_a} | B_top{k}={cand_b}")
        print(f"Regressions ({len(reg)} total):")
        for q in reg[:args.examples]:
            ans = sorted(A_common[q]['answers'])
            cand_b = [m for m, _ in B_common[q]['cand'][:k]]
            cand_a = [m for m, _ in A_common[q]['cand'][:k]]
            print(f"  [REG] Q='{q[:80]}' | ans={ans} | A_top{k}={cand_a} | B_top{k}={cand_b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
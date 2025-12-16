#!/usr/bin/env python3
"""
ESM‑1v zero‑shot scoring for multi‑FASTA(s) — with epistasis and batch mode.

Now supports:
  • One or **many** input FASTA files in a single run
  • Per‑file outputs to --out-dir (default: current directory)
  • Single WT per file (via --wt-id / first record / --wt-seq)
  • Epistasis metrics as before

See usage examples near the end of this header.

Given (per file):
  • A multi‑FASTA of sequences (WT + mutants of the **same protein**) OR a WT sequence

This script (per file):
  1) Derives the list of amino‑acid substitutions for each mutant vs WT
  2) Computes per‑site PLLR in the mutant background (conditional)
  3) (Optional) Computes additive expectation from single‑mutant scores in WT background
  4) Reports ε = conditional_sum − additive_sum and optional pairwise estimates
  5) Writes <input>_scores.csv to --out-dir (or --out if exactly one input file)

Install once:
    pip install --upgrade torch transformers

Usage examples:
    # Single file, auto‑WT by header substring
    python esm1v_multifasta_scoring.py mutants.fasta --wt-id WT --epistasis --pairwise

    # Multiple files at once → results/<name>_scores.csv for each input
    python esm1v_multifasta_scoring.py data/*.fasta --wt-id WT --epistasis --pairwise --out-dir results

Notes
  • **One WT per file.** If you have multiple different proteins (multiple WTs), split them into separate files first.
  • New: optional **embeddings export** (mutated sites, all residues, and/or sequence-level).
  • Substitutions only (A123T). Indels are skipped.
  • ESM‑1v context limit is ~1022 aa; longer needs windowing (not implemented here).
  • CPU works but is slow; GPU recommended for ensemble and pairwise.
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))  # standard 20

ESM1V_MODELS = [
    "facebook/esm1v_t33_650M_UR90S_1",
    "facebook/esm1v_t33_650M_UR90S_2",
    "facebook/esm1v_t33_650M_UR90S_3",
    "facebook/esm1v_t33_650M_UR90S_4",
    "facebook/esm1v_t33_650M_UR90S_5",
]


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records = []
    header, seq = None, []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq).upper().replace(" ", "")))
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if header is not None:
            records.append((header, "".join(seq).upper().replace(" ", "")))
    if not records:
        raise ValueError("No records read from FASTA.")
    return records


def pick_wt(records: List[Tuple[str, str]], wt_id: str = None, wt_seq: str = None) -> Tuple[str, str]:
    if wt_seq:
        wt_seq = wt_seq.strip().upper()
        return ("WT", wt_seq)
    if wt_id:
        for h, s in records:
            if wt_id in h:
                return (h, s)
        raise ValueError(f"WT id '{wt_id}' not found in FASTA headers.")
    # Default: first record is WT
    return records[0]


def find_substitutions(wt: str, mut: str) -> List[Tuple[int, str, str]]:
    """Return list of (position 1‑based, wtAA, mutAA) for substitutions only.
       Raise on length mismatch; skip non‑AA20 characters."""
    if len(wt) != len(mut):
        raise ValueError("Length mismatch (indels not supported in this script).")
    subs = []
    for i, (a, b) in enumerate(zip(wt, mut), start=1):
        if a != b and a in AA20 and b in AA20:
            subs.append((i, a, b))
    return subs


def apply_mutations(seq: str, muts: Iterable[Tuple[int, str]]) -> str:
    s = list(seq)
    for pos1, newaa in muts:
        s[pos1 - 1] = newaa
    return "".join(s)


def load_models(names: List[str], device: torch.device):
    tok = AutoTokenizer.from_pretrained(names[0])
    models = []
    for nm in names:
        mdl = AutoModelForMaskedLM.from_pretrained(nm)
        mdl.to(device)
        mdl.eval()
        models.append(mdl)
    return tok, models


def token_id_for_aa(tok, aa: str) -> int:
    return tok.convert_tokens_to_ids(aa)


def _site_logprob(masked_ids, attn, tok_pos: int, target_token_id: int, models) -> float:
    with torch.no_grad():
        vals = []
        for mdl in models:
            out = mdl(input_ids=masked_ids, attention_mask=attn)
            logits = out.logits[0, tok_pos, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            vals.append(log_probs[target_token_id].item())
    return sum(vals) / len(vals)


def llr_for_site(seq: str, pos1: int, wt_aa: str, mut_aa: str, tok, models, device: torch.device) -> float:
    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    tok_pos = pos1  # accounts for initial BOS token

    masked = input_ids.clone()
    masked[0, tok_pos] = tok.mask_token_id

    wt_id = token_id_for_aa(tok, wt_aa)
    mut_id = token_id_for_aa(tok, mut_aa)

    lp_mut = _site_logprob(masked, attn, tok_pos, mut_id, models)
    lp_wt = _site_logprob(masked, attn, tok_pos, wt_id, models)
    return lp_mut - lp_wt

def sequence_logpl(seq: str, tok, models, device: torch.device) -> float:
    """
    Pseudo-log-likelihood of a full sequence under the ensemble.

    Masks each position once and sums log P(residue | remaining context).
    Works even if the sequence length is different from the WT.
    """
    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # ESM-1v uses a BOS + EOS token → actual residues are positions 1..L
    L = input_ids.size(1) - 2

    total = 0.0
    for pos1 in range(1, L + 1):
        tok_pos = pos1
        masked = input_ids.clone()
        target_id = int(input_ids[0, tok_pos])
        masked[0, tok_pos] = tok.mask_token_id
        lp = _site_logprob(masked, attn, tok_pos, target_id, models)
        total += lp
    return float(total)


def sequence_pllr(wt_seq: str, mut_seq: str, tok, models, device: torch.device) -> float:
    """
    Whole-sequence pseudo-log-likelihood ratio mut vs WT.

    This is what you can use for indels: it does *not* require the
    sequences to be the same length.
    """
    wt_lp = sequence_logpl(wt_seq, tok, models, device)
    mut_lp = sequence_logpl(mut_seq, tok, models, device)
    return float(mut_lp - wt_lp)


def conditional_sum_pllr(mut_seq: str, subs: List[Tuple[int, str, str]], tok, models, device) -> Tuple[float, List[Dict]]:
    per_site = []
    llrs = []
    for pos1, wt_aa, mut_aa in subs:
        llr = llr_for_site(mut_seq, pos1, wt_aa, mut_aa, tok, models, device)
        llrs.append(llr)
        per_site.append({"pos": pos1, "wt": wt_aa, "mut": mut_aa, "llr_conditional": float(llr), "label": f"{wt_aa}{pos1}{mut_aa}"})
    return float(sum(llrs)), per_site


def additive_sum_from_wt(wt_seq: str, subs: List[Tuple[int, str, str]], tok, models, device, cache: Dict) -> Tuple[float, List[float]]:
    llrs = []
    for pos1, wt_aa, mut_aa in subs:
        key = (pos1, mut_aa)
        if key not in cache:
            llr = llr_for_site(wt_seq, pos1, wt_aa, mut_aa, tok, models, device)
            cache[key] = float(llr)
        llrs.append(cache[key])
    return float(sum(llrs)), llrs


def pairwise_epistasis(wt_seq: str, subs: List[Tuple[int, str, str]], tok, models, device, single_cache: Dict) -> Tuple[float, Tuple[str, float], float]:
    if len(subs) < 2:
        return 0.0, ("", 0.0), 0.0
    eps_sum = 0.0
    min_label, min_eps = "", 0.0
    for (i, (p1, wt1, m1)), (j, (p2, wt2, m2)) in itertools.combinations(enumerate(subs), 2):
        dm_seq = apply_mutations(wt_seq, [(p1, m1), (p2, m2)])
        cond2, _ = conditional_sum_pllr(dm_seq, [(p1, wt1, m1), (p2, wt2, m2)], tok, models, device)
        add1, _ = additive_sum_from_wt(wt_seq, [(p1, wt1, m1)], tok, models, device, single_cache)
        add2, _ = additive_sum_from_wt(wt_seq, [(p2, wt2, m2)], tok, models, device, single_cache)
        eps = cond2 - (add1 + add2)
        eps_sum += eps
        label = f"{wt1}{p1}{m1}+{wt2}{p2}{m2}"
        if eps < min_eps:
            min_eps = eps
            min_label = label
    return float(eps_sum), (min_label, float(min_eps)), float(min_eps)


def token_embeddings(seq: str, tok, models, device, agg: str = "last"):
    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    reps = []
    with torch.no_grad():
        for mdl in models:
            out = mdl(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            hs = out.hidden_states  # tuple: [embeddings, layer1, ..., layerN]
            if agg == "last":
                rep = hs[-1][0]  # [tokens, hidden]
            else:
                rep = torch.stack([hs[-1][0], hs[-2][0], hs[-3][0], hs[-4][0]], dim=0).mean(dim=0)
            reps.append(rep)
    rep_avg = torch.stack(reps, dim=0).mean(dim=0)  # [tokens, hidden]
    return rep_avg.cpu().numpy()


def process_one_file(
    in_fa: Path,
    wt_id: str,
    wt_seq: str,
    tok,
    models,
    device,
    out_csv: Path,
    log_path: Path,
    do_epistasis: bool,
    do_pairwise: bool,
    dump_emb: bool = False,
    emb_all: bool = False,
    dump_seq_emb: bool = False,
    emb_agg: str = "last",
):
    records = read_fasta(in_fa)
    wt_header, wt_sequence = pick_wt(records, wt_id, wt_seq)
    wt_pll = sequence_logpl(wt_sequence, tok, models, device)

    import csv
    out_fh = out_csv.open("w", newline="")
    writer = csv.writer(out_fh)
    header_cols = [
        "record_id", "n_mut", "mutations",
        "pllr_conditional_sum", "pllr_additive_sum", "epsilon_epistasis", "class",
        "pairwise_min_label", "pairwise_min_epsilon", "pairwise_sum_epsilon",
        "pll_mut", "pll_wt", "len_mut","len_wt",
        "per_site_json",
    ]
    writer.writerow(header_cols)

    # Prepare embedding outputs if requested
    emb_writer = None
    seqemb_writer = None
    if dump_emb:
        emb_path = out_csv.with_name(f"{in_fa.stem}_siteemb.csv")
        emb_fh = emb_path.open("w", newline="")
        emb_writer = csv.writer(emb_fh)
        wt_emb_tokens = token_embeddings(wt_sequence, tok, models, device, emb_agg)
        hidden = wt_emb_tokens.shape[1]
        emb_header = ["record_id", "context", "pos", "wt", "mut", "label"] + [f"e{i+1}" for i in range(hidden)]
        emb_writer.writerow(emb_header)
    else:
        wt_emb_tokens = None
        hidden = None

    if dump_seq_emb:
        seqemb_path = out_csv.with_name(f"{in_fa.stem}_seqemb.csv")
        seqemb_fh = seqemb_path.open("w", newline="")
        seqemb_writer = csv.writer(seqemb_fh)
        if wt_emb_tokens is None:
            wt_emb_tokens = token_embeddings(wt_sequence, tok, models, device, emb_agg)
            hidden = wt_emb_tokens.shape[1]
        seqemb_header = ["record_id", "context"] + [f"e{i+1}" for i in range(hidden)]
        seqemb_writer.writerow(seqemb_header)
        # WT sequence-level
        seqemb_writer.writerow([wt_header, "WT"] + wt_emb_tokens[0, :].tolist())

    skipped = []
    total = 0
    scored = 0
    single_cache = {}

    for header, seq in records:
        total += 1
        if (header, seq) == (wt_header, wt_sequence):
            continue
        try:
            # --- NEW: try to detect indels safely ---
            has_indel = False
            try:
                # Normal path: substitutions only, same length
                subs = find_substitutions(wt_sequence, seq)
                substitution_percentage = len(subs) / len(wt_sequence)

            except ValueError as e:
                msg = str(e)
                if "indels not supported" in msg or "Length mismatch" in msg:
                    # New path: mutant has deletions / insertions
                    has_indel = True
                    subs = []

                if substitution_percentage > 0.3:
                    has_indel = True

                else:
                    # Some other ValueError we don't recognize → bubble up
                    raise

            # --- NEW: handle deletion/indel sequences ---
            if has_indel:
                # crude count: how many residues are missing relative to WT
                n_del = max(len(wt_sequence) - len(seq), 0)
                mut_label = f"{n_del}del" if n_del else "indel"

                # Use whole-sequence PLLR as the "conditional" score
                mut_pll = sequence_logpl(seq, tok, models, device)
                seq_pllr = mut_pll - wt_pll  # whole-sequence PLLR (mut - WT)
                #seq_pllr = sequence_pllr(wt_sequence, seq, tok, models, device)

                # Epistasis isn't really defined here → set additive / eps to 0 or NaN
                add_sum = 0.0
                eps = 0.0
                _cls = "indel_only"
                if substitution_percentage > 0.3:
                    _cls = "too_many_subs"

                writer.writerow([
                    header,
                    n_del,
                    mut_label,
                    float(seq_pllr),    # pllr_conditional_sum
                    float(add_sum),     # pllr_additive_sum
                    float(eps),         # epsilon_epistasis
                    _cls,               # class
                    "",                 # pairwise_min_label
                    0.0,                # pairwise_min_epsilon
                    0.0,                # pairwise_sum_epsilon
                    float(mut_pll),     # pll_mut
                    float(wt_pll),      # pll_wt
                    int(len(seq)),
                    int(len(wt_sequence)),
                    json.dumps([]),     # per_site_json (empty for now)
                ])

                if dump_seq_emb:
                    mut_tokens = token_embeddings(seq, tok, models, device, emb_agg)
                    seqemb_writer.writerow([header, "mutant"] + mut_tokens[0, :].tolist())

                scored += 1
                continue
            # --- END NEW BLOCK ---

            # Original substitution-only behavior
            if not subs:
                mut_pll = sequence_logpl(seq, tok, models, device)
                
                writer.writerow([header, 0, "", 0.0, 0.0, 0.0, "neutral", "", 0.0, 0.0,
                float(mut_pll), float(wt_pll),int(len(seq)),
                    int(len(wt_sequence)),
                json.dumps([])])

                if dump_seq_emb:
                    mut_tokens = token_embeddings(seq, tok, models, device, emb_agg)
                    seqemb_writer.writerow([header, "mutant"] + mut_tokens[0, :].tolist())
                scored += 1
                continue

            # scoring
            cond_sum, per_site = conditional_sum_pllr(seq, subs, tok, models, device)
            add_sum, singles = (0.0, [])
            if do_epistasis:
                add_sum, singles = additive_sum_from_wt(wt_sequence, subs, tok, models, device, single_cache)
                for d, s in zip(per_site, singles):
                    d["llr_single_WT"] = float(s)
            eps = cond_sum - add_sum
            _cls = "synergistic" if eps < -0.5 else ("buffering" if eps > 0.5 else "approximately_additive")

            pmin_label, pmin_eps, psum_eps = "", 0.0, 0.0
            if do_pairwise and len(subs) >= 2:
                psum_eps, (pmin_label, pmin_eps), _ = pairwise_epistasis(
                    wt_sequence, subs, tok, models, device, single_cache
                )

            muts_label = ";".join([p["label"] for p in per_site])
            writer.writerow([
                header, len(per_site), muts_label,
                float(cond_sum), float(add_sum), float(eps), _cls,
                pmin_label, float(pmin_eps), float(psum_eps),
                float(mut_pll), float(wt_pll), int(len(seq)),
                    int(len(wt_sequence)),
                json.dumps(per_site)
            ])

            # embeddings
            if dump_emb or dump_seq_emb:
                mut_tokens = token_embeddings(seq, tok, models, device, emb_agg)
                if dump_seq_emb:
                    seqemb_writer.writerow([header, "mutant"] + mut_tokens[0, :].tolist())
                if dump_emb:
                    if emb_all:
                        for pos1 in range(1, len(seq) + 1):
                            label = f"{wt_sequence[pos1-1]}{pos1}{seq[pos1-1]}"
                            emb_writer.writerow(
                                [header, "WT", pos1, wt_sequence[pos1-1], seq[pos1-1], label]
                                + wt_emb_tokens[pos1, :].tolist()
                            )
                            emb_writer.writerow(
                                [header, "mutant", pos1, wt_sequence[pos1-1], seq[pos1-1], label]
                                + mut_tokens[pos1, :].tolist()
                            )
                    else:
                        for d in per_site:
                            p = d["pos"]
                            wt = d["wt"]; mu = d["mut"]; label = d["label"]
                            emb_writer.writerow(
                                [header, "WT", p, wt, mu, label] + wt_emb_tokens[p, :].tolist()
                            )
                            emb_writer.writerow(
                                [header, "mutant", p, wt, mu, label] + mut_tokens[p, :].tolist()
                            )

            scored += 1

        except Exception as e:
            skipped.append((header, str(e)))
            continue

    out_fh.close()
    if dump_emb:
        emb_fh.close()
    if dump_seq_emb:
        seqemb_fh.close()

    if skipped:
        with log_path.open("w") as lf:
            for h, msg in skipped:
                lf.write(f"{h}\t{msg}\n")

    print(f"[OK] {in_fa.name}: Scored {scored}/{total} (see {log_path}). Output → {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="ESM‑1v scoring for multi‑FASTA(s) (with epistasis, batch mode, and embeddings)")
    ap.add_argument("fastas", nargs="+", type=Path, help="One or more multi‑FASTA files (one WT per file)")
    ap.add_argument("--wt-id", dest="wt_id", type=str, default=None, help="Header substring identifying WT record in each FASTA")
    ap.add_argument("--wt-seq", dest="wt_seq", type=str, default=None, help="Explicit WT amino‑acid sequence (applies to all files)")
    ap.add_argument("--single", action="store_true", help="Use a single ESM‑1v checkpoint (faster)")
    ap.add_argument("--model", type=str, default=ESM1V_MODELS[0], help="Model to use if --single (default: esm1v_..._1)")
    ap.add_argument("--epistasis", action="store_true", help="Compute additive expectation from singles (WT background) and ε")
    ap.add_argument("--pairwise", action="store_true", help="Also estimate pairwise epistasis by scoring double mutants")
    ap.add_argument("--out", type=Path, default=None, help="Output CSV path (allowed only with a single input file)")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Directory for per‑file outputs when multiple inputs are given")
    ap.add_argument("--log-dir", type=Path, default=Path("."), help="Directory for per‑file log files")
    ap.add_argument("--dump-emb", action="store_true", help="Write per‑site embeddings to <basename>_siteemb.csv (mutated sites by default)")
    ap.add_argument("--emb-all-residues", action="store_true", help="With --dump-emb, export embeddings for **all** residues (not just mutated sites)")
    ap.add_argument("--dump-seq-emb", action="store_true", help="Write sequence‑level [CLS] embeddings to <basename>_seqemb.csv")
    ap.add_argument("--emb-agg", choices=["last", "meanlast4"], default="last", help="Layer aggregation for embeddings (default: last layer; alt: mean of last 4)")
    args = ap.parse_args()

    if len(args.fastas) > 1 and args.out is not None:
        raise SystemExit("--out is only allowed when a single input FASTA is provided. Use --out-dir for batch mode.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_names = [args.model] if args.single else ESM1V_MODELS
    tok, models = load_models(model_names, device)
  

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    for in_fa in args.fastas:
        out_csv = args.out if (args.out is not None) else (args.out_dir / f"{in_fa.stem}_scores.csv")
        log_path = args.log_dir / f"{in_fa.stem}_skipped.log"
        process_one_file(
            in_fa, args.wt_id, args.wt_seq, tok, models, device,
            out_csv, log_path, args.epistasis, args.pairwise,
            dump_emb=args.dump_emb, emb_all=args.emb_all_residues,
            dump_seq_emb=args.dump_seq_emb, emb_agg=args.emb_agg,
        )


if __name__ == "__main__":
    main()


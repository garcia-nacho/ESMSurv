#!/usr/bin/env python3
"""
Quick and dirty pll calculation 

"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np
import csv
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))  # standard 20

ESM1V_MODELS = [
    "facebook/esm1v_t33_650M_UR90S_1",
    "facebook/esm1v_t33_650M_UR90S_2",
    "facebook/esm1v_t33_650M_UR90S_3",
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



def load_models(names: List[str], device: torch.device):
    tok = AutoTokenizer.from_pretrained(names[0])
    models = []
    for nm in names:
        mdl = AutoModelForMaskedLM.from_pretrained(nm,torch_dtype=torch.float16)
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


def _per_site_logprobs(
    seq: str,
    tok,
    models,
    device: torch.device,
    chunk_size: int = 64,
) -> np.ndarray:
    """
    Compute per-residue log-probs for a sequence that *fits* in the model
    context window (no sliding). Returns an array [L] with log p(res_i | rest).
    """
    # Tokenize once
    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)        # (1, T)
    attn      = enc["attention_mask"].to(device)   # (1, T)

    T = input_ids.size(1)
    L = T - 2  # residues at positions 1..L
    if L <= 0:
        raise ValueError(f"Sequence too short after tokenization: T={T}")

    # Positions of residues to mask (in token space)
    pos_all = torch.arange(1, L + 1, device=device, dtype=torch.long)  # [1..L]
    target_token_ids = input_ids[0, pos_all]                           # (L,)

    # Accumulate log-probs across models
    per_site_lp = torch.zeros(L, device=device)

    with torch.no_grad():
        for mdl in models:
            # sum over models
            for start in range(0, L, chunk_size):
                end = min(L, start + chunk_size)
                pos_chunk = pos_all[start:end]        # (B,)
                chunk_len = end - start

                # Repeat sequence chunk_len times
                masked_ids = input_ids.repeat(chunk_len, 1)   # (B, T)
                attn_rep   = attn.repeat(chunk_len, 1)        # (B, T)

                # Mask each position in its own copy
                batch_idx = torch.arange(chunk_len, device=device, dtype=torch.long)
                masked_ids[batch_idx, pos_chunk] = tok.mask_token_id

                # Forward pass
                out = mdl(input_ids=masked_ids, attention_mask=attn_rep)
                logits = out.logits                           # (B, T, V)
                log_probs = torch.log_softmax(logits, dim=-1) # (B, T, V)

                # Gather log p(target AA) at the masked positions
                targets_chunk = target_token_ids[start:end]   # (B,)
                lp_chunk = log_probs[batch_idx, pos_chunk, targets_chunk]  # (B,)

                # Map token positions 1..L → residue indices 0..L-1
                local_idx = pos_chunk - 1                     # (B,)
                per_site_lp[local_idx] += lp_chunk

    per_site_lp = per_site_lp / float(len(models))            # average across models
    return per_site_lp.detach().cpu().numpy()                 # [L,]

def sequence_logpl(
    seq: str,
    tok,
    models,
    device: torch.device,
    chunk_size: int = 64,
    window_len: int | None = None,
    overlap: int = 128,
) -> float:
    """
    Pseudo-log-likelihood of a full sequence under the ensemble.

    - If len(seq) <= effective_window_len: compute PLL exactly (full context).
    - If len(seq)  > effective_window_len: use a sliding window with overlap,
      compute per-residue log-probs in each window, and average per-residue
      log-probs across windows.

    chunk_size: how many positions to mask per forward pass.
    window_len: window length in residues; if None, use model max (config).
    overlap:   residues of overlap between adjacent windows (for long seqs).
    """
    # Model context limit in tokens, convert to residues (BOS+EOS)
    max_tokens = getattr(models[0].config, "max_position_embeddings", 1024)
    max_residues = max_tokens - 2

    if window_len is None:
        window_len = max_residues
    else:
        window_len = min(window_len, max_residues)

    L = len(seq)
    if L <= window_len:
        # Fits in a single context: exact PLL via full sequence
        per_site_lp = _per_site_logprobs(seq, tok, models, device, chunk_size)
        return float(per_site_lp.sum())

    # --- Sliding-window mode for long sequences ---
    step = max(1, window_len - overlap)

    per_site_lp_sum = np.zeros(L, dtype=np.float32)
    per_site_cnt    = np.zeros(L, dtype=np.float32)

    start = 0
    while start < L:
        end = min(L, start + window_len)
        subseq = seq[start:end]          # residues [start:end]
        # per-residue log-probs for this subseq (length W = end-start)
        local_lp = _per_site_logprobs(subseq, tok, models, device, chunk_size)
        W = end - start
        if len(local_lp) != W:
            raise RuntimeError(
                f"Window PLL length mismatch: expected {W}, got {len(local_lp)}"
            )

        per_site_lp_sum[start:end] += local_lp
        per_site_cnt[start:end]    += 1.0

        start += step

    # Average per-residue log-probs across windows
    per_site_cnt[per_site_cnt == 0] = 1.0
    per_site_lp_avg = per_site_lp_sum / per_site_cnt

    return float(per_site_lp_avg.sum())


def process_one_file(in_fa: Path, tok, models, device, out_csv: Path):
    in_fa = Path(in_fa)
    out_csv = Path(out_csv)

    records = read_fasta(in_fa)

    out_fh = out_csv.open("w", newline="")
    writer = csv.writer(out_fh)
    header_cols = [
        "record_id",
        "pll",
        "len",
    ]
    writer.writerow(header_cols)

    skipped = []
    total = 0
    scored = 0

    for header, seq in records:
        total += 1
        print(f"Running {header}")
        try:
            mut_pll = sequence_logpl(seq, tok, models, device,chunk_size=164,  window_len=1000)

            writer.writerow([
                header,
                float(mut_pll),
                int(len(seq)),
            ])

            scored += 1

        except Exception as e:
            print(f"  -> skipped {header}: {e}")
            skipped.append((header, str(e)))
            continue

    out_fh.close()

    # Optional: write a simple log for skipped sequences
    log_path = out_csv.with_suffix(out_csv.suffix + ".log")
    if skipped:
        with log_path.open("w") as lf:
            for h, msg in skipped:
                lf.write(f"{h}\t{msg}\n")
        print(f"[OK] {in_fa.name}: Scored {scored}/{total} (see {log_path}). Output → {out_csv}")
    else:
        print(f"[OK] {in_fa.name}: Scored {scored}/{total}. Output → {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="ESM‑1v PLL for multi‑FASTA(s)")
    ap.add_argument("fastas", nargs="+", type=Path, help="One or more multi‑FASTA files (one WT per file)")
    ap.add_argument("--model", type=str, default=ESM1V_MODELS[0], help="Model to use if --single (default: esm1v_..._1)")
    ap.add_argument("--out", type=Path, default=None, help="Output CSV path (allowed only with a single input file)")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Directory for per‑file outputs when multiple inputs are given")
    args = ap.parse_args()

    if len(args.fastas) > 1 and args.out is not None:
        raise SystemExit("--out is only allowed when a single input FASTA is provided. Use --out-dir for batch mode.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_names = ESM1V_MODELS
    tok, models = load_models(model_names, device)
  
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for in_fa in args.fastas:
        out_csv = args.out if (args.out is not None) else (args.out_dir / f"{in_fa.stem}_scores.csv")
        process_one_file(
            in_fa,  
            tok, 
            models, 
            device,
            out_csv, 
        )

if __name__ == "__main__":
    main()


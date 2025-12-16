#!/usr/bin/env python3
"""
esm_embed_fastas.py — Compute ESM-2 embeddings for sequences in one or more FASTA files.

What it does
------------
• Reads one or more multi-FASTA files.
• For every record, computes:
    - Sequence-level embedding (mean over residues by default).
    - (Optional) per-residue embeddings (saved to NPZ).
• No WT/mutation logic; sequences are treated independently.
• Handles long proteins via sliding windows.

Outputs
-------
• <basename>_seqemb.csv       (sequence-level embeddings)
• (optional) <basename>_siteemb.npz  (per-residue arrays keyed by record_id)

Usage
-----
python esm_embed_fastas.py data/*.fasta \
  --out-dir embeds/ \
  --model-id facebook/esm2_t33_650M_UR50D \
  --pool mean --layer-agg last

# also dump per-residue embeddings:
python esm_embed_fastas.py proteins.faa --dump-site-emb

Notes
-----
• Uses ESM-2 via Hugging Face transformers.
• Default context ~1022 residues; longer sequences are windowed with overlap (merge by averaging).
• GPU is auto-detected; use --cpu to force CPU.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import csv
import sys
import math

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
        raise ValueError(f"No records read from FASTA: {path}")
    return records

def load_esm2(model_id: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForMaskedLM.from_pretrained(model_id)
    mdl.eval().to(device)
    return tok, mdl

@torch.inference_mode()
def residue_embeddings(
    seq: str,
    tok: AutoTokenizer,
    mdl: AutoModelForMaskedLM,
    device: torch.device,
    layer_agg: str = "last",
    window_len: int = 1022,
    overlap: int = 128,
) -> np.ndarray:
    """
    Return per-residue embeddings [L, D] for a protein sequence.
    Uses sliding windows for sequences longer than window_len.
    layer_agg: 'last' or 'meanlast4'
    """
    # Try to infer supported max length, fallback to 1022
    try:
        # subtract 2 for special tokens BOS/EOS
        supported = int(getattr(mdl.config, "max_position_embeddings", window_len)) - 2
        window_len = max(1, min(window_len, supported))
    except Exception:
        pass

    L = len(seq)
    # Accumulators for overlap-averaging
    emb_sum = None
    emb_cnt = None

    def _chunk_embed(subseq: str) -> np.ndarray:
        enc = tok(subseq, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = mdl(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        if layer_agg == "meanlast4":
            hs = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)[0]  # [T, D]
        else:
            hs = out.hidden_states[-1][0]  # [T, D]
        # strip specials (BOS/EOS)
        hs = hs[1:-1]  # [len(subseq), D]
        return hs.detach().cpu().numpy()

    if L <= window_len:
        return _chunk_embed(seq)

    # Sliding window with overlap
    step = max(1, window_len - overlap)
    D = None
    emb_sum = np.zeros((L, 1), dtype=np.float32)  # placeholder to derive shape
    # We don't know D until we run once; do a first window to get D
    first = _chunk_embed(seq[:window_len])
    D = first.shape[1]
    emb_sum = np.zeros((L, D), dtype=np.float32)
    emb_cnt = np.zeros((L, 1), dtype=np.float32)

    # place first chunk
    emb_sum[:window_len] += first
    emb_cnt[:window_len] += 1.0

    # remaining chunks
    start = step
    while start < L:
        end = min(L, start + window_len)
        chunk = _chunk_embed(seq[start:end])
        emb_sum[start:end] += chunk
        emb_cnt[start:end] += 1.0
        start += step

    emb_cnt[emb_cnt == 0] = 1.0
    return emb_sum / emb_cnt

@torch.inference_mode()
def batch_residue_embeddings(
    seqs,
    tok: AutoTokenizer,
    mdl: AutoModelForMaskedLM,
    device: torch.device,
    layer_agg: str = "last",
):
    """
    Batched version of residue_embeddings for sequences that fit
    within the model context (no sliding windows).

    seqs: list of sequences (all with len(seq) <= window_len)

    Returns: list of numpy arrays [L_i, D] per sequence.
    """
    if not seqs:
        return []

    enc = tok(
        list(seqs),
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    ids = enc["input_ids"].to(device)        # (B, T_max)
    attn = enc["attention_mask"].to(device)  # (B, T_max)

    out = mdl(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    if layer_agg == "meanlast4":
        hs_all = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)  # (B, T_max, D)
    else:
        hs_all = out.hidden_states[-1]  # (B, T_max, D)

    res = []
    B, T_max, D = hs_all.shape

    for i in range(B):
        hs = hs_all[i]        # (T_max, D)
        mask = attn[i]        # (T_max,)
        T_eff = int(mask.sum().item())  # non-pad tokens

        if T_eff <= 2:
            # degenerate edge-case; no real residues
            emb = hs.new_zeros((0, D)).cpu().numpy()
        else:
            # tokens: 0 = BOS, T_eff-1 = EOS, so residues = 1..T_eff-2
            emb = hs[1:T_eff-1].detach().cpu().numpy()  # (L_i, D)

        res.append(emb)

    return res



def pool_sequence(emb_res: np.ndarray, pool: str = "mean") -> np.ndarray:
    """
    Pool per-residue embeddings [L, D] into [D].
    pool: 'mean' or 'cls' (CLS not available from residue embs; keep mean here)
    """
    if pool == "mean":
        v = emb_res.mean(axis=0)
    else:
        # CLS isn’t in per-residue output; fall back to mean
        v = emb_res.mean(axis=0)
    # L2-normalize (common for cosine metrics)
    norm = np.linalg.norm(v) + 1e-12
    return (v / norm).astype(np.float32)

def write_seqemb_csv(out_csv: Path, rows: List[Tuple[str, int, np.ndarray]], file_tag: str):
    """
    rows: list of (record_id, length, vector)
    """
    if not rows:
        return
    D = rows[0][2].shape[0]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["record_id", "file", "length"] + [f"e{i+1}" for i in range(D)]
        w.writerow(header)
        for rid, length, vec in rows:
            w.writerow([rid, file_tag, length] + vec.tolist())

def process_fasta_batched(
    in_fa: Path,
    tok: AutoTokenizer,
    mdl: AutoModelForMaskedLM,
    device: torch.device,
    out_dir: Path,
    pool: str = "mean",
    layer_agg: str = "last",
    dump_site_emb: bool = False,
    batch_size: int = 32,
    window_len: int = 1022,
    overlap: int = 128,
):
    """
    Faster version: batch short sequences together, fall back to
    residue_embeddings() with sliding windows for long ones.
    """
    records = read_fasta(in_fa)
    file_tag = in_fa.name

    seq_rows = []  # (record_id, length, vec)
    site_embs: Dict[str, np.ndarray] = {} if dump_site_emb else None

    # Split into short/long relative to window_len
    short = []
    long = []
    for rid, seq in records:
        if len(seq) <= window_len:
            short.append((rid, seq))
        else:
            long.append((rid, seq))

    print(f"[{in_fa.name}] {len(records)} records: {len(short)} short, {len(long)} long")

    # --- Short sequences: batched, no sliding windows ---
    for start in range(0, len(short), batch_size):
        batch = short[start:start + batch_size]
        batch_ids = [rid for rid, _ in batch]
        batch_seqs = [seq for _, seq in batch]

        emb_list = batch_residue_embeddings(
            batch_seqs,
            tok,
            mdl,
            device,
            layer_agg=layer_agg,
        )

        for rid, seq, emb_res in zip(batch_ids, batch_seqs, emb_list):
            seq_vec = pool_sequence(emb_res, pool)
            seq_rows.append((rid, len(seq), seq_vec))
            if dump_site_emb:
                site_embs[rid] = emb_res

    # --- Long sequences: one-by-one using sliding windows ---
    for rid, seq in long:
        print(f"  [long] {rid} (L={len(seq)})")
        emb_res = residue_embeddings(
            seq,
            tok,
            mdl,
            device,
            layer_agg=layer_agg,
            window_len=window_len,
            overlap=overlap,
        )
        seq_vec = pool_sequence(emb_res, pool)
        seq_rows.append((rid, len(seq), seq_vec))
        if dump_site_emb:
            site_embs[rid] = emb_res

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{in_fa.stem}_seqemb.csv"
    write_seqemb_csv(out_csv, seq_rows, file_tag)

    if dump_site_emb:
        site_out = out_dir / f"{in_fa.stem}_siteemb.npz"
        np.savez_compressed(site_out, **site_embs)

    print(f"[DONE] {in_fa.name} → {out_csv}")
    if dump_site_emb:
        print(f"        site embeddings → {site_out}")

def main():
    ap = argparse.ArgumentParser(description="Compute ESM-2 embeddings for sequences in FASTA(s).")
    ap.add_argument("fastas", nargs="+", type=Path, help="One or more FASTA files")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Directory for outputs")
    ap.add_argument("--model-id", type=str, default="facebook/esm2_t33_650M_UR50D",
                    help="ESM-2 model id (e.g., facebook/esm2_t30_150M_UR50D)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument("--layer-agg", choices=["last", "meanlast4"], default="last",
                    help="Hidden-state aggregation per token")
    ap.add_argument("--pool", choices=["mean"], default="mean",
                    help="Pooling over residues to a sequence vector")
    ap.add_argument("--window-len", type=int, default=1022,
                    help="Window length for long sequences (sliding windows)")
    ap.add_argument("--overlap", type=int, default=128,
                    help="Overlap between windows for long sequences")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for short sequences (<= window-len)")
    ap.add_argument("--dump-site-emb", action="store_true",
                    help="Also save per-residue embeddings to <stem>_siteemb_esm2.npz")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok, mdl = load_esm2(args.model_id, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fa in args.fastas:
        recs = read_fasta(fa)

        # Split into short vs long, using the same cutoff as residue_embeddings
        short = []
        long = []
        for rid, seq in recs:
            if len(seq) <= args.window_len:
                short.append((rid, seq))
            else:
                long.append((rid, seq))

        print(f"[{fa.name}] {len(recs)} records: {len(short)} short, {len(long)} long")

        seq_rows: List[Tuple[str, int, np.ndarray]] = []
        site_store: Dict[str, np.ndarray] = {} if args.dump_site_emb else {}

        # ---- Short sequences: batched, no sliding windows ----
        for start in range(0, len(short), args.batch_size):
            batch = short[start:start + args.batch_size]
            batch_ids = [rid for rid, _ in batch]
            batch_seqs = [seq for _, seq in batch]

            emb_list = batch_residue_embeddings(
                batch_seqs,
                tok,
                mdl,
                device,
                layer_agg=args.layer_agg,
            )

            for rid, seq, emb_res in zip(batch_ids, batch_seqs, emb_list):
                seq_vec = pool_sequence(emb_res, pool=args.pool)
                seq_rows.append((rid, len(seq), seq_vec))
                if args.dump_site_emb:
                    site_store[rid] = emb_res.astype(np.float32)

        # ---- Long sequences: one-by-one with sliding windows ----
        for rid, seq in long:
            print(f"  [long] {rid} (L={len(seq)})")
            emb_res = residue_embeddings(
                seq,
                tok,
                mdl,
                device,
                layer_agg=args.layer_agg,
                window_len=args.window_len,
                overlap=args.overlap,
            )
            seq_vec = pool_sequence(emb_res, pool=args.pool)
            seq_rows.append((rid, len(seq), seq_vec))
            if args.dump_site_emb:
                site_store[rid] = emb_res.astype(np.float32)

        # ---- Write outputs ----
        out_csv = args.out_dir / f"{fa.stem}_seqemb_esm2.csv"
        write_seqemb_csv(out_csv, seq_rows, file_tag=fa.name)

        if args.dump_site_emb and site_store:
            npz_path = args.out_dir / f"{fa.stem}_siteemb_esm2.npz"
            np.savez_compressed(npz_path, **site_store)
            print(f"[OK] {fa.name}: wrote {out_csv.name} and {npz_path.name}")
        else:
            print(f"[OK] {fa.name}: wrote {out_csv.name}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
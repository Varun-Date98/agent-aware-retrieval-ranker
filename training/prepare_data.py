"""Build reranker training data from HotpotQA with easy + hard negatives.

Run:  python -m training.prepare_data
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import load_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _passage_id(title: str, text: int) -> str:
    return hashlib.md5(f"{title}::{text}".encode()).hexdigest()[:12]


def _build_corpus_and_samples(cfg: dict):
    """Return passage corpus and per-sample metadata, using disk cache when available."""
    corpus_path = DATA_DIR / "corpus.jsonl"
    samples_path = DATA_DIR / "samples.jsonl"

    if corpus_path.exists() and samples_path.exists():
        print("  Found cached corpus and samples, loading from disk...")
        corpus = {}
        with open(corpus_path) as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["doc_id"]] = doc
        samples = []
        with open(samples_path) as f:
            for line in f:
                samples.append(json.loads(line))
        return corpus, samples

    print("  Downloading and parsing HotpotQA...")
    ds = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["split"],
        split="train",
    )

    max_samples = cfg["training"]["max_train_samples"] + cfg["training"]["max_val_samples"]
    samples = []
    corpus: dict[str, dict] = {}

    for row in tqdm(ds, desc="Loading HotpotQA", total=min(len(ds), max_samples)):
        if len(samples) >= max_samples:
            break

        question = row["question"]
        answer = row["answer"]

        sup_titles = set(row["supporting_facts"]["title"])
        sup_indices: dict[str, set] = {}
        for t, idx in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
            sup_indices.setdefault(t, set()).add(idx)

        positives = []
        same_sample_negatives = []

        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            full_text = " ".join(sentences)
            doc_id = _passage_id(title, full_text)
            doc = {"doc_id": doc_id, "title": title, "text": full_text}
            corpus[doc_id] = doc

            if title in sup_titles:
                positives.append(doc_id)
            else:
                same_sample_negatives.append(doc_id)

        samples.append({
            "question": question,
            "answer": answer,
            "positives": positives,
            "same_sample_negatives": same_sample_negatives,
        })

    # Cache to disk for future runs
    with open(corpus_path, "w") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + "\n")
    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Cached corpus ({len(corpus)} passages) and samples ({len(samples)}) to {DATA_DIR}")

    return corpus, samples


def _generate_triplets(corpus: dict, samples: list, cfg: dict):
    """Generate (query, passage, label) triplets with easy + hard negatives.

    Hard negatives: same-sample distractor paragraphs (topically related but
    not supporting facts -- already curated by HotpotQA authors).
    Easy negatives: random unrelated passages from the full corpus.
    """
    hard_ratio = cfg["training"]["hard_negative_ratio"]
    easy_ratio = cfg["training"]["easy_negative_ratio"]

    all_doc_ids = list(corpus.keys())
    triplets = []

    for sample in tqdm(samples, desc="Generating triplets"):
        q = sample["question"]
        pos_set = set(sample["positives"])
        same_neg_set = set(sample["same_sample_negatives"])
        hard_candidates = list(same_neg_set)

        for pos_id in sample["positives"]:
            triplets.append({
                "query": q,
                "passage": corpus[pos_id]["text"],
                "label": 1,
                "doc_id": pos_id,
            })

            random.shuffle(hard_candidates)
            for neg_id in hard_candidates[:hard_ratio]:
                if neg_id in corpus:
                    triplets.append({
                        "query": q,
                        "passage": corpus[neg_id]["text"],
                        "label": 0,
                        "doc_id": neg_id,
                    })

            for _ in range(easy_ratio):
                neg_id = random.choice(all_doc_ids)
                while neg_id in pos_set or neg_id in same_neg_set:
                    neg_id = random.choice(all_doc_ids)
                triplets.append({
                    "query": q,
                    "passage": corpus[neg_id]["text"],
                    "label": 0,
                    "doc_id": neg_id,
                })

    return triplets


def main():
    cfg = load_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1/4: Loading HotpotQA and building corpus...")
    corpus, samples = _build_corpus_and_samples(cfg)
    print(f"  Corpus: {len(corpus)} passages, Samples: {len(samples)}")

    print("Step 2/4: Generating triplets...")
    triplets = _generate_triplets(corpus, samples, cfg)
    random.shuffle(triplets)
    print(f"  Total triplets: {len(triplets)}")

    split_idx = int(len(triplets) * 0.9)
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]

    print(f"Step 3/4: Saving JSONL splits (train={len(train_triplets)}, val={len(val_triplets)})...")
    for name, data in [("train", train_triplets), ("val", val_triplets)]:
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for t in data:
                f.write(json.dumps(t) + "\n")
        print(f"  Saved {path}")

    print("Step 4/4: Pre-tokenizing for training...")
    rcfg = cfg["reranker"]
    max_len = cfg["training"]["max_seq_length"]
    tokenizer = AutoTokenizer.from_pretrained(rcfg["model"])

    for name, data in [("train", train_triplets), ("val", val_triplets)]:
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_passage_tokens = []

        for t in tqdm(data, desc=f"Tokenizing {name}"):
            encoded = tokenizer(
                t["query"],
                t["passage"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            all_input_ids.append(encoded["input_ids"].squeeze(0))
            all_attention_mask.append(encoded["attention_mask"].squeeze(0))
            all_labels.append(t["label"])
            passage_toks = len(tokenizer.encode(
                t["passage"], add_special_tokens=False, truncation=True, max_length=max_len
            ))
            all_passage_tokens.append(passage_toks)

        pt_data = {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_mask),
            "labels": torch.tensor(all_labels, dtype=torch.float),
            "passage_tokens": torch.tensor(all_passage_tokens, dtype=torch.float),
        }
        pt_path = DATA_DIR / f"{name}.pt"
        torch.save(pt_data, pt_path)
        print(f"  Saved {pt_path} ({len(data)} samples)")

    print("Done.")


if __name__ == "__main__":
    main()

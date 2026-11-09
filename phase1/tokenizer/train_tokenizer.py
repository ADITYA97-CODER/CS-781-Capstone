"""Stage 3: train the byte-level BPE tokenizer shared by all four Phase-1 models.

The training sample is drawn from the cleaned *training* split only (never the
held-out validation documents), with each source weighted by --mix so the
vocabulary reflects the pretraining mix rather than the raw corpus sizes.

Special tokens (fixed ids, in this order):
  0 <|endoftext|>   document separator / end of generation
  1 <pad>           padding for downstream batches
  2 <mask>          masked-LM objective (Arch C)
  3 <cls>, 4 <sep>  classification / pair inputs in Phases 3-7
  5.. <extra_id_0> .. <extra_id_99>   span-corruption sentinels (Arch B)

Output: artifacts/tokenizer/legal_bpe_<vocab>k/ (Hugging Face fast-tokenizer files).
"""

import argparse
import json
import os
import random

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN_DIR = os.path.join(PROJECT_DIR, "data", "corpus", "clean")
OUT_ROOT = os.path.join(PROJECT_DIR, "artifacts", "tokenizer")

SPECIAL_TOKENS = ["<|endoftext|>", "<pad>", "<mask>", "<cls>", "<sep>"] + [f"<extra_id_{i}>" for i in range(100)]
DEFAULT_MIX = "supreme_court=0.25,high_courts=0.40,tribunals=0.20,acts=0.15"


def parse_mix(spec):
    mix = {k: float(v) for k, v in (item.split("=") for item in spec.split(","))}
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def sample_texts(mix, total_words, seed):
    """Yield whole documents from each source until its share of total_words is reached."""
    rng = random.Random(seed)
    for name, share in mix.items():
        table = pq.read_table(os.path.join(CLEAN_DIR, f"{name}.train.parquet"), columns=["text", "n_words"])
        texts, counts = table.column("text").to_pylist(), table.column("n_words").to_pylist()
        order = list(range(len(texts)))
        rng.shuffle(order)
        budget, used = share * total_words, 0
        for i in order:
            if used >= budget:
                break
            yield texts[i]
            used += counts[i]
        print(f"  {name}: {used / 1e6:.0f}M words", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--sample-words", type=int, default=300_000_000)
    p.add_argument("--mix", default=DEFAULT_MIX)
    p.add_argument("--min-frequency", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"sampling {args.sample_words / 1e6:.0f}M words with mix {parse_mix(args.mix)}", flush=True)
    tokenizer.train_from_iterator(sample_texts(parse_mix(args.mix), args.sample_words, args.seed), trainer=trainer)

    out_dir = os.path.join(OUT_ROOT, f"legal_bpe_{args.vocab_size // 1000}k")
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>", bos_token="<|endoftext|>", pad_token="<pad>",
        mask_token="<mask>", cls_token="<cls>", sep_token="<sep>",
        additional_special_tokens=[f"<extra_id_{i}>" for i in range(100)],
        model_max_length=8192,
    )
    hf_tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    print(f"saved {len(hf_tok):,}-token tokenizer -> {out_dir}")


if __name__ == "__main__":
    main()

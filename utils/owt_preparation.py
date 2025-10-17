import argparse
import logging
from itertools import chain

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


# Constants
DATASET_NAME = "Skylion007/openwebtext"
BASE_TOKENIZER = "bert-base-cased"
MEAN_SYMBOLS_PER_TOKEN = 3.5
DEFAULT_NUM_TOKENS = 128
SENTENCE_THRESHOLD_TOKENS = 150
HISTOGRAM_SAMPLE_SIZE = 100_000
TEST_SPLIT_SIZE = 50_000
RANDOM_SEED = 42
NUM_PROC = 30
MAP_BATCH_SIZE = 1000
MAX_HIST_LENGTH = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare OpenWebText dataset.")
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument("--output_dir_template", type=str, default="./data/openwebtext-{}")
    parser.add_argument("--num_tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--num_proc", type=int, default=NUM_PROC)
    return parser.parse_args()


def split_by_newline(batch):
    result = [text.split("\n\n") for text in batch["text"]]
    return {"text": list(chain(*result))}


def setup_nltk_tokenizer():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    return nltk.data.load('tokenizers/punkt/english.pickle')


def create_sentence_splitter(sentence_tokenizer, threshold):
    def split_and_join_into_sents(batch):
        result = []
        for text in batch["text"]:
            sents = sentence_tokenizer.tokenize(text)
            current_chunk = ''
            for sent in sents:
                if len(current_chunk.split()) + len(sent.split()) < threshold:
                    current_chunk = f"{current_chunk} {sent}" if current_chunk else sent
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    current_chunk = sent
            if current_chunk:
                result.append(current_chunk)
        return {"text": result}
    return split_and_join_into_sents


def plot_histogram(lengths, num_tokens):
    logging.info("Generating and saving histogram.")
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 6))
    
    font_size = 20
    num_bins = 100
    bins = np.linspace(0, MAX_HIST_LENGTH, num_bins + 1)
    
    sns.histplot(lengths, bins=bins, color="green", label="OpenWebText", alpha=0.7, edgecolor="black", stat="density")
    
    plt.xlabel("Text length in tokens", fontsize=font_size + 5)
    plt.ylabel("Density", fontsize=font_size + 5)
    plt.xlim(0, MAX_HIST_LENGTH)
    plt.xticks(fontsize=font_size - 5)
    plt.yticks(fontsize=font_size - 5)
    
    output_path = f"./notebooks/openwebtext-{num_tokens}-histogram.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    logging.info(f"Histogram saved to {output_path}")
    plt.show()


def main():
    args = parse_args()
    
    logging.info(f"Loading dataset: {args.dataset_name}")
    dt = load_dataset(args.dataset_name)["train"]
    
    logging.info("Splitting dataset by newlines.")
    dt = dt.map(
        split_by_newline,
        batched=True,
        num_proc=args.num_proc,
        desc="Splitting texts by newline",
        batch_size=MAP_BATCH_SIZE,
    )
    
    logging.info("Filtering short texts.")
    min_symbols = MEAN_SYMBOLS_PER_TOKEN * args.num_tokens
    dt = dt.filter(lambda b: len(b["text"]) >= min_symbols, num_proc=args.num_proc)
    
    logging.info("Splitting and joining sentences.")
    nltk_tokenizer = setup_nltk_tokenizer()
    sentence_splitter = create_sentence_splitter(nltk_tokenizer, SENTENCE_THRESHOLD_TOKENS)
    joined_dt = dt.map(
        sentence_splitter,
        batched=True,
        num_proc=args.num_proc,
        desc="Splitting and joining sentences",
        batch_size=MAP_BATCH_SIZE,
    )
    
    logging.info("Tokenizing for histogram.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    
    sampled_dt = joined_dt.shuffle(seed=RANDOM_SEED).select(range(HISTOGRAM_SAMPLE_SIZE))
    texts = sampled_dt["text"]
    
    lengths = []
    for i in tqdm(range(0, len(texts), MAP_BATCH_SIZE), desc="Tokenizing sample"):
        batch = tokenizer(texts[i:i + MAP_BATCH_SIZE])
        lengths.extend([len(t) for t in batch["input_ids"]])
        
    total_tokens_estimate = sum(lengths) / HISTOGRAM_SAMPLE_SIZE * len(joined_dt)
    logging.info(f"Estimated total number of tokens: {total_tokens_estimate:,.0f}")

    plot_histogram(lengths, args.num_tokens)
    
    joined_dt = joined_dt.rename_column("text", "text_trg")
    
    logging.info(f"Splitting dataset with test size: {TEST_SPLIT_SIZE}")
    train_test_split = joined_dt.train_test_split(test_size=TEST_SPLIT_SIZE, shuffle=True, seed=RANDOM_SEED)
    
    output_dir = args.output_dir_template.format(args.num_tokens)
    logging.info(f"Saving dataset to {output_dir}")
    train_test_split.save_to_disk(output_dir, max_shard_size="2GB")
    logging.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
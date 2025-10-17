import os
from itertools import chain
from pathlib import Path

import nltk
from datasets import load_dataset


# Constants
DATASET_NAME = "bigscience-data/roots_en_wikipedia"
MIN_SYMBOLS = 150
MAX_WORDS_PER_CHUNK = 128
TEST_SIZE = 50000
# Use all available cores, default to 4 if cpu_count is None
NUM_PROC = os.cpu_count() or 4
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "wikipedia"


def split_paragraphs(batch):
    """Splits texts into paragraphs."""
    result = []
    for text in batch["text"]:
        texts = text.split("\n\n")
        result.append(texts)
    result = list(chain(*result))
    return {"text": result}


def create_sentence_splitter():
    """Creates and returns NLTK sentence tokenizer."""
    try:
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt")
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer


def split_and_join_into_sents(batch, tokenizer):
    """Splits text into sentences and joins them into chunks of a maximum word count."""
    result = []
    for text in batch["text"]:
        sentences = tokenizer.tokenize(text)

        current_chunk = ""
        for sentence in sentences:
            # Check if adding the next sentence exceeds the max word count
            if len(current_chunk.split()) + len(sentence.split()) < MAX_WORDS_PER_CHUNK:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # If the chunk is not empty, add it to the result
                if current_chunk:
                    result.append(current_chunk)
                # Start a new chunk with the current sentence
                current_chunk = sentence

        # Add the last chunk if it exists
        if current_chunk:
            result.append(current_chunk)

    return {"text": result}


def main():
    """
    Downloads and prepares the Wikipedia dataset.
    The pipeline includes:
    1. Loading the 'bigscience-data/roots_en_wikipedia' dataset.
    2. Splitting articles into paragraphs.
    3. Filtering out short paragraphs.
    4. Splitting paragraphs into sentences and grouping them into chunks.
    5. Splitting the data into training and testing sets.
    6. Saving the processed dataset to disk.
    """
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.remove_columns("meta")

    print("Splitting into paragraphs...")
    dataset = dataset.map(
        split_paragraphs,
        batched=True,
        num_proc=NUM_PROC,
        desc="Splitting into paragraphs",
        batch_size=1000,
    )

    print(f"Filtering out texts with less than {MIN_SYMBOLS} symbols...")
    dataset = dataset.filter(lambda b: len(b["text"]) >= MIN_SYMBOLS, num_proc=NUM_PROC)

    print("Splitting into sentences and joining into chunks...")
    sentence_tokenizer = create_sentence_splitter()
    dataset = dataset.map(
        lambda batch: split_and_join_into_sents(batch, sentence_tokenizer),
        batched=True,
        num_proc=NUM_PROC,
        desc=f"Joining sentences into chunks of ~{MAX_WORDS_PER_CHUNK} words",
        batch_size=1000,
    )

    print("Renaming 'text' column to 'text_trg'...")
    dataset = dataset.rename_column("text", "text_trg")

    print(f"Splitting into train and test sets (test_size={TEST_SIZE})...")
    train_test_split = dataset.train_test_split(test_size=TEST_SIZE, shuffle=True, seed=42)

    print(f"Saving dataset to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_test_split.save_to_disk(str(OUTPUT_DIR), max_shard_size="2GB")

    print("Dataset preparation finished successfully!")
    print(f"Train set size: {len(train_test_split['train'])}")
    print(f"Test set size: {len(train_test_split['test'])}")


if __name__ == "__main__":
    main()

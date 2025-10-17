from transformers import AutoTokenizer
import spacy
from nltk.util import ngrams
import pickle
import os
from datasets import load_from_disk


def truncate_text(texts, max_length, min_required_length):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_texts = tokenizer(texts, padding=False, truncation=True, max_length=max_length)["input_ids"]
    tokenized_texts = [text for text in tokenized_texts if len(text) >= min_required_length]
    truncated_texts = tokenizer.batch_decode(tokenized_texts, skip_special_tokens=True)
    return truncated_texts

def get_unique_four_grams(train_unique_four_grams, train_dataset_path):
    if os.path.exists(train_unique_four_grams):
        with open(train_unique_four_grams, 'rb') as f:
            unique_four_grams = pickle.load(f)
    else:
        N = 100000
        n = 4
        train_dataset = load_from_disk(train_dataset_path)[:N]["text_trg"]

        tokenizer = spacy.load("en_core_web_sm").tokenizer
        unique_four_grams = set()
        for sentence in train_dataset:
            unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

        with open(train_unique_four_grams, 'wb') as f:
            pickle.dump(unique_four_grams, f)
    
    return unique_four_grams
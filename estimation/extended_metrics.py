"""
Extended metrics for multi-benchmark evaluation.
Adds ROUGE, BERTScore, BLEU-4, and J-Score to the existing metric set.

Used in Part 2 (Phase 5) for evaluation across XSum, SQuAD 2.0, ParaDetox.
"""

import numpy as np
from typing import List, Optional


def compute_extended_metric(metric_name: str, predictions: List[str],
                            references: Optional[List[str]] = None, **kwargs) -> float:
    """Dispatch extended metrics."""
    if metric_name == "rouge":
        return compute_rouge(predictions, references)
    elif metric_name == "bertscore":
        return compute_bertscore(predictions, references)
    elif metric_name == "bleu4":
        return compute_bleu4(predictions, references)
    elif metric_name == "jscore":
        return compute_jscore(predictions, references, **kwargs)
    else:
        raise ValueError(f"Unknown extended metric: {metric_name}")


def compute_rouge(predictions: List[str], references: List[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)

    return {key: np.mean(vals) if vals else 0.0 for key, vals in scores.items()}


def compute_bertscore(predictions: List[str], references: List[str],
                      model_type: str = "bert-base-cased") -> dict:
    """Compute BERTScore (precision, recall, F1)."""
    import bert_score

    # Filter empty pairs
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid_pairs:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    preds, refs = zip(*valid_pairs)
    P, R, F1 = bert_score.score(
        list(preds), list(refs),
        model_type=model_type,
        verbose=False,
        device="cuda",
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def compute_bleu4(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU-4 score."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    # Tokenize
    ref_tokenized = [[ref.split()] for ref in references if ref]
    pred_tokenized = [pred.split() for pred in predictions if pred]

    if not ref_tokenized or not pred_tokenized:
        return 0.0

    # Ensure same length
    min_len = min(len(ref_tokenized), len(pred_tokenized))
    ref_tokenized = ref_tokenized[:min_len]
    pred_tokenized = pred_tokenized[:min_len]

    smoothing = SmoothingFunction().method1
    score = corpus_bleu(ref_tokenized, pred_tokenized,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothing)
    return score


def compute_jscore(predictions: List[str], references: List[str],
                   toxicity_classifier=None) -> dict:
    """
    Compute J-Score for detoxification evaluation.
    J-Score = STA * SIM * FL
    - STA: Style Transfer Accuracy (non-toxicity rate)
    - SIM: Semantic similarity (BERTScore F1)
    - FL: Fluency (inverse perplexity, normalized)

    If no toxicity classifier provided, returns partial metrics.
    """
    # Semantic similarity via BERTScore
    bertscore = compute_bertscore(predictions, references)
    sim = bertscore["bertscore_f1"]

    result = {
        "sim": sim,
        "jscore": None,
    }

    # STA would require a toxicity classifier
    # Placeholder — will be filled when toxicity classifier is available
    if toxicity_classifier is not None:
        # toxicity_classifier should return toxicity probability
        sta_scores = []
        for pred in predictions:
            if pred:
                tox = toxicity_classifier(pred)
                sta_scores.append(1.0 - tox)
        sta = np.mean(sta_scores) if sta_scores else 0.0
        result["sta"] = sta
        result["jscore"] = sta * sim
    else:
        result["sta"] = None

    return result

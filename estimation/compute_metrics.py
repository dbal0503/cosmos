import argparse
import json
from transformers import AutoTokenizer
import numpy as np
from omegaconf import OmegaConf

from .metrics import compute_metric
from .util import truncate_text
from utils.hydra_utils import load_config

def main(args, config):
    num_times = args.num_times
    max_length = args.max_length
    num_samples = args.num_samples
    min_required_length = args.min_required_length

    with open(args.pred_file, "r") as f:
        pred_data = json.load(f)

    predictions = [pred["GEN"] for pred in pred_data]
    predictions = truncate_text(predictions, max_length, min_required_length)

    references = [pred["TRG"] for pred in pred_data]
    references = truncate_text(references, max_length, min_required_length)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of references: {len(references)}")

    import random
    
    # Randomly sample num_samples texts if there are more than num_samples

    metrics_dict = {
        "ppl": [],
        "mauve": [],
        "div": [],
        "mem": [],
    }

    for i in range(num_times):
        indices = random.sample(range(len(predictions)), num_samples)
        batch_predictions = [predictions[i] for i in indices]
        indices = random.sample(range(len(references)), num_samples)
        batch_references = [references[i] for i in indices]

        for metric_name in ["mem", "ppl", "mauve", "div"]:
            metrics_dict[metric_name].append(compute_metric(
                metric_name, 
                predictions=batch_predictions, 
                references=batch_references,
                train_unique_four_grams=config.diffusion.generation.train_unique_four_grams,
                train_dataset_path=f"{config.dataset.dataset_path}/train",
            ))

    print(f"Pred_file: {args.pred_file}")
    for key, value in metrics_dict.items():
        print(f"{key}: {np.mean(value):0.5f} ± {np.std(value):0.5f}")
        print(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--num_times", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--min_required_length", type=int, default=500)
    args = parser.parse_args()

    config = load_config(project_root="/home/vime725h/LatentDiffusion", config_dir_path="../conf")

    main(args, config)

"""
CUDA_VISIBLE_DEVICES=0 \
python -m estimation.compute_metrics \
    --pred_file ./generated_texts/diffusion-openwebtext-512-64-d=3-final-512/500000-N=200-len=3000.json \
    --num_times 1 \
    --num_samples 512 \
    --min_required_length 0 \
    --max_length 512

CUDA_VISIBLE_DEVICES=0 \
python -m estimation.compute_metrics \
    --pred_file ./generated_texts/diffusion-openwebtext-512-512-d=5-48-layers-512/500000-N=256-len=512.json \
    --num_times 1 \
    --num_samples 512 \
    --min_required_length 0 \
    --max_length 512




python -m estimation.compute_metrics \
    --pred_file ./generated_texts/gidd-small.json \
    --num_times 1 \
    --num_samples 512 \
    --min_required_length 0 \
    --max_length 512

"""
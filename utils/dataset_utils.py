import os
import gc
import torch
import numpy as np
from random import random
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from itertools import cycle
from transformers import AutoTokenizer
from typing import List, Union, Optional, Dict, Any
from collections import UserDict


class DatasetDDP:
    def __init__(self, config, split):
        self.split = split
        self.config = config
        self.dataset_name = config.dataset.name
        self.swap_cfg_coef = config.dataset.swap_cfg_coef
        
        self.base_path = config.dataset.dataset_path
        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        
        self.epoch = 0
        self.files = self.get_files()
        self.iter_files = cycle(self.files)

    def get_files(self):
        path = f"{self.base_path}/{self.split}/"
        files = list(os.listdir(path))
        files = [t for t in files if ".arrow" in t and t.startswith("data")]
        files = sorted(files, key = lambda f: int(f.split("-")[1]))
        return files

    def spilt_data_across_gpu(self, dt: List[str]):
        self.epoch += 1
        if self.split == "train":
            indexes = np.random.default_rng(seed=self.epoch).permutation(len(dt))
        else:
            indexes = np.arange(len(dt))
        
        start_ind = self.device_id * (len(dt) // self.total_device_number)
        end_ind = (self.device_id + 1) * (len(dt) // self.total_device_number)
        if (self.device_id + 1) == self.total_device_number:
            indexes = indexes[start_ind:]
        else:
            indexes = indexes[start_ind: end_ind]
        
        return Dataset.from_dict(dt[indexes])

    def load_data(self):
        file = next(self.iter_files)
        path = f"{self.base_path}/{self.split}/{file}"
        dt = Dataset.from_file(path)
        dt = self.spilt_data_across_gpu(dt)

        rename_dict = {
            "text": "text_trg",
            "source": "text_src",
            "question1": "text_trg",
            "question2": "text_src",
            "original": "text_trg",
            "paraphrase": "text_src",
            "target": "text_trg",
        }

        for k, v in rename_dict.items():
            if k in dt.features:
                dt = dt.rename_column(k, v)

        if "text_src" in dt.features:
            if self.swap_cfg_coef and self.split == "train":
                dt = dt.map(
                    self.cfg_swap_function,
                    batched=True,
                    load_from_cache_file=False,
                    num_proc=30,
                    desc="Applying classifier free guidance swap",
                    batch_size=1000,
                )
        self.dt = dt
        return self.dt
    
    def cfg_swap_function(self, batch):
        batch["text_src"] = ["" if p < self.swap_cfg_coef else sample for p, sample in zip(torch.rand(len(batch["text_src"])), batch["text_src"])]
        return batch
            
    def get_dataset_iter(self):
        while True:
            yield self.load_data()
            del self.dt
            gc.collect()


class BatchEncoding(UserDict):
    def __init__(self, data: Optional[Dict[str, Any]] = None, return_tp="pt"):
        super().__init__(data)
        self.return_tp = return_tp

    def __getitem__(self, item: str) -> Union[str, torch.Tensor]:
        if isinstance(item, str):
            value = self.data[item]
            if item.startswith("text"):
                return value
            else:
                if self.return_tp == "pt":
                    return torch.Tensor(value)
                else:
                    return value
        elif isinstance(item, slice):
            return BatchEncoding({key: self.data[key][item] for key in self.data.keys()})
        else:
            raise KeyError(
                "Invalid key. Only two types of key are available: "
                "(1) string, and (2) slices for data subsetting."
            )

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        for k, v in self.data.items():
            if not k.startswith("text"):
                self.data[k] = v.to(device=device)
        return self

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

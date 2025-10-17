import os
import queue
import threading
import time
from typing import Optional
from datasets import load_from_disk
from datasets import Dataset


class ShardedDataset:
    def __init__(self, config, split="train", prefetch_shards=2, logger=None):
        self.config = config
        self.split = split
        self.prefetch_shards = prefetch_shards
        
        self.dataset_path = os.path.join(config.dataset.dataset_path, split)
        self.logger = logger
        
        # Load dataset metadata only (not the actual data)
        self.shards = self.read_shards(self.dataset_path)
        self.shards.sort()
        self.num_shards = len(self.shards)
        
        self.current_shard_idx = 0
        self.shard_queue = queue.Queue(maxsize=prefetch_shards)
        self.loading_thread = None
        self.stop_loading = False
        
        if self.logger:
            self.logger.info(f"ShardedDataset initialized: {self.num_shards} shards, prefetch_shards={prefetch_shards}")
        
        # Start prefetching thread
        self._start_prefetch_thread()

    def read_shards(self, dataset_path):
        shards = []
        for file in os.listdir(dataset_path):
            if file.endswith(".arrow") or os.path.isdir(os.path.join(dataset_path, file)):
                shards.append(file)
        return shards
    
    def _start_prefetch_thread(self):
        """Start thread for prefetching shards"""
        self.loading_thread = threading.Thread(target=self._prefetch_worker)
        self.loading_thread.daemon = True
        self.loading_thread.start()
        
        if self.logger:
            self.logger.info("Prefetch thread started")
    
    def _prefetch_worker(self):
        """Worker thread for prefetching shards with retry logic"""
        shard_idx = 0
        while not self.stop_loading:
            try:
                if self.logger and shard_idx % 10 == 0:  # Log every 10th shard
                    self.logger.info(f"Prefetching shard {shard_idx}/{len(self.shards)}")
                
                # Use retry logic for shard loading
                shard = self.get_shard_with_retry(shard_idx)
                self.shard_queue.put(shard, timeout=100)
                shard_idx = (shard_idx + 1) % self.num_shards
                
            except queue.Full:
                # Queue is full, skip this iteration
                continue
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error prefetching shard {shard_idx}: {e}")
                break
        
        if self.logger:
            self.logger.info("Prefetch worker thread finished")
    
    def get_shard_with_retry(self, shard_idx: int, max_retries: int = 10) -> Dataset:
        """Load shard with retry logic for race conditions"""
        for attempt in range(max_retries):
            try:
                return self.get_shard(shard_idx)
            except FileNotFoundError as e:
                if attempt < max_retries - 1:
                    if self.logger:
                        self.logger.warning(f"File not found for shard {shard_idx}, attempt {attempt + 1}/{max_retries}, retrying in 1s...")
                    time.sleep(1)  # Wait for file to be created
                    continue
                else:
                    if self.logger:
                        self.logger.error(f"Failed to load shard {shard_idx} after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Unexpected error loading shard {shard_idx}: {e}")
                raise
    
    def get_next_shard(self) -> Optional[Dataset]:
        """Get next prefetched shard"""
        try:
            shard = self.shard_queue.get(timeout=100)
            if self.logger:
                self.logger.debug(f"Retrieved shard from queue, size: {len(shard)}")
            return shard
        except queue.Empty:
            if self.logger:
                self.logger.warning("No prefetched shards available, timeout reached")
            return None
    
    def stop(self):
        """Stop prefetching"""
        if self.logger:
            self.logger.info("Stopping ShardedDataset...")
        
        self.stop_loading = True
        if self.loading_thread:
            self.loading_thread.join(timeout=10)  # Wait max 10 seconds
            
        if self.logger:
            self.logger.info("ShardedDataset stopped")
    
    def reset(self):
        """Reset the shard index"""
        self.current_shard_idx = 0

    def get_shard(self, shard_idx: int) -> Dataset:
        """Load specific shard of dataset"""
        if self.logger:
            self.logger.debug(f"Loading shard {shard_idx}")
        
        # Load only the required shard
        shard_name = self.shards[shard_idx]
        shard_path = os.path.join(self.dataset_path, shard_name)
        
        if os.path.isdir(shard_path):
            shard = load_from_disk(shard_path)
        else:
            shard = Dataset.from_file(shard_path)
        
        # rename text column
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
            if k in shard.features:
                shard = shard.rename_column(k, v)
        
        return shard
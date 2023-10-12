import random
from .tensor_util import trees_to_batch_tensors


class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def make_mini_batch_iterator(self, buckets):
        bucket_ids = list(buckets.keys())
        random.shuffle(bucket_ids)
        for bucket_idx in bucket_ids:
            trees = buckets[bucket_idx]
            random.shuffle(trees)
            
            batch_trees = []
            samples = 0
            for i, tree in enumerate(trees):
                batch_trees.append(tree)
                samples += 1
                
                if samples >= self.batch_size:
                    batch_obj = trees_to_batch_tensors(batch_trees)
                
                    yield batch_obj
                    batch_trees = []
                    samples = 0


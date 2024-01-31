import random
import logging
from tensor_util import *


class DataLoader:
    LOGGER = logging.getLogger('DataLoader')

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def make_minibatch_iterator(self, buckets, num_pos, num_neg):
        bucket_ids = list(buckets.keys())
        random.shuffle(bucket_ids)
        for bucket_idx in bucket_ids:
            self.LOGGER.debug("Switching bucket...")
            trees = buckets[bucket_idx]
            self.LOGGER.debug(f"Num items in bucket {len(trees)}")
            random.shuffle(trees)

            batch_trees = []
            samples = 0
            for i, tree in enumerate(trees):
                batch_trees.append(tree)
                samples += 1

                if samples >= self.batch_size:
                    flatten_batch_trees = []
                    for _ in range(1 + num_pos + num_neg):
                        for all_samples in batch_trees:
                            flatten_batch_trees.append(all_samples[_])
                    batch_obj = trees_to_batch_tensors(flatten_batch_trees)

                    yield batch_obj
                    batch_trees = []
                    samples = 0


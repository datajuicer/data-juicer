from itertools import chain, repeat

import numpy as np


def random_sample(dataset, weight=1.0, sample_number=0, seed=None):
    """
    Randomly sample a subset from a dataset with weight or number,
    if sample number is bigger than 0, we will use sample
    number instead of weight.
    :param dataset: a HuggingFace dataset
    :param weight: sample ratio of dataset
    :param sample_number: sample number of dataset
    :param seed: random sample seed, if None, 42 as default
    :return: a subset of dataset
    """
    if seed is None:
        seed = 42

    ds_samples = dataset.num_rows
    if sample_number <= 0:
        sample_number = int(np.ceil(ds_samples * weight))

    if sample_number == ds_samples:
        return dataset

    # When sampling more samples than the dataset size, repeat the dataset
    if sample_number > ds_samples:
        sample_index = range(sample_number)
        n_repeat = int(np.ceil(sample_number / ds_samples)) - 1
        if n_repeat > 0:
            remain_samples = sample_number - n_repeat * ds_samples
            sample_index = chain(*repeat(range(ds_samples), n_repeat), range(remain_samples))
        return dataset.shuffle(seed=seed).select(sample_index)

    # `datasets.Dataset.shuffle(seed)` is implemented as
    # `np.random.default_rng(seed).permutation(n)` followed by `select`, so
    # slicing the permutation keeps exactly the rows that `main` selects while
    # avoiding the full shuffled index mapping. `random.sample` is O(k) but
    # builds a Python list and changes the selected rows for the same seed.
    rng = np.random.default_rng(seed)
    sample_index = rng.permutation(ds_samples)[:sample_number]
    return dataset.select(sample_index)

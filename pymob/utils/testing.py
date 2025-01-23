import numpy as np

def assert_no_infs_in_dataset(dataset):
    infs = bool((dataset == np.inf).sum().to_array().any())
    if infs:
        raise ValueError("Dataset contained infinity values")

def assert_no_nans_in_dataset(dataset):
    nans = bool(dataset.isnull().sum().to_array().any())
    if nans:
        raise ValueError("Dataset contained NaN values")


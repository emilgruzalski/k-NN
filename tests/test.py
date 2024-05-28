import pytest
import sys

sys.path.insert(0, "../")

import pandas as pd
from app.main import KNNModel

# Fixture for loading sample data


@pytest.fixture()
def knn_model():
    model = KNNModel()

    model.load_data(
        "sample_data.csv"
    )  # Ensure this file exists and is the correct format

    return model


def test_load_data(knn_model):
    assert knn_model.data is not None, "Data should be loaded"

    assert not knn_model.data.empty, "Data should not be empty"

    assert knn_model.norm_data is not None, "Normalized data should be available"


def test_normalization(knn_model):
    norm_data = knn_model.norm_data

    min_values = norm_data.min()

    max_values = norm_data.max()

    assert (min_values[:2] == 0).all(), "Normalized data should have a min of 0"

    assert (max_values[:2] == 1).all(), "Normalized data should have a max of 1"


def test_k_update(knn_model):
    knn_model.k = 5

    assert knn_model.k == 5, "k should be updated to 5"


def test_metric_update(knn_model):
    knn_model.metric = "manhattan"

    assert knn_model.metric == "manhattan", "Metric should be updated to Manhattan"


def test_vote_update(knn_model):
    knn_model.vote = "weighted"

    assert knn_model.vote == "weighted", "Vote type should be updated to weighted"


def test_classification_euclidean_simple(knn_model):
    knn_model.metric = "euclidean"

    knn_model.vote = "simple"

    knn_model.k = 3

    category, nearest_indexes, nearest_dists = knn_model.classify(pd.Series([0.5, 0.5]))

    assert isinstance(category, int), "Category should be an integer"

    assert (
        len(nearest_indexes) == knn_model.k
    ), f"Should have {knn_model.k} nearest neighbors"

    assert all(nearest_dists >= 0), "Distances should be non-negative"


def test_classification_manhattan_weighted(knn_model):
    knn_model.metric = "manhattan"

    knn_model.vote = "weighted"

    knn_model.k = 3

    category, nearest_indexes, nearest_dists = knn_model.classify(pd.Series([0.5, 0.5]))

    assert isinstance(category, int), "Category should be an integer"

    assert (
        len(nearest_indexes) == knn_model.k
    ), f"Should have {knn_model.k} nearest neighbors"

    assert all(nearest_dists >= 0), "Distances should be non-negative"


if __name__ == "__main__":
    pytest.main()

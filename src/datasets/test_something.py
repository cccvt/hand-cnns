import pytest

from src.datasets.something import SomethingSomething
import src.datasets.something as something


@pytest.fixture(scope="module")
def smthg_dataset():
    return SomethingSomething()


def test_get_samples(smthg_dataset):
    assert len(something.get_samples(smthg_dataset.video_path)) == 108499


def test_get_split_ids(smthg_dataset):
    assert len(something.get_split_ids(smthg_dataset.valid_path)) == 11522
    assert len(something.get_split_ids(smthg_dataset.test_path)) == 10960


def test_get_split_labels(smthg_dataset):
    assert len(something.get_split_labels(smthg_dataset.valid_path)) == 11522
    assert len(something.get_split_labels(smthg_dataset.train_path)) == 86017


def test_get_dense_samples(smthg_dataset):
    assert len(smthg_dataset.get_dense_samples(10)) > 0

"""Unit tests for APTOS and Messidor-2 PyTorch Datasets."""

import torch
from src.data.aptos_dataset import APTOSDataset
from src.data.messidor_dataset import Messidor2Dataset
from src.data.samplers import create_weighted_sampler
from src.data.transforms import get_val_transforms


def test_aptos_dataset_iteration(mock_dataset_dir):
    """Test APTOSDataset initialization, length, and sample output."""
    transforms = get_val_transforms(image_size=380)
    dataset = APTOSDataset(
        df_or_csv=mock_dataset_dir["csv_path"],
        images_dir=mock_dataset_dir["images_dir"],
        transform=transforms,
        image_size=380,
    )

    assert len(dataset) == 10
    img_tensor, label, img_path = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 380, 380)
    assert 0 <= label <= 4
    assert isinstance(img_path, str)


def test_aptos_class_weights(mock_dataset_dir):
    """Test class weights computation for loss balancing."""
    dataset = APTOSDataset(
        df_or_csv=mock_dataset_dir["csv_path"],
        images_dir=mock_dataset_dir["images_dir"],
    )
    weights = dataset.get_class_weights()
    assert isinstance(weights, torch.Tensor)
    assert len(weights) == 5
    assert torch.all(weights > 0)


def test_messidor2_filtering(mock_messidor_dir):
    """Verify Messidor-2 dataset strictly filters adjudicated_gradable == 1."""
    dataset = Messidor2Dataset(
        df_or_csv=mock_messidor_dir["csv_path"],
        images_dir=mock_messidor_dir["images_dir"],
        filter_gradable=True,
    )
    # 8 of 10 samples were gradable
    assert len(dataset) == 8


def test_weighted_sampler(mock_dataset_dir):
    """Verify weighted random sampler generation."""
    labels = mock_dataset_dir["df"]["diagnosis"].values
    sampler = create_weighted_sampler(labels)
    assert len(sampler) == len(labels)

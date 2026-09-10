"""Pytest fixtures for creating synthetic datasets and mock images."""

import os
import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_fundus_image():
    """Generate a synthetic 400x400 retinal fundus image with dark borders and blood vessels."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw circular retina fundus region (reddish/orange)
    center = (200, 200)
    radius = 170
    cv2.circle(img, center, radius, (30, 70, 200), -1)

    # Draw simulated optic disc (yellow/bright circle)
    cv2.circle(img, (140, 200), 25, (100, 220, 240), -1)

    # Draw simulated blood vessels
    cv2.line(img, (140, 200), (280, 130), (10, 20, 100), 3)
    cv2.line(img, (140, 200), (270, 270), (10, 20, 100), 2)
    return img


@pytest.fixture
def mock_dataset_dir(tmp_path, sample_fundus_image):
    """Create a temporary dataset directory with CSV and sample images."""
    data_dir = tmp_path / "aptos_mock"
    images_dir = data_dir / "train_images"
    images_dir.mkdir(parents=True)

    records = []
    for i in range(10):
        img_id = f"test_sample_{i}"
        img_filename = f"{img_id}.png"
        img_path = str(images_dir / img_filename)
        # Save synthetic fundus image
        cv2.imwrite(img_path, cv2.cvtColor(sample_fundus_image, cv2.COLOR_RGB2BGR))
        # Assign variety of grades 0..4
        grade = i % 5
        records.append({"id_code": img_id, "diagnosis": grade})

    df = pd.DataFrame(records)
    csv_path = str(data_dir / "train.csv")
    df.to_csv(csv_path, index=False)

    return {
        "data_dir": str(data_dir),
        "csv_path": csv_path,
        "images_dir": str(images_dir),
        "df": df,
    }


@pytest.fixture
def mock_messidor_dir(tmp_path, sample_fundus_image):
    """Create a temporary Messidor-2 directory with adjudicated_gradable column."""
    data_dir = tmp_path / "messidor_mock"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)

    records = []
    for i in range(10):
        img_id = f"messidor_{i}"
        img_filename = f"{img_id}.png"
        img_path = str(images_dir / img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(sample_fundus_image, cv2.COLOR_RGB2BGR))

        # Make 8 gradable and 2 ungradable
        gradable = 1 if i < 8 else 0
        grade = i % 5
        records.append(
            {
                "image_id": img_id,
                "adjudicated_dr_grade": grade,
                "adjudicated_gradable": gradable,
            }
        )

    df = pd.DataFrame(records)
    csv_path = str(data_dir / "messidor_data.csv")
    df.to_csv(csv_path, index=False)

    return {
        "data_dir": str(data_dir),
        "csv_path": csv_path,
        "images_dir": str(images_dir),
        "df": df,
    }

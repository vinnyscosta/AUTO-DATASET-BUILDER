import os
import pytest
from dataset_builder.builder import Dataset, ImageLabel


@pytest.fixture
def mock_dataset_structure(tmp_path):
    # Cria estrutura de pastas temporárias para images e labels
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Cria imagens e labels fictícios
    for i in range(5):
        image_path = images_dir / f"img{i}.jpg"
        label_path = labels_dir / f"img{i}.txt"
        image_path.write_bytes(b"fake image data")
        label_path.write_text("label data")

    return tmp_path


def test_image_label_success(mock_dataset_structure):
    origin_path = str(mock_dataset_structure)
    filename = "img0.jpg"
    filetype = "train"

    # Cria pastas de destino
    os.makedirs(f"dataset/images/{filetype}", exist_ok=True)
    os.makedirs(f"dataset/labels/{filetype}", exist_ok=True)

    label = ImageLabel(filetype, filename, origin_path)  # noqa: F841
    assert os.path.exists(f"dataset/images/{filetype}/img0.jpg")
    assert os.path.exists(f"dataset/labels/{filetype}/img0.txt")


def test_dataset_split_and_creation(mock_dataset_structure):
    dataset = Dataset(str(mock_dataset_structure))

    assert len(dataset.itens["train"]) > 0
    assert len(dataset.itens["val"]) > 0
    assert len(dataset.itens["test"]) > 0

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


class ImageLabel:
    """Class to handle image and label files for a dataset."""
    def __init__(self, filetype: str, filename: str, origin_path: str):
        self.filetype = filetype
        self.filename = filename

        # Image
        self.image_filename = filename
        self.image_path = os.path.join(origin_path, 'images', self.image_filename)  # noqa: E501

        # Label
        self.label_filename = filename.replace('.jpg', '.txt')
        self.label_path = os.path.join(origin_path, 'labels', self.label_filename)  # noqa: E501

        # Check if image and label exist
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file {self.image_path} does not exist.")  # noqa: E501
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file {self.label_path} does not exist.")  # noqa: E501

        # Add to dataset
        self.add_to_dataset()

    def add_to_dataset(self):
        """Add image and label files to the dataset."""
        shutil.copy(self.image_path, f"dataset/images/{self.filetype}/{self.image_filename}")  # noqa: E501
        shutil.copy(self.label_path, f"dataset/labels/{self.filetype}/{self.label_filename}")  # noqa: E501

    def __str__(self):
        return f"Image: {self.image_path}, Label: {self.label_path}"


class Dataset:
    """Cria um dataset para treinamento de um modelo de machine learning.
    O dataset é dividido em três partes: treino, validação e teste.
    """

    paths = [
        'dataset/images/train',
        'dataset/images/test',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/test',
        'dataset/labels/val',
    ]

    def split_base(self, images_path: str):
        """Split the dataset into train, validation, and test sets."""
        # Get image path
        images_path = os.path.abspath(os.path.join(images_path, 'images'))

        # List all images in the specified directory
        self.images = [i for i in os.listdir(images_path) if i.endswith('.jpg')]

        # Primeiro, separamos treino + validação de teste (por exemplo, 80% para treino+val e 20% para teste)  # noqa: E501
        train_val, self.itens['test'] = train_test_split(self.images, test_size=0.2, random_state=42)  # noqa: E501

        # Agora separamos o treino e a validação (por exemplo, 75% do restante para treino, 25% para validação)  # noqa: E501
        self.itens['train'], self.itens['val'] = train_test_split(train_val, test_size=0.25, random_state=42)  # 0.25 de 80% = 20%  # noqa: E501

    def create_directories(self):
        """Create directories for the dataset."""
        for path in Dataset.paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def set_dataset_items(self, filetype: str):
        """Set the dataset items for the specified filetype."""
        self.itens[filetype] = [
            ImageLabel(filetype, filename, self.origin_path)
            for filename in self.itens[filetype]
        ]

    def __init__(self, files_path: str):
        """Initialize the Dataset class."""
        self.itens = {
            'train': [],
            'val': [],
            'test': [],
        }

        self.origin_path = os.path.abspath(files_path)

        self.create_directories()
        self.split_base(files_path)

        for filetype in self.itens.keys():
            self.set_dataset_items(filetype)

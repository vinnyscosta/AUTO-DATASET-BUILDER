import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class ImageLabel:
    """Class to handle image and label files for a dataset."""
    def __init__(self, filetype: str, filename: str, origin_path: str):
        self.filetype = filetype
        self.filename = filename

        # Image
        self.image_filename = filename
        self.image_path = os.path.join(origin_path, 'images', self.image_filename)

        # Label
        self.label_filename = filename.replace('.jpg', '.txt')
        self.label_path = os.path.join(origin_path, 'labels', self.label_filename)

        # Check if image and label exist
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file {self.image_path} does not exist.")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file {self.label_path} does not exist.")

        # Add to dataset
        self.add_to_dataset()

    def add_to_dataset(self):
        """Add image and label files to the dataset."""
        shutil.copy(self.image_path, f"dataset/images/{self.filetype}/{self.image_filename}")
        shutil.copy(self.label_path, f"dataset/labels/{self.filetype}/{self.label_filename}")

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

        # Primeiro, separamos treino + validação de teste (por exemplo, 80% para treino+val e 20% para teste)
        train_val, self.itens['test'] = train_test_split(self.images, test_size=0.2, random_state=42)

        # Agora separamos o treino e a validação (por exemplo, 75% do restante para treino, 25% para validação)
        self.itens['train'], self.itens['val'] = train_test_split(train_val, test_size=0.25, random_state=42)  # 0.25 de 80% = 20%

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

    def create_negatives(
        self,
        base_dir: str = "dataset",
        structure: dict = {"train": 0, "val": 0, "test": 0},
        width: int = 2480,
        height: int = 3508,
        background_color: tuple = (255, 255, 255),  # branco
        prefix: str = "sem_carimbo"
    ):
        """
        Cria imagens negativas (sem carimbo) para os conjuntos de treino, validação e teste.

        Parâmetros:
        - base_dir: diretório raiz onde ficam os subdiretórios 'images/train', 'labels/train' etc.
        - estrutura: dicionário com o nome do conjunto e a quantidade de imagens negativas a serem criadas.
            Ex: {"train": 240, "val": 80, "test": 81}
        - width, height: tamanho da imagem
        - background_color: cor RGB (255,255,255 para branco)
        - prefix: prefixo usado nos nomes dos arquivos
        """

        for conjunto, qtd in structure.items():
            dir_img = os.path.join(base_dir, "images", conjunto)
            dir_lbl = os.path.join(base_dir, "labels", conjunto)

            os.makedirs(dir_img, exist_ok=True)
            os.makedirs(dir_lbl, exist_ok=True)

            for i in range(round(qtd*0.75)):  # 75% para treino
                nome_base = f"{prefix}_{i+1:03d}"

                # Imagem
                imagem = np.full((height, width, 3), background_color, dtype=np.uint8)
                caminho_img = os.path.join(dir_img, nome_base + ".jpg")
                cv2.imwrite(caminho_img, imagem)

                # Label vazio
                caminho_lbl = os.path.join(dir_lbl, nome_base + ".txt")
                with open(caminho_lbl, 'w') as f:  # noqa: F841
                    pass

                print(f"[{conjunto.upper()}] Criado: {caminho_img} + label vazio")

    def __init__(self, files_path: str, create_negatives: bool = False):
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

        if create_negatives:
            self.create_negatives(
                base_dir="dataset/",
                structure={
                    "train": len(self.itens['train']),
                    "val": len(self.itens['val']),
                    "test": len(self.itens['test'])
                },
                width=2480,
                height=3508,
                background_color=(255, 255, 255),  # branco
                prefix="sem_carimbo"
            )

# Auto Dataset Builder 🧠📂

Este projeto organiza imagens e labels no estilo YOLO automaticamente em diretórios de treino, validação e teste.

## 📁 Estrutura Esperada dos Dados

O script lê os arquivos `.jpg` e `.txt` da seguinte estrutura:

```
seu_diretorio/
├── images/
│   ├── image1.jpg
│   └── ...
└── labels/
    ├── image1.txt
    └── ...
```

E os reorganiza automaticamente para a seguinte estrutura:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 🚀 Como Usar

1. Clone o repositório ou copie os arquivos para seu projeto.
2. Instale as dependências:
```bash
pip install -r requirements.txt
```
3. Execute o script principal:

```python
from dataset_builder.builder import Dataset

# Caminho contendo as pastas 'images' e 'labels'
dataset = Dataset('caminho/para/seus_arquivos')
```

## 🧪 Exemplo de Execução

```python
from dataset_builder.builder import Dataset

if __name__ == '__main__':
    Dataset('examples/data')
```

## 📦 Requisitos

- Python 3.8+
- scikit-learn

Instale com:

```bash
pip install -r requirements.txt
```

## 📄 Licença

Este projeto está licenciado sob a licença MIT.
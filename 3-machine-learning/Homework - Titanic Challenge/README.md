# Machine Learning — How to Run

This folder holds **two** notebooks:

| notebook | task | status |
|---|---|---|
| `MNIST_Homework.ipynb` | classify handwritten digits 0-9 | done |
| `Titanic_Challenge.ipynb` | predict passenger survival | model not defined yet |

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter
pip install tensorflow matplotlib
```

## Run

```
jupyter notebook
```

## Known issue: the Titanic data path

`Titanic_Challenge.ipynb` downloads the dataset from Google Drive into `/content/input/data.csv`,
which only exists on Colab. Running it locally gives a `FileNotFoundError`.

The dataset is already in this folder as `data.csv`, so locally the read becomes:

```python
data = pd.read_csv('data.csv')
```

MNIST has no such problem — Keras downloads that dataset itself.

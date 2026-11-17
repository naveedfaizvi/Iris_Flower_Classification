# Iris Flower Species Classification

An exploratory machine-learning project that classifies Iris flowers as **setosa**, **versicolor**, or **virginica** from four measurements: sepal length, sepal width, petal length, and petal width.

## Project files

- [`Iris_Flower_Classification.ipynb`](./Iris_Flower_Classification.ipynb) - Main Jupyter notebook with data exploration, feature selection, model tuning, evaluation, and an example prediction.
- [`Iris_Flower_Classification_Report.html`](./Iris_Flower_Classification_Report.html) - HTML export of the notebook.
- [`Iris_Dataset/iris.csv`](./Iris_Dataset/iris.csv) - CSV copy of the Iris data. The notebook loads the equivalent built-in dataset from scikit-learn, so the CSV is not required to run it.

## Analysis workflow

The notebook:

1. Loads and audits the 150-row Iris dataset.
2. Explores feature distributions, species differences, and correlations.
3. Creates a stratified 80/20 train/test split (random seed 42).
4. Compares all four measurements with a training-only, two-feature selection.
5. Tunes Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest using five-fold stratified cross-validation on the training data.
6. Reports final metrics on the held-out test set, compares models, and predicts the species of an example flower.

Preprocessing and feature selection are fitted within cross-validation pipelines to reduce data leakage. The test set is reserved for final evaluation and is not used to select the model. Results are specific to this small dataset and should not be assumed to generalize to all flowers or measurement conditions.

## Requirements

- Python 3
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- IPython

Install the packages with:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn ipython jupyter
```

## Run the notebook

From the project directory, start Jupyter:

```bash
jupyter notebook
```

Open `Iris_Flower_Classification.ipynb` and run the cells from top to bottom. The dataset is bundled with scikit-learn and is loaded locally; no separate download is needed.

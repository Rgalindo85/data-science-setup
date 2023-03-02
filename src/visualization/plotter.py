import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def draw_histogram(data: pd.DataFrame, col: str, title: str, filename: str):
    """Plot a histogram with bins = 30.

    Args:
        data (pd.DataFrame): data
        col (str): column name to plot
        title (str): title plot
        filename (str): path to store the plot
    """
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    sns.histplot(data=data, x=col, color="g", bins=30, ax=ax)

    plt.title(title)
    plt.savefig(filename)


def draw_barplot(data: pd.DataFrame, col: str, title: str, filename: str):

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    sns.countplot(data=data, x=col, ax=ax)

    plt.title(title)
    plt.savefig(filename)


def draw_categorical_features(data: pd.DataFrame, features: list, outpath: str):

    for col in features:
        draw_barplot(
            data=data,
            col=col,
            title=col,
            filename=os.path.join(outpath, f"{col}_barplot.png"),
        )


def draw_pca_plots(model: any, outpath: str):

    # Explained variance
    fig, _ = model.plot()
    fig.savefig(os.path.join(outpath, "models", "pca", "explained_variance.png"))

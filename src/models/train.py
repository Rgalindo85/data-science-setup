import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pca import pca
from visualization import plotter


@hydra.main(config_path="../../config", config_name="main")
def main(config: DictConfig):
    pass

    data_path = abspath(config.input_model.train_path)
    df = pd.read_csv(data_path)

    target = config.target
    plots_path = abspath(config.plots.path)
    explore_dim_reduction(data=df, target=target, outpath=plots_path)

    train_model(data=df, config=config)


def train_model(data: pd.DataFrame, config: DictConfig):
    logger = logging.getLogger("train_model")

    model = Pipeline(
        [
            ("std_scaler", StandardScaler()),
            ("pca", PCA(0.95)),
            ("lin_reg", LinearRegression()),
        ]
    )

    target = config.target

    data.dropna(inplace=True)
    X = data.drop(target, axis=1)
    y = data[target]

    model.fit(X, y)

    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_true=y, y_pred=predictions))

    r2 = r2_score(y_true=y, y_pred=predictions)

    logger.info(f"Train RMSE: {round(rmse, 3)}")
    logger.info(f"Train R2: {round(r2, 3)}")

    model_path = abspath(config.models.path)
    joblib.dump(model, os.path.join(model_path, "price_liner_reg.joblib"))


def explore_dim_reduction(data: pd.DataFrame, target: str, outpath: str):

    run_pca(data=data, target=target, outpath=outpath)


def run_pca(data: pd.DataFrame, target: str, outpath: str):

    X = data.drop(target, axis=1)
    y = data[target]

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    model = pca(normalize=True)
    results = model.fit_transform(X_std)

    plotter.draw_pca_plots(model=model, outpath=outpath)


if __name__ == "__main__":
    # Basic configuration for logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    logging.captureWarnings(True)
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        # filename=f'data/logs/fermentation_{farmid}_{dt_string}.log'
    )

    main()

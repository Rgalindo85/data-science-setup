import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pca import pca
from visualization import plotter

import wandb


@hydra.main(config_path="../../config", config_name="main")
def main(config: DictConfig):
    pass

    train_path = abspath(config.input_model.train_path)
    df_train = pd.read_csv(train_path)

    test_path = abspath(config.input_model.test_path)
    df_test = pd.read_csv(test_path)

    # target = config.target
    # plots_path = abspath(config.plots.path)
    # explore_dim_reduction(data=df, target=target, outpath=plots_path)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
    }

    for name, model in models.items():
        train_model(
            data_train=df_train,
            data_test=df_test,
            config=config,
            name=name,
            model=model,
        )


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    config: DictConfig,
    name: str,
    model: any,
):
    logger = logging.getLogger("train_model")

    wandb.init(project="housing", group=name, reinit=True)  # Group experiments by model

    model_pipe = Pipeline(
        [
            ("std_scaler", StandardScaler()),
            # ('poly_feat', PolynomialFeatures()),
            ("pca", PCA(0.95)),
            (name, model),
        ]
    )

    target = config.target

    data_train.dropna(inplace=True)
    X_train = data_train.drop(target, axis=1)
    y_train = data_train[target]

    data_test.dropna(inplace=True)
    X_test = data_test.drop(target, axis=1)
    y_test = data_test[target]

    model_pipe.fit(X_train, y_train)

    predictions = model_pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))

    r2 = r2_score(y_true=y_test, y_pred=predictions)

    wandb.log({"rmse": rmse, "r2": r2})

    logger.info(f"Train RMSE: {round(rmse, 3)}")
    logger.info(f"Train R2: {round(r2, 3)}")

    wandb.sklearn.plot_regressor(
        model_pipe, X_train, X_test, y_train, y_test, model_name=name
    )

    model_path = abspath(config.models.path)
    joblib.dump(model_pipe, os.path.join(model_path, f"price_{name}.joblib"))


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
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.captureWarnings(True)
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        # filename=f'data/logs/fermentation_{farmid}_{dt_string}.log'
    )

    main()

import os

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

import logging

from visualization import plotter


@hydra.main(config_path="../../config", config_name="main")
def main(config: DictConfig):
    """Function to process the data"""

    raw_path = abspath(config.raw.path)
    logging.info(f"Process data using {raw_path}")

    df = pd.read_csv(raw_path)

    outpath = abspath(config.plots.path)
    exploratory_data_analysis(data=df, outpath=outpath)


def exploratory_data_analysis(data: pd.DataFrame, outpath: str):
    """Make simple EDA for given dataframe

    Args:
        data (pd.DataFrame): data to explore
        outpath (str): path to save plots
    """
    target = "price"
    num_features, cat_features = get_cat_num_features(data=data, target=target)

    plotter.draw_histogram(
        data=data,
        col=target,
        title="Target Variable Distribution",
        filename=os.path.join(outpath, "eda", "price_eda_histogram.png"),
    )

    plotter.draw_categorical_features(
        data=data, features=cat_features, outpath=os.path.join(outpath, "eda")
    )


def get_cat_num_features(data: pd.DataFrame, target: str) -> tuple:
    """Return a tuple of lists with column names for
    numerical and categorical features in the dataframe

    Args:
        data (pd.DataFrame): data to explore
        target (str): name of target col

    Returns:
        tuple: list numerical, list categorical
    """
    logger = logging.getLogger("get_cat_num_features")
    features = [i for i in data.columns if i not in [target]]

    # numerical & categorical features
    num_unique_rows = data[features].nunique().sort_values()
    num_features = []
    cat_features = []

    for i in range(data[features].shape[1]):
        if num_unique_rows.values[i] <= 16:
            cat_features.append(num_unique_rows.index[i])
        else:
            num_features.append(num_unique_rows.index[i])

    logger.info(f"Numerical features: {len(num_features)}")
    logger.info(f"Categorical features: {len(cat_features)}")

    return (num_features, cat_features)


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

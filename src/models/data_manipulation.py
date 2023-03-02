import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

import logging
from sklearn.model_selection import train_test_split


@hydra.main(config_path="../../config", config_name="main")
def main(config: DictConfig):
    input_path = abspath(config.feature.path)
    logging.info(f"Process data using {input_path}")

    df = pd.read_csv(input_path)
    make_datasets(data=df, config=config)


def make_datasets(data: pd.DataFrame, config: DictConfig):

    target = config.target

    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42
    )

    X_train["price"] = y_train
    X_train.to_csv(abspath(config.input_model.train_path), index=False)

    X_test["price"] = y_test
    X_test.to_csv(abspath(config.input_model.test_path), index=False)


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

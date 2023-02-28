import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

import logging


@hydra.main(config_path="../../config", config_name="main")
def main(config: DictConfig):

    raw_path = abspath(config.raw.path)
    logging.info(f"Process data using {raw_path}")

    df = pd.read_csv(raw_path)
    df_fe = clean_data(data=df)

    interim_path = abspath(config.feature.path)
    df_fe.to_csv(interim_path, index=False)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    data = remove_duplicates(data=data)
    check_empty_values(data=data)
    data = transform_categorical_features(data=data)
    data = remove_numerical_outliers(data=data)

    return data


def remove_numerical_outliers(data: pd.DataFrame):
    logger = logging.getLogger("remove_numerical_outliers")

    original_shape = data.shape

    for col in data.columns:
        if data[col].nunique() < 17:
            continue

        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        data = data.query(f"{col} >= {q1 - 1.5*iqr}")
        data = data.query(f"{col} <= {q3 + 1.5*iqr}")

    logger.info(f"Dataframe \n {data.head()}")

    output_shape = data.shape

    logger.info(f"Original data shape: {original_shape}")
    logger.info(f"Output data shape: {output_shape}")
    return data


def transform_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("transform_categorical_features")

    for col in data.columns:
        if data[col].nunique() == 2:
            logger.info(f"One-Hot Encoding on feature: {col}")
            data[col] = pd.get_dummies(data[col], drop_first=True, prefix=col)

        if data[col].nunique() > 2 and data[col].nunique() < 17:
            logger.info(f"Dummy Encoding on feature: {col}")

            data = pd.concat(
                [
                    data.drop([col], axis=1),
                    pd.DataFrame(
                        pd.get_dummies(data[col], drop_first=True, prefix=col)
                    ),
                ],
                axis=1,
            )

    logger.info(f"Data shape: {data.shape}")
    return data


def check_empty_values(data: pd.DataFrame):
    logger = logging.getLogger("clean_data")

    # Check for empty elements
    nvc = pd.DataFrame(data.isnull().sum().sort_values(), columns=["Count"])
    nvc["Percentage"] = round(nvc["Count"] / data.shape[0], 3) * 100
    logger.info(f"\n{nvc}")


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("remove_duplicates")

    # remove duplicates
    original_shape = data.shape
    data.drop_duplicates(inplace=True)
    output_shape = data.shape

    if original_shape == output_shape:
        logger.info("No duplicated rows found")
    else:
        logger.info(f"{original_shape[0] - output_shape[0]} duplicates found")

    return data


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

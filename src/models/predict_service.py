import bentoml
from bentoml.io import NumpyNdarray, JSON

import numpy as np
import pandas as pd

from src.features.data_preprocessing import transform_categorical_features
from pydantic import BaseModel

# Load Model
model = bentoml.sklearn.get("price_lasso_reg:latest").to_runner()

# create service with the model
service_model = bentoml.Service("price_lasso_reg", runners=[model])


class House(BaseModel):

    area: int = 7420,
    bedrooms: int = 4,
    bathrooms: int = 2,
    stories: int = 3,
    mainroad: str = 'yes',
    guestroom: str = 'no',
    basement: str = 'no',
    hotwaterheating: str = 'no',
    airconditioning: str = 'yes',
    parking: int = 2,
    prefarea: str = 'yes',
    furnishingstatus: str = 'furnished'


# Create and API function
@service_model.api(input=JSON(pydantic_model=House), output=NumpyNdarray())
def predict(house: House) -> np.ndarray:
    data = pd.DataFrame(house.dict(), index=[0])
    data = transform_categorical_features(data)
    results = model.run(data)

    return np.array(results)

# Data Science SetUp
-----------
This repository is intended to structure the project and the correct usage of different tools to guarantee reproducibility and easy deployment of models.

As an example I am using the Housing dataset, from kaggle [link](https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms), to predict the price. I adapt the notebook into python scripts


## Great Expectations
A very useful tool to test and compared new data, only tabular data. To identify if a new set satisfies the training distribution, what we expect.

1. Initialize great expectations:

```
great_expectations --v3-api init
```

This will create a directory called `great_expectations` with the next structure

```
 great_expectations
├──  checkpoints
├──  expectations
├──  great_expectations.yml
├──  plugins
│  └──  custom_data_docs
│     ├──  renderers
│     ├──  styles
│     │  └──  data_docs_custom_styles.css
│     └──  views
├──  profilers
└──  uncommitted
   ├──  config_variables.yml
   ├──  data_docs
   └──  validations
```

2. Connect to Data:
As an example 2 csv files can be found at `data/model_input`, `train.csv` and `test.csv`.

````
$ great_expectations --v3-api datasource new
````
Here you will be asked about the data source and prompt to a jupyter notebook, replace datasource_name to train and run the cells. This notebook will assist you to write the yaml file to configure you data source.

3. Create the expectations:
````
great_expectations --v3-api suite new
````
This will prompt a notebook to assist you to create the JSON file with the expectations that will be located at `great_expectations/expectations/train/warning.json`

4. Validate new data:
````
great_expectations --v3-api checkpoint new test_checkpoint
````
This will validate every column in the `test.csv`. The checkpoint will be saved at `great_expectations/checkpoints/test_checkpoint.yml`

to rerun the validation
````
great_expectations --v3-api checkpoint run test_checkpoint
````

## Prefect
After running you can monitor your flows using the prefect server
````
prefect server start
````

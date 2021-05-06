<h1 align="center">Toy Machine Learning Pipeline</h3>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#ml-task-description-and-evaluation-procedure">ML task description and evaluation procedure</a></li>
    <li><a href="#dataset-description">Dataset description</a></li>
    <li><a href="#repository-structure">Repository structure</a>
    <ul>
        <li><a href="#pipeline-components">Pipeline components</a></li>
        <li><a href="#data-storage">Data storage</a></li>
      </ul>
    </li>
    <li><a href="#utils">Utils documentation</a>
    <ul>
        <li><a href="#io">io</a></li>
        <li><a href="#feature-generators">Feature generators</a></li>
        <li><a href="#models">Models</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About

This is a toy example of a standalone ML pipeline written **entirely** in Python. No external tools are incorporated into the master branch. I built this for two reasons:

1. To experiment with my own ideas for MLOps tools, as it is hard to develop devtools in a vacuum :) 
2. To have something to integrate existing MLOps tools with so I can have real opinions

The following diagram describes the pipeline at a high level. The README describes it in more detail.

![Diagram](./toy-ml-pipeline-diagram.svg)

## Getting started

This pipeline is broken down into several components, described in a high level by the directories in this repository. To serve the inference API locally, you can do the following:

1. `git clone` the repository
2. In the root directory of the repo, run `python inference/app.py`
3. [OPTIONAL] In a new tab, run `python inference/inference.py` to ping the API with some sample records

All Python dependencies and virtual environment creation is handled by the Makefile. See `setup.py` to see the packages installed into the virtual environment, which mainly consist of basic Python packages such as `pandas` or `sklearn`.

## ML task description and evaluation procedure

We train a model to predict whether a passenger in a NYC taxicab ride will give the driver a large tip. This is a **binary classification task.** A large tip is arbitrarily defined as greater than 20% of the total fare (before tip). To evaluate the model or measure the efficacy of the model, we measure the [**F1 score**](https://en.wikipedia.org/wiki/F-score).

The current best model is an instance of `sklearn.ensemble.RandomForestClassifier` with `max_depth` of 10 and other default parameters. The test set F1 score is 0.716. I explored this toy task earlier in my [debugging ML talk](https://github.com/shreyashankar/debugging-ml-talk).

## Dataset description

We use the yellow taxicab trip records from the NYC Taxi & Limousine Comission [public dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), which is stored in a public aws S3 bucket. The data dictionary can be found [here](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) and is also shown below:

| Field Name      | Description |
| ----------- | ----------- |
| VendorID      | A code indicating the TPEP provider that provided the record. 1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.       |
| tpep_pickup_datetime   | The date and time when the meter was engaged.        |
| tpep_dropoff_datetime   | The date and time when the meter was disengaged.        |
| Passenger_count   | The number of passengers in the vehicle. This is a driver-entered value.      |
| Trip_distance   | The elapsed trip distance in miles reported by the taximeter.      |
| PULocationID   | TLC Taxi Zone in which the taximeter was engaged.      |
| DOLocationID   | TLC Taxi Zone in which the taximeter was disengaged      |
| RateCodeID   | The final rate code in effect at the end of the trip. 1= Standard rate, 2=JFK, 3=Newark, 4=Nassau or Westchester, 5=Negotiated fare, 6=Group ride     |
| Store_and_fwd_flag | This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, aka “store and forward,” because the vehicle did not have a connection to the server. Y= store and forward trip, N= not a store and forward trip |
| Payment_type | A numeric code signifying how the passenger paid for the trip. 1= Credit card, 2= Cash, 3= No charge, 4= Dispute, 5= Unknown, 6= Voided trip |
| Fare_amount | The time-and-distance fare calculated by the meter. | 
| Extra | Miscellaneous extras and surcharges. Currently, this only includes the $0.50 and $1 rush hour and overnight charges. |
| MTA_tax | $0.50 MTA tax that is automatically triggered based on the metered rate in use. | 
| Improvement_surcharge | $0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015. | 
| Tip_amount | Tip amount – This field is automatically populated for credit card tips. Cash tips are not included. | 
| Tolls_amount | Total amount of all tolls paid in trip. | 
| Total_amount | The total amount charged to passengers. Does not include cash tips. |

## Repository structure

The pipeline contains multiple components, each organized into the following high-level subdirectories:

* `etl`
* `training`
* `inference`

### Pipeline components

Any applied ML pipeline is essentially a series of functions applied one after the other, such as data transformations, models, and output transformations. This pipeline was initially built in a lightweight fashion to run on a regular laptop with around 8 GB of RAM. *The logic in these components is a first pass; there is a lot of room to improve.*

The following table describes the components of this pipeline, in order:

_Before running, build the Docker image: `docker build -t toy-ml-pipeline .`._

| Name      | Description | How to run | File(s) |
| ----------- | ----------- | --- | -- |
| Cleaning | Reads the dataset (stored in a public S3 bucket) and performs very basic cleaning (drops rows outside the time range or with $0-valued fares) | `docker run --env-file=./.env toy-ml-pipeline cleaning` | `etl/cleaning.py` |
| Featuregen | Generates basic features for the ML model | `docker run --env-file=./.env toy-ml-pipeline featuregen` | `etl/featuregen.py` | 
| Split | Splits the features into train and test sets | `docker run --env-file=./.env toy-ml-pipeline split` | `training/split.py` |
| Training | Trains a random forest classifier on the train set and evaluates it on the test set | `docker run --env-file=./.env toy-ml-pipeline training` | `training/train.py` |
| Inference | Locally serves an API that is essentially a wrapper around the `predict` function | `docker run -p 5000:5000 --env-file=./.env toy-ml-pipeline serve, docker run --env-file=./.env toy-ml-pipeline inference` | `[inference/app.py, inference/inference.py]` |
| tests | Runs unit tests (currently only for `io`) | `docker run --env-file=./.env toy-ml-pipeline sh -c "pytest -s ./app/utils/tests.py"` | `utils/tests.py` | 

### Data storage

The inputs and outputs for the pipeline components, as well as other artifacts, are stored in a public S3 bucket named `toy-applied-ml-pipeline` located in `us-west-1`. Read access is universal and doesn't require special permissions. Write access is limited to those with credentials. If you are interested in contributing and want write access, please contact me directly describing how you would like to be involved, and I can send you keys. 

The bucket has a `scratch` folder, where random scratch files live. These random scratch files were likely generated by the `write_file` function in `utils.io`. The bulk of the bucket lies in the `dev` directory, or `s3://toy-applied-ml-pipeline/dev`.

The dev directory's subdirectories represent the components in the pipeline. These subdirectories contain the outputs of each component respectively, where the outputs are versioned with the timestamp the component was run. The `utils.io` library contains helper functions to write outputs and load the latest component output as input to another component. To inspect the filesystem structure further, you can call `io.list_files(dirname)`, which returns the immediate files in `dirname`.

If you have write permissions, store your keys/ids in an `.env` file and export them as environment variables. If you do not have write permissions, you will run into an error if you try to write to the S3 bucket.

## Utils documentation

The `utils` directory contains helper functions and abstractions for expanding upon the current pipeline. Tests are in `utils/tests.py`. Note that only the `io` functions are tested as of now.

### io

`utils/io.py` contains various helper functions to interface with S3. The two most useful functions are:

```python
def load_output_df(component: str, dev: bool = True, version: str = None) -> pd.DataFrame:
  """
    This function loads the latest version of data that was produced by a component.
    Args:
        component (str): component name that we want to get the output from
        dev (bool): whether this is run in development or "production" mode
        version (str, optional): specified version of the data
    Returns:
        df (pd.DataFrame): dataframe corresponding to the data in the latest version of the output for the specified component
    """
    ...

def save_output_df(df: pd.DataFrame, component: str, dev: bool = True, overwrite: bool = False, version: str = None) -> str:
    """
    This function writes the output of a pipeline component (a dataframe) to a parquet file.
    Args:
        df (pd.DataFrame): dataframe representing the output
        component (str): name of the component that produced the output (ex: clean)
        dev (bool, optional): whether this is run in development or "production" mode
        overwrite (bool, optional): whether to overwrite a file with the same name
        version (str, optional): optional version for the output. If not specified, the function will create the version number.
    Returns:
        path (str): Full path that the file can be accessed at
    """
    ...
```

Note that `save_output_df`'s default parameters are set such that you cannot overwrite an existing file. You can change this by setting `overwrite = True`.

### Feature generators

`utils.feature_generators.py` contains the lightweight abstraction for a feature generator to make it easy for someone to create a new feature. The abstraction is as follows:

```python
class FeatureGenerator(ABC):
    """Abstract class for a feature generator."""

    def __init__(self, name: str, required_columns: typing.List[str]):
        """Constructor stores the name of the feature and columns required in a df to construct that feature."""
        self.name = name
        self.required_columns = required_columns

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def schema(self):
        pass
```

See `utils.feature_generators.py` for examples on how to create specific feature types and `etl/featuregen.py` for an example on how to create the actual instances of the features themselves.

### Models

`utils/models.py` contains the `ModelWrapper` abstraction. This abstraction is essentially a wrapper around a model and consists of:

- the model binary
- pointer to dataset(s)
- metric values

To use this abstraction, you must create a subclass of `ModelWrapper` and implement the `preprocess`, `train`, `predict`, and `score` methods. The base class also provides methods to save and load the `ModelWrapper` object. It will fail to save if the client has not added data paths and metrics to the object.

An example of a subclass of `ModelWrapper` is the `RandomForestModelWrapper`, which is also found in `utils/models.py`. The `RandomForestModelWrapper` client usage example is in `training/train.py` and is partially shown below:

```python
from utils import models

# Create and train model
mw = models.RandomForestModelWrapper(
    feature_columns=feature_columns, model_params=model_params)
mw.train(train_df, label_column)

# Score model
train_score = mw.score(train_df, label_column)
test_score = mw.score(test_df, label_column)

mw.add_data_path('train_df', train_file_path)
mw.add_data_path('test_df', test_file_path)
mw.add_metric('train_f1', train_score)
mw.add_metric('test_f1', test_score)

# Save model
print(mw.save('training/models'))

# Load latest model version
reloaded_mw = models.RandomForestModelWrapper.load('training/models')
test_preds = reloaded_mw.predict(test_df)
```

## Roadmap

See the [open issues](https://github.com/shreyashankar/toy-ml-pipeline/issues) for tickets corresponding to feature ideas. The issues in this repo are mainly tagged either `data science` or `engineering`.

## Contributing

Having a toy example of an ML pipeline isn't just nice to have for people experimenting with MLOps tools. ML beginners or data science enthusiasts looking to understand how to build pipelines around ML models can also benefit from this repository.

Anyone is welcome to contribute, and your contribution is greatly appreciated! Feel free to either create issues or pull requests to address issues.

1. Fork the repo
2. Create your branch (`git checkout -b YOUR_GITHUB_USERNAME/somefeature`)
3. Make changes and add files to the commit (`git add .`)
3. Commit your changes (`git commit -m 'Add something'`)
4. Push to your branch (`git push origin YOUR_GITHUB_USERNAME/somefeature`)
5. Make a pull request

## Contact

Original author: [Shreya Shankar](https://www.twitter.com/sh_reya)

Email: shreya@cs.stanford.edu


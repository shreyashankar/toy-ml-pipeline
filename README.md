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
    <li><a href="#utils">Utils</a>
    <ul>
        <li><a href="#io">io</a></li>
        <li><a href="feature-generators">Feature generators</a></li>
        <li><a href="models">Models</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About

This is a toy example of a standalone ML pipeline written **entirely** in Python. No external tools are incorporated into the master branch. I built this for two reasons:

1. To experiment with my own ideas for MLOps tools, as it is hard to develop devtols in a vacuum :) 
2. To have something to integrate existing MLOps tools with so I can have real opinions

## Outline

- [x] Description of ML task and evaluation procedure
- [x] Dataset description 
- [x] Description of repository organization / structure
- [x] Description of data storage and filesystem organization
- [x] How to run
- [ ] `io`, `feature_generators`, and `models` documentation
- [ ] "Future work" / how to contribute

## Getting started

This pipeline is broken down into several components, described in a high level by the directories in this repository. See the Makefile for various commands you can run, but to serve the inference API locally, you can do the following:

1. `git clone` the repository
2. In the root directory of the repo, run `make serve`
3. [OPTIONAL] In a new tab, run `make inference` to ping the API with some sample records

All Python dependencies and virtual environment creation is handled by the Makefile. See `setup.py` to see the packages installed into the virtual environment, which mainly consist of basic Python packages such as `pandas` or `sklearn`.

## ML task description and evaluation procedure

We train a model to predict whether a passenger in a NYC taxicab ride will give the driver a large tip. This is a **binary classification task.** A large tip is arbitrarily defined as greater than 20% of the total fare (before tip). To evaluate the model or measure the efficacy of the model, we measure the [**F1 score**](https://en.wikipedia.org/wiki/F-score).

The current best model is an instance of `sklearn.ensemble.RandomForestClassifier` with `max_depth` of 10 and other default parameters. I explored this toy task earlier in my [debugging ML talk](https://github.com/shreyashankar/debugging-ml-talk).

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

| Name      | Description | How to run | File(s) |
| ----------- | ----------- | --- | -- |
| Cleaning | Reads the dataset (stored in a public S3 bucket) and performs very basic cleaning (drops rows outside the time range or with $0-valued fares) | `make cleaning` | `etl/cleaning.py` |
| Featuregen | Generates basic features for the ML model | `make featuregen` | `etl/featuregen.py` | 
| Split | Splits the features into train and test sets | `make split` | `training/split.py` |
| Training | Trains a random forest classifier on the train set and evaluates it on the test set | `make training` | `training/train.py` |
| Inference | Locally serves an API that is essentially a wrapper around the `predict` function | `make serve, make inference` | `[inference/app.py, inference/inference.py]` |

### Data storage

The inputs and outputs for the pipeline components, as well as other artifacts, are stored in a public S3 bucket named `toy-applied-ml-pipeline` located in `us-west-1`. Read access is universal and doesn't require special permissions. Write access is limited to those with credentials. If you are interested in contributing and want write access, please contact me directly describing how you would like to be involved, and I can send you keys. 

The bucket has a `scratch` folder, where random scratch files live. These random scratch files were likely generated by the `write_file` function in `utils.io`. The bulk of the bucket lies in the `dev` directory, or `s3://toy-applied-ml-pipeline/dev`.

The dev directory's subdirectories represent the components in the pipeline. These subdirectories contain the outputs of each component respectively, where the outputs are versioned with the timestamp the component was run. The `utils.io` library contains helper functions to write outputs and load the latest component output as input to another component. To inspect the filesystem structure further, you can call `io.list_files(dirname)`, which returns the immediate files in `dirname`.

If you have write permissions, store your keys/ids in an `.env` file, and the `Makefile` will automatically pick it up. If you do not have write permissions, you will run into an error if you try to write to the S3 bucket.

<!-- ## Repository structure

Issues correspond to software tickets. Each PR is associated with a ticket.

Currently I have enabled repository interaction limits on anyone who is *not* a collaborator. -->

<!-- This is designed such that you need credentials (which you can store in `.env` file) to write to the data bucket, but you can read any files in the data bucket without credentials. -->


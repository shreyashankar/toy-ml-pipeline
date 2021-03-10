# toy-applied-ml-pipeline

This is a toy example of a standalone ML pipeline written **entirely** in Python. No external tools are incorporated into the master branch to keep it as lightweight as possible. 

I am making this public in case other people are interested, but I am not fully committed to perfect documentation right now. I built this for two reasons:

1. To experiment with my own ideas for MLOps tools, as it is hard to develop devtols in a vacuum :) 
2. To have something to integrate existing MLOps tools with so I can have real opinions

## Outline

- [x] Description of ML task
- [x] Dataset description 
- [ ] Description of repository organization / structure
- [ ] Diagram
- [ ] Description of data storage and filesystem organization
- [ ] How to run
- [ ] "Future work" / how to contribute

## ML task description

We train a model to predict whether a passenger in a NYC taxicab ride will give the driver a large tip. This is a **binary classification task.** A large tip is arbitrarily defined as greater than 20% of the total fare (before tip). The current best model is an instance of `sklearn.ensemble.RandomForestClassifier` with `max_depth` of 10 and other default parameters.

I explored this toy task earlier in my [debugging ML talk](https://github.com/shreyashankar/debugging-ml-talk).

## Dataset description

We use the yellow taxicab trip records from the NYC Taxi & Limousine Comission [public dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The data dictionary can be found [here](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) and is also shown below:

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

## Pipeline components

At a basic high leve, this pipeline will consist of data transformations, models, and output transformations. **For a first pass, I will build this system in a lightweight fashion to run on my machine.** The raw data is stored in a public `s3` bucket. I will write the following components:

* data cleaning
* featurization
* preprocessing
* model inference
* [OPTIONAL] output postprocessing

I will train a model in an offline setting and use that for the model inference setting. For the first pass, I won't be dealing with versioning on the *model* -- just the data, if any. 

For the second pass, I will incorporate MLFlow models to do model versioning. I will think of some clever way to do data versioning (without using extra tools) -- if anything, it wil be another partition in a table.

## Repository structure

Issues correspond to software tickets. Each PR is associated with a ticket.

Currently I have enabled repository interaction limits on anyone who is *not* a collaborator.

This is designed such that you need credentials (which you can store in `.env` file) to write to the data bucket, but you can read any files in the data bucket without credentials.


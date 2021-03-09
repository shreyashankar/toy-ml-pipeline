# toy-applied-ml-pipeline

This is a toy example of a standalone ML pipeline written **entirely** in Python. No external tools are incorporated into the master branch to keep it as lightweight as possible. 

I am making this public in case other people are interested, but I am not fully committed to perfect documentation right now. I built this for two reasons:

1. To experiment with my own ideas for MLOps tools, as it is hard to develop devtols in a vacuum :) 
2. To have something to integrate existing MLOps tools with so I can have real opinions

## Outline

- [ ] Description of ML task
- [ ] Dataset description 
- [ ] Description of repository organization / structure
- [ ] Diagram
- [ ] Description of data storage and filesystem organization
- [ ] "Future work"

## Prediction task

I will leverage the toy task I have trained models for before in [this notebook](https://github.com/shreyashankar/debugging-ml-talk/blob/main/nyc_taxi_2020.ipynb). For any record coming in, we want to predict whether the passenger will give a high tip or not. 

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


.PHONY: help test cleaning featuregen split train serve inference
include .env

# Makefile variables
VENV_NAME:=venv
PYTHON=${VENV_NAME}/bin/python3 

# Include your variables here
# AIRFLOW_HOME=~/.airflow

# Export environment variables
VARS:=$(shell sed -ne 's/ *\#.*$$//; /./ s/=.*$$// p' .env )
$(foreach v,$(VARS),$(eval $(shell echo export $(v)="$($(v))")))

.DEFAULT: help
help:
	@echo "make venv"
	@echo "       prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests"
	@echo "make cleaning"
	@echo "       run data cleaning"
	@echo "make featuregen"
	@echo "       run feature generation"
	@echo "make split"
	@echo "       run train/test split"
	@echo "make train"
	@echo "       run model training"
	@echo "make serve"
	@echo "       serve API locally"
	@echo "make inference"
	@echo "       run inference on random samples"

# Install dependencies whenever setup.py is changed.
venv: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: setup.py
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -e .
	rm -rf ./*.egg-info
	touch $(VENV_NAME)/bin/activate

# lint: venv
# 	${PYTHON} -m pylint main.py

test: venv
	${PYTHON} -m pytest -s utils/tests.py

cleaning: venv
	${PYTHON} etl/cleaning.py

featuregen: venv
	${PYTHON} etl/featuregen.py

split: venv
	${PYTHON} training/split.py

train: venv
	${PYTHON} training/train.py

serve: venv
	${PYTHON} inference/app.py

inference: venv
	${PYTHON} inference/inference.py

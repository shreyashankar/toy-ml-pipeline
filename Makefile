.PHONY: help lint etl
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
	@echo "make lint"
	@echo "       run pylint"
	@echo "make run"
	@echo "       run project"

# Install dependencies whenever setup.py is changed.
venv: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: setup.py
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -e .
	rm -rf ./*.egg-info
	touch $(VENV_NAME)/bin/activate

lint: venv
	${PYTHON} -m pylint main.py

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
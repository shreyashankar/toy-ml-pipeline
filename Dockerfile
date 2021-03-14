FROM python:3.8

ENV WORKDIR=/app

COPY requirements.txt ${WORKDIR}/requirements.txt
RUN pip install -r ${WORKDIR}/requirements.txt

COPY . ${WORKDIR}
RUN pip install -e ${WORKDIR}

WORKDIR ${WORKDIR}

# Include your variables here and/or 
# use --env-file=<paht to .env file> on `docker run`
# AIRFLOW_HOME=~/.airflow

FROM python:3.8

ARG NUM_CPUS=4
ARG DEBIAN_FRONTEND=noninteractive

# Installing Virtualenv
RUN pip install virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="VIRTUAL_ENV/bin:$PATH"

# Working with Application
ENV WORKDIR=app
COPY ./ /${WORKDIR}
RUN pip install	--upgrade pip
RUN pip install -r /${WORKDIR}/requirements.txt
RUN pip install -e /${WORKDIR}/.

# Expose port 
EXPOSE 5000

# Run the application:
# ENTRYPOINT [ "cd", "/app" ]
CMD ["python", "./app/inference/app.py"]


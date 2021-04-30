FROM python:3.8

# set work directory
WORKDIR /src

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV GIT_PYTHON_REFRESH=quiet

# copy project
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e mltrace/.

# Set credentials

# Expose port 
EXPOSE 8000

# Run the application:
CMD ["python", "./inference/app.py"]
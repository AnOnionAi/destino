# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=True \
    POETRY_VIRTUALENVS_CREATE=False \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PATH="${PATH}:/root/.poetry/bin:/root/.cargo/bin:${PATH}"

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install Rust toolchain, curl, wget
RUN apt-get update && apt-get install -y curl \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Install dependency manager
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
# Install dependecies 
RUN poetry install --no-dev --no-interaction --no-ansi && rm -rf "$POETRY_CACHE_DIR"

# Download models
RUN poetry run python install.py

# Run the web service on container startup. Here we use the gunicorn
# webserver, with four worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :$PORT -k uvicorn.workers.UvicornWorker --workers 4 app:app

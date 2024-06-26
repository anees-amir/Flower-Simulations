# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . .

# Install the project as a Python package using pip
RUN pip install .

# Set the entrypoint to the script and use CMD for default arguments
CMD ["python", "client.py"]
# # Use the official Python image from the Docker Hub
# FROM python:3.8-slim

# # Set environment variables
# ENV PYTHONUNBUFFERED 1

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements.txt file into the container at /app
# COPY requirements.txt /app/

# # Install the dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc libglib2.0-0 libsm6 libxext6 libxrender-dev && \
#     pip install --upgrade pip && \
#     pip install -r requirements.txt && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*


# # check pytorch version
# RUN python -c "import torch; print(torch.__version__)"

# # Copy the models directory into the container at /app/models
# COPY ../models /app/models

# # Copy the current directory contents into the container at /app
# COPY . /app/

# # Expose port 5000 for the Flask app to run on
# EXPOSE 5000

# # Define the command to run the application
# CMD ["python", "app.py"]







# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install the dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the model file into the container at /app/models
# COPY serving_model_v0.pth /app/models/serving_model_v0.pth
# Remove the line that copies the model into the container

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose port 5000 for the Flask app to run on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]

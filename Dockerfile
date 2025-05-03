# Use a specific version of Python
FROM python:3.10-slim

# Set a working directory for the app
WORKDIR /app

# Copy the content of the current directory to the /app directory in the container
COPY . /app

# Install pip and dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port if your app is a web service (e.g., Flask app)
EXPOSE 5000

# Define the default command to run when the container starts
CMD ["python", "prediction.py"]

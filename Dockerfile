# Use official Python image
FROM python:3.10

# Set working directory inside the container
WORKDIR /fraud_detection

# Copy requirements and install dependencies
COPY ./requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and models separately to maintain correct structure
COPY ./apps /fraud_detection/apps
COPY ./models /fraud_detection/models
COPY ./data /fraud_detection/data
# Expose the Flask port
EXPOSE 5000

# Start the Flask app
CMD ["python", "/fraud_detection/apps/app.py"]

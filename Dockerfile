
# Base image
FROM python:3.9

# Set work directory
WORKDIR /fraud_detection

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY ./apps /fraud_detection/apps
COPY ./models /fraud_detection/models
COPY . .

# Run the app
CMD ["python", "apps/app.py"]

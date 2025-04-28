FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y python3-distutils python3-venv gcc ffmpeg libsndfile1

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the application, now correctly reading $PORT
CMD ["python", "main.py"]

# Use Python 3.12.7 as the base image
FROM python:3.12.7-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and model into the container
COPY requirements.txt /app/requirements.txt
COPY models/best_model.pt /app/models/best_model.pt


# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for persistent storage
RUN mkdir -p /app/models /app/logs /app/data

# Copy the source code into the container
COPY src/ ./src/
COPY README.md ./


EXPOSE 8501  

# Set the entry point for the application
CMD ["streamlit", "run", "src/app.py"]

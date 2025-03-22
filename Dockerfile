# Use the official Python image as base
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Cloud Run (Google uses port 8080)
ENV PORT 8080
EXPOSE 8080

# Start FastAPI using Gunicorn with Uvicorn workers
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app1:app", "--bind", "0.0.0.0:8080"]

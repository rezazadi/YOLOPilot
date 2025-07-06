# Use official Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements.txt first (for caching)
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Now copy the rest of the app code (after installing dependencies)
COPY . .

# Expose port for Gradio app
EXPOSE 7861

# Run the app
CMD ["python", "yolopilot_app.py"]

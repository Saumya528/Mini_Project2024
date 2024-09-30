# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all the files from the current directory into the container's /app directory
COPY . /app

# Copy the model file specifically (optional)
COPY model.pkl /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app runs on
EXPOSE 5000

# Set the default command to run the Flask app
CMD ["python", "app.py"]

# docker build -t crema-d-sentiment
# docker run -p 5000:5000 crema-d-sentiment

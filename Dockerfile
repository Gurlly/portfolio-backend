# STEP 1: Use an official Python base image
FROM python:3.10-slim

# STEP 2: Set the working directory inside the container
WORKDIR /app

# STEP 3: Copy the requirements file first
COPY requirements.txt .

# STEP 4: Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# STEP 5: Copy the rest of your application code
COPY . .

# STEP 6: Configure Permissions for Hugging Face
RUN chown -R 1000:1000 /app

# STEP 7: Switch to the non-root user
USER 1000

# STEP 8: Run the Application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
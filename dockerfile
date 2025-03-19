FROM python:3.10-slim-buster  

# Install scikit-learn 1.4.2
RUN pip install scikit-learn==1.4.2

# Install other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Define the entrypoint (assuming you run App.py)
CMD ["streamlit", "run", "App.py"]
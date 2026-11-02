
# 1. Python environment
FROM python:3.14-slim

# 2. Application directory
WORKDIR /app

# 3. Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 4. Copy the API source code
COPY src/ ./src/

# 5. Copy the trained model
COPY models/tfidf_logreg.joblib ./models/tfidf_logreg.joblib

# 6. Document the application port
EXPOSE 8000

# 7. Start FastAPI
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

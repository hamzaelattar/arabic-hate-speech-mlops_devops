
FROM apache/airflow:3.3.1-python3.12

USER airflow

# ML dependencies compatible with our trained model
RUN pip install --no-cache-dir \
    mlflow==3.17.2 \
    scikit-learn==1.9.1 \
    joblib==1.6.0 \
    pandas==3.1.0 \
    numpy==2.5.4 \
    scipy==1.18.1

ENV PYTHONPATH=/opt/airflow/project
ENV MLFLOW_ALLOW_FILE_STORE=true

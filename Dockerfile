FROM python:3.9-slim

WORKDIR /app

COPY breast_cancer_model.pkl ./breast_cancer_model.pkl
COPY app.py ./app.py

RUN pip install flask pandas scikit-learn joblib

EXPOSE 8000

CMD ["python", "app.py"]

FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy

EXPOSE 9696

COPY [ "predict.py", "./" ]
COPY [ "xgb_cls.bin", "preprocessor.bin", "./"]

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

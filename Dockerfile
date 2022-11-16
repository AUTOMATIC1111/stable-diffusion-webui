FROM python:3.10.6-slim-bullseye
ENV APP_HOME /modules/api
ENV ENV prod
WORKDIR $APP_HOME
COPY . ./
RUN pip install -r requirements.txt
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 modules.api.api:app --timeout 540
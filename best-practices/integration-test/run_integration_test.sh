#!/usr/bin/env python
# coding: utf-8

cd "$(dirname "$0")"

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then 
    LOCAL_TAG=$(date +"%Y-%m-%d")
    export LOCAL_IMAGE_NAME="model-deployment:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} ..
else
    echo "No need to build image ${LOCAL_IMAGE_NAME}"
fi

docker-compose up -d

sleep 5

pipenv run python test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down

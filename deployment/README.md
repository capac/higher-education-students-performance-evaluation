# Deployment using MLflow

You can test the code in this folder by creating an isolated environment with `pipenv`, using the `Pipfile` and `Pipfile.lock` files. Once the environment has been generated, launch MLflow with the following command:

```language-bash
> mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts
```

At this point in the same Pipenv environment but in a separate terminal, you can run the following command to launch the Flask web service for making predictions:

```language-bash
> python predict.py
```

Finally in a third terminal and in the same Pipenv environment, run the last Python script to test the prediction web service, using new student test data contained in the test script:

```language-bash
> python test.py
```

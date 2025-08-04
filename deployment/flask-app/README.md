# Flask web service

You can test the code in this folder by creating an isolated environment with `pipenv`, using the `Pipfile` and `Pipfile.lock` files. After creating the environment, run the following command to launch the Flask web service for making predictions:

```language-bash
> python predict.py
```

In a second terminal and using the same Pipenv environment, run the Python script to test the prediction web service, using new student test data contained in the test script:

```language-bash
> python test.py
```

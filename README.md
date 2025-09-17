# Kidney-Disease-Classification-MLflow-DVC


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py


## Project Structure

├── src/cnnClassifier/ # Core package
│ ├── components/ # Data ingestion, training, evaluation modules
│ ├── pipeline/ # Training & prediction pipelines
│ ├── utils/ # Common utility functions
│ ├── config/ # Config management
│ └── entity/ # Dataclass-based configs
├── artifacts/ # Generated artifacts (gitignored)
├── templates/ # Flask HTML templates
├── app.py # Flask entrypoint
├── dvc.yaml # DVC pipeline
├── params.yaml # Model/training parameters
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build instructions
├── setup.py # Packaging setup
└── tests/ # Unit tests (pytest)

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Roshini-K/Kidney-disease-classification/
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.8 -y
```

```bash
conda activate venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)
MLFLOW_TRACKING_URI=https://dagshub.com/Roshini-K/Kidney-disease-classification.mlflow/


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)


## Docker

### Docker commands to run and build locally

1. build docker image
```bash
docker build -t kidney-app:local .
```

2. Run container (from powershell/terminal)
```bash
docker run --rm -p 8080:8080 `
  -e MLFLOW_TRACKING_URI=https://dagshub.com/Roshini-K/Kidney-disease-classification.mlflow `
  -e MLFLOW_TRACKING_USERNAME=Roshini-K `
  -e MLFLOW_TRACKING_PASSWORD=YOUR_DAGSHUB_TOKEN `
  kidney-app:local
```
then open : https://localhost:8080

## Docker
- Unit tests are provided in the tests/ folder. Run them using:

```bash
pip install pytest
pytest -v
```
# G203 - MP269 | Mobile Data Use (Surat)

An end-to-end Machine Learning Operations (MLOps) project developed as part of the **SCSE3040 MLOps Mini Project**. The project focuses on building a reproducible machine learning pipeline to predict daily mobile data consumption for users in Surat.

---

## Project Information

| Field | Details |
|---|---|
| Course | SCSE3040 - MLOps |
| Group ID | G203 |
| Problem ID | MP269 |
| Problem Title | Mobile Data Use (Surat) |
| Faculty | Dr. Saurabh Shrivastava |
| Batch | 48 |
| Prediction Type | Regression |
| Target Variable | `gb_per_day` |
| Evaluation Metric | Mean Absolute Error (MAE) |

## Team Members

| Enrollment Number | Name |
|---|---|
| S24CSEU1883 | MittulGulia |
| S24CSEU1890 | piyushjhakumar |
| S24CSEU0271 | MayankSharma |

---

## Project Objective

The objective is to develop an end-to-end MLOps workflow that predicts a customer's daily mobile data usage in gigabytes (`gb_per_day`) using available usage and customer-related information.

Beyond model development, the project covers the complete lifecycle of a machine learning application, including data validation, version control, experiment tracking, model management, API development, containerization, automated testing, deployment configuration, monitoring, and retraining.

## Dataset Overview

The project package provides three datasets:

| Dataset | Purpose |
|---|---|
| `train.csv` | Historical data used for exploration, preprocessing and initial model training |
| `later.csv` | Later-arriving data used for drift analysis, evaluation and retraining |
| `holdout_features.csv` | Records for which final daily data usage predictions must be generated |

The original training and later datasets will be managed using **DVC**. The supplied raw data will be preserved so that its integrity can be verified.

## Technology Stack

- **Python** - Main programming language
- **Pandas** - Data loading, exploration and preprocessing
- **Scikit-learn** - Machine learning models and evaluation
- **DVC (Data Version Control)** - Dataset version management
- **MLflow** - Experiment tracking and model registry
- **FastAPI** - Prediction API
- **Docker** - Application containerization
- **GitHub Actions** - Continuous integration and automated testing
- **Kubernetes** - Deployment configuration and service management
- **Pytest** - Automated testing

## Planned MLOps Workflow

1. **Data Exploration and Validation:** Inspect the supplied datasets and investigate data quality issues.
2. **Data Version Control:** Initialize DVC and track the required datasets.
3. **Data Preprocessing:** Build reproducible cleaning and preprocessing code.
4. **Model Development:** Train regression models and evaluate them using MAE.
5. **Experiment Tracking:** Record model parameters, metrics and data provenance in MLflow.
6. **Model Registry:** Register model versions and manage the champion model.
7. **API Development:** Create prediction and health endpoints using FastAPI.
8. **Containerization:** Package the API using Docker.
9. **Continuous Integration:** Run automated tests through GitHub Actions.
10. **Deployment Configuration:** Define the required Kubernetes Deployment and Service.
11. **Monitoring and Retraining:** Measure data drift, evaluate model performance and apply the model-promotion rule.
12. **Final Prediction:** Generate predictions for every record in the supplied holdout dataset.

## Repository Structure

```text
G203_MP269_MLOps/
├── data/
│   └── raw/
├── src/
│   └── mp269/
├── tests/
├── app/
├── models/
├── predictions/
├── reports/
├── k8s/
├── .github/
│   └── workflows/
├── README.md
├── requirements.txt
├── params.yaml
├── Dockerfile
└── .gitignore
```

The repository structure will be expanded as the project stages are implemented.

## Project Status

**Current Stage:** Stage 1 - Repository Setup and Data Exploration

The project is being developed in stages, following the instructions and submission requirements provided for group G203.

---

## Evaluation

The primary regression evaluation metric is **Mean Absolute Error (MAE)**. The final pipeline will also be assessed on reproducibility, data checks, experiment tracking, API validation, testing, CI, deployment configuration, monitoring and holdout predictions.

## Academic Note

This repository is maintained by the members of Group G203 for the SCSE3040 MLOps mini project. All project-specific analysis and results are to be generated using the dataset supplied to the group.

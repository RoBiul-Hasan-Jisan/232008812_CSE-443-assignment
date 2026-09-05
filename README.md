# CSE-443 Assignment

This repository contains two machine learning projects completed for CSE-443.

## Projects

### 1. [CNN - Chest X-Ray Classification](./CNN%20-%20Chest%20X-Ray%20Classification)

A convolutional neural network, built from scratch (no transfer learning or pretrained weights), that classifies pediatric chest X-rays as **Normal** or **Pneumonia**. Includes training/evaluation code, a Jupyter notebook, and demo outputs (diagnostics plots and Grad-CAM visualizations). Trained model artifacts are hosted on Hugging Face.

- Code: `pediatric_cxr_cnn.py`, `pediatric_cxr_cnn.ipynb`
- Demo outputs: `demo/` ,`https://huggingface.co/robiulhasanjisan88/p1/tree/main`
- Details: see the project's own [README](./CNN%20-%20Chest%20X-Ray%20Classification/README.md)

### 2. [Logistic Regression - Bank Marketing](./Logistic%20Regression%20-%20Bank%20Marketing)

A logistic regression model that predicts whether a bank customer will subscribe to a term deposit, based on demographic and campaign data from the UCI Bank Marketing dataset. Chosen for interpretability so stakeholders can see which factors drive subscription likelihood.

- Code: `bank.py`, `bank.ipynb`
- Data: `bank-full.csv`
- Outputs: `output/` (plots, metrics, deployable model)
- Details: see the project's own [README](./Logistic%20Regression%20-%20Bank%20Marketing/README.md)

## Structure

```
CSE-443-assignment-main/
├── CNN - Chest X-Ray Classification/
│   ├── notebook/
│   ├── report/
│   ├── .gitignore
│   └── Readme.md
│
├── Logistic Regression - Bank Marketing/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── reports/
│   ├── .gitignore
│   └── Readme.md
│
└── requirements.txt
```

Each project folder contains its own detailed README with methodology, results, and instructions.

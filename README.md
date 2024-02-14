# Breast Cancer Classification with SVM Hyperparameter Tuning

## Overview
This repository contains code for a machine learning project focused on classifying breast cancer using Support Vector Machine (SVM) with hyperparameter tuning. The project utilizes the Breast Cancer Wisconsin (Diagnostic) Dataset, implementing GridSearchCV to optimize the SVM model's performance.

## Project Structure
- `main.py`: python file containing the code for the project.
- `README.md`: This file providing an overview of the project.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes features describing characteristics of cell nuclei present in the image.

## Dependencies
- Python 3.x
- scikit-learn
- Jupyter Notebook

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/abhipatel35/SVM-Hyperparameter-Optimization-for-Breast-Cancer.git
2. Navigate to the project directory:
   ```bash
   cd breast-cancer-svm
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Open and run the 'main.py' python file in Jupyter Notebook or PyCharm.

## Results
- The initial SVM model performance is evaluated without hyperparameter tuning.
- GridSearchCV is employed to optimize SVM hyperparameters (C, gamma, kernel).
- Model performance is compared before and after hyperparameter tuning using metrics like classification reports.

# Credit Card Fraud Detection using Machine Learning

## Project Overview
This project focuses on detecting fraudulent credit card transactions using various machine learning techniques. The goal is to build and compare different classification models to accurately identify fraudulent transactions while minimizing false positives.

## Dataset
The project uses the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle, which contains:
- 284,807 transactions
- 492 fraudulent transactions
- 30 features (28 of which are PCA-transformed for privacy)
- Target variable: Binary classification (0: legitimate, 1: fraudulent)

## Models Implemented
1. **K-Nearest Neighbors (KNN)**
   - Implemented with n_neighbors=3
   - Achieved high accuracy with balanced precision and recall

2. **Random Forest**
   - Demonstrated excellent performance in detecting fraud

3. **Logistic Regression**
   - Achieved high accuracy with balanced precision and recall

4. **MLP**
   - Achieved high accuracy with balanced precision and recall
  
## Performance Metrics
All models were evaluated using:
- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- Confusion Matrix

## Results
### KNN Model
- Accuracy: 0.99
- Precision: 0.93
- Recall: 0.71
- Specificity: 0.99
- F1 Score: 0.81

### Random Forest Model
- Accuracy: 0.9996
- Precision: 0.9740
- Recall: 0.7653
- Specificity: 0.9999
- F1 Score: 0.8571

### Logistic Regression Model
- Accuracy: 0.99
- Precision: 0.75
- Recall: 0.84
- Specificity: 0.99
- F1 Score: 0.79

### MLP Model
- Accuracy: 0.9996
- Precision: 0.9231
- Recall: 0.8571
- Specificity: 0.9999
- F1 Score: 0.8889


## Code Structure
- `KNN.ipynb`: Implementation of K-Nearest Neighbors classifier
- `RandomForest.ipynb`: Implementation of Random Forest classifier
- `Logistic.ipynb`: Implementation of Logistic Regression classifier
- `MLP.ipynb`: Implementation of MLP classifier
- `utils.py`: Contains utility functions for model evaluation

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
```

## Usage
1. Download the dataset from Kaggle
2. Place the `creditcard.csv` file in the project directory
3. Run the Jupyter notebooks to train and evaluate the models

### Note
- The dataset is not included in the repository due to size constraints. Please download it from the provided link. It's included, but divided into two parts due to GitHub's file size limit.

## Future Improvements
- Implement additional models
- Apply feature engineering techniques
- Implement cross-validation
- Add hyperparameter tuning
- Explore anomaly detection techniques

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Machine Learning course materials and guidance 
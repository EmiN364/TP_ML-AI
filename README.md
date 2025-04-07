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
   - Achieved 99% accuracy with high precision (0.93) but moderate recall (0.71)

2. **Random Forest**
   - Best overall performance with 99.96% accuracy
   - Highest precision (0.97) among all models
   - Good recall (0.77) and excellent specificity (0.9999)

3. **Logistic Regression**
   - Achieved 99% accuracy
   - Balanced performance with moderate precision (0.75) and good recall (0.84)

4. **MLP (Multi-Layer Perceptron)**
   - Strong performance with 99.94% accuracy
   - Good precision (0.85) and recall (0.79)
   - High specificity (0.9998)
  
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
- Accuracy: 0.9994
- Precision: 0.8462
- Recall: 0.7857
- Specificity: 0.9998
- F1 Score: 0.8148

## Conclusion
### Key Achievements
- Successfully implemented and compared 4 ML models for fraud detection
- Achieved exceptional accuracy rates (99%+) across all models
- Demonstrated the effectiveness of different approaches to fraud detection

### Model Performance Summary
| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Random Forest | 99.96%   | 0.974     | 0.765  | 0.857    |
| MLP           | 99.94%   | 0.846     | 0.786  | 0.815    |
| KNN           | 99.00%   | 0.930     | 0.710  | 0.810    |
| Log Reg       | 99.00%   | 0.750     | 0.840  | 0.790    |

### Key Insights

#### Random Forest (Best Overall Performance)
- Highest accuracy at 99.96%
- Best precision score of 0.974
- Excellent specificity of 0.9999

#### MLP (Strong Runner-up)
- Close second with 99.94% accuracy
- Well-balanced precision and recall scores
- Demonstrated robust overall performance

#### KNN & Logistic Regression (Solid Baselines)
- Both achieved impressive 99% accuracy
- Complementary strengths:
  - KNN excelled in precision (0.930)
  - Logistic Regression showed strong recall (0.840)

### Technical Implementation Details
- Leveraged PCA-transformed features for enhanced privacy
- Implemented comprehensive evaluation metrics suite
- Developed modular codebase with dedicated model notebooks

### Future Development Roadmap
1. Model Enhancement
   - Integration of additional algorithms
   - Advanced feature engineering implementation
   - Robust hyperparameter optimization

2. System Improvements
   - Real-time detection capabilities
   - Cross-validation implementation
   - Enhanced model validation framework

### Project Impact
- Successfully demonstrated ML application in financial security
- Validated feasibility for real-world fraud detection systems
- Highlighted critical balance between precision and recall

### Conclusion
This project effectively demonstrates machine learning's capability in credit card fraud detection, with Random Forest emerging as the optimal solution for this specific use case. The comprehensive evaluation across multiple models provides valuable insights for real-world implementation.

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
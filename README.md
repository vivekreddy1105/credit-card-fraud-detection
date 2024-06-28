# Credit Card Fraud Detection Using Random Forest Classifier

This project demonstrates how to build a credit card fraud detection system using a Random Forest classifier. The goal is to classify credit card transactions as either fraudulent or legitimate based on the transaction details. The dataset used for this project is a collection of credit card transactions labeled as fraudulent or non-fraudulent.

## Libraries and Techniques Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning algorithms and tools, including:
  - `train_test_split`: To split the dataset into training and testing sets.
  - `RandomForestClassifier`: The machine learning model used for classification.
  - `confusion_matrix`, `accuracy_score`, `precision_score`, `recall_score`, `classification_report`: To evaluate the performance of the model.
- **Imbalanced-learn**: For handling imbalanced datasets using techniques like Random Over Sampling.

## Code Explanation

1. **Importing Libraries**:
   The necessary libraries are imported, including `pandas`, `numpy`, `matplotlib`, `seaborn`, and several modules from `scikit-learn` and `imbalanced-learn`.

2. **Loading and Preprocessing the Data**:
   The credit card transaction dataset is loaded into a DataFrame using `pandas.read_csv()`. The shape of the dataset and the number of missing values are checked. The dataset is explored to understand the distribution of the 'Class' variable.

3. **Visualizing Data**:
   A count plot of the 'Class' variable and a heatmap of the correlation matrix are generated to visualize the data.

4. **Handling Imbalanced Data**:
   The dataset is highly imbalanced, so Random Over Sampling is used to balance the classes.

5. **Splitting the Data**:
   The data is split into training and testing sets using `train_test_split()`.

6. **Training the Model**:
   A Random Forest classifier is trained on the resampled training data.

7. **Evaluating the Model**:
   The accuracy, precision, recall, and classification report of the model are evaluated on the test data. A confusion matrix is also generated to visualize the performance of the model.

Training data: https://drive.google.com/file/d/1VvqnClLB4qWCvhfD8lKqXNbn1JcH1Xlt/view?usp=drive_link

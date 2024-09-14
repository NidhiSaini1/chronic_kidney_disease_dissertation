# Chronic Kidney Disease Prediction Using Machine Learning and Visualization

## Project Overview

This project focuses on applying machine learning models to predict Chronic Kidney Disease (CKD) using patient data. The dataset consists of 1,659 patient records containing clinical and laboratory data. The project aims to improve the accuracy of CKD prediction while enhancing the interpretability of the models through visualization techniques.

The repository includes code, documentation, and the final dissertation submitted in partial fulfillment of the requirements for the MSc Data Science degree at the University of Nottingham.

## Motivation

Chronic Kidney Disease (CKD) is a significant health issue worldwide. It often goes undiagnosed until the disease has progressed to a severe stage. This project seeks to utilize machine learning to predict CKD in its early stages, allowing for timely intervention and improved patient outcomes. Traditional diagnostic methods may not capture early-stage CKD, but machine learning can identify complex patterns in patient data.

## Features

- **Data Preprocessing**: Cleaning, normalization, handling missing values, and categorical encoding.
- **Machine Learning Models**: Includes Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).
- **Class Imbalance Handling**: Used SMOTE (Synthetic Minority Over-sampling Technique) to address imbalanced classes.
- **Feature Selection**: Employed Recursive Feature Elimination (RFE) to select the most relevant features for model training.
- **Visualization**: Implemented model interpretability tools such as TensorBoard and the Language Interpretability Tool (LIT).

## Technologies

- **Programming Language**: Python
- **Libraries**: Pandas, Scikit-learn, Matplotlib, TensorFlow
- **Development Environments**: Jupyter Notebook, Google Colab
- **Version Control**: Git, GitHub

## Repository Structure

- `/src/`: Python scripts for data preprocessing, model training, and evaluation.
- `/data/`: Contains the preprocessed dataset (if applicable, anonymized).
- `/docs/`: Dissertation document and related materials.
- `/notebooks/`: Jupyter notebooks for model development and testing.
- `/results/`: Model performance results, visualizations, and logs.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/CKD-Prediction.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the preprocessing and training scripts:
    ```bash
    python src/preprocess_data.py
    python src/train_models.py
    ```

4. Visualize the results using the `visualize.py` script or by launching TensorBoard:
    ```bash
    python src/visualize.py
    tensorboard --logdir=results/logs
    
    ```
5.Visualising Boards:
![image](https://github.com/user-attachments/assets/f1e907a9-feaa-499c-bb2b-016687568100)
![image](https://github.com/user-attachments/assets/545bef9a-9301-421d-8d00-a226dcdc47bd)


## Results

- The **Random Forest** model achieved the best performance with high recall and precision for CKD-positive predictions.
- Despite the successful prediction, the project faced challenges with model interpretability, particularly for medical professionals who require clear, actionable insights.
- **Model Metrics**: The final model achieved a recall score of 98.36% for CKD-positive cases.

## Future Improvements

- Extend the dataset to include more diverse patient data.
- Experiment with deep learning models for more complex prediction patterns.
- Enhance visualization methods to improve interpretability for healthcare professionals.

## References

For detailed references and studies related to CKD prediction, machine learning, and model interpretability, please refer to the dissertation in the `/docs/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

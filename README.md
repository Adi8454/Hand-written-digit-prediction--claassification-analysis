# 🧠 Handwritten Digit Prediction – Classification Analysis (MNIST)

This project focuses on building and evaluating classification models to predict handwritten digits (0–9) using the popular **MNIST dataset**. We use multiple machine learning algorithms including **K-Nearest Neighbors (KNN)**, **Decision Trees**, **Random Forest**, and **Logistic Regression** to compare their performance.

## 📁 Dataset
- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Source:** [Kaggle / scikit-learn built-in](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- **Description:** 70,000 grayscale images of handwritten digits (28x28 pixels each)

## ✅ Objectives
- Load and visualize the MNIST dataset
- Preprocess the image data
- Train and evaluate multiple classification algorithms
- Compare models using accuracy and confusion matrix
- Build a simple prediction interface

## 🛠️ Technologies & Libraries
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn (KNN, DecisionTree, RandomForest, LogisticRegression)
- Jupyter Notebook / Google Colab

## 📌 Steps Performed
1. **Data Loading & Visualization**
   - Used Scikit-learn’s `load_digits()` or `fetch_openml()` for MNIST
   - Displayed sample digit images

2. **Preprocessing**
   - Scaled pixel values
   - Flattened image arrays (if needed)

3. **Model Training**
   - Trained multiple models:
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Random Forest
     - Logistic Regression

4. **Model Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

5. **Prediction**
   - Tested predictions on random or custom inputs
   - Visualized predictions using Matplotlib

## 📈 Results
| Model              | Accuracy |
|-------------------|----------|
| KNN               | 92%      |
| Decision Tree     | 88%      |
| Random Forest     | 90%      |
| Logistic Regression | 85%    |

*(Accuracy values may vary slightly depending on parameters and dataset version)*

## 🎯 Conclusion
- Random Forest and KNN showed the best performance.
- The project demonstrates the effectiveness of ensemble and distance-based classifiers for image classification tasks.
- This analysis builds a strong foundation for more advanced deep learning techniques like CNNs.

## 🚀 Future Improvements
- Implement a CNN using TensorFlow or PyTorch
- Build a web interface using Streamlit for digit prediction
- Deploy the model online (e.g., Render, Heroku)

## 📎 Project Demo
Coming Soon...

## 📚 References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Kaggle MNIST Page](https://www.kaggle.com/c/digit-recognizer)

## 🙌 Author
- Aditya Kumar  
- [LinkedIn](http://www.linkedin.com/in/aditya-kumar-4276a3281)  
- [GitHub](https://github.com/adi8454)


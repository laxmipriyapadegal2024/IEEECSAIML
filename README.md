# IEEECSAIML
#  Image Classification 

## Level 0: Data Understanding & Preprocessing
### Objectives:
- Load and explore the dataset.
- Visualize sample images and understand the data structure.
- Perform basic data cleaning.

### Steps:
1. **Load Dataset**  
   - Read the dataset into a Pandas DataFrame.  
   - Check the shape and first few rows of the dataset.  

2. **Data Visualization**  
   - Display sample images using `matplotlib` to understand pixel distribution.  
   - Plot class distribution to identify class imbalance.  

3. **Data Cleaning**  
   - Check for missing values and handle them.  
   - Convert labels into numeric form (if needed).  

---

##  Level 1: Exploratory Data Analysis (EDA) & Feature Engineering
###  Objectives:
- Perform in-depth exploratory data analysis.
- Normalize and preprocess the dataset.
- Extract meaningful insights.

### Steps:
1. **Descriptive Statistics & Visualizations**  
   - Summary statistics of pixel values.  
   - Class distribution visualization.  
   - Plot pixel intensity histograms.  

2. **Feature Engineering**  
   - Normalize pixel values (scale to [0,1]).  
   - Convert data to NumPy arrays for model training.  
   - Apply dimensionality reduction techniques (if needed).  

3. **Data Splitting**  
   - Split dataset into **training (80%)** and **testing (20%)** sets using `train_test_split()`.  
   - Ensure stratification to maintain class balance.  

---

## Level 2: Basic Classification Model
### Objectives:
- Train a basic logistic regression classifier.
- Evaluate model performance.
- Apply explainability techniques.

### Steps:
1. **Develop a Classifier using Logistic Regression**  
   - Train a logistic regression model using `LogisticRegression()` from `sklearn`.  
   - Use `solver='saga'` for better performance on large datasets.  
   - Experiment with hyperparameter tuning (`C`, `penalty`).  

2. **Preprocess and Normalize Data**  
   - Scale features using `StandardScaler()`.  
   - Ensure labels are correctly encoded.  

3. **Split the Dataset**  
   - Divide data into training & test sets.  
   - Ensure no data leakage.  

4. **Model Training & Evaluation**  
   - Train the model on the training set.  
   - Make predictions and compute **accuracy, precision, recall, and F1-score**.  

5. **Explainable AI (XAI) Techniques using SHAP**  
   - Use **SHAP (SHapley Additive exPlanations)** to interpret model decisions.  
   - Generate **SHAP summary plots** to understand feature importance.  

---

## Level 3 : Neural Network Implementation

## Project Overview  
This project focuses on classifying grayscale images of fashion items into **10 categories** using a **Convolutional Neural Network (CNN)**. The dataset consists of **70,000 images (28x28 pixels)**, divided into training and testing sets. The goal is to develop a deep learning model that accurately classifies images into different clothing categories.  

---

## 1. Dataset Loading & Preprocessing  

### 1.1 Mounting Google Drive & Loading the Dataset  
- The dataset is stored in Google Drive and is loaded as a NumPy array.  
- It contains both **training and testing images** along with their respective labels.  

### 1.2 Exploring the Dataset  
- The structure of the dataset is examined by checking the shapes of **training and test sets**.  
- This ensures a proper understanding of the number of samples available for training and evaluation.  

### 1.3 Data Visualization  
- Sample images from the dataset are visualized using **Matplotlib** to understand the dataset distribution.  
- This step helps in verifying that the dataset contains expected images before proceeding with preprocessing.  

### 1.4 Data Preprocessing  
- The images are **normalized** by scaling pixel values between **0 and 1** to improve model convergence.  
- Since CNN models expect 3D input, images are reshaped to **(28,28,1)** to include a single-channel grayscale dimension.  

---

## 2. Building the CNN Model  

### 2.1 Model Architecture  
The **Convolutional Neural Network (CNN)** consists of:  
- **Convolutional Layers**: Extract features from images using different filters.  
- **MaxPooling Layers**: Reduce spatial dimensions to make computations more efficient.  
- **Flatten Layer**: Converts 2D feature maps into a 1D array for classification.  
- **Fully Connected Layers**: Learn complex patterns and classify the images.  
- **Output Layer**: Uses the **Softmax activation function** to assign probabilities to 10 classes.  

### 2.2 Model Compilation  
- The model is compiled using **Adam Optimizer** for efficient learning.  
- **Sparse Categorical Crossentropy** is used as the loss function since the dataset has multiple classes.  
- The model tracks **accuracy** as the evaluation metric.  

---

## 3. Training the CNN Model  

- The model is trained on the **training dataset** with validation on the **test dataset**.  
- **Epochs** are used to iterate through the dataset multiple times to improve learning.  
- The training process helps the model optimize its weights to classify fashion items accurately.  

---

## 4. Model Evaluation  

### 4.1 Accuracy & Loss Evaluation  
- After training, the model is evaluated on the test set.  
- **Accuracy and loss values** are measured to assess model performance.  

### 4.2 Performance Visualization  
- **Graphs for accuracy and loss** over epochs are plotted.  
- This helps in understanding the model's learning curve and detecting any underfitting or overfitting issues.  

---

## 5. Making Predictions  

- The trained model is used to classify **new images** from the dataset.  
- The predictions are compared with actual labels to analyze performance.  

---

## Conclusion  
- A **CNN-based image classification model** was successfully built to classify fashion items.  
- The model achieves reasonable accuracy and can be further fine-tuned for better performance.  
- Future improvements include **hyperparameter tuning, data augmentation, and deeper architectures**.  


Breast Cancer Prediction Using a Simple Neural Network

# Breast Cancer Classification with a Neural Network

A beginner-friendly deep learning project that classifies breast tumors as **Malignant** (cancerous) or **Benign** (non-cancerous) using a simple feed-forward Neural Network built with **TensorFlow / Keras**.

> ⚠️ **Disclaimer:** This is an educational project. It is not a medical device and must not be used for real diagnosis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Tech Stack](#3-tech-stack)
4. [Project Workflow](#4-project-workflow)
5. [Step-by-Step Explanation](#5-step-by-step-explanation)
6. [Model Architecture](#6-model-architecture)
7. [Results](#7-results)
8. [Predictive System (Inference)](#8-predictive-system-inference)
9. [How to Run](#9-how-to-run)
10. [Limitations & Future Improvements](#10-limitations--future-improvements)

---

## 1. Project Overview

| Item | Detail |
|---|---|
| **Problem type** | Binary classification |
| **Goal** | Predict whether a tumor is malignant or benign from measurements of cell nuclei |
| **Model** | Fully connected Neural Network (1 hidden layer) |
| **Framework** | TensorFlow / Keras |
| **Test accuracy** | **~93.9%** |

The notebook walks through the full machine-learning pipeline: loading data, exploring it, preprocessing, building and training a neural network, evaluating it, and finally using it to predict on a brand-new sample.

---

## 2. Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset**, loaded directly from `sklearn.datasets.load_breast_cancer()`.

- **Samples:** 569 patients
- **Features:** 30 numeric features
- **Missing values:** None
- **Target (`label`):**
  - `0` → **Malignant** (212 samples)
  - `1` → **Benign** (357 samples)

### About the 30 features

The features are computed from digitized images of a fine needle aspirate (FNA) of a breast mass. They describe the characteristics of the cell nuclei. Ten base measurements are each recorded in three forms: **mean**, **standard error**, and **worst** (largest value).

| Base measurement | Meaning |
|---|---|
| radius | Mean distance from center to points on the perimeter |
| texture | Standard deviation of gray-scale values |
| perimeter | Size of the core tumor boundary |
| area | Area of the nucleus |
| smoothness | Local variation in radius lengths |
| compactness | perimeter² / area − 1.0 |
| concavity | Severity of concave portions of the contour |
| concave points | Number of concave portions of the contour |
| symmetry | Symmetry of the nucleus |
| fractal dimension | "Coastline approximation" − 1 |

10 measurements × 3 forms (mean, error, worst) = **30 features**.

### Class balance

The dataset is moderately imbalanced: roughly **63% benign** and **37% malignant**.

### Key insight from exploration

Grouping by label shows clear differences between classes. For example, malignant tumors have a much larger average radius (~17.5) and area (~978) than benign tumors (~12.1 and ~463). This is a strong sign that the features carry useful signal for classification.

---

## 3. Tech Stack

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations, array handling |
| `pandas` | Data frame creation and exploration |
| `matplotlib` | Plotting accuracy and loss curves |
| `scikit-learn` | Dataset loading, train/test split, feature scaling |
| `tensorflow` / `keras` | Building, training, and evaluating the neural network |

---

## 4. Project Workflow

```
Load Data  →  Explore Data  →  Separate X / Y  →  Train-Test Split
      →  Standardize Features  →  Build NN  →  Compile  →  Train
      →  Visualize Metrics  →  Evaluate on Test Set  →  Predict on New Data
```

---

## 5. Step-by-Step Explanation

### 5.1 Importing the dependencies
Imports NumPy, Pandas, Matplotlib, scikit-learn's dataset module and `train_test_split`.

### 5.2 Data collection & processing
- The dataset is loaded with `load_breast_cancer()`, which returns a dictionary-like object containing `data`, `target`, and `feature_names`.
- It is converted into a Pandas DataFrame so it is easier to inspect.
- A new column named `label` is added to hold the target (0 or 1).

### 5.3 Exploratory data analysis (EDA)
| Command | What it tells us |
|---|---|
| `data_frame.head()` / `.tail()` | Look at the first and last rows |
| `data_frame.shape` | `(569, 31)` → 569 rows, 30 features + 1 label |
| `data_frame.info()` | All columns are `float64` (label is integer) with no nulls |
| `data_frame.isnull().sum()` | Confirms **zero missing values** |
| `data_frame.describe()` | Mean, std, min, max, and quartiles for every feature |
| `data_frame['label'].value_counts()` | 357 benign vs 212 malignant |
| `data_frame.groupby('label').mean()` | Average of each feature per class |

### 5.4 Separating features and target
```python
X = data_frame.drop(columns='label', axis=1)   # 30 input features
Y = data_frame['label']                        # target
```

### 5.5 Train-test split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```
- **80%** training → 455 samples
- **20%** testing → 114 samples
- `random_state=2` makes the split reproducible.

### 5.6 Standardizing the data
```python
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)
```
Features live on very different scales (e.g., *area* is in the hundreds/thousands while *smoothness* is around 0.1). Neural networks train poorly in that situation, so each feature is rescaled to **mean 0 and standard deviation 1**.

> ✅ The scaler is **fit only on the training data** and then applied to the test data. This prevents **data leakage** (the model never "sees" test statistics during training).

### 5.7 Building the neural network
See [Model Architecture](#6-model-architecture) below.

### 5.8 Compiling the model
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
- **Optimizer – Adam:** an adaptive learning-rate optimizer that works well by default.
- **Loss – Sparse Categorical Crossentropy:** suited to integer class labels (0, 1) when the output layer has one neuron per class.
- **Metric – Accuracy:** the share of correct predictions.

### 5.9 Training the model
```python
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
```
- **10 epochs:** the model passes over the training data 10 times.
- **`validation_split=0.1`:** 10% of the training data is held out to monitor performance on unseen data during training.
- The `history` object stores the metrics per epoch for plotting.
- `tf.random.set_seed(3)` is used for reproducibility.

### 5.10 Visualizing accuracy and loss
Two plots are produced:
- **Model accuracy:** training vs. validation accuracy per epoch
- **Model loss:** training vs. validation loss per epoch

These show whether the model is learning and whether it is overfitting.

### 5.11 Evaluating on test data
```python
loss, accuracy = model.evaluate(X_test_std, Y_test)
```
Tests the trained model on the 114 samples it has never seen.

### 5.12 Converting probabilities to class labels
`model.predict()` returns a probability pair for every sample, e.g. `[0.248, 0.538]` → `[P(malignant), P(benign)]`.

`np.argmax` picks the index of the larger value, which is the predicted class:

```python
Y_pred_labels = [np.argmax(i) for i in Y_pred]
```

---

## 6. Model Architecture

```
Input (30 features)
      │
 Flatten layer        → input_shape = (30,)
      │
 Dense (20 neurons)   → ReLU activation
      │
 Dense (2 neurons)    → Sigmoid activation
      │
Output: [P(Malignant), P(Benign)]
```

| Layer | Details | Purpose |
|---|---|---|
| `Flatten(input_shape=(30,))` | Input layer | Receives the 30 features (the data is already flat, so this is mainly a formal input layer) |
| `Dense(20, activation='relu')` | Hidden layer, 20 neurons | Learns non-linear combinations of features |
| `Dense(2, activation='sigmoid')` | Output layer, 2 neurons | One output per class |

**Trainable parameters:** (30 × 20 + 20) + (20 × 2 + 2) = **662**

---

## 7. Results

### Training log (final epoch)

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 95.84% | 95.65% |
| Loss | 0.1556 | 0.1221 |

Training progressed smoothly: accuracy rose from ~45% in epoch 1 to ~96% in epoch 10, while loss fell from 0.77 to 0.16.

### Final test performance

| Metric | Value |
|---|---|
| **Test accuracy** | **93.86%** |
| **Test loss** | 0.1693 |

### Interpretation
- Training and validation curves track each other closely, which suggests **little overfitting**.
- The model is still improving at epoch 10, so training for more epochs may give a small gain.
- A simple one-hidden-layer network already reaches roughly 94% accuracy on this dataset.

---

## 8. Predictive System (Inference)

The notebook ends with a system that classifies a **single new tumor** from its 30 measurements:

1. Put the 30 values in a tuple.
2. Convert to a NumPy array and reshape to `(1, -1)` (one sample, 30 features).
3. **Standardize** with the same `scaler` used in training.
4. Run `model.predict()`.
5. Use `np.argmax` to pick the class.
6. Print the result.

```python
if prediction_label[0] == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')
```

**Example output:**
```
[[0.42537233 0.9805447 ]]
[1]
The tumor is Benign
```

> ℹ️ A `UserWarning` about feature names appears because the scaler was fitted on a DataFrame (which has column names) but the new input is a plain NumPy array. It is harmless here.

---

## 9. How to Run

### Install the requirements
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

### Launch the notebook
```bash
jupyter notebook Breast_Cancer_Classification_with_NN.ipynb
```
Then run all cells from top to bottom. No external dataset download is needed because the data ships with scikit-learn.

---

## 10. Limitations & Future Improvements

### Things worth knowing
- **Sigmoid + sparse categorical crossentropy:** this works, but the more standard pairing is **softmax** for a 2-neuron output (probabilities then sum to 1). Alternatively, use **1 neuron + sigmoid + binary crossentropy**. With sigmoid on two neurons the outputs are independent and do not sum to 1 (e.g. `[0.425, 0.981]`).
- **Accuracy alone is not enough** for medical problems. Missing a malignant tumor (false negative) is far worse than a false alarm, so recall/sensitivity matters.
- **Small dataset** (569 rows) and a single train/test split, so the result may vary with a different split.
- **Short training** (10 epochs) with no early stopping.

### Ideas to improve
- Add a **confusion matrix**, **precision, recall, F1-score**, and **ROC-AUC**
- Use **softmax** or **binary crossentropy** as described above
- Train longer with **EarlyStopping**
- Add **Dropout** / regularization and try deeper architectures
- Use **k-fold cross-validation** for a more reliable estimate
- Compare against classic models (Logistic Regression, Random Forest, SVM)
- Save the model and scaler (`model.save()`, `joblib.dump(scaler)`) and serve it via a small web app (Streamlit / Flask)

---

## Summary

This project demonstrates an end-to-end ML workflow: **load → explore → preprocess → build → train → evaluate → predict**. A compact neural network with just 662 parameters reaches about **94% test accuracy** at distinguishing malignant from benign tumors, making it a solid starting point for learning neural-network classification.

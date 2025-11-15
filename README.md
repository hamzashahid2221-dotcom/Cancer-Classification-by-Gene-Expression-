# Cancer-Classification-by-Gene-Expression-
# 🧬 Leukemia Subtype Classification using Ensemble Machine Learning

This repository contains my research project on **classifying Leukemia subtypes** —  
**Acute Lymphoblastic Leukemia (ALL)** and **Acute Myeloid Leukemia (AML)** — using **gene expression microarray data**.

The study applies multiple machine learning models and a final **Stacking Ensemble** to improve diagnostic robustness for precision oncology.

---

## 🎯 Project Overview

Leukemia is one of the most common life-threatening cancers, and accurate subtype identification is essential for treatment decisions.  
However, microarray-based diagnosis is complex because gene expression space is **high-dimensional** while clinical samples are **limited**.

To address this, I combined:

### ✔ Classical ML models  
### ✔ Gradient boosting algorithms  
### ✔ Deep learning-based gene classifiers  
### ✔ Final **Stacking Ensemble** (best)

This approach leverages diverse learning patterns and improves clinical reliability.

---

## 🧩 Dataset

| Dataset | Description |
|--------|-------------|
| `data_set_ALL_AML_train.csv` | Training dataset (gene expression) |
| `data_set_ALL_AML_independent.csv` | Independent test dataset |
| `actual.csv` | Patient label file (ALL / AML) |

🧪 Gene count (after cleaning): **7130**  
👥 Final Training Samples: **≈ 63**  
👥 Independent Testing Samples: **34 patients**  

**Classes:**
- `ALL` — Acute Lymphoblastic Leukemia  
- `AML` — Acute Myeloid Leukemia  

---

## 🧠 Models Used

| Model | Framework |
|-------|-----------|
| Support Vector Classifier (Linear SVM) | `scikit-learn` |
| CatBoost Classifier | `catboost` |
| Deep Neural Network (MLP) | `TensorFlow / Keras` |
| **Ensembles: Soft Voting & Stacking** | `scikit-learn` |

### 🔗 Final Ensemble
```python
VotingClassifier(
    estimators=[('svc', model_svc), ('cat', model_cat), ('dl', keras_clf)],
    voting='soft'
)

StackingClassifier(
    estimators=[('svc', model_svc), ('cat', model_cat), ('dl', keras_clf)],
    final_estimator=LogisticRegression()
)


Training Strategy:

| Step             | Method                           |
| ---------------- | -------------------------------- |
| Feature Scaling  | StandardScaler                   |
| Target Encoding  | LabelEncoder                     |
| Split            | 80% Train — 20% Validation       |
| Cross-Validation | 5-Fold CV                        |
| Metric           | Accuracy + Classification Report |

Result:

| Metric              | Score     |
| ------------------- | --------- |
| Average CV Accuracy | **~0.91** |

Independent Test Result:

Classification Report:

              precision    recall  f1-score   support
ALL              0.90      0.95      0.93        20
AML              0.92      0.86      0.89        14

accuracy                             0.91        34
macro avg         0.91      0.90      0.91        34
weighted avg      0.91      0.91      0.91        34

Tech Stack:
| Library / Tool        | Usage                        |
| --------------------- | ---------------------------- |
| Python, Pandas, NumPy | Data preparation             |
| Scikit-learn          | ML Models, CV, Preprocessing |
| XGBoost               | Gradient boosting            |
| CatBoost              | Categorical boosting         |
| TensorFlow / Keras    | Deep learning                |
| Matplotlib, Seaborn   | Visualization                |

Research Significance:
This study demonstrates that combining classical ML, boosting, and neural networks yields strong diagnostic potential for hematological cancers.

The approach supports precision medicine and can be extended to other genomic biomarker discovery tasks.
Author

Hamza Shahid
Bachelor of Biomedical Engineering (with Distinction)
University of Engineering & Technology (UET), Lahore

🔍 Research Interests:
AI in Healthcare • Computational Oncology • Bioinformatics • Precision Medicine

License

Released under the MIT License — open for academic and research use.

Acknowledgment

Thanks to global open-source biomedical research communities enabling AI-driven healthcare innovation.

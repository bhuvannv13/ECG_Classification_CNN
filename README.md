# MIT-BIH Arrhythmia Dataset - EDA, Class Balancing & Deep Learning Model

This repository contains a comprehensive exploratory data analysis (EDA) of the MIT-BIH Arrhythmia dataset, preprocessing steps to address class imbalance, and a deep learning model to classify ECG signals into five heartbeat types.

---

## 📊 Project Highlights

- Performed class distribution analysis and visualizations
- Upsampled minority classes to achieve balanced representation
- Visualized sample ECG signals and intensity heatmaps
- Built and evaluated a 1D CNN model for heartbeat classification

---

## 🧪 Dataset

MIT-BIH Arrhythmia Dataset (from PhysioNet):

- Each record contains 186 features representing an ECG beat
- Classes:
  - 0: Non-ectopic beat (N)
  - 1: Supraventricular ectopic beat (S)
  - 2: Ventricular ectopic beat (V)
  - 3: Fusion beat (F)
  - 4: Unknown beat (Q)

---

## 📁 Files

- `EDA_MITBIH.ipynb`: Colab notebook with EDA, upsampling, and model training
- `train.csv`: Original imbalanced training data
- `test.csv`: Original test data
- `README.md`: This file

---

## ⚙️ Dependencies

Install via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
📈 Results
📊 Data Balancing
Before: Extreme class imbalance (e.g., Class 0 had ~90,000 samples, Class 3 had ~800)

After: All classes balanced to 20,000 samples (100,000 total training instances)

🤖 Model
Model Used: 1D Convolutional Neural Network (CNN)

Architecture:

Input Layer (186,)

2 Conv1D layers + MaxPooling

Flatten → Dense → Softmax (5 classes)

🧪 Evaluation on Test Set
Accuracy: 95.43%

Precision (macro avg): 94.92%

Recall (macro avg): 95.41%

F1 Score (macro avg): 95.08%

Confusion Matrix: Shows strong per-class performance with minimal misclassifications

🔍 Visualizations
Heatmaps for each class

Sample ECG plots

Training loss/accuracy curves

🚀 Next Steps
Experiment with LSTM or hybrid CNN-LSTM

Test generalizability on real-time ECG signal windows

Deploy model using Hugging Face or Flask

This Project is done During my bachelors for Learning purpose

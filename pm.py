"""
Bearing Fault Diagnosis Pipeline using Vibration Signals (CWRU Dataset)
========================================================================

Author: Mahdi
Domain: Industrial Predictive Maintenance & Vibration Analysis

Overview:
---------
This module implements an end-to-end classification pipeline to detect and 
diagnose bearing defects based on accelerometer vibration signals. 

Data Strategy & Generalization:
-------------------------------
To simulate real-world blind inference and avoid data leakage:
- Baseline training is conducted on data under 0 HP motor load.
- Blind model evaluation is performed on data under 1 HP motor load.
- Feature scalers (StandardScaler) and label encoders (LabelEncoder) are 
  strictly fitted on the training split only and subsequently transformed 
  on the unseen test split.

Target Classes:
---------------
0: Ball fault
1: Inner race fault
2: Normal baseline
3: Outer race fault

Key Features:
-------------
- Leakage-free feature scaling and encoding.
- Cross-load performance evaluation.
- Confusion matrix generation and visualization for diagnostic verification.
"""
# ==========================================
#  Importing the Required Libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import urllib.request
import os
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


# ==========================================================================================
# Dataset Configuration: Maps fault classes to their local filenames and remote source URLs.
# ==========================================================================================
DATASET_FILES = {
    "Normal": ("97.mat", "https://engineering.case.edu/sites/default/files/97.mat"),
    "Inner_Race": ("105.mat", "https://engineering.case.edu/sites/default/files/105.mat"),
    "Ball": ("118.mat", "https://engineering.case.edu/sites/default/files/118.mat"),
    "Outer_Race": ("130.mat", "https://engineering.case.edu/sites/default/files/130.mat")
}


# =============================
# Signal Processing Pipeline
# =============================
def download_and_extract_all(window_size=2048):
    """
    Downloads raw CWRU vibration signals, performs signal segmentation, and extracts 
    time-frequency domain features to construct a training dataset.

    Features Extracted:
        - Time-Domain:
            - RMS: Measures signal energy content.
            - Kurtosis: Measures the impulsiveness/spikiness (detects early bearing faults).
            - Skewness: Measures the asymmetry of the signal distribution.
            - Crest Factor: Ratio of peak to RMS; indicates severity of impact shocks.
        - Frequency-Domain (Spectral):
            - Peak Frequency: The dominant spectral component (correlates to fault frequencies).
            - Total Power: Represents total signal energy in the frequency domain.

    Args:
        window_size (int): Number of samples per signal segment. Defaults to 2048.

    Returns:
        pd.DataFrame: A structured dataset containing extracted feature vectors and fault labels.
    """
    all_data = []
    for label, (filename, url) in DATASET_FILES.items():
        #Download file if not present
        if not os.path.exists(filename):
            print(f"Downloading file {filename} for class: {label}...")
            urllib.request.urlretrieve(url, filename)

        raw = loadmat(filename)
        # Find Drive End (DE) sensor data
        key = [k for k in raw.keys() if 'DE_time' in k][0]
        signal = raw[key].flatten()
        fs = 12000  
        for start in range(0, len(signal) - window_size, window_size):
            win = signal[start:start + window_size]
            rms = np.sqrt(np.mean(win**2))
            kurt = kurtosis(win)
            sk = skew(win)
            peak = np.max(np.abs(win))
            crest = peak / (rms + 1e-8)
            n = len(win)
            fft_vals = np.abs(np.fft.rfft(win))  # Frequency domain analysis
            freqs = np.fft.rfftfreq(n, d=1 / fs)
            peak_freq = freqs[1:][np.argmax(fft_vals[1:])]  # Frequency axis (Hz)
            total_power = np.sum(fft_vals**2) / n


            all_data.append({
                'RMS': rms,
                'Kurtosis': kurt,
                'Skewness': sk,
                'Crest_Factor': crest,
                'peak':peak_freq,
                'total power':total_power,
                'Label': label
            })

    return pd.DataFrame(all_data)


### Pipeline execution and data integrity validation
df_all = download_and_extract_all()
print("\n--- Statistics by Fault Class ---")
print(df_all['Label'].value_counts())
print("\nPreview of processed training data:")
print(df_all.sample(5))

### Shuffle dataset and reset indices for randomized order.
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)


# ===========================================
# Scale features and encode target labels.
# ===========================================
scaler = StandardScaler()
X = df_all.drop('Label', axis=1) 
y = df_all['Label']              
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

for index, class_name in enumerate(le.classes_):
    print(f"Class ID {index} -> {class_name}")


# print(X_scaled[100:130])  
# print(y_encoded[100:130])


# ==============================================
# Split data, train Random Forest, and predict.
# ==============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


# ==================================================================================
# Evaluate model performance, display classification metrics, and store predictions.
# ==================================================================================
print("\n" + "="*45)
print(f"Overall Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("="*45)

print("\n📊 Classification Performance Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

result = pd.DataFrame(X_test)
result["y_test"] = y_test
result["y_pred"] = y_pred
print(result)

### Configure 1HP test dataset for model evaluation.
files_1hp = {
    "Normal":     ("98.mat",  "https://engineering.case.edu/sites/default/files/98.mat"),
    "Inner_Race": ("106.mat", "https://engineering.case.edu/sites/default/files/106.mat"),
    "Ball":       ("119.mat", "https://engineering.case.edu/sites/default/files/119.mat"),
    "Outer_Race": ("131.mat", "https://engineering.case.edu/sites/default/files/131.mat"),
}
DATASET_FILES = files_1hp
df_test_1hp = download_and_extract_all()

print("\nTest dataset shape:")
print(df_test_1hp.shape)

print("\nSample count per class:")
print(df_test_1hp["Label"].value_counts())


# =============================================================
# Scale test features and perform inference on the 1HP dataset.
# =============================================================
X = df_test_1hp.drop('Label', axis=1) 
y = df_test_1hp['Label']             
X_scaled = scaler.transform(X)
y_encoded = le.transform(y)

for index, class_name in enumerate(le.classes_):
    print(f"Label Mapping: {index} -> {class_name}")

y_pred2 = rf_model.predict(X_scaled)


# ==========================================================
# Evaluate inference results and visualize confusion matrix.
# ==========================================================
check = pd.DataFrame({'Actual': y_encoded, 'Predicted': y_pred2})
print(check.sample(30))
cm = confusion_matrix(y_encoded, y_pred2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)


plt.title('Confusion Matrix for 1HP Dataset', fontsize=15)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)


plt.tight_layout()
plt.show()
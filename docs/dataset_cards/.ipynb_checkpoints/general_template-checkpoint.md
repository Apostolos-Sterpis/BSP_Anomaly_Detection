# Dataset Card: <Dataset Name>

## 1. Overview
A short description of the dataset:
- Domain (e.g., biomedical, IoT, motion)
- What the signal represents
- What the anomaly represents
- Why this dataset is interesting or relevant

---

## 2. File Information
**Filename:** `<filename>.txt`  
**Length:** `<number of samples>`  
**Sampling Frequency:** `<Hz or "Not provided">`  
**Value Range:**  
- Min: `<value>`  
- Max: `<value>`

---

## 3. Anomaly Metadata
**Anomaly Type:** `<collective, contextual, point, shape distortion, noise burst, clipped plateau, etc.>`  

**Indices:**  
- Normal region end (train_end): `<train_end>`  
- Anomaly start: `<anomaly_start>`  
- Anomaly end: `<anomaly_end>`  
- Anomaly length: `<anomaly_end - anomaly_start>`  

**Description:**  
A short sentence describing the anomaly.

---

## 4. Visual Characteristics
- Periodicity: `<yes/no/weak>`  
- Trend/Drift: `<yes/no>`  
- Noise Level: `<low/medium/high>`  
- Amplitude Stability: `<stable/variable>`  
- Local Shape Complexity: `<low/medium/high>`  

Short notes on how the dataset looks based on the plots.

---

## 5. Method Suitability

### **5.1 Z-Score Thresholding**
Describe how well Z-Score is expected to perform:
- Works best on clean, stable signals with consistent amplitude.
- Struggles with noise, drift, or shape-based anomalies.

### **5.2 Moving Average & Residual Analysis**
Describe suitability:
- Good for trend removal and contextual anomalies.
- Performs well on IoT-like signals with slow transitions.
- Struggles when data is chaotic or highly irregular.

### **5.3 Statistical Process Control (SPC)**
Describe suitability:
- Very effective at detecting spikes, shifts, plateaus, or sustained deviations.
- Ideal for industrial, physiological, or rhythmic signals.
- Assumes relatively stationary behavior; weaker on chaotic, short, or drifting data.

### **5.4 Matrix Profile (Discord Discovery)**
Describe suitability:
- Excellent for shape and motif anomalies.
- Works well on periodic or semi-periodic signals.
- Struggles if the dataset is too short, extremely noisy, or lacks repeated structure.

### **5.5 Isolation Forest**
Describe suitability:
- Strong general-purpose anomaly detector.
- Works well on noisy or irregular datasets.
- May be unstable with short sequences or when training data is too limited.

### **5.6 One-Class SVM**
Describe suitability:
- Performs well on clean, well-scaled signals.
- Sensitive to noise and parameter choices.
- May underperform on high-noise or chaotic datasets.

### **5.7 LSTM Autoencoder**
Describe suitability:
- Best for datasets with learnable temporal patterns (periodic or semi-structured).
- Strong at detecting shape distortions and sequence-level anomalies.
- Requires sufficient clean training data; may struggle on short or chaotic signals.

---

## 6. Preprocessing Requirements
- Cleaning required: `<yes/no + details>`  
- Normalization needed: `<yes/no + recommended type>`  
- Detrending needed: `<yes/no>`  
- Windowing recommendations: `<window sizes>`  

---

## 7. Challenges & Notes
Any difficulties or particularities:
- Heavy noise  
- Very short length  
- Irregular waveform  
- Multi-scale anomaly  
- Sensor drift, etc.

---

## 8. Justification for Inclusion
Why this dataset was chosen and what aspect of anomaly detection it helps evaluate.

---

## 9. Visualizations

### Raw Signal
![raw](../../results/plots/<dataset_name>/raw.png)

### Anomaly Highlighted
![highlight](../../results/plots/<dataset_name>/highlight.png)

### Zoomed Anomaly Region
![zoom-anomaly](../../results/plots/<dataset_name>/zoom_anomaly.png)

### Zoomed Normal Region
![zoom-normal](../../results/plots/<dataset_name>/zoom_normal.png)

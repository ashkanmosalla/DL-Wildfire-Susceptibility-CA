# DL-Wildfire-Susceptibility-CA

# National-Scale Wildfire Susceptibility and Recurring Risk Mapping across Canada

This repository contains the implementation of a deep learning-based framework for mapping wildfire frequency and likelihood across Canada. By integrating multi-source hydrogeoclimatic data, the study benchmarks several advanced architectures to identify optimal predictors for national-scale wildfire risk assessment.

## 🖼️ Methodology & Workflow
Based on the study's **Graphical Abstract**, the pipeline consists of:
1.  **Data Preparation:** Processing 4,240 large fire locations (CNFDB) and 13 hydrogeoclimatic variables.
2.  **Target Definition:** KDE-based calculation of **Frequency** (expected counts) and **Likelihood** (occurrence probability).
3.  **Feature Selection:** A hybrid **Boruta-CART** approach to identify the most salient predictors.
4.  **Spatial Validation:** **10-Fold Spatial Cross-Validation** via K-means clustering to mitigate spatial autocorrelation.
5.  **Model Calibration:** Benchmarking BiRNN, ED-BiRNN, LSTM, RNN, and DNN architectures.

## 📂 Repository Structure
* `src/architectures.py`: Definitions of all Deep Learning models (BiRNN, ED-BiRNN, etc.).
* `src/feature_selection.py`: Implementation of the Boruta-CART hybrid selection logic.
* `src/spatial_cv.py`: K-means spatial partitioning logic.
* `src/tuner.py`: Automated grid search engine for hyperparameter calibration.
* `main.py`: Master script for executing the pipeline for both target variables.

## 🛠 Usage
Install dependencies:
```bash
pip install -r requirements.txt

python main.py
```
## 📊 Key Results
The pipeline generates:
* **Performance Metrics:** R², RMSE, MAE for each fold and target variable.
* **Spatial Maps:** Susceptibility and recurring risk outputs in CSV/Raster compatible formats.
* **Feature Importance:** SHAP-based interpretability outputs to identify key hydrogeoclimatic drivers.

## 📦 Dependencies
* Python 3.10+
* TensorFlow 2.x
* Scikit-learn
* Boruta
* Pandas & NumPy
* Matplotlib & Seaborn

# DL-Wildfire-Susceptibility-CA
## National-Scale Wildfire Susceptibility and Recurring Risk Mapping across Canada

This repository contains the official implementation of a deep learning-based framework for mapping wildfire frequency and likelihood across Canada. By integrating multi-source hydrogeoclimatic data, the study benchmarks several advanced architectures (BiRNN, ED-BiRNN, LSTM, RNN, and DNN) to identify optimal predictors for national-scale risk assessment.

## 🖼️ Methodology & Workflow
Based on the study's **Graphical Abstract**, the pipeline consists of:
1.  **Data Preparation:** Processing 4,240 large fire locations (CNFDB) and 13 hydrogeoclimatic variables.
2.  **Target Definition:** KDE-based calculation of **Frequency** (expected counts) and **Likelihood** (occurrence probability).
3.  **Feature Selection:** A hybrid **Boruta-CART** approach for identifying the most salient predictors.
4.  **Spatial Validation:** **10-Fold Spatial Cross-Validation** via K-means clustering to mitigate spatial autocorrelation.
5.  **Model Calibration:** Benchmarking and hyperparameter tuning for various DL architectures.



## 📂 Repository Structure
* `src/architectures.py`: Definitions of all Deep Learning models.
* `src/feature_selection.py`: Implementation of the Boruta-CART hybrid selection logic.
* `src/spatial_cv.py`: K-means spatial partitioning logic.
* `src/tuner.py`: Automated grid search engine for hyperparameter calibration.
* `main.py`: Master script for executing the pipeline for both target variables.

## 👥 Research Team & Acknowledgments
This research was a collaborative effort. Special thanks to the core contributors:

* **Ashkan Mosallanejad** 
* **Khabat Khosravi**
* **Alireza Shahvaran**


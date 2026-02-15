import pandas as pd
from src.data_utils import WildfireDataProcessor
from src.feature_engineering import FeatureSelector
from src.spatial_cv import SpatialCVManager
from src.tuner import ModelCalibrator
from src.architectures import WildfireArchitectures

def run_wildfire_pipeline(data_path, target_col):
    """
    Main execution pipeline following the Graphical Abstract flow:
    Data -> Preprocessing -> Feature Selection -> Spatial CV -> Grid Search -> Final Evaluation
    """
    
    # 1. Load and Clean Data
    num_vars = ['Wind speed', 'Maximum Temperature', 'Soil moisture', 'Precipitation', 
                'Slope', 'Elevation', 'Humidity', 'NDVI', 'Evaptranspiration', 
                'Shortwave radiation flux', 'Climatic water deficit']
    
    processor = WildfireDataProcessor(numerical_cols=num_vars)
    df = processor.load_and_clean(data_path)
    
    # 2. Assign Spatial Folds (K-Means Clustering)
    spatial_cv = SpatialCVManager(n_folds=10)
    df = spatial_cv.assign_spatial_folds(df)
    
    # 3. Preprocessing (Scaling & One-Hot Encoding)
    X_processed = processor.fit_transform(df)
    y = df[target_col]
    
    # 4. Hybrid Feature Selection (Boruta + CART)
    selector = FeatureSelector()
    selected_features = selector.run_selection(X_processed, y)
    X_final = X_processed[selected_features]
    
    print(f"Selected Features for {target_col}: {selected_features}")

    # 5. 10-Fold Spatial Cross-Validation Loop
    results = []
    for fold in range(10):
        train_idx, val_idx = spatial_cv.get_split(df, fold)
        
        X_train, y_train = X_final.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X_final.loc[val_idx], y.loc[val_idx]
        
        # 6. Hyperparameter Calibration (Grid Search) for the Top Model (BiRNN)
        tuner = ModelCalibrator(model_type='BiRNN', input_shape=X_final.shape[1])
        best_cfg = tuner.calibrate(X_train, y_train, X_val, y_val)
        
        # Training final model for this fold with best_cfg...
        # Store results for Table 3
        
    print(f"Pipeline completed for {target_col}.")

if __name__ == "__main__":
    # Running for Frequency (Target 1)
    run_wildfire_pipeline('wildfire_frequency.csv', 'ndfb_kd_co')
    
    # Running for Likelihood (Target 2)
    run_wildfire_pipeline('wildfire_likelihood.csv', 'ndfb_kd_pr')
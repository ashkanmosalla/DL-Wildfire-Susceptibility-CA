import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

class FeatureSelector:
    """
    Implements the Hybrid Feature Selection approach (Boruta-CART) 
    as described in the study's methodology to identify optimal 
    hydrogeoclimatic predictors for wildfire risk mapping.
    """
    
    def __init__(self, cart_threshold=0.0005, random_state=42):
        """
        Initializes the feature selector.
        :param cart_threshold: Minimum importance value for CART selection (Gini index).
        :param random_state: Seed for reproducibility.
        """
        self.cart_threshold = cart_threshold
        self.random_state = random_state

    def run_selection(self, X, y):
        """
        Executes the dual-stage selection process:
        1. Boruta: Identifies all relevant features.
        2. CART: Ranks features based on relative importance.
        
        Returns a list of features that satisfy both criteria.
        """
        print("Starting Hybrid Feature Selection (Boruta-CART)...")

        # --- Stage 1: Boruta Selection ---
        # Using Random Forest as the shadow-feature generator for Boruta
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=self.random_state)
        
        # Initialize Boruta
        boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=self.random_state)
        
        # Fit Boruta (Note: Boruta requires numpy arrays)
        boruta_selector.fit(X.values, y.values)
        
        # Get confirmed features from Boruta
        boruta_confirmed = X.columns[boruta_selector.support_].tolist()
        print(f"Boruta confirmed {len(boruta_confirmed)} relevant predictors.")

        # --- Stage 2: CART Importance (Gini Index/MSE) ---
        # Decision Tree is used to rank the relative importance of features
        cart_model = DecisionTreeRegressor(random_state=self.random_state)
        cart_model.fit(X, y)
        
        # Get feature importances from CART
        cart_importances = pd.Series(cart_model.feature_importances_, index=X.columns)
        
        # Filter features that meet the importance threshold
        cart_selected = cart_importances[cart_importances > self.cart_threshold].index.tolist()
        print(f"CART identified {len(cart_selected)} features above the importance threshold.")

        # --- Final Hybrid Selection ---
        # Logic: Feature must be confirmed by Boruta AND significant in CART
        final_selected_features = [feat for feat in boruta_confirmed if feat in cart_selected]
        
        print(f"Hybrid selection completed. Final set includes {len(final_selected_features)} features.")
        
        return final_selected_features

    def get_importance_report(self, X, y):
        """
        Generates a summary of feature importance for reporting in results tables.
        """
        cart_model = DecisionTreeRegressor(random_state=self.random_state)
        cart_model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance_Score': cart_model.feature_importances_
        }).sort_values(by='Importance_Score', ascending=False)
        
        return importance_df
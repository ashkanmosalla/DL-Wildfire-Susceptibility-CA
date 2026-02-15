import pandas as pd
from sklearn.cluster import KMeans

class SpatialCVManager:
    """
    Manages 10-Fold Spatial Cross-Validation logic to mitigate 
    spatial autocorrelation as described in the study.
    This module uses K-means clustering on Latitude and Longitude 
    to create geographically independent folds.
    """
    
    def __init__(self, n_folds=10, random_state=42):
        """
        Initializes the spatial cross-validation manager.
        :param n_folds: Number of spatial clusters (folds), default is 10.
        :param random_state: Random seed for reproducibility.
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_folds, 
                             random_state=self.random_state, 
                             n_init=10)

    def assign_spatial_folds(self, df, lat_col='Latitude', lon_col='Longitude'):
        """
        Partitions the study area into geographically independent clusters.
        Each cluster represents a unique fold for the cross-validation process.
        """
        # Extract geographic coordinates for clustering
        coords = df[[lat_col, lon_col]]
        
        # Fit K-means and assign cluster labels to each data point
        df['spatial_fold'] = self.kmeans.fit_predict(coords)
        
        print(f"Data successfully partitioned into {self.n_folds} spatial clusters.")
        return df

    def get_fold_data(self, df, fold_id):
        """
        Returns training and validation indices for a specific spatial fold.
        :param df: The dataframe containing 'spatial_fold' labels.
        :param fold_id: The index of the current fold to be used as validation.
        """
        # Data points NOT in the current cluster are used for training
        train_idx = df[df['spatial_fold'] != fold_id].index
        
        # Data points IN the current cluster are used for validation
        val_idx = df[df['spatial_fold'] == fold_id].index
        
        return train_idx, val_idx

    def drop_meta_columns(self, df, extra_cols=None):
        """
        Removes metadata columns (Latitude, Longitude, Fold IDs) 
        before feeding the data into the models.
        """
        cols_to_drop = ['spatial_fold', 'Latitude', 'Longitude']
        if extra_cols:
            cols_to_drop.extend(extra_cols)
        
        return df.drop(columns=cols_to_drop, errors='ignore')
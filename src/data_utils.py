import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class WildfireDataProcessor:
    """
    Handles loading, cleaning, scaling, and categorical encoding 
    as described in the study's data preprocessing section.
    """
    def __init__(self, numerical_cols, categorical_cols=['LULC']):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def load_and_clean(self, path):
        df = pd.read_csv(path)
        # Replacing missing values reported as -9999
        df.replace(-9999, pd.NA, inplace=True)
        return df.dropna()

    def fit_transform(self, df):
        """
        Applies Standard Scaling to numerical features and 
        One-Hot Encoding to categorical features (e.g., LULC).
        """
        # 1. Scale numerical features
        scaled_num = self.scaler.fit_transform(df[self.numerical_cols])
        df_num = pd.DataFrame(scaled_num, columns=self.numerical_cols, index=df.index)

        # 2. One-Hot Encode categorical features
        encoded_cat = self.encoder.fit_transform(df[self.categorical_cols])
        cat_names = self.encoder.get_feature_names_out(self.categorical_cols)
        df_cat = pd.DataFrame(encoded_cat, columns=cat_names, index=df.index)

        # 3. Combine processed features
        return pd.concat([df_num, df_cat], axis=1)

    def transform(self, df):
        """Used for validation and test sets to avoid data leakage."""
        scaled_num = self.scaler.transform(df[self.numerical_cols])
        df_num = pd.DataFrame(scaled_num, columns=self.numerical_cols, index=df.index)

        encoded_cat = self.encoder.transform(df[self.categorical_cols])
        cat_names = self.encoder.get_feature_names_out(self.categorical_cols)
        df_cat = pd.DataFrame(encoded_cat, columns=cat_names, index=df.index)

        return pd.concat([df_num, df_cat], axis=1)
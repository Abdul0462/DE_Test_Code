import pandas as pd
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataLoader:
    """
    Loads and preprocesses the datasets
    """

    def __init__(self, item_file, sales_file, promotion_file, supermarket_file):
        self.files = {
            'items': item_file,
            'sales': sales_file,
            'promotion': promotion_file,
            'supermarkets': supermarket_file
        }
        self.data = {}

    def load_data(self):
        logging.info("Loading datasets...")
        self.data = {key: pd.read_csv(file) for key, file in self.files.items()}
        logging.info("Datasets loaded successfully.")
        return self.data


class DataPreprocessor:
    """
    Cleans and preprocesses the data.
    """

    def __init__(self, data):
        self.data = data

    def clean_data(self):
        logging.info("Cleaning data...")
        for key in self.data:
            self.data[key].dropna(inplace=True)
        logging.info("Data cleaning complete.")
        return self.data

    def merge_data(self):
        logging.info("Merging data...")
        merged_df = (self.data['sales']
                     .merge(self.data['items'], on='code', how='left')
                     .rename(columns={'supermarket': 'supermarkets'})
                     .merge(self.data['promotion'], on=['code', 'supermarkets'], how='left')
                     .rename(columns={'supermarkets': 'supermarket_No'})
                     .merge(self.data['supermarkets'], on='supermarket_No', how='left')
                     .rename(columns={'province_x': 'province', 'week_x': 'week'}))
        logging.info("Data merging complete. Columns: %s", merged_df.columns)
        return merged_df


class FeatureEngineer:
    """
    Handles feature engineering.
    """

    def __init__(self, data):
        self.data = data

    def create_features(self):
        logging.info("Creating features...")
        self.data['Total_Sales'] = self.data['units'] * self.data['amount']
        self.data.fillna(0, inplace=True)
        logging.info("Feature engineering complete.")
        return self.data


class ModelTrainer:
    """
    Trains a supervised learning model.
    """

    def __init__(self, data):
        self.data = data

    def train_model(self):
        logging.info("Training optimized model...")
        start_time = time.time()

        features = ['units', 'amount', 'week', 'province', 'postal-code']
        target = 'Total_Sales'

        X = self.data[features]
        y = self.data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        training_time = time.time() - start_time

        logging.info("Model training complete with MSE: %f and R² Score: %f", mse, r2)
        logging.info("Model training time: %.2f seconds", training_time)

        return model, mse, r2, training_time


if __name__ == "__main__":
    logging.info("Starting the pipeline...")
    loader = DataLoader('item.csv', 'sales.csv', 'promotion.csv', 'supermarkets.csv')
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    cleaned_data = preprocessor.clean_data()
    merged_data = preprocessor.merge_data()

    engineer = FeatureEngineer(merged_data)
    feature_data = engineer.create_features()

    trainer = ModelTrainer(feature_data)
    model, mse, r2, training_time = trainer.train_model()

    logging.info("Pipeline execution complete.")
    print(f"Model trained with MSE: {mse}, R² Score: {r2}, Training Time: {training_time:.2f} seconds")

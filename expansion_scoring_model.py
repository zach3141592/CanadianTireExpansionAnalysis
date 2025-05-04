from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ExpansionScoringModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def create_model(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Output score between 0 and 1
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=validation_split, random_state=42)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return history
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = load_model(path)

# Example usage
if __name__ == "__main__":
    # Example factors (you should replace with real data)
    factors = [
        'Market Size',
        'GDP Growth Rate',
        'Retail Market Growth',
        'Competition Level',
        'Regulatory Environment',
        'Infrastructure Quality',
        'Consumer Spending Power',
        'E-commerce Penetration',
        'Political Stability',
        'Cultural Similarity'
    ]
    
    # Create sample data (replace with real data)
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples, 10 factors
    y = np.random.rand(100)      # Target scores
    
    # Initialize and train model
    model = ExpansionScoringModel()
    model.create_model()
    history = model.train(X, y)
    
    # Example prediction
    new_country_data = np.random.rand(1, 10)
    score = model.predict(new_country_data)
    print(f"Expansion Score: {score[0][0]:.4f}") 
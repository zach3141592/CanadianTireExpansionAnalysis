import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generate_expansion_data import generate_country_data
from expansion_scoring_model import ExpansionScoringModel

def prepare_data(data):
    # Separate features and target
    feature_columns = [
        'Market_Size', 'GDP_Growth', 'Retail_Growth', 'Competition_Level',
        'Regulatory_Environment', 'Infrastructure_Quality', 'Consumer_Spending_Power',
        'Ecommerce_Penetration', 'Political_Stability', 'Cultural_Similarity'
    ]
    
    X = data[feature_columns].values
    y = data['Expansion_Score'].values
    
    return X, y

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_predictions(y_true, y_pred, countries):
    plt.figure(figsize=(15, 8))
    indices = np.arange(len(countries))
    width = 0.35
    
    plt.bar(indices - width/2, y_true, width, label='True Score', alpha=0.7)
    plt.bar(indices + width/2, y_pred, width, label='Predicted Score', alpha=0.7)
    
    plt.xlabel('Countries')
    plt.ylabel('Expansion Score')
    plt.title('True vs Predicted Expansion Scores')
    plt.xticks(indices, countries, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('predictions_comparison.png')
    plt.close()

def main():
    # Generate the dataset
    print("Generating expansion data...")
    data = generate_country_data()
    
    # Prepare data for training
    X, y = prepare_data(data)
    
    # Initialize and train the model
    print("\nInitializing and training the model...")
    model = ExpansionScoringModel()
    model.create_model()
    history = model.train(X, y, epochs=200, batch_size=16)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X)
    
    # Plot predictions vs actual
    print("Plotting predictions vs actual scores...")
    plot_predictions(y, predictions.flatten(), data['Country'].values)
    
    # Print top 5 countries for expansion
    data['Predicted_Score'] = predictions.flatten()
    top_countries = data.sort_values('Predicted_Score', ascending=False).head(5)
    
    print("\nTop 5 Countries for Expansion:")
    print(top_countries[['Country', 'Predicted_Score']].to_string(index=False))
    
    # Save the model
    model.save_model('expansion_scoring_model.h5')
    print("\nModel saved as 'expansion_scoring_model.h5'")

if __name__ == "__main__":
    main() 
import numpy as np
import pandas as pd
from datetime import datetime

def generate_country_data(num_countries=50):
    # List of potential target countries
    countries = [
        'Mexico', 'Brazil', 'Chile', 'Colombia', 'Peru',
        'Germany', 'France', 'UK', 'Italy', 'Spain',
        'Japan', 'South Korea', 'Australia', 'New Zealand', 'Singapore',
        'South Africa', 'Nigeria', 'Kenya', 'Egypt', 'Morocco',
        'India', 'Indonesia', 'Thailand', 'Vietnam', 'Malaysia',
        'Poland', 'Czech Republic', 'Hungary', 'Romania', 'Greece',
        'Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Oman',
        'Argentina', 'Uruguay', 'Costa Rica', 'Panama', 'Ecuador',
        'Sweden', 'Norway', 'Denmark', 'Finland', 'Netherlands',
        'Philippines', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Myanmar'
    ]
    
    # Generate synthetic data for each factor
    np.random.seed(42)
    
    # Market Size (in billions USD)
    market_size = np.random.uniform(10, 500, num_countries)
    
    # GDP Growth Rate (percentage)
    gdp_growth = np.random.uniform(-2, 8, num_countries)
    
    # Retail Market Growth (percentage)
    retail_growth = np.random.uniform(0, 15, num_countries)
    
    # Competition Level (0-1, where 1 is most competitive)
    competition = np.random.uniform(0.2, 0.9, num_countries)
    
    # Regulatory Environment (0-1, where 1 is most favorable)
    regulatory = np.random.uniform(0.3, 0.95, num_countries)
    
    # Infrastructure Quality (0-1, where 1 is best)
    infrastructure = np.random.uniform(0.4, 1.0, num_countries)
    
    # Consumer Spending Power (0-1, where 1 is highest)
    spending_power = np.random.uniform(0.2, 0.95, num_countries)
    
    # E-commerce Penetration (percentage)
    ecommerce = np.random.uniform(5, 40, num_countries)
    
    # Political Stability (0-1, where 1 is most stable)
    political_stability = np.random.uniform(0.3, 0.95, num_countries)
    
    # Cultural Similarity (0-1, where 1 is most similar to Canada)
    cultural_similarity = np.random.uniform(0.2, 0.9, num_countries)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Country': countries[:num_countries],
        'Market_Size': market_size,
        'GDP_Growth': gdp_growth,
        'Retail_Growth': retail_growth,
        'Competition_Level': competition,
        'Regulatory_Environment': regulatory,
        'Infrastructure_Quality': infrastructure,
        'Consumer_Spending_Power': spending_power,
        'Ecommerce_Penetration': ecommerce,
        'Political_Stability': political_stability,
        'Cultural_Similarity': cultural_similarity
    })
    
    # Generate target scores (expansion success probability)
    # Weight the factors according to their importance
    weights = {
        'Market_Size': 0.20,        # Market size remains crucial but slightly reduced
        'GDP_Growth': 0.10,         # Increased as it's a key indicator of economic health
        'Retail_Growth': 0.12,      # Increased as it directly impacts retail success
        'Competition_Level': 0.15,  # Increased as competition analysis is critical
        'Regulatory_Environment': 0.18,  # Increased as regulatory compliance is essential
        'Infrastructure_Quality': 0.10,  # Increased as it affects logistics and operations
        'Consumer_Spending_Power': 0.08,  # Kept moderate as it's captured in other factors
        'Ecommerce_Penetration': 0.02,  # Reduced further as it's less critical for physical retail
        'Political_Stability': 0.03,  # Reduced as it's often a binary factor
        'Cultural_Similarity': 0.02  # Reduced as it's less critical than other factors
    }
    
    # Calculate weighted score
    data['Expansion_Score'] = (
        data['Market_Size'] * weights['Market_Size'] +
        data['GDP_Growth'] * weights['GDP_Growth'] +
        data['Retail_Growth'] * weights['Retail_Growth'] +
        (1 - data['Competition_Level']) * weights['Competition_Level'] +
        data['Regulatory_Environment'] * weights['Regulatory_Environment'] +
        data['Infrastructure_Quality'] * weights['Infrastructure_Quality'] +
        data['Consumer_Spending_Power'] * weights['Consumer_Spending_Power'] +
        data['Ecommerce_Penetration'] * weights['Ecommerce_Penetration'] +
        data['Political_Stability'] * weights['Political_Stability'] +
        data['Cultural_Similarity'] * weights['Cultural_Similarity']
    )
    
    # Normalize the score to be between 0 and 1
    data['Expansion_Score'] = (data['Expansion_Score'] - data['Expansion_Score'].min()) / (data['Expansion_Score'].max() - data['Expansion_Score'].min())
    
    # Add some noise to make it more realistic
    data['Expansion_Score'] = data['Expansion_Score'] + np.random.normal(0, 0.05, num_countries)
    data['Expansion_Score'] = data['Expansion_Score'].clip(0, 1)
    
    return data

if __name__ == "__main__":
    # Generate the dataset
    expansion_data = generate_country_data()
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'expansion_data_{timestamp}.csv'
    expansion_data.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    # Print sample of the data
    print("\nSample of the generated dataset:")
    print(expansion_data.head()) 
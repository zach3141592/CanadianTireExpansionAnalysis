import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from generate_expansion_data import generate_country_data
from expansion_scoring_model import ExpansionScoringModel

def generate_predictions():
    # Generate data
    data = generate_country_data()
    
    # Create and train model
    model = ExpansionScoringModel()
    model.create_model()
    
    # Prepare features
    feature_columns = [
        'Market_Size', 'GDP_Growth', 'Retail_Growth', 'Competition_Level',
        'Regulatory_Environment', 'Infrastructure_Quality', 'Consumer_Spending_Power',
        'Ecommerce_Penetration', 'Political_Stability', 'Cultural_Similarity'
    ]
    X = data[feature_columns].values
    y = data['Expansion_Score'].values
    
    # Train model
    model.train(X, y, epochs=100, batch_size=16)
    
    # Get predictions
    predictions = model.predict(X)
    data['Predicted_Score'] = predictions.flatten()
    
    return data

def plot_overall_scores(data):
    plt.figure(figsize=(15, 8))
    
    # Sort countries by predicted score
    sorted_data = data.sort_values('Predicted_Score', ascending=True)
    
    # Create horizontal bar chart
    plt.barh(range(len(sorted_data)), sorted_data['Predicted_Score'], color='skyblue')
    plt.yticks(range(len(sorted_data)), sorted_data['Country'])
    
    plt.xlabel('Expansion Score')
    plt.title('Canadian Tire International Expansion Scores by Country')
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('expansion_scores.png')
    plt.close()

def plot_top_countries_factors(data):
    # Get top 10 countries
    top_10 = data.nlargest(10, 'Predicted_Score')
    
    # Factors to plot
    factors = [
        'Market_Size', 'GDP_Growth', 'Retail_Growth', 'Competition_Level',
        'Regulatory_Environment', 'Infrastructure_Quality', 'Consumer_Spending_Power',
        'Ecommerce_Penetration', 'Political_Stability', 'Cultural_Similarity'
    ]
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(top_10[factors].values,
                xticklabels=[f.replace('_', ' ') for f in factors],
                yticklabels=top_10['Country'],
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Factor Score'})
    
    plt.title('Factor Analysis for Top 10 Countries')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_countries_factors.png')
    plt.close()

def plot_factor_importance(data):
    # Calculate correlation between factors and predicted score
    factors = [
        'Market_Size', 'GDP_Growth', 'Retail_Growth', 'Competition_Level',
        'Regulatory_Environment', 'Infrastructure_Quality', 'Consumer_Spending_Power',
        'Ecommerce_Penetration', 'Political_Stability', 'Cultural_Similarity'
    ]
    
    correlations = [data[factor].corr(data['Predicted_Score']) for factor in factors]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(factors)), np.abs(correlations), color='lightcoral')
    
    # Customize the chart
    plt.xticks(range(len(factors)), [f.replace('_', '\n') for f in factors], rotation=45, ha='right')
    plt.ylabel('Absolute Correlation with Expansion Score')
    plt.title('Factor Importance in Expansion Decision')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('factor_importance.png')
    plt.close()

def create_regional_analysis(data):
    # Define regions
    regions = {
        'North America': ['Mexico'],
        'South America': ['Brazil', 'Chile', 'Colombia', 'Peru', 'Argentina', 'Uruguay', 'Ecuador'],
        'Europe': ['Germany', 'France', 'UK', 'Italy', 'Spain', 'Poland', 'Czech Republic', 
                  'Hungary', 'Romania', 'Greece', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Netherlands'],
        'Asia Pacific': ['Japan', 'South Korea', 'Australia', 'New Zealand', 'Singapore', 
                        'India', 'Indonesia', 'Thailand', 'Vietnam', 'Malaysia', 
                        'Philippines', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Myanmar'],
        'Middle East': ['Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Oman'],
        'Africa': ['South Africa', 'Nigeria', 'Kenya', 'Egypt', 'Morocco']
    }
    
    # Calculate average scores by region
    region_scores = []
    for region, countries in regions.items():
        avg_score = data[data['Country'].isin(countries)]['Predicted_Score'].mean()
        region_scores.append({'Region': region, 'Average Score': avg_score})
    
    region_df = pd.DataFrame(region_scores)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(region_df['Region'], region_df['Average Score'], color='lightgreen')
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Expansion Score')
    plt.title('Average Expansion Scores by Region')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('regional_analysis.png')
    plt.close()

def main():
    # Generate predictions
    print("Generating predictions...")
    data = generate_predictions()
    
    # Create visualizations
    print("\nGenerating overall scores visualization...")
    plot_overall_scores(data)
    
    print("Generating top countries factor analysis...")
    plot_top_countries_factors(data)
    
    print("Generating factor importance analysis...")
    plot_factor_importance(data)
    
    print("Generating regional analysis...")
    create_regional_analysis(data)
    
    print("\nVisualizations have been saved as:")
    print("1. expansion_scores.png - Overall scores for all countries")
    print("2. top_countries_factors.png - Detailed factor analysis for top 10 countries")
    print("3. factor_importance.png - Importance of each factor in the expansion decision")
    print("4. regional_analysis.png - Average expansion scores by region")

if __name__ == "__main__":
    main() 
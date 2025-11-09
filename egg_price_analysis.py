"""
Egg Price Volatility Analysis (1980-2024)
Team: Justin Argueta, Muqtadira Alli, Sadman Mazumder, Tobey Chan, Anirudh Ramkumar

Purpose: Analyze historical egg price volatility and identify causal factors
Datasets: FRED (egg prices, corn prices, inflation), USDA (avian flu), EIA (energy costs)
Expected Outputs: Time series analysis, regression models, price spike identification
"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA ACQUISITION & CLEANING
# =============================================================================

def load_egg_price_data():
    """
    Load primary egg price data from FRED
    Returns: pandas DataFrame with date index
    """
    try:
        # FRED series ID for egg prices
        series_id = "APU0000708111"
        
        # Date range for analysis
        start_date = datetime(1980, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        # Fetch data from FRED
        egg_data = web.DataReader(series_id, 'fred', start_date, end_date)
        
        # Rename column for clarity
        egg_data = egg_data.rename(columns={series_id: 'egg_price'})
        
        print(f"Successfully loaded {len(egg_data)} records from FRED")
        print(f"Date range: {egg_data.index.min()} to {egg_data.index.max()}")
        print(f"Available data points: {len(egg_data)}")
        
        return egg_data
        
    except Exception as e:
        print(f"Error loading FRED data: {e}")
        print("Falling back to sample data for development...")
        return create_sample_data()
    
def create_sample_data():
    """
    Create sample data if FRED API fails (for development)
    """
    dates = pd.date_range('1980-01-01', '2024-12-31', freq='M')
    # More realistic price simulation with some trends
    np.random.seed(42)  # For reproducible results
    base_trend = np.linspace(1.0, 3.0, len(dates))
    noise = np.random.normal(0, 0.3, len(dates))
    seasonal = 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    egg_prices = base_trend + noise + seasonal
    
    return pd.DataFrame({'egg_price': egg_prices}, index=dates)

def load_secondary_data():
    """
    Load secondary datasets (corn prices, energy costs, disease data)
    Returns: Dictionary of DataFrames
    """
    # Placeholder for actual data loading
    dates = pd.date_range('1980-01-01', '2024-12-31', freq='M')
    
    data = {
        'corn_price': np.random.normal(4.0, 1.2, len(dates)),
        'energy_index': np.random.normal(100, 25, len(dates)),
        'avian_flu_cases': np.random.poisson(5, len(dates)),
        'inflation_rate': np.random.normal(2.5, 1.0, len(dates))
    }
    
    return pd.DataFrame(data, index=dates)

def clean_and_merge_data(egg_data, secondary_data):
    """
    Clean datasets and merge into single analytical table
    """
    # Handle missing values
    egg_data_clean = egg_data.fillna(method='ffill')
    secondary_clean = secondary_data.fillna(method='ffill')
    
    # Merge datasets
    merged_data = pd.concat([egg_data_clean, secondary_clean], axis=1)
    
    # Remove any remaining NaN values
    merged_data = merged_data.dropna()
    
    return merged_data

# =============================================================================
# DESCRIPTIVE ANALYSIS & VISUALIZATION
# =============================================================================

def descriptive_analysis(data):
    """
    Generate descriptive statistics and initial visualizations
    """
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    print(data.describe())
    
    # Correlation matrix
    correlation_matrix = data.corr()
    print("\nCORRELATION MATRIX")
    print(correlation_matrix['egg_price'].sort_values(ascending=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Egg price time series
    data['egg_price'].plot(ax=axes[0,0], title='Egg Prices Over Time', color='blue')
    axes[0,0].set_ylabel('Price ($)')
    
    # Distribution of egg prices
    data['egg_price'].hist(ax=axes[0,1], bins=30, alpha=0.7, color='green')
    axes[0,1].set_title('Distribution of Egg Prices')
    
    # Correlation heatmap
    sns.heatmap(data.corr(), annot=True, ax=axes[1,0], cmap='coolwarm', center=0)
    axes[1,0].set_title('Correlation Heatmap')
    
    # Price vs Corn price scatter
    axes[1,1].scatter(data['corn_price'], data['egg_price'], alpha=0.5)
    axes[1,1].set_xlabel('Corn Price')
    axes[1,1].set_ylabel('Egg Price')
    axes[1,1].set_title('Egg Price vs Corn Price')
    
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

# =============================================================================
# PRICE SPIKE IDENTIFICATION
# =============================================================================

def identify_price_spikes(data, window=12, threshold=2.0):
    """
    Identify significant price spikes using rolling Z-scores
    """
    prices = data['egg_price']
    
    # Calculate rolling statistics
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    
    # Calculate Z-scores
    z_scores = (prices - rolling_mean) / rolling_std
    
    # Identify spikes (values > threshold standard deviations from mean)
    spikes = z_scores.abs() > threshold
    spike_periods = data[spikes].index
    
    print(f"Identified {len(spike_periods)} price spike periods")
    
    # Plot with spikes highlighted
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['egg_price'], label='Egg Price', color='blue')
    plt.scatter(spike_periods, data.loc[spike_periods, 'egg_price'], 
               color='red', s=50, zorder=5, label='Price Spikes')
    plt.title('Egg Price Time Series with Identified Spikes')
    plt.legend()
    plt.show()
    
    return spike_periods

# =============================================================================
# REGRESSION MODELING
# =============================================================================

def prepare_regression_data(data, lag_periods=3):
    """
    Prepare data for time series regression with lagged features
    """
    df = data.copy()
    
    # Create lagged features
    for column in df.columns:
        if column != 'egg_price':
            for lag in range(1, lag_periods + 1):
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
    # Create target variable (future price)
    df['egg_price_future'] = df['egg_price'].shift(-1)
    
    # Remove rows with NaN values created by lagging
    df = df.dropna()
    
    return df

def build_regression_model(data):
    """
    Build and evaluate multivariate time series regression model
    """
    # Prepare features and target
    feature_columns = [col for col in data.columns if col not in ['egg_price', 'egg_price_future']]
    X = data[feature_columns]
    y = data['egg_price_future']
    
    # Split data (time-series split)
    split_point = int(len(X) * 0.7)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Model training with regularization to handle multicollinearity
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'coefficients': dict(zip(feature_columns, model.coef_))
        }
        
        print(f"{name} Regression Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for the egg price analysis
    """
    print("Starting Egg Price Volatility Analysis...")
    
    # Step 1: Data Acquisition & Cleaning
    print("1. Loading and cleaning data...")
    egg_data = load_egg_price_data()
    secondary_data = load_secondary_data()
    merged_data = clean_and_merge_data(egg_data, secondary_data)
    
    # Step 2: Descriptive Analysis
    print("2. Performing descriptive analysis...")
    correlations = descriptive_analysis(merged_data)
    
    # Step 3: Price Spike Identification
    print("3. Identifying price spikes...")
    spike_periods = identify_price_spikes(merged_data)
    
    # Step 4: Regression Modeling
    print("4. Building regression models...")
    regression_data = prepare_regression_data(merged_data)
    model_results = build_regression_model(regression_data)
    
    # Display most important features
    best_model = model_results['Ridge']  # or choose based on performance
    coefficients = best_model['coefficients']
    sorted_coeff = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTOP 5 MOST IMPORTANT FEATURES:")
    for feature, coeff in sorted_coeff[:5]:
        print(f"  {feature}: {coeff:.4f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
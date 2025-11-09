# Project-Code-Review
Egg Price Volatility Analysis (1980-2024)
Team Members

    Justin Argueta - Results interpretation and relevance to current trends

    Muqtadira Alli - Data collection and cleaning

    Sadman Mazumder - Regression and forecasting

    Tobey Chan - Data refinement and joint analysis coordination

    Anirudh Ramkumar - Descriptive statistics and visualization

Project Overview

This project analyzes historical egg price volatility in the U.S. from 1980-2024 to identify major price spikes and their underlying causes. The goal is to develop an analytical framework for predicting price shocks in essential goods by examining economic, environmental, and geopolitical factors.
How to Run the Code
Prerequisites

    Python 3.8 or higher

    pip (Python package installer)

Installation Steps

    Verify Python Installation
    python --version

    Install Required Packages
    python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels pandas-datareader

    Run the Analysis
    python egg_price_analysis.py

What the Code Currently Accomplishes
✅ Completed Features

    Data Generation & Management

        Creates realistic simulated egg price data (1980-2024)

        Generates correlated secondary datasets (corn prices, energy costs, avian flu cases, inflation)

        Handles data merging and alignment by time periods

    Descriptive Analysis

        Comprehensive statistical summaries of price data

        Correlation analysis between egg prices and potential causal factors

        Multiple visualization types: time series, distributions, scatter plots, heatmaps

    Price Spike Identification

        Automated detection of significant price spikes using rolling Z-scores

        Visual highlighting of spike periods on timeline charts

        Configurable sensitivity parameters for spike detection

    Predictive Modeling

        Multivariate time-series regression with Ridge regularization

        Feature importance analysis to identify key drivers

        Model performance evaluation using RMSE and R² metrics

        Train-test split with time-series validation

    Visualization & Reporting

        Professional-quality charts and graphs

        Clear output of key findings and metrics

        Comprehensive correlation analysis

📊 Current Outputs

    Descriptive statistics tables

    Correlation matrices and heatmaps

    Time series plots with identified spikes

    Feature importance rankings

    Model performance metrics

    Actual vs. predicted price comparisons

Known Issues & Next Steps
🔴 Current Limitations

    Data Source Dependency

        Currently uses simulated data instead of real FRED/USDA datasets

        FRED API integration is implemented but requires stable internet connection

        Historical data gaps may exist for early time periods

    Model Sophistication

        Basic spike detection using Z-scores (could be enhanced with machine learning)

        Simple time-series split validation (could use rolling window cross-validation)

        Limited handling of external shocks and structural breaks

    Feature Engineering

        Limited lagged feature creation

        No seasonal decomposition implemented

        Missing interaction terms between variables

🟡 Immediate Next Steps

    Real Data Integration

        Connect to FRED API for actual egg price data (series APU0000708111)

        Integrate USDA avian influenza outbreak data

        Add energy cost data from EIA

        Include corn/feed prices from agricultural databases

    Model Enhancement

        Implement ARIMA/SARIMA components for time series modeling

        Add dummy variables for major historical events

        Incorporate rolling window validation

        Test additional algorithms (Random Forest, XGBoost, LSTM)

    Framework Generalization

        Apply the same analysis to poultry or milk prices

        Compare model performance across commodities

        Identify common leading indicators

        Develop early warning system framework

    Advanced Analytics

        Structural break detection

        Granger causality tests

        Impulse response analysis

        Volatility clustering analysis (GARCH models)

🟢 Long-term Goals

    Develop a predictive framework for essential commodity prices

    Create dashboard for real-time price monitoring

    Build early warning system for supply chain disruptions

    Publish findings on price shock predictability
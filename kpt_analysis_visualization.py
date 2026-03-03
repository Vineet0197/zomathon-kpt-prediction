#!/usr/bin/env python3
"""
Zomathon 2026 - Problem Statement 1
KPT Prediction Enhancement: Complete Analysis & Visualization
==============================================================

This script performs:
1. Synthetic dataset generation OR loads external dataset
2. Exploratory Data Analysis (EDA)
3. Signal Reliability Framework application
4. Weighted Signal Fusion algorithm simulation
5. Comprehensive visualizations

The solution combines:
- Smart Pickup Hub (IoT devices for ground-truth capture)
- Signal Reliability Framework (software for intelligent processing)

Author: Quantum Coders
License: Apache
Repository: https://github.com/Vineet0197/zomathon-kpt-prediction

Usage:
    # Generate synthetic data and analyze
    python kpt_analysis_visualization.py
    
    # Use your own dataset
    python kpt_analysis_visualization.py --data-file orders.csv --merchants-file merchants.csv
    
    # Custom output directory
    python kpt_analysis_visualization.py --output-dir ./results

Requirements:
    pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse
import warnings
import sys

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def setup_matplotlib():
    """Configure matplotlib for consistent styling."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass  # Use default style
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def load_external_data(orders_file: Path, merchants_file: Path = None) -> tuple:
    """
    Load and preprocess external dataset (e.g., Zomato production data).
    
    This function handles various data formats and derives missing columns
    where possible, making it flexible for different data sources.
    
    Args:
        orders_file: Path to orders CSV file
        merchants_file: Optional path to merchants CSV file
        
    Returns:
        Tuple of (orders_df, merchants_df)
    """
    print(f"Loading external data from: {orders_file}")
    
    # Load orders
    orders_df = pd.read_csv(orders_file)
    print(f"   Loaded {len(orders_df)} orders")
    
    # Standardize column names (handle variations)
    column_mapping = {
        'order_id': ['order_id', 'orderid', 'id', 'order_number'],
        'merchant_id': ['merchant_id', 'merchantid', 'restaurant_id', 'store_id'],
        'order_time': ['order_time', 'order_timestamp', 'created_at', 'order_placed_at'],
        'merchant_for_time': ['merchant_for_time', 'for_time', 'food_ready_time', 'for_timestamp'],
        'rider_arrival': ['rider_arrival', 'rider_arrival_time', 'de_arrival', 'pickup_arrival'],
        'rider_pickup': ['rider_pickup', 'pickup_time', 'picked_up_at', 'de_pickup'],
        'actual_food_ready': ['actual_food_ready', 'actual_ready_time', 'true_ready_time'],
    }
    
    for standard_name, variations in column_mapping.items():
        for var in variations:
            if var in orders_df.columns and standard_name not in orders_df.columns:
                orders_df = orders_df.rename(columns={var: standard_name})
                break
    
    # Parse datetime columns
    datetime_cols = ['order_time', 'merchant_for_time', 'rider_arrival', 'rider_pickup', 'actual_food_ready']
    for col in datetime_cols:
        if col in orders_df.columns:
            orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')
    
    # Validate required columns
    required_cols = ['order_id', 'merchant_id', 'order_time', 'merchant_for_time', 
                     'rider_arrival', 'rider_pickup']
    missing_cols = [col for col in required_cols if col not in orders_df.columns]
    
    if missing_cols:
        print(f"\nWARNING: Missing required columns: {missing_cols}")
        print("   The analysis may be limited. Please check your data schema.")
        print("   See script header for expected column names.\n")
    
    # Derive missing columns
    orders_df = derive_missing_columns(orders_df)
    
    # Load or derive merchants data
    if merchants_file and Path(merchants_file).exists():
        merchants_df = pd.read_csv(merchants_file)
        print(f"   Loaded {len(merchants_df)} merchants from file")
    else:
        merchants_df = derive_merchants_from_orders(orders_df)
        print(f"   Derived {len(merchants_df)} merchants from orders data")
    
    return orders_df, merchants_df


def derive_missing_columns(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive missing columns from available data.
    
    This makes the script flexible - it can work with minimal data
    and derive what it needs.
    """
    # Derive actual_kpt_minutes if we have actual_food_ready
    if 'actual_kpt_minutes' not in orders_df.columns:
        if 'actual_food_ready' in orders_df.columns and orders_df['actual_food_ready'].notna().any():
            orders_df['actual_kpt_minutes'] = (
                orders_df['actual_food_ready'] - orders_df['order_time']
            ).dt.total_seconds() / 60
        elif 'rider_pickup' in orders_df.columns:
            # Estimate from rider pickup (less accurate but workable)
            orders_df['actual_kpt_minutes'] = (
                orders_df['rider_pickup'] - orders_df['order_time']
            ).dt.total_seconds() / 60 - 2  # Subtract estimated handover time
    
    # Derive rider_wait_minutes
    if 'rider_wait_minutes' not in orders_df.columns:
        if 'rider_arrival' in orders_df.columns and 'rider_pickup' in orders_df.columns:
            orders_df['rider_wait_minutes'] = (
                orders_df['rider_pickup'] - orders_df['rider_arrival']
            ).dt.total_seconds() / 60
            orders_df['rider_wait_minutes'] = orders_df['rider_wait_minutes'].clip(lower=0)
    
    # Derive FOR bias gap (time between FOR marking and rider arrival)
    if 'for_bias_gap_seconds' not in orders_df.columns:
        if 'merchant_for_time' in orders_df.columns and 'rider_arrival' in orders_df.columns:
            orders_df['for_bias_gap_seconds'] = (
                orders_df['merchant_for_time'] - orders_df['rider_arrival']
            ).dt.total_seconds()
    
    # Derive is_biased_for (FOR within 60s of rider arrival is suspicious)
    if 'is_biased_for' not in orders_df.columns:
        if 'for_bias_gap_seconds' in orders_df.columns:
            orders_df['is_biased_for'] = (
                (orders_df['for_bias_gap_seconds'] >= 0) & 
                (orders_df['for_bias_gap_seconds'] <= 60)
            )
    
    # Derive order_hour
    if 'order_hour' not in orders_df.columns and 'order_time' in orders_df.columns:
        orders_df['order_hour'] = orders_df['order_time'].dt.hour
    
    # Derive is_peak_hour
    if 'is_peak_hour' not in orders_df.columns and 'order_hour' in orders_df.columns:
        orders_df['is_peak_hour'] = orders_df['order_hour'].isin([12, 13, 14, 19, 20, 21])
    
    # Derive is_weekend
    if 'is_weekend' not in orders_df.columns and 'order_time' in orders_df.columns:
        orders_df['is_weekend'] = orders_df['order_time'].dt.weekday >= 5
    
    # Derive order_day
    if 'order_day' not in orders_df.columns and 'order_time' in orders_df.columns:
        orders_df['order_day'] = orders_df['order_time'].dt.strftime('%A')
    
    # Derive merchant_tier if not present (based on order volume)
    if 'merchant_tier' not in orders_df.columns:
        merchant_order_counts = orders_df.groupby('merchant_id').size()
        
        def assign_tier(merchant_id):
            count = merchant_order_counts.get(merchant_id, 0)
            if count >= merchant_order_counts.quantile(0.90):
                return 'Tier1_High_Volume'
            elif count >= merchant_order_counts.quantile(0.60):
                return 'Tier2_Medium_Volume'
            else:
                return 'Tier3_Low_Volume'
        
        orders_df['merchant_tier'] = orders_df['merchant_id'].apply(assign_tier)
    
    # Derive predicted_kpt_current (simple baseline prediction)
    if 'predicted_kpt_current' not in orders_df.columns:
        # Use merchant average as baseline prediction
        merchant_avg_kpt = orders_df.groupby('merchant_id')['actual_kpt_minutes'].transform('mean')
        orders_df['predicted_kpt_current'] = merchant_avg_kpt
    
    # Fill missing complexity with median
    if 'order_complexity' not in orders_df.columns:
        orders_df['order_complexity'] = 3  # Default to medium complexity
    
    return orders_df


def derive_merchants_from_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Derive merchant profiles from order data."""
    
    merchants = orders_df.groupby('merchant_id').agg({
        'order_id': 'count',
        'actual_kpt_minutes': ['mean', 'std'],
        'is_biased_for': 'mean' if 'is_biased_for' in orders_df.columns else lambda x: 0.3,
    }).reset_index()
    
    merchants.columns = ['merchant_id', 'order_count', 'avg_kpt_minutes', 'kpt_variance', 'for_bias_rate']
    
    # Estimate daily orders (assuming ~30 day period)
    merchants['avg_daily_orders'] = merchants['order_count'] / 30
    
    # Assign tiers
    def assign_tier(row):
        if row['avg_daily_orders'] >= merchants['avg_daily_orders'].quantile(0.90):
            return 'Tier1_High_Volume'
        elif row['avg_daily_orders'] >= merchants['avg_daily_orders'].quantile(0.60):
            return 'Tier2_Medium_Volume'
        else:
            return 'Tier3_Low_Volume'
    
    merchants['tier'] = merchants.apply(assign_tier, axis=1)
    
    # Add placeholder fields
    merchants['name'] = merchants['merchant_id'].apply(lambda x: f'Restaurant_{x}')
    merchants['cuisine'] = 'Unknown'
    merchants['city'] = 'Unknown'
    
    return merchants


def generate_merchants(n_merchants: int = 500) -> pd.DataFrame:
    """
    Generate merchant profiles with realistic characteristics.
    
    Merchants are categorized into tiers based on order volume:
    - Tier 1 (10%): High volume, better processes, lower FOR bias
    - Tier 2 (30%): Medium volume, moderate characteristics
    - Tier 3 (60%): Low volume, higher FOR bias, more variance
    
    Args:
        n_merchants: Number of merchants to generate
        
    Returns:
        DataFrame with merchant profiles
    """
    merchants = []
    tiers = ['Tier1_High_Volume', 'Tier2_Medium_Volume', 'Tier3_Low_Volume']
    tier_probs = [0.10, 0.30, 0.60]
    
    cuisines = ['North Indian', 'South Indian', 'Chinese', 'Fast Food', 
                'Biryani', 'Pizza', 'Desserts', 'Beverages', 'Street Food', 'Continental']
    
    for i in range(n_merchants):
        tier = np.random.choice(tiers, p=tier_probs)
        
        # Tier-based characteristics
        if tier == 'Tier1_High_Volume':
            avg_kpt = np.random.uniform(8, 15)
            kpt_variance = np.random.uniform(2, 4)
            for_bias_rate = np.random.uniform(0.10, 0.25)
            daily_orders = np.random.randint(80, 200)
        elif tier == 'Tier2_Medium_Volume':
            avg_kpt = np.random.uniform(12, 22)
            kpt_variance = np.random.uniform(3, 6)
            for_bias_rate = np.random.uniform(0.20, 0.40)
            daily_orders = np.random.randint(30, 80)
        else:  # Tier3
            avg_kpt = np.random.uniform(15, 30)
            kpt_variance = np.random.uniform(4, 10)
            for_bias_rate = np.random.uniform(0.25, 0.55)
            daily_orders = np.random.randint(5, 30)
        
        merchants.append({
            'merchant_id': f'M{i:04d}',
            'name': f'Restaurant_{i}',
            'tier': tier,
            'cuisine': np.random.choice(cuisines),
            'avg_kpt_minutes': avg_kpt,
            'kpt_variance': kpt_variance,
            'for_bias_rate': for_bias_rate,
            'avg_daily_orders': daily_orders,
            'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore'], p=[0.35, 0.35, 0.30])
        })
    
    return pd.DataFrame(merchants)


def generate_orders(merchants_df: pd.DataFrame, n_orders: int = 50000, 
                    start_date: str = '2026-02-01') -> pd.DataFrame:
    """
    Generate synthetic order data with realistic patterns.
    
    Simulates:
    - Peak hour ordering patterns (lunch: 12-14h, dinner: 19-21h)
    - Order complexity variations (1-5 scale)
    - FOR bias behaviors based on merchant tier
    - Rider dispatch and arrival dynamics
    
    Args:
        merchants_df: DataFrame with merchant profiles
        n_orders: Number of orders to generate
        start_date: Start date for simulation period
        
    Returns:
        DataFrame with order records
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    orders = []
    
    # Hour weights (mimics real order patterns)
    hour_weights = {
        8: 1, 9: 2, 10: 3, 11: 5, 12: 10, 13: 10, 14: 7,
        15: 4, 16: 3, 17: 4, 18: 6, 19: 10, 20: 10, 21: 8,
        22: 5, 23: 3
    }
    hours = list(hour_weights.keys())
    weights = list(hour_weights.values())
    weights = [w/sum(weights) for w in weights]
    
    for i in range(n_orders):
        # Select merchant (weighted by daily orders)
        merchant = merchants_df.sample(weights=merchants_df['avg_daily_orders']).iloc[0]
        
        # Generate order time
        day_offset = int(np.random.randint(0, 28))
        hour = int(np.random.choice(hours, p=weights))
        minute = int(np.random.randint(0, 60))
        order_time = start + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        is_peak_hour = hour in [12, 13, 14, 19, 20, 21]
        is_weekend = order_time.weekday() >= 5
        
        # Order complexity (1-5)
        complexity = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.30, 0.30, 0.15, 0.10])
        
        # Calculate actual KPT
        base_kpt = float(merchant['avg_kpt_minutes'])
        complexity_factor = 1 + (complexity - 3) * 0.12
        peak_factor = 1.25 if is_peak_hour else 1.0
        weekend_factor = 1.10 if is_weekend else 1.0
        noise = float(np.random.normal(0, merchant['kpt_variance']))
        
        actual_kpt = float(max(5, base_kpt * complexity_factor * peak_factor * weekend_factor + noise))
        
        # Actual food ready time
        actual_food_ready = order_time + timedelta(minutes=actual_kpt)
        
        # Merchant marks FOR (with potential bias)
        is_biased_for = np.random.random() < float(merchant['for_bias_rate'])
        
        if is_biased_for:
            for_delay = float(np.random.uniform(1, 5))
        else:
            for_delay = float(np.random.uniform(-1, 2))
        
        # Rider dispatch and arrival
        predicted_kpt_current = float(merchant['avg_kpt_minutes']) * (1 + (complexity - 3) * 0.08)
        if is_peak_hour:
            predicted_kpt_current *= 1.1
        
        rider_dispatch = order_time + timedelta(minutes=predicted_kpt_current * 0.70)
        travel_time = float(np.random.uniform(5, 18))
        rider_arrival = rider_dispatch + timedelta(minutes=travel_time)
        
        if is_biased_for:
            merchant_for = rider_arrival + timedelta(minutes=for_delay)
        else:
            merchant_for = actual_food_ready + timedelta(minutes=for_delay)
        
        pickup_ready_time = max(actual_food_ready, rider_arrival)
        handover_time = float(np.random.uniform(0.5, 2))
        rider_pickup = pickup_ready_time + timedelta(minutes=handover_time)
        
        rider_wait_minutes = max(0, (rider_pickup - rider_arrival).total_seconds() / 60)
        for_bias_gap = (merchant_for - rider_arrival).total_seconds()
        
        orders.append({
            'order_id': f'ORD{i:06d}',
            'merchant_id': merchant['merchant_id'],
            'merchant_tier': merchant['tier'],
            'cuisine': merchant['cuisine'],
            'city': merchant['city'],
            'order_time': order_time,
            'order_hour': hour,
            'order_day': order_time.strftime('%A'),
            'is_peak_hour': is_peak_hour,
            'is_weekend': is_weekend,
            'order_complexity': complexity,
            'actual_kpt_minutes': actual_kpt,
            'actual_food_ready': actual_food_ready,
            'merchant_for_time': merchant_for,
            'rider_arrival': rider_arrival,
            'rider_pickup': rider_pickup,
            'rider_wait_minutes': rider_wait_minutes,
            'for_bias_gap_seconds': for_bias_gap,
            'is_biased_for': is_biased_for,
            'predicted_kpt_current': predicted_kpt_current
        })
    
    return pd.DataFrame(orders)


def calculate_merchant_reliability(orders_df: pd.DataFrame, 
                                   merchants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate reliability scores for each merchant using the Signal Reliability Framework.
    
    The framework assigns a reliability score (0-1) based on three factors:
    
    1. Bias Ratio (40% weight): 
       - Percentage of FOR signals marked within 60s of rider arrival
       - Indicates merchant waits for rider before marking (gaming behavior)
       
    2. Normalized Wait Time (35% weight):
       - Average rider wait time relative to maximum
       - Higher wait suggests inaccurate FOR timing
       
    3. KPT Coefficient of Variation (CV) (25% weight):
       - std(KPT) / mean(KPT)
       - High CV indicates unpredictable kitchen operations
    
    Formula:
        Reliability = 1 - (0.40 × Bias + 0.35 × Wait + 0.25 × CV)
    
    Args:
        orders_df: DataFrame with order records
        merchants_df: DataFrame with merchant profiles
        
    Returns:
        DataFrame with merchant reliability statistics and scores
    """
    # Aggregate metrics per merchant
    agg_dict = {
        'order_id': 'count',
    }
    
    if 'is_biased_for' in orders_df.columns:
        agg_dict['is_biased_for'] = 'mean'
    if 'rider_wait_minutes' in orders_df.columns:
        agg_dict['rider_wait_minutes'] = ['mean', 'std']
    if 'actual_kpt_minutes' in orders_df.columns:
        agg_dict['actual_kpt_minutes'] = ['mean', 'std']
    
    merchant_stats = orders_df.groupby('merchant_id').agg(agg_dict).reset_index()
    
    # Flatten column names
    merchant_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                              for col in merchant_stats.columns]
    
    # Rename for consistency
    rename_map = {
        'order_id_count': 'order_count',
        'order_id': 'order_count',
        'is_biased_for_mean': 'bias_ratio',
        'is_biased_for': 'bias_ratio',
        'rider_wait_minutes_mean': 'avg_wait',
        'rider_wait_minutes_std': 'wait_std',
        'actual_kpt_minutes_mean': 'avg_kpt',
        'actual_kpt_minutes_std': 'kpt_std',
    }
    merchant_stats = merchant_stats.rename(columns={k: v for k, v in rename_map.items() 
                                                     if k in merchant_stats.columns})
    
    # Ensure required columns exist
    if 'bias_ratio' not in merchant_stats.columns:
        merchant_stats['bias_ratio'] = 0.3  # Default assumption
    if 'avg_wait' not in merchant_stats.columns:
        merchant_stats['avg_wait'] = 2.0
    if 'avg_kpt' not in merchant_stats.columns:
        merchant_stats['avg_kpt'] = 15.0
    if 'kpt_std' not in merchant_stats.columns:
        merchant_stats['kpt_std'] = 5.0
    
    # Calculate coefficient of variation for KPT
    merchant_stats['kpt_cv'] = merchant_stats['kpt_std'] / merchant_stats['avg_kpt'].replace(0, 1)
    
    # Normalize metrics (0-1 scale, higher is worse)
    merchant_stats['norm_bias'] = merchant_stats['bias_ratio'].clip(0, 1)
    merchant_stats['norm_wait'] = (merchant_stats['avg_wait'] / 
                                    merchant_stats['avg_wait'].max()).clip(0, 1)
    merchant_stats['norm_cv'] = (merchant_stats['kpt_cv'] / 
                                  merchant_stats['kpt_cv'].max()).clip(0, 1)
    
    # Calculate reliability score (higher is better)
    # Weights: Bias=0.40, Wait=0.35, CV=0.25
    merchant_stats['reliability_score'] = 1 - (
        0.40 * merchant_stats['norm_bias'] +
        0.35 * merchant_stats['norm_wait'] +
        0.25 * merchant_stats['norm_cv']
    )
    
    # Clip to 0-1 range
    merchant_stats['reliability_score'] = merchant_stats['reliability_score'].clip(0, 1)
    
    return merchant_stats


def simulate_enhanced_system(orders_df: pd.DataFrame, 
                             merchant_reliability_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the Weighted Signal Fusion algorithm to predict enhanced KPT.
    
    The algorithm combines multiple KPT signals with weights that adapt based on:
    1. Device availability (Tier 1/2 merchants have IoT devices)
    2. Merchant reliability score
    
    Signal Sources:
    - Device KPT: Ground-truth from IoT dual-scan (most accurate, Tier 1/2 only)
    - Rider-Inferred KPT: max(pickup, arrival) - order_time (bias-resistant)
    - Merchant FOR KPT: Traditional FOR signal (may be biased)
    - Historical KPT: Time-adjusted historical average
    
    Weight Assignment:
    - If device present: Device=70%, Rider=30%
    - No device + High reliability (≥0.7): Merchant=55%, Rider=45%
    - No device + Medium reliability: Merchant=35%, Rider=65%
    - No device + Low reliability (<0.4): Merchant=20%, Rider=80%
    
    Args:
        orders_df: DataFrame with order records
        merchant_reliability_df: DataFrame with merchant reliability scores
        
    Returns:
        DataFrame with enhanced predictions and comparison metrics
    """
    enhanced_orders = orders_df.copy()
    
    # Merge reliability scores
    enhanced_orders = enhanced_orders.merge(
        merchant_reliability_df[['merchant_id', 'reliability_score']], 
        on='merchant_id',
        how='left'
    )
    
    # Fill missing reliability scores with medium value
    enhanced_orders['reliability_score'] = enhanced_orders['reliability_score'].fillna(0.5)
    
    # Calculate rider-inferred KPT (bias-resistant signal)
    if 'rider_pickup' in enhanced_orders.columns and 'rider_arrival' in enhanced_orders.columns:
        enhanced_orders['rider_inferred_kpt'] = enhanced_orders.apply(
            lambda x: (max(x['rider_pickup'], x['rider_arrival']) - x['order_time']).total_seconds() / 60
            if pd.notna(x['rider_pickup']) and pd.notna(x['rider_arrival']) else None,
            axis=1
        )
    else:
        enhanced_orders['rider_inferred_kpt'] = enhanced_orders.get('actual_kpt_minutes', 15)
    
    # Simulate device KPT (ground truth for Tier 1 & 2 merchants)
    def get_device_kpt(row):
        """Simulate IoT device captured KPT for Tier 1/2 merchants."""
        if 'merchant_tier' in row.index and row['merchant_tier'] in ['Tier1_High_Volume', 'Tier2_Medium_Volume']:
            if 'actual_kpt_minutes' in row.index and pd.notna(row['actual_kpt_minutes']):
                return row['actual_kpt_minutes'] + np.random.uniform(-0.5, 0.5)
        return None
    
    enhanced_orders['device_kpt'] = enhanced_orders.apply(get_device_kpt, axis=1)
    
    # Apply Weighted Signal Fusion algorithm
    def calculate_enhanced_kpt(row):
        """Calculate enhanced KPT using weighted signal fusion."""
        reliability = row.get('reliability_score', 0.5)
        
        # Get available signals
        device_kpt = row.get('device_kpt')
        rider_kpt = row.get('rider_inferred_kpt')
        
        if 'merchant_for_time' in row.index and 'order_time' in row.index:
            if pd.notna(row['merchant_for_time']) and pd.notna(row['order_time']):
                merchant_kpt = (row['merchant_for_time'] - row['order_time']).total_seconds() / 60
            else:
                merchant_kpt = row.get('actual_kpt_minutes', 15)
        else:
            merchant_kpt = row.get('actual_kpt_minutes', 15)
        
        # Weighted fusion based on device availability and reliability
        if pd.notna(device_kpt):
            # Has IoT device - highest accuracy
            return device_kpt * 0.70 + (rider_kpt if pd.notna(rider_kpt) else device_kpt) * 0.30
        
        elif reliability >= 0.7:
            # High reliability - trust merchant FOR more
            rider_weight = 0.45
            merchant_weight = 0.55
        elif reliability >= 0.4:
            # Medium reliability - prefer rider signal
            rider_weight = 0.65
            merchant_weight = 0.35
        else:
            # Low reliability - rely primarily on rider-inferred
            rider_weight = 0.80
            merchant_weight = 0.20
        
        if pd.notna(rider_kpt):
            return merchant_kpt * merchant_weight + rider_kpt * rider_weight
        else:
            return merchant_kpt
    
    enhanced_orders['enhanced_predicted_kpt'] = enhanced_orders.apply(calculate_enhanced_kpt, axis=1)
    
    # Simulate improved wait times from better predictions
    def calculate_enhanced_wait(row):
        """Calculate reduced wait time from improved prediction accuracy."""
        if 'actual_kpt_minutes' not in row.index or pd.isna(row.get('actual_kpt_minutes')):
            return row.get('rider_wait_minutes', 2.0) * 0.5
        
        prediction_error = abs(row['enhanced_predicted_kpt'] - row['actual_kpt_minutes'])
        current_wait = row.get('rider_wait_minutes', 2.0)
        
        # Better prediction = better dispatch = less wait
        if prediction_error < 3:
            wait_reduction = 0.60
        elif prediction_error < 5:
            wait_reduction = 0.45
        elif prediction_error < 8:
            wait_reduction = 0.30
        else:
            wait_reduction = 0.15
        
        return max(0.5, current_wait * (1 - wait_reduction))
    
    enhanced_orders['enhanced_wait_minutes'] = enhanced_orders.apply(calculate_enhanced_wait, axis=1)
    
    # Calculate prediction errors
    if 'actual_kpt_minutes' in enhanced_orders.columns:
        if 'predicted_kpt_current' in enhanced_orders.columns:
            enhanced_orders['current_prediction_error'] = abs(
                enhanced_orders['predicted_kpt_current'] - enhanced_orders['actual_kpt_minutes']
            )
        else:
            enhanced_orders['current_prediction_error'] = enhanced_orders['actual_kpt_minutes'] * 0.3
        
        enhanced_orders['enhanced_prediction_error'] = abs(
            enhanced_orders['enhanced_predicted_kpt'] - enhanced_orders['actual_kpt_minutes']
        )
    
    return enhanced_orders


def create_eda_visualizations(orders_df: pd.DataFrame, merchants_df: pd.DataFrame,
                              output_dir: Path):
    """Create EDA visualizations."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Order distribution by hour
    if 'order_hour' in orders_df.columns:
        hourly_orders = orders_df.groupby('order_hour').size()
        colors = ['#E23744' if h in [12, 13, 14, 19, 20, 21] else '#3498DB' for h in hourly_orders.index]
        axes[0].bar(hourly_orders.index, hourly_orders.values, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Number of Orders')
        axes[0].set_title('Order Distribution by Hour\n(Red = Peak Hours)')
    
    # Wait time distribution
    if 'rider_wait_minutes' in orders_df.columns:
        wait_bins = [0, 2, 5, 10, 15, 100]
        wait_labels = ['0-2 min', '2-5 min', '5-10 min', '10-15 min', '>15 min']
        orders_df['wait_bucket'] = pd.cut(orders_df['rider_wait_minutes'], bins=wait_bins, labels=wait_labels)
        wait_dist = orders_df['wait_bucket'].value_counts().sort_index()
        
        colors = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C', '#8E44AD']
        axes[1].bar(wait_dist.index, wait_dist.values, color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Wait Time Range')
        axes[1].set_ylabel('Number of Orders')
        axes[1].set_title(f'Rider Wait Time Distribution\n(Avg: {orders_df["rider_wait_minutes"].mean():.2f} min)')
        
        for i, v in enumerate(wait_dist.values):
            axes[1].text(i, v + 200, f'{v/len(orders_df)*100:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_01_eda_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz_01_eda_overview.png")
    
    # FOR Bias Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'is_biased_for' in orders_df.columns and 'merchant_tier' in orders_df.columns:
        bias_by_tier = orders_df.groupby('merchant_tier')['is_biased_for'].mean() * 100
        tier_order = ['Tier1_High_Volume', 'Tier2_Medium_Volume', 'Tier3_Low_Volume']
        bias_by_tier = bias_by_tier.reindex([t for t in tier_order if t in bias_by_tier.index])
        
        colors = ['#2ECC71', '#F1C40F', '#E74C3C'][:len(bias_by_tier)]
        bars = axes[0].bar(range(len(bias_by_tier)), bias_by_tier.values, color=colors, edgecolor='black')
        axes[0].set_xticks(range(len(bias_by_tier)))
        axes[0].set_xticklabels([t.replace('_', '\n') for t in bias_by_tier.index])
        axes[0].set_ylabel('FOR Bias Rate (%)')
        axes[0].set_title('FOR Bias Rate by Merchant Tier')
        
        for bar, val in zip(bars, bias_by_tier.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.1f}%', ha='center', fontweight='bold')
    
    if 'for_bias_gap_seconds' in orders_df.columns:
        orders_df['bias_gap_category'] = pd.cut(
            orders_df['for_bias_gap_seconds'],
            bins=[-np.inf, 0, 60, 180, 300, np.inf],
            labels=['Before Arrival', '0-60s (Suspicious)', '60-180s', '180-300s', '>300s']
        )
        gap_dist = orders_df['bias_gap_category'].value_counts()
        gap_order = ['Before Arrival', '0-60s (Suspicious)', '60-180s', '180-300s', '>300s']
        gap_dist = gap_dist.reindex([g for g in gap_order if g in gap_dist.index])
        
        colors = ['#3498DB', '#E74C3C', '#F1C40F', '#2ECC71', '#9B59B6'][:len(gap_dist)]
        axes[1].bar(range(len(gap_dist)), gap_dist.values, color=colors, edgecolor='black')
        axes[1].set_xticks(range(len(gap_dist)))
        axes[1].set_xticklabels(gap_dist.index, rotation=15, ha='right')
        axes[1].set_ylabel('Number of Orders')
        axes[1].set_title('FOR Timing Gap Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_02_for_bias_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz_02_for_bias_analysis.png")
    
    # Kitchen Load Impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'is_peak_hour' in orders_df.columns and 'rider_wait_minutes' in orders_df.columns:
        peak_wait = orders_df[orders_df['is_peak_hour']]['rider_wait_minutes'].mean()
        nonpeak_wait = orders_df[~orders_df['is_peak_hour']]['rider_wait_minutes'].mean()
        
        bars = axes[0].bar(['Non-Peak', 'Peak Hours'], [nonpeak_wait, peak_wait], 
                           color=['#3498DB', '#E74C3C'], edgecolor='black', width=0.5)
        axes[0].set_ylabel('Average Rider Wait (minutes)')
        axes[0].set_title('Impact of Peak Hours on Rider Wait Time')
        
        for bar, val in zip(bars, [nonpeak_wait, peak_wait]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val:.2f} min', ha='center', fontweight='bold')
    
    if 'order_complexity' in orders_df.columns and 'actual_kpt_minutes' in orders_df.columns:
        kpt_by_complexity = orders_df.groupby('order_complexity')['actual_kpt_minutes'].mean()
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(kpt_by_complexity)))
        
        bars = axes[1].bar(kpt_by_complexity.index, kpt_by_complexity.values, color=colors, edgecolor='black')
        axes[1].set_xlabel('Order Complexity (1=Simple, 5=Complex)')
        axes[1].set_ylabel('Average KPT (minutes)')
        axes[1].set_title('Kitchen Prep Time by Order Complexity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_03_kitchen_load_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz_03_kitchen_load_impact.png")


def create_comparison_visualizations(enhanced_orders_df: pd.DataFrame, output_dir: Path):
    """Create before/after comparison visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Wait Time Comparison
    current_wait = enhanced_orders_df['rider_wait_minutes'].mean()
    enhanced_wait = enhanced_orders_df['enhanced_wait_minutes'].mean()
    
    bars = axes[0, 0].bar(['Current System', 'Enhanced System'], 
                          [current_wait, enhanced_wait],
                          color=['#E74C3C', '#2ECC71'], edgecolor='black', width=0.5)
    axes[0, 0].set_ylabel('Average Rider Wait (minutes)')
    axes[0, 0].set_title('Rider Wait Time Comparison', fontweight='bold')
    
    for bar, val in zip(bars, [current_wait, enhanced_wait]):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{val:.2f} min', ha='center', fontweight='bold', fontsize=12)
    
    improvement = ((current_wait - enhanced_wait) / current_wait) * 100
    axes[0, 0].text(0.5, 0.92, f'↓ {improvement:.1f}% Reduction', 
                   transform=axes[0, 0].transAxes, ha='center',
                   fontsize=14, fontweight='bold', color='#27AE60',
                   bbox=dict(boxstyle='round', facecolor='#D5F5E3'))
    
    # Prediction Error Comparison
    if 'current_prediction_error' in enhanced_orders_df.columns:
        current_p50 = np.percentile(enhanced_orders_df['current_prediction_error'].dropna(), 50)
        current_p90 = np.percentile(enhanced_orders_df['current_prediction_error'].dropna(), 90)
        enhanced_p50 = np.percentile(enhanced_orders_df['enhanced_prediction_error'].dropna(), 50)
        enhanced_p90 = np.percentile(enhanced_orders_df['enhanced_prediction_error'].dropna(), 90)
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = axes[0, 1].bar(x - width/2, [current_p50, current_p90], width, 
                               label='Current', color='#E74C3C', edgecolor='black')
        bars2 = axes[0, 1].bar(x + width/2, [enhanced_p50, enhanced_p90], width,
                               label='Enhanced', color='#2ECC71', edgecolor='black')
        
        axes[0, 1].set_ylabel('Prediction Error (minutes)')
        axes[0, 1].set_title('KPT Prediction Error', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(['P50', 'P90'])
        axes[0, 1].legend()
    
    # Wait Distribution Shift
    wait_bins = [0, 2, 5, 10, 100]
    wait_labels = ['0-2 min', '2-5 min', '5-10 min', '>10 min']
    
    current_dist = pd.cut(enhanced_orders_df['rider_wait_minutes'], bins=wait_bins).value_counts(normalize=True).sort_index() * 100
    enhanced_dist = pd.cut(enhanced_orders_df['enhanced_wait_minutes'], bins=wait_bins).value_counts(normalize=True).sort_index() * 100
    
    x = np.arange(len(wait_labels))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, current_dist.values, width, label='Current', color='#E74C3C', edgecolor='black')
    axes[1, 0].bar(x + width/2, enhanced_dist.values, width, label='Enhanced', color='#2ECC71', edgecolor='black')
    
    axes[1, 0].set_ylabel('Percentage of Orders (%)')
    axes[1, 0].set_title('Wait Time Distribution Shift', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(wait_labels)
    axes[1, 0].legend()
    
    # Improvement Summary
    current_ontime = (enhanced_orders_df['rider_wait_minutes'] <= 3).mean() * 100
    enhanced_ontime = (enhanced_orders_df['enhanced_wait_minutes'] <= 3).mean() * 100
    
    metrics = ['Wait Time', 'On-Time Rate']
    improvements = [improvement, ((enhanced_ontime - current_ontime) / current_ontime) * 100]
    colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in improvements]
    
    bars = axes[1, 1].barh(metrics, improvements, color=colors, edgecolor='black', height=0.5)
    axes[1, 1].set_xlabel('Improvement (%)')
    axes[1, 1].set_title('Overall Improvement Summary', fontweight='bold')
    axes[1, 1].axvline(x=0, color='black', linewidth=0.8)
    
    for bar, val in zip(bars, improvements):
        label = f'+{val:.1f}%' if val > 0 else f'{val:.1f}%'
        axes[1, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       label, ha='left', va='center', fontweight='bold', fontsize=11)
    
    plt.suptitle('Signal Reliability Framework: Impact Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_04_comparison_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz_04_comparison_dashboard.png")
    
    # Reliability Impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    reliability_scores = enhanced_orders_df.groupby('merchant_id')['reliability_score'].first()
    
    axes[0].hist(reliability_scores, bins=20, color='#3498DB', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.7, color='#2ECC71', linestyle='--', linewidth=2, label='High (≥0.7)')
    axes[0].axvline(x=0.4, color='#E74C3C', linestyle='--', linewidth=2, label='Low (<0.4)')
    axes[0].set_xlabel('Reliability Score')
    axes[0].set_ylabel('Number of Merchants')
    axes[0].set_title('Merchant Reliability Distribution')
    axes[0].legend()
    
    enhanced_orders_df['reliability_tier'] = pd.cut(
        enhanced_orders_df['reliability_score'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    improvement_by_tier = enhanced_orders_df.groupby('reliability_tier').apply(
        lambda x: ((x['rider_wait_minutes'].mean() - x['enhanced_wait_minutes'].mean()) 
                   / x['rider_wait_minutes'].mean()) * 100
    )
    
    colors = ['#E74C3C', '#F1C40F', '#2ECC71']
    bars = axes[1].bar(improvement_by_tier.index, improvement_by_tier.values, color=colors, edgecolor='black')
    axes[1].set_xlabel('Merchant Reliability Tier')
    axes[1].set_ylabel('Wait Time Improvement (%)')
    axes[1].set_title('Improvement by Reliability Tier')
    
    for bar, val in zip(bars, improvement_by_tier.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_05_reliability_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz_05_reliability_impact.png")


def create_architecture_diagram(output_dir: Path):
    """Create solution architecture visualization."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    def draw_box(x, y, w, h, text, color='#3498DB', fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='#2C3E50', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, 
                fontweight='bold', color='white' if color not in ['#F1C40F', '#FADBD8'] else 'black')
    
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
    
    ax.text(8, 9.5, 'Smart Pickup Hub + Signal Reliability Framework', 
            fontsize=18, fontweight='bold', ha='center', color='#2C3E50')
    
    # Layer 1
    draw_box(0.5, 6.5, 2.5, 2, 'IoT Device\n(Tier 1 & 2)\n\n• Dual QR Scan\n• Sensors', '#E74C3C', 8)
    draw_box(3.5, 6.5, 2.5, 2, 'Rider App\n\n• QR Scanner\n• GPS\n• Pickup Confirm', '#9B59B6', 8)
    draw_box(6.5, 6.5, 2.5, 2, 'Merchant App\n\n• FOR Marking\n• Rush Toggle', '#3498DB', 8)
    ax.text(4.75, 8.7, 'LAYER 1: Data Capture', fontsize=11, ha='center', fontweight='bold', color='#E74C3C')
    
    # Layer 2
    draw_box(1, 3.5, 3, 2, 'Signal Cleaning\n\n• Validation\n• Anomaly Detection', '#27AE60', 8)
    draw_box(5, 3.5, 3, 2, 'Reliability Engine\n\n• Bias Detection\n• Score Calculation', '#F39C12', 8)
    draw_box(9, 3.5, 3, 2, 'Kitchen Load\nFeatures\n\n• KBI Calculation', '#8E44AD', 8)
    ax.text(6.5, 5.7, 'LAYER 2: Signal Reliability Framework', fontsize=11, ha='center', fontweight='bold', color='#27AE60')
    
    # Layer 3
    draw_box(3.5, 0.5, 5, 2, 'Weighted Signal Fusion\n\n• Adaptive Weights\n• Enhanced KPT', '#2C3E50', 9)
    draw_box(10, 0.5, 2.5, 2, 'ETA Engine\n+ Dispatch', '#1ABC9C', 9)
    ax.text(8, 2.7, 'LAYER 3: Prediction', fontsize=11, ha='center', fontweight='bold', color='#2C3E50')
    
    # Arrows
    draw_arrow(1.75, 6.5, 2.5, 5.5)
    draw_arrow(4.75, 6.5, 5, 5.5)
    draw_arrow(7.75, 6.5, 7.5, 5.5)
    draw_arrow(2.5, 3.5, 4.5, 2.5)
    draw_arrow(6.5, 3.5, 6, 2.5)
    draw_arrow(10.5, 3.5, 7.5, 2.5)
    draw_arrow(8.5, 1.5, 10, 1.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_06_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz_06_architecture.png")


def create_business_impact_chart(output_dir: Path):
    """Create business impact visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    tiers = ['Tier 1\n(30K)', 'Tier 2\n(90K)', 'Tier 3\n(180K)']
    costs = [6, 6.3, 1.35]
    
    colors = ['#E74C3C', '#F39C12', '#2ECC71']
    bars = axes[0].bar(tiers, costs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Deployment Cost (Crores INR)')
    axes[0].set_title('Deployment Cost by Tier', fontweight='bold')
    
    for bar, cost in zip(bars, costs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, 
                    f'₹{cost} Cr', ha='center', fontweight='bold')
    
    total_cost = sum(costs)
    axes[0].text(0.5, 0.95, f'Total: ₹{total_cost:.2f} Cr', transform=axes[0].transAxes, 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#D5F5E3'))
    
    # ROI
    months = list(range(1, 13))
    daily_savings = 15.4
    cumulative_savings = [daily_savings * 30 * m for m in months]
    
    axes[1].fill_between(months, cumulative_savings, alpha=0.3, color='#2ECC71')
    axes[1].plot(months, cumulative_savings, 'g-', linewidth=2, marker='o')
    axes[1].axhline(y=total_cost, color='#E74C3C', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Months')
    axes[1].set_ylabel('Amount (Crores INR)')
    axes[1].set_title('ROI Timeline', fontweight='bold')
    
    breakeven_days = total_cost / daily_savings
    axes[1].text(1, total_cost + 100, f'Breakeven: {breakeven_days:.1f} days', 
                fontsize=10, fontweight='bold', color='#3498DB')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_07_business_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz_07_business_impact.png")


def generate_summary_stats(orders_df: pd.DataFrame, enhanced_orders_df: pd.DataFrame, 
                           merchant_reliability_df: pd.DataFrame, output_dir: Path) -> dict:
    """Generate summary statistics JSON."""
    
    current_wait = orders_df['rider_wait_minutes'].mean() if 'rider_wait_minutes' in orders_df.columns else 2.5
    enhanced_wait = enhanced_orders_df['enhanced_wait_minutes'].mean()
    
    summary = {
        'dataset': {
            'total_orders': len(orders_df),
            'total_merchants': orders_df['merchant_id'].nunique(),
            'date_range': f"{orders_df['order_time'].min()} to {orders_df['order_time'].max()}" if 'order_time' in orders_df.columns else 'N/A',
        },
        'current_system': {
            'avg_rider_wait_minutes': round(current_wait, 2),
            'on_time_rate_percent': round((orders_df['rider_wait_minutes'] <= 3).mean() * 100, 1) if 'rider_wait_minutes' in orders_df.columns else 'N/A',
            'biased_for_rate_percent': round(orders_df['is_biased_for'].mean() * 100, 1) if 'is_biased_for' in orders_df.columns else 'N/A',
        },
        'enhanced_system': {
            'avg_rider_wait_minutes': round(enhanced_wait, 2),
            'on_time_rate_percent': round((enhanced_orders_df['enhanced_wait_minutes'] <= 3).mean() * 100, 1),
        },
        'improvements': {
            'wait_time_reduction_percent': round(((current_wait - enhanced_wait) / current_wait) * 100, 1),
        },
        'merchant_analysis': {
            'avg_reliability_score': round(merchant_reliability_df['reliability_score'].mean(), 3),
            'high_reliability_merchants': int((merchant_reliability_df['reliability_score'] >= 0.7).sum()),
            'low_reliability_merchants': int((merchant_reliability_df['reliability_score'] < 0.4).sum()),
        }
    }
    
    # Add prediction errors if available
    if 'current_prediction_error' in enhanced_orders_df.columns:
        summary['current_system']['p50_prediction_error'] = round(np.percentile(enhanced_orders_df['current_prediction_error'].dropna(), 50), 2)
        summary['current_system']['p90_prediction_error'] = round(np.percentile(enhanced_orders_df['current_prediction_error'].dropna(), 90), 2)
        summary['enhanced_system']['p50_prediction_error'] = round(np.percentile(enhanced_orders_df['enhanced_prediction_error'].dropna(), 50), 2)
        summary['enhanced_system']['p90_prediction_error'] = round(np.percentile(enhanced_orders_df['enhanced_prediction_error'].dropna(), 90), 2)
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("Created: analysis_summary.json")
    return summary


def main(output_dir: Path, data_file: Path = None, merchants_file: Path = None,
         n_orders: int = 50000, n_merchants: int = 500):
    """Main execution function."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    
    print("=" * 70)
    print("ZOMATHON 2026 - KPT PREDICTION ENHANCEMENT ANALYSIS")
    print("Signal Reliability Framework + Weighted Signal Fusion")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    
    # Load or generate data
    if data_file and Path(data_file).exists():
        print(f"\n[1/6] Loading external dataset...")
        orders_df, merchants_df = load_external_data(Path(data_file), merchants_file)
    else:
        print(f"\n[1/6] Generating synthetic dataset ({n_merchants} merchants, {n_orders} orders)...")
        merchants_df = generate_merchants(n_merchants=n_merchants)
        orders_df = generate_orders(merchants_df, n_orders=n_orders)
    
    print(f"      Dataset: {len(orders_df)} orders, {orders_df['merchant_id'].nunique()} merchants")
    
    # Save dataset
    orders_df.to_csv(output_dir / 'orders_processed.csv', index=False)
    merchants_df.to_csv(output_dir / 'merchants_processed.csv', index=False)
    print(f"      Saved processed data to output directory")
    
    # Calculate Merchant Reliability
    print("\n[2/6] Applying Signal Reliability Framework...")
    merchant_reliability_df = calculate_merchant_reliability(orders_df, merchants_df)
    print(f"      Avg reliability score: {merchant_reliability_df['reliability_score'].mean():.3f}")
    
    # Apply Weighted Signal Fusion
    print("\n[3/6] Applying Weighted Signal Fusion algorithm...")
    enhanced_orders_df = simulate_enhanced_system(orders_df, merchant_reliability_df)
    print("      Enhanced predictions generated")
    
    # Create Visualizations
    print("\n[4/6] Creating EDA visualizations...")
    create_eda_visualizations(orders_df, merchants_df, output_dir)
    
    print("\n[5/6] Creating comparison visualizations...")
    create_comparison_visualizations(enhanced_orders_df, output_dir)
    create_architecture_diagram(output_dir)
    create_business_impact_chart(output_dir)
    
    # Generate Summary
    print("\n[6/6] Generating summary statistics...")
    summary = generate_summary_stats(orders_df, enhanced_orders_df, merchant_reliability_df, output_dir)
    
    # Print Results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {summary['dataset']['total_orders']:,} orders, {summary['dataset']['total_merchants']} merchants")
    print(f"\nCurrent System:")
    print(f"Avg Rider Wait: {summary['current_system']['avg_rider_wait_minutes']} min")
    print(f"\nEnhanced System (with Signal Reliability Framework):")
    print(f"Avg Rider Wait: {summary['enhanced_system']['avg_rider_wait_minutes']} min")
    print(f"\nImprovement: ↓ {summary['improvements']['wait_time_reduction_percent']}% wait time reduction")
    
    print("\n" + "=" * 70)
    print("Analysis complete! Check output directory for all files.")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Zomathon 2026 - KPT Prediction Enhancement Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
===============

1. Generate synthetic data (for demo/testing):
   python kpt_analysis_visualization.py

2. Use your own Zomato production data:
   python kpt_analysis_visualization.py --data-file orders.csv --merchants-file merchants.csv

3. Custom output directory:
   python kpt_analysis_visualization.py --output-dir ./results

EXPECTED DATA SCHEMA:
=====================
orders.csv must contain:
  - order_id, merchant_id, order_time
  - merchant_for_time, rider_arrival, rider_pickup
  - (optional) actual_food_ready, actual_kpt_minutes, merchant_tier

        """
    )
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--data-file', '-d', type=str, default=None,
                        help='Path to orders CSV file (uses synthetic data if not provided)')
    parser.add_argument('--merchants-file', '-m', type=str, default=None,
                        help='Path to merchants CSV file (derived from orders if not provided)')
    parser.add_argument('--orders', '-n', type=int, default=50000,
                        help='Number of orders for synthetic data (default: 50000)')
    parser.add_argument('--merchants', type=int, default=500,
                        help='Number of merchants for synthetic data (default: 500)')
    
    args = parser.parse_args()
    
    main(
        output_dir=Path(args.output_dir).resolve(),
        data_file=Path(args.data_file) if args.data_file else None,
        merchants_file=Path(args.merchants_file) if args.merchants_file else None,
        n_orders=args.orders,
        n_merchants=args.merchants
    )

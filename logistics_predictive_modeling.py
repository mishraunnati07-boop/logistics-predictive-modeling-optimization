"""
================================================================================
Week 4: Predictive Modeling and Optimization in Logistics Systems
Author: Unnati Mishra
Domain: Logistics Data Analyst Internship
Task: Predictive Modeling, Evaluation & Optimization Framework
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# STEP 1: LOGISTICS DATA SIMULATION WITH EMPIRICAL PARAMETERS
# ------------------------------------------------------------------------------
def generate_logistics_dataset(n_samples=1200, random_seed=42):
    """
    Generates a realistic freight logistics dataset based on industry parameters.
    Consistent naming conventions used throughout:
      - Shipment_ID, Vehicle_Type, Distance_KM, Shipment_Weight_KG
      - Warehouse_Processing_HR, Traffic_Congestion_Index, Weather_Severity
      - Transit_Time_HR, Transport_Cost_INR, Is_Delayed
    """
    np.random.seed(random_seed)
    
    # Haul Distance (km): Regional to inter-state distribution (25 to 850 km)
    distance_km = np.random.uniform(25, 850, n_samples)
    
    # Consignment Weight (kg): Partial parcel to LTL shipments
    shipment_weight_kg = np.random.uniform(5, 1200, n_samples)
    
    # Corridor Congestion: 1.0 (free-flow) to 5.0 (severe gridlock)
    traffic_congestion_index = np.random.uniform(1.0, 5.0, n_samples)
    
    # Weather Severity: 0 (Clear - 65%), 1 (Rain - 25%), 2 (Storm - 10%)
    weather_severity = np.random.choice([0, 1, 2], size=n_samples, p=[0.65, 0.25, 0.10])
    
    # Fleet Vehicle Types
    vehicle_types = np.random.choice(['Van', 'Medium Truck', 'Heavy Freight'], size=n_samples, p=[0.45, 0.35, 0.20])
    
    # Warehouse Dwell Time (hours): Modeled via Gamma distribution (Mean ~ 2 hrs)
    warehouse_processing_hr = np.random.gamma(shape=2.5, scale=0.8, size=n_samples)
    
    # Base cruising speed per fleet type (km/h)
    base_speed = {'Van': 55.0, 'Medium Truck': 45.0, 'Heavy Freight': 38.0}
    speeds = np.array([base_speed[v] for v in vehicle_types])
    
    # Actual Transit Time calculation with weather & traffic multipliers
    traffic_factor = 1.0 + 0.12 * (traffic_congestion_index - 1.0)
    weather_factor = 1.0 + 0.20 * weather_severity
    stochastic_noise = np.random.normal(0, 0.6, n_samples)
    
    transit_time_hr = (distance_km / speeds) * traffic_factor * weather_factor + warehouse_processing_hr + stochastic_noise
    transit_time_hr = np.clip(transit_time_hr, 1.2, None)
    
    # Freight Cost Model (INR)
    base_cost_km = {'Van': 12.0, 'Medium Truck': 22.0, 'Heavy Freight': 35.0}
    cost_rates = np.array([base_cost_km[v] for v in vehicle_types])
    fuel_surcharge = traffic_congestion_index * 2.5
    transport_cost_inr = (distance_km * cost_rates) + (shipment_weight_kg * 1.5) + (fuel_surcharge * distance_km * 0.1) + np.random.normal(0, 150, n_samples)
    
    # Delivery SLA baseline and delay classification
    expected_sla_hr = (distance_km / 45.0) + 3.0
    is_delayed = (transit_time_hr > expected_sla_hr).astype(int)
    
    df = pd.DataFrame({
        'Shipment_ID': [f'LOG-{10000 + i}' for i in range(n_samples)],
        'Vehicle_Type': vehicle_types,
        'Distance_KM': np.round(distance_km, 1),
        'Shipment_Weight_KG': np.round(shipment_weight_kg, 1),
        'Warehouse_Processing_HR': np.round(warehouse_processing_hr, 2),
        'Traffic_Congestion_Index': np.round(traffic_congestion_index, 2),
        'Weather_Severity': weather_severity,
        'Transit_Time_HR': np.round(transit_time_hr, 2),
        'Transport_Cost_INR': np.round(transport_cost_inr, 2),
        'Is_Delayed': is_delayed
    })
    return df

# ------------------------------------------------------------------------------
# STEP 2: MODEL TRAINING AND BENCHMARKING
# ------------------------------------------------------------------------------
def evaluate_models(df):
    """
    Trains and validates Linear Regression, Random Forest, and Gradient Boosting.
    """
    feature_cols = ['Vehicle_Type', 'Distance_KM', 'Shipment_Weight_KG', 
                    'Warehouse_Processing_HR', 'Traffic_Congestion_Index', 'Weather_Severity']
    target_col = 'Transit_Time_HR'
    
    X = df[feature_cols]
    y = df[target_col]
    
    categorical_features = ['Vehicle_Type']
    numerical_features = ['Distance_KM', 'Shipment_Weight_KG', 'Warehouse_Processing_HR', 
                          'Traffic_Congestion_Index', 'Weather_Severity']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    models = {
        'Multiple Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
        'Gradient Boosting (XGBoost/GBR)': GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, subsample=0.85, random_state=42)
    }
    
    results = {}
    print("=" * 65)
    print("MODEL BENCHMARKING & EVALUATION (WEEK 4 TASK)")
    print("=" * 65)
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # 5-Fold Cross Validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
        
        # Fit on train and evaluate on hold-out test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'pipeline': pipeline, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv_scores.mean()}
        print(f"[{name}]")
        print(f"  Test MAE:       {mae:.3f} hrs")
        print(f"  Test RMSE:      {rmse:.3f} hrs")
        print(f"  Test R2:        {r2:.3f}")
        print(f"  5-Fold CV R2:   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n")
        
    return results, X_test, y_test

# ------------------------------------------------------------------------------
# STEP 3: LOGISTICS OPTIMIZATION & DISPATCH PRESCRIPTIONS
# ------------------------------------------------------------------------------
def prescribe_dispatch_action(predicted_transit_hr, expected_sla_hr, congestion_index):
    """
    Applies prescriptive rules based on predicted transit time vs agreed customer SLA.
    """
    buffer_margin = expected_sla_hr - predicted_transit_hr
    
    if buffer_margin < 0.5:
        return "HIGH RISK (SLA Breach imminent) -> Action: Switch to Expedited Van or Reroute via Secondary Corridor"
    elif buffer_margin < 1.5:
        return "MODERATE RISK -> Action: Fast-track Warehouse Cross-dock by 45 mins to ensure on-time delivery"
    else:
        return "LOW RISK (Optimal Buffer) -> Action: Proceed with Standard Consolidated Freight Dispatch"

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("Simulating dataset...")
    df = generate_logistics_dataset()
    print(f"Dataset generated with shape: {df.shape}")
    print(df.head(3))
    
    results, X_test, y_test = evaluate_models(df)
    
    # Sample optimization test
    print("=" * 65)
    print("PRESCRIPTIVE OPTIMIZATION SYSTEM TEST")
    print("=" * 65)
    sample_pred = 14.2
    sample_sla = 14.5
    print(f"Predicted Transit: {sample_pred} hrs | SLA: {sample_sla} hrs")
    print("Prescription:", prescribe_dispatch_action(sample_pred, sample_sla, congestion_index=4.2))

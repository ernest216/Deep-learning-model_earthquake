import pandas as pd
import numpy as np
from itertools import combinations
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, Add, Embedding, Flatten, Concatenate, Lambda, Multiply, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler
import matplotlib.pyplot as plt
import os
import gc
import multiprocessing
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import time
import sys
import contextlib
import io
import traceback
from datetime import datetime
import csv
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Lambda, GlobalAveragePooling1D
import json
from sklearn.preprocessing import StandardScaler
import joblib
import math
import keras
from keras.saving import register_keras_serializable
from tensorflow.keras import backend as K

# Set TensorFlow logging to ERROR level only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# Limit TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Also add this custom callback to completely suppress progress bar
class SuppressCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        # This effectively silences per-batch output
        pass
        
    def on_train_batch_end(self, batch, logs=None):
        # This effectively silences per-batch output
        pass
        
    def on_test_batch_begin(self, batch, logs=None):
        # This effectively silences per-batch output
        pass
        
    def on_test_batch_end(self, batch, logs=None):
        # This effectively silences per-batch output
        pass

# Create a context manager to suppress all output
@contextlib.contextmanager
def suppress_stdout():
    """Context manager to completely suppress stdout"""
    # Save the original stdout
    old_stdout = sys.stdout
    # Redirect to a dummy IO stream
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        # Restore original stdout
        sys.stdout = old_stdout

def safe_inverse_log10(y_log, y_orig=None):
    """Safely convert log10 values back to original scale with zero preservation"""
    # First apply standard inverse log with clipping
    result = np.power(10, np.clip(y_log, -10, 10))
    
    # If original values provided, restore zeros where they existed 
    if y_orig is not None:
        result = np.where(y_orig == 0, 0, result)
        
    return result

# Create a diagnostics logger function
def log_diagnostic(message):
    """Log a diagnostic message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # Also write to a log file
    log_file = os.path.expanduser("~/Desktop/fyp 2_debug1/model_diagnostic.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create empty file at the beginning of execution
    if not hasattr(log_diagnostic, "file_initialized"):
        log_diagnostic.file_initialized = True
        with open(log_file, "w") as f:  # Use 'w' to create empty file on first call
            f.write(f"[{timestamp}] Log started - {message}\n")
    else:
        with open(log_file, "a") as f:  # Use 'a' for subsequent writes
            f.write(f"[{timestamp}] {message}\n")

# Initialize the diagnostic file at the start
def initialize_diagnostic_log():
    """Create a fresh diagnostic log file"""
    log_path = os.path.expanduser("~/Desktop/fyp 2_debug1/model_diagnostics.txt")
    with open(log_path, 'w') as f:
        f.write(f"SEISMIC MODEL TRAINING DIAGNOSTICS - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

# -------------------------
# ✅ Step 1: Data Preparation
# -------------------------
# ORIGINAL EVENT LOADING CODE (COMMENTED OUT FOR SPEED)
from sklearn.preprocessing import StandardScaler

def prepare_features(group):
    """Prepare features for a single group with both log and original values"""
    sensor_ids = group["Sensor_ID"].unique()
    if len(sensor_ids) < 6:
        return None, None, None, None, None, None, None
    
    static_features = group.iloc[0][[
        "Magnitude", "PGA", "Depth", "Repi", "Stories",
        "Total_Height (in)", "Base Length (in)", "Base Breadth (in)",
        "Typical Floor Length (in)", "Typical Floor Breadth (in)"
    ]].values

    X_batch, y_accel_batch, y_vel_batch, y_arias_batch = [], [], [], []
    # Original values batches
    y_accel_orig_batch, y_vel_orig_batch, y_arias_orig_batch = [], [], []
    
    for combo in combinations(sensor_ids, 5):
        input_sensors = list(combo)
        remaining_sensors = [s for s in sensor_ids if s not in input_sensors]
        
        input_data = []
        for sensor in input_sensors:
            sensor_row = group[group["Sensor_ID"] == sensor]
            sensor_features = sensor_row[[
                "Height (in)", "Breadth (in)", "Length (in)", 
                "Log_Peak_Acceleration", "Log_Peak_Velocity", "Log_Arias_Intensity"
            ]].values.flatten()
            
            direction = sensor_row["Sensor_Direction"].iloc[0]
            direction_encoding = [1 if direction == d else 0 for d in ['X', 'Y', 'Z']]
            input_data.extend(np.concatenate([sensor_features, direction_encoding]))

        full_features = np.concatenate([input_data, static_features])
        
        for target_sensor in remaining_sensors:
            target_row = group[group["Sensor_ID"] == target_sensor]
            
            # Get log values for training (these are already log-transformed in the dataset)
            target_acc_log = target_row["Log_Peak_Acceleration"].values[0]
            target_vel_log = target_row["Log_Peak_Velocity"].values[0]
            target_arias_log = target_row["Log_Arias_Intensity"].values[0]
            
            # Get original values from the dataset
            target_acc_orig = target_row["Peak Acceleration"].values[0]
            target_vel_orig = target_row["Peak Velocity"].values[0]
            target_arias_orig = target_row["Arias Intensity"].values[0]
            
            X_batch.append(full_features)
            y_accel_batch.append(target_acc_log)
            y_vel_batch.append(target_vel_log)
            y_arias_batch.append(target_arias_log)
            
            # Store original values
            y_accel_orig_batch.append(target_acc_orig)
            y_vel_orig_batch.append(target_vel_orig)
            y_arias_orig_batch.append(target_arias_orig)
    
    return (np.array(X_batch, dtype=np.float32), 
            np.array(y_accel_batch, dtype=np.float32),
            np.array(y_vel_batch, dtype=np.float32),
            np.array(y_arias_batch, dtype=np.float32),
            np.array(y_accel_orig_batch, dtype=np.float32),
            np.array(y_vel_orig_batch, dtype=np.float32),
            np.array(y_arias_orig_batch, dtype=np.float32))

def engineer_features(X):
    """Apply enhanced feature engineering to X data"""
    # Assuming the structure: 5 sensors with 4 values each, then static features
    base_features = X[:, :20].reshape(-1, 5, 4)
    static_features = X[:, 20:]
    
    # Basic statistical features (already in your code)
    sensor_means = np.mean(base_features, axis=1)
    sensor_maxs = np.max(base_features, axis=1)
    sensor_mins = np.min(base_features, axis=1)
    sensor_std = np.std(base_features, axis=1)
    
    # NEW: Add more sophisticated features
    
    # Height differences between floors (captures structural arrangement)
    height_diffs = np.diff(base_features[:, :, 0], axis=1)
    
    # Ratios between sensors (captures relative response)
    sensor_ratios = base_features[:, 1:, :] / np.maximum(base_features[:, :-1, :], 1e-10)
    sensor_ratios = sensor_ratios.reshape(sensor_ratios.shape[0], -1)
    
    # DEBUG: Check for zero variance before calculating correlations
    log_diagnostic("Checking sensor data variance...")
    for i in range(min(10, len(X))):
        sensor_data = base_features[i, :, 1:]
        std_vals = np.std(sensor_data, axis=1)
        min_std = np.min(std_vals)
        if min_std < 1e-10:
            log_diagnostic(f"Sample {i} has near-zero variance: {std_vals}")
            log_diagnostic(f"Problematic sensor data:\n{sensor_data}")
    
    # Correlations between sensor measurements with safety check
    correlations = []
    for i in range(len(X)):
        sensor_data = base_features[i, :, 1:] # Skip height, use only measurements
        
        # Check if any row has zero variance
        std_vals = np.std(sensor_data, axis=1)
        if np.any(std_vals < 1e-10):
            # Use a placeholder correlation (zeros) if we have constant sensors
            correlations.append(np.zeros(10))  # 10 = number of pairs in upper triangle
        else:
            # Calculate correlation normally
            corr = np.corrcoef(sensor_data.reshape(5, 3), rowvar=True)
            correlations.append(corr[np.triu_indices(5, k=1)])
    
    correlations = np.array(correlations)
    
    # Interaction features between building properties and earthquake intensity
    # e.g., height × magnitude, stories × PGA
    building_features = static_features[:, 4:9]  # Assuming these are building features
    quake_features = static_features[:, 0:4]    # Assuming these are earthquake features
    interactions = np.zeros((X.shape[0], building_features.shape[1] * quake_features.shape[1]))
    
    idx = 0
    for i in range(building_features.shape[1]):
        for j in range(quake_features.shape[1]):
            interactions[:, idx] = building_features[:, i] * quake_features[:, j]
            idx += 1
    
    # Combine all new engineered features
    return np.hstack([
        X, sensor_means, sensor_maxs, sensor_mins, sensor_std, 
        height_diffs, sensor_ratios, correlations, interactions
    ])

    # Add this debug code to your engineer_features function
    for i in range(min(10, len(X))):
        sensor_data = base_features[i, :, 1:]
        std_vals = np.std(sensor_data, axis=1)
        min_std = np.min(std_vals)
        if min_std < 1e-10:
            print(f"Sample {i} has near-zero variance: {std_vals}")
            print(f"Problematic sensor data:\n{sensor_data}")

# -------------------------
# ✅ Step 2: Model Building
# -------------------------

# Add these global variables for scalers
sensor_geometry_scaler = None
earthquake_param_scaler = None
building_param_scaler = None
derived_feature_scaler = None

def normalize_features(X, feature_indices, is_training=False):
    """
    Normalize features by groups while preserving important physical measurements
    
    Args:
        X: Input feature array
        feature_indices: Dictionary mapping feature types to their column indices
        is_training: Whether this is the training phase (fit scalers) or inference
    
    Returns:
        X_normalized: Array with normalized features
    """
    global sensor_geometry_scaler, earthquake_param_scaler, building_param_scaler, derived_feature_scaler
    
    # Make a copy to avoid modifying the original
    X_normalized = X.copy()
    
    # Extract feature groups
    sensor_geom_cols = feature_indices.get('sensor_geometry', [])
    earthquake_cols = feature_indices.get('earthquake_params', [])
    building_cols = feature_indices.get('building_params', [])
    derived_cols = feature_indices.get('derived_features', [])
    
    # Skip normalization for these measurement columns
    skip_cols = []
    skip_cols.extend(feature_indices.get('pfa', []))
    skip_cols.extend(feature_indices.get('pfv', []))
    skip_cols.extend(feature_indices.get('arias', []))
    
    # Remove measurement columns from normalization
    sensor_geom_cols = [col for col in sensor_geom_cols if col not in skip_cols]
    
    # Initialize scalers during training
    if is_training:
        log_diagnostic("Creating new feature scalers for normalization")
        sensor_geometry_scaler = StandardScaler()
        earthquake_param_scaler = StandardScaler()
        building_param_scaler = StandardScaler()
        derived_feature_scaler = StandardScaler()
    
    # Normalize each feature group separately
    if sensor_geom_cols and len(sensor_geom_cols) > 0:
        if is_training:
            log_diagnostic(f"Fitting sensor geometry scaler on {len(sensor_geom_cols)} columns")
            X_normalized[:, sensor_geom_cols] = sensor_geometry_scaler.fit_transform(X[:, sensor_geom_cols])
        else:
            X_normalized[:, sensor_geom_cols] = sensor_geometry_scaler.transform(X[:, sensor_geom_cols])
    
    if earthquake_cols and len(earthquake_cols) > 0:
        if is_training:
            log_diagnostic(f"Fitting earthquake parameter scaler on {len(earthquake_cols)} columns")
            X_normalized[:, earthquake_cols] = earthquake_param_scaler.fit_transform(X[:, earthquake_cols])
        else:
            X_normalized[:, earthquake_cols] = earthquake_param_scaler.transform(X[:, earthquake_cols])
    
    if building_cols and len(building_cols) > 0:
        if is_training:
            log_diagnostic(f"Fitting building parameter scaler on {len(building_cols)} columns")
            X_normalized[:, building_cols] = building_param_scaler.fit_transform(X[:, building_cols])
        else:
            X_normalized[:, building_cols] = building_param_scaler.transform(X[:, building_cols])
    
    if derived_cols and len(derived_cols) > 0:
        if is_training:
            log_diagnostic(f"Fitting derived features scaler on {len(derived_cols)} columns")
            X_normalized[:, derived_cols] = derived_feature_scaler.fit_transform(X[:, derived_cols])
        else:
            X_normalized[:, derived_cols] = derived_feature_scaler.transform(X[:, derived_cols])
    
    return X_normalized

def define_feature_indices(input_dim, engineered_dim):
    """Define indices for different feature types based on input structure"""
    # Assumes structure:
    # - First 20 features are sensor data (5 sensors x 4 values)
    # - Next ~10-15 features are static earthquake/building features
    # - Remaining features are engineered
    
    # Identify features by position in the array
    feature_indices = {
        # Original sensor features - split by type
        'sensor_geometry': [0, 4, 8, 12, 16],           # Heights (1st value of each sensor)
        'pfa': [1, 5, 9, 13, 17],                       # Peak Floor Acceleration (2nd value)
        'pfv': [2, 6, 10, 14, 18],                      # Peak Floor Velocity (3rd value)
        'arias': [3, 7, 11, 15, 19],                    # Arias Intensity (4th value)
        
        # Static features
        'earthquake_params': [20, 21, 22, 23],          # Magnitude, PGA, Depth, Repi
        'building_params': list(range(24, 34)),         # Building parameters
        
        # Engineered features start at original dim (input_dim)
        'derived_features': list(range(input_dim, engineered_dim))
    }
    
    return feature_indices

def engineer_features_and_normalize(X, is_training=True):
    """Apply enhanced feature engineering and normalization in sequence"""
    
    # Step 1: Apply your existing feature engineering
    X_engineered = engineer_features(X)
    log_diagnostic(f"Applied feature engineering: {X.shape} → {X_engineered.shape}")
    
    # Step 2: Define feature indices based on your engineered data structure
    # Adjust these indices to match your actual data columns
    feature_indices = {
        # Sensor features (positions and measurements)
        'sensor_geometry': [i for i in range(20) if i % 4 == 0],  # Every 4th column in first 20
        'pfa': [i for i in range(20) if i % 4 == 1],              # PFA features - EXCLUDED FROM NORMALIZATION
        'pfv': [i for i in range(20) if i % 4 == 2],              # PFV features - EXCLUDED FROM NORMALIZATION
        'arias': [i for i in range(20) if i % 4 == 3],            # AI features - EXCLUDED FROM NORMALIZATION
        
        # Static features
        'earthquake_params': list(range(20, 24)),                 # Magnitude, PGA, Depth, Repi
        'building_params': list(range(24, 35)),                   # Building parameters
        
        # Derived features from engineer_features
        'derived_features': list(range(35, X_engineered.shape[1]))  # All additional engineered features
    }
    
    # Step 3: Apply normalization while preserving PFA, PFV, AI values
    X_normalized = normalize_features(X_engineered, feature_indices, is_training)
    log_diagnostic("Applied feature normalization, preserving PFA, PFV, and AI measurements")
    
    return X_normalized

# Modified function with normalization integrated
def load_and_prepare_data(data_path, debug_mode=False, max_groups=None):
    """Load and prepare data with proper event and sensor grouping"""
    try:
        # Very explicit debug mode marker - FIXED THIS LINE
        log_diagnostic(f"Loading data from {data_path} ... (debug mode: {debug_mode}), (max groups: {max_groups})")
        
        if debug_mode:
            log_diagnostic("🔴🔴🔴 DEBUG MODE ACTIVE - WILL LIMIT TO 10 GROUPS 🔴🔴🔴")
        else:
            log_diagnostic("Normal mode - processing all groups")
        
        if debug_mode:
            log_diagnostic(f"DEBUG MODE ENABLED: Loading data from {data_path} with max_groups={max_groups}")
        else:
            log_diagnostic(f"Loading data from {data_path}...")
        
        # Load data
        df = pd.read_csv(data_path)
        log_diagnostic(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        log_diagnostic(f"Column names: {df.columns.tolist()}")
        
        # Target columns
        accel_col = "Peak Acceleration"
        vel_col = "Peak Velocity"
        arias_col = "Arias Intensity"
        
        # Log columns
        log_accel_col = "Log_Peak_Acceleration"
        log_vel_col = "Log_Peak_Velocity" 
        log_arias_col = "Log_Arias_Intensity"
        
        log_diagnostic(f"Using target columns: {accel_col}, {vel_col}, {arias_col}")
        log_diagnostic(f"Using log columns: {log_accel_col}, {log_vel_col}, {log_arias_col}")
        
        # Handle missing columns - create log columns if they don't exist
        if log_accel_col not in df.columns:
            log_diagnostic(f"Creating log column {log_accel_col}")
            df[log_accel_col] = np.log10(np.maximum(df[accel_col].values, 1e-10))
            
        if log_vel_col not in df.columns:
            log_diagnostic(f"Creating log column {log_vel_col}")
            df[log_vel_col] = np.log10(np.maximum(df[vel_col].values, 1e-10))
            
        if log_arias_col not in df.columns:
            log_diagnostic(f"Creating log column {log_arias_col}")
            df[log_arias_col] = np.log10(np.maximum(df[arias_col].values, 1e-10))
        
        # Create chunks for processing
        if len(df) < 10000:
            chunk_gen = [df]  # Single chunk with entire dataframe
            log_diagnostic(f"Using single chunk processing mode, dataframe size: {len(df)}")
        else:
            # Split into chunks of 5000 rows
            chunk_size = 5000
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            chunk_gen = [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
            log_diagnostic(f"Using multi-chunk processing, {num_chunks} chunks of size {chunk_size}")
            
        all_X, all_y_accel, all_y_vel, all_y_arias = [], [], [], []
        all_y_accel_orig, all_y_vel_orig, all_y_arias_orig = [], [], []
        
        for chunk in chunk_gen:
            log_diagnostic(f"Processing chunk of size {len(chunk)}...")
            
            # Filter and clean data
            chunk = chunk.dropna(subset=[
                "STID", "Earthquake Name", 
                log_accel_col, log_vel_col, log_arias_col,
                accel_col, vel_col, arias_col,
                "Magnitude", "PGA", "Depth", "Repi", "Stories",
                "Total_Height (in)", "Base Length (in)", "Base Breadth (in)", 
                "Typical Floor Length (in)", "Typical Floor Breadth (in)", 
                "Height (in)", "Breadth (in)", "Length (in)",
                "Sensor_Direction", "Sensor_ID"
            ])
            log_diagnostic(f"Chunk size after dropping NA: {len(chunk)}")
            
            # Remove duplicate Sensor_IDs within groups
            chunk = chunk.drop_duplicates(subset=["STID", "Earthquake Name", "Sensor_ID"])
            log_diagnostic(f"Chunk size after removing duplicates: {len(chunk)}")
            
            # Group by (STID, Earthquake)
            grouped = chunk.groupby(["STID", "Earthquake Name"])
            grouped_dict = {g: d for g, d in grouped if len(d) >= 6}
            log_diagnostic(f"Number of valid groups (with >= 6 sensors): {len(grouped_dict)}")
            
            # Right after you create grouped_dict:
            full_group_count = len(grouped_dict)
            log_diagnostic(f"BEFORE LIMITING: Found {full_group_count} groups with >= 6 sensors")
            
            # FIXED THIS CONDITION - Make it very explicit
            if debug_mode is True:  # Force boolean comparison
                log_diagnostic("🔴 DEBUG MODE IS TRUE - LIMITING GROUPS NOW 🔴")
                
                # Get list of all group keys
                all_groups = list(grouped_dict.keys())
                
                # Randomly select max_groups from the list
                import random
                random.seed(42)  # For reproducibility
                selected_groups = random.sample(all_groups, max_groups)
                
                # Create a new dictionary with ONLY the selected groups
                new_grouped_dict = {}
                for group_key in selected_groups:
                    new_grouped_dict[group_key] = grouped_dict[group_key]
                
                # Replace the original dictionary completely
                grouped_dict = new_grouped_dict
                
                log_diagnostic(f"🔴🔴🔴 DEBUG MODE: Limited from {full_group_count} to {len(grouped_dict)} groups 🔴🔴🔴")
                log_diagnostic(f"Selected groups: {selected_groups}")
            else:
                log_diagnostic("Debug mode is FALSE - processing all groups")
            
            # Verify the group count after potential limitation
            log_diagnostic(f"AFTER LIMITING: Processing {len(grouped_dict)} groups")
            
            # Prepare features for each group
            for (stid, quake), group in grouped_dict.items():
                log_diagnostic(f"Preparing features for group ({stid}, {quake}) with {len(group)} sensors...")
                
                # FIX: Don't pass debug_mode to prepare_features
                result = prepare_features(group)  # Remove debug_mode parameter
                
                if result is not None:
                    X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig = result
                    all_X.append(X)
                    all_y_accel.append(y_accel)
                    all_y_vel.append(y_vel)
                    all_y_arias.append(y_arias)
                    all_y_accel_orig.append(y_accel_orig)
                    all_y_vel_orig.append(y_vel_orig)
                    all_y_arias_orig.append(y_arias_orig)
                    log_diagnostic(f"Added {len(X)} examples from group ({stid}, {quake})")
                    del X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig
                else:
                    log_diagnostic(f"Group ({stid}, {quake}) returned no valid combinations")
                
                gc.collect()
        
        # Combine all data
        if not all_X:
            raise ValueError("No valid groups found in the data.")
        
        X = np.concatenate(all_X)
        y_accel = np.concatenate(all_y_accel)
        y_vel = np.concatenate(all_y_vel)
        y_arias = np.concatenate(all_y_arias)
        
        # Original values
        y_accel_orig = np.concatenate(all_y_accel_orig)
        y_vel_orig = np.concatenate(all_y_vel_orig)
        y_arias_orig = np.concatenate(all_y_arias_orig)
        
        log_diagnostic(f"Final dataset size: X={X.shape}, y_accel={y_accel.shape}, y_vel={y_vel.shape}, y_arias={y_arias.shape}")
        
        # NEW APPROACH: Apply feature engineering with normalization
        X = engineer_features_and_normalize(X, is_training=True)
        log_diagnostic("Feature engineering and normalization applied successfully!")
        
        # Save scalers for future use
        os.makedirs('scalers', exist_ok=True)
        joblib.dump(sensor_geometry_scaler, 'scalers/sensor_geometry_scaler.pkl')
        joblib.dump(earthquake_param_scaler, 'scalers/earthquake_param_scaler.pkl')
        joblib.dump(building_param_scaler, 'scalers/building_param_scaler.pkl')
        joblib.dump(derived_feature_scaler, 'scalers/derived_feature_scaler.pkl')
        log_diagnostic("Saved feature scalers to 'scalers/' directory")
        
        return X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig
        
    except Exception as e:
        log_diagnostic(f"Error loading data: {str(e)}")
        traceback.print_exc()
        raise

# Add this function for model predictions
def predict_with_normalized_features(model, X_new):
    """Make predictions with a trained model, applying the same normalization"""
    try:
        global sensor_geometry_scaler, earthquake_param_scaler, building_param_scaler, derived_feature_scaler
        
        # Load scalers if they're not already loaded
        if sensor_geometry_scaler is None:
            log_diagnostic("Loading feature scalers for prediction...")
            sensor_geometry_scaler = joblib.load('scalers/sensor_geometry_scaler.pkl')
            earthquake_param_scaler = joblib.load('scalers/earthquake_param_scaler.pkl')
            building_param_scaler = joblib.load('scalers/building_param_scaler.pkl')
            derived_feature_scaler = joblib.load('scalers/derived_feature_scaler.pkl')
        
        # Apply same feature engineering and normalization as during training
        X_processed = engineer_features_and_normalize(X_new, is_training=False)
        
        # Make predictions
        predictions = model.predict(X_processed)
        log_diagnostic(f"Generated predictions for {len(X_new)} samples")
        
        return predictions
    
    except Exception as e:
        log_diagnostic(f"Error in prediction: {str(e)}")
        traceback.print_exc()

def feature_cross_layer(x, hidden_dim=32):
    """
    Create pairwise feature interactions to model complex feature relationships.
    Uses a more efficient implementation to avoid memory issues.
    """
    # First get original dimensionality
    input_dim = tf.shape(x)[1]
    
    # Create feature crosses more efficiently
    # Instead of explicit cross products, use factorized approach
    x1 = Dense(hidden_dim, kernel_regularizer=l2(0.001))(x)
    x2 = Dense(hidden_dim, kernel_regularizer=l2(0.001))(x)
    
    # Element-wise multiplication for interaction
    cross = Multiply()([x1, x2])
    
    # Combine original features with cross features
    combined = Concatenate()([x, cross])
    
    return combined

def preprocess_with_embeddings(X, categorical_cols=[0, 1, 2]):  # Assuming cols 0,1,2 are building/earthquake IDs
    # Split data into categorical and continuous
    X_cat = X[:, categorical_cols]
    X_cont = np.delete(X, categorical_cols, axis=1)
    
    # Create embeddings for each categorical feature
    embedding_layers = []
    embedded_features = []
    
    for i in range(X_cat.shape[1]):
        num_unique = int(np.max(X_cat[:, i])) + 1
        embedding_dim = min(50, (num_unique + 1) // 2)  # Rule of thumb for embedding size
        
        # Create embedding layers
        input_layer = Input(shape=(1,))
        embedding = Embedding(num_unique, embedding_dim, input_length=1)(input_layer)
        embedding = Flatten()(embedding)
        
        embedding_layers.append(input_layer)
        embedded_features.append(embedding)
    
    # Concatenate embeddings with continuous features
    cont_input = Input(shape=(X_cont.shape[1],))
    merged_features = Concatenate()(embedded_features + [cont_input])
    
    # First dense layer with residual connection
    x = Dense(128, activation='relu')(merged_features)
    x = BatchNormalization()(x)
    
    # Residual block 1
    x_skip = x
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = Add()([x, x_skip])  # Add residual connection
    x = Activation('relu')(x)
    
    # Residual block 2
    x_skip = x
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = Add()([x, x_skip])  # Add residual connection
    x = Activation('relu')(x)
    
    # Output branches
    acceleration = Dense(64, activation='relu')(x)
    acceleration = Dense(1, name='acceleration')(acceleration)
    
    velocity = Dense(64, activation='relu')(x)
    velocity = Dense(1, name='velocity')(velocity)
    
    arias = Dense(64, activation='relu')(x)
    arias = Dense(1, name='arias')(arias)
    
    return Model(inputs=[*embedding_layers, cont_input], outputs=[acceleration, velocity, arias])

# Fix the PositionEmbedding class with correct parameter ordering
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length=200):
        super().__init__()
        self.max_length = max_length
        
    def build(self, input_shape):
        _, seq_len, d_model = input_shape
        self.pos_embedding = self.add_weight(
            name="pos_embedding",  # Use 'name' parameter instead of positional
            shape=(self.max_length, d_model),
            initializer="uniform",
            trainable=True,
        )
        
    def call(self, inputs):
        # Take just the needed embeddings
        position_embeddings = self.pos_embedding[:tf.shape(inputs)[1], :]
        return inputs + tf.expand_dims(position_embeddings, axis=0)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.05):
    """Transformer encoder block with multi-head attention"""
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = attention_output + inputs

    # Feed Forward Part
    ff = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(ff_dim, activation="relu")(ff)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    
    return x + ff

def build_transformer_model(input_dim, learning_rate=0.0005):
    """Build transformer-based multi-output neural network with optimized hyperparameters"""
    
    # Input layer
    inputs = Input(shape=(input_dim,), name='input')
    
    # Reshape flat input into sensor-based sequence
    sensor_features = 4
    num_sensors = 5
    static_features_dim = input_dim - (sensor_features * num_sensors)
    
    reshaped = Lambda(lambda x: tf.reshape(
        x[:, :sensor_features*num_sensors], 
        [-1, num_sensors, sensor_features]))(inputs)
    
    static = Lambda(lambda x: x[:, sensor_features*num_sensors:])(inputs)
    
    # Position embedding with larger dimension
    x = PositionEmbedding(max_length=num_sensors)(reshaped)
    
    # Transformer blocks with increased capacity
    num_transformer_blocks = 12  # Increase from 8
    head_size = 256  # Increase from 128
    num_heads = 16   # Increase from 12
    ff_dim = 2048    # Increase from 1024
    
    for i in range(num_transformer_blocks):
        x = transformer_encoder(
            x, 
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=0.1
        )
    
    # Global context
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Combine with static features
    x = Concatenate()([x, static])
    
    # Wider shared layers with residual connections
    x1 = Dense(2048, kernel_regularizer=l2(0.0001))(x)  # Increased width, reduced regularization
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.2)(x1)
    
    # Residual connection
    x2 = Dense(2048, kernel_regularizer=l2(0.0001))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.2)(x2)
    x = Add()([x1, x2])  # Residual connection
    
    # Branch for Acceleration
    accel = Dense(512, kernel_regularizer=l2(0.0001))(x)
    accel = BatchNormalization()(accel)
    accel = Activation('relu')(accel)
    accel = Dropout(0.1)(accel)
    accel = Dense(256, kernel_regularizer=l2(0.0001))(accel)
    accel = BatchNormalization()(accel)
    accel = Activation('relu')(accel)
    accel = Dense(1, name='output_accel')(accel)
    
    # Branch for Velocity
    vel = Dense(512, kernel_regularizer=l2(0.0001))(x)
    vel = BatchNormalization()(vel)
    vel = Activation('relu')(vel)
    vel = Dropout(0.1)(vel)
    vel = Dense(256, kernel_regularizer=l2(0.0001))(vel)
    vel = BatchNormalization()(vel)
    vel = Activation('relu')(vel)
    vel = Dense(1, name='output_vel')(vel)
    
    # Branch for Arias Intensity
    arias = Dense(512, kernel_regularizer=l2(0.0001))(x)
    arias = BatchNormalization()(arias)
    arias = Activation('relu')(arias)
    arias = Dropout(0.1)(arias)
    arias = Dense(256, kernel_regularizer=l2(0.0001))(arias)
    arias = BatchNormalization()(arias)
    arias = Activation('relu')(arias)
    arias = Dense(1, name='output_arias')(arias)
    
    model = Model(inputs=inputs, outputs=[accel, vel, arias])
    
    # Use AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True,
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            'output_accel': 'huber',  # Changed to huber loss for better robustness
            'output_vel': 'huber',
            'output_arias': 'huber'
        },
        metrics={
            'output_accel': ['mae', 'mse'],
            'output_vel': ['mae', 'mse'],
            'output_arias': ['mae', 'mse']
        }
    )
    
    return model

def should_generate_plots():
    """Check if plots should be generated"""
    return not os.path.exists(os.path.expanduser('~/Desktop/fyp 2_debug1/disable_plots.flag'))
# -------------------------
# ✅ Step 3: Training Setup
# -------------------------
def mae_loss(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

def get_optimal_training_params(data_size):
    """
    Get optimal training parameters based on dataset size
    """
    if data_size < 10000:
        batch_size = 32
        initial_lr = 0.001
    elif data_size < 50000:
        batch_size = 64
        initial_lr = 0.0008
    elif data_size < 100000:
        batch_size = 128
        initial_lr = 0.0005
    else:
        batch_size = 256
        initial_lr = 0.0003
    
    return batch_size, initial_lr

def step_decay_schedule(initial_lr=0.001, decay_factor=0.75, step_size=10):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    
    Args:
        initial_lr: Initial learning rate
        decay_factor: Factor by which to decay learning rate
        step_size: Epoch interval at which to decay learning rate
        
    Returns:
        Schedule function for LearningRateScheduler
    """
    def schedule(epoch):
        lr = initial_lr * (decay_factor ** (epoch // step_size))
        # Log the learning rate changes
        if epoch % step_size == 0:
            tf.print(f"\nReducing learning rate to {lr} at epoch {epoch}\n")
        return lr
        
    return schedule

def cosine_decay_schedule(initial_lr=0.001, min_lr=1e-6, epochs=50):
    """
    Wrapper function to create a LearningRateScheduler with cosine decay.
    
    Args:
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        epochs: Total number of epochs
        
    Returns:
        Schedule function for LearningRateScheduler
    """
    def schedule(epoch):
        # Cosine decay formula
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / epochs))
        # Log significant learning rate changes
        if epoch % 5 == 0:
            tf.print(f"\nLearning rate at epoch {epoch}: {lr:.6f}\n")
        return lr
        
    return schedule

def train_model(X_train, y_accel_train, y_vel_train, y_arias_train,
                X_val, y_accel_val, y_vel_val, y_arias_val,
                batch_size=32, epochs=100, patience=20):
    """Train model with original Arias intensity values"""
    
    # Define feature dimension
    input_dim = X_train.shape[1]
    
    # Build model with same architecture
    model = build_transformer_model(input_dim)
    
    # Compile with mixed loss strategy - MAE for log values, MSE for original Arias
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'output_accel': 'mae',      # Still using log values
            'output_vel': 'mae',        # Still using log values
            'output_arias': 'mse'       # Better for original scale values
        },
        metrics={
            'output_accel': ['mae', 'mse'],
            'output_vel': ['mae', 'mse'],
            'output_arias': ['mae', 'mse']  # Now comparing in original scale
        }
    )
    
    # Use callbacks as before
    callbacks = [
        EarlyStopping(patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.00001),
        # Other callbacks...
    ]
    
    # Train with original Arias values
    history = model.fit(
        X_train, 
        {
            'output_accel': y_accel_train,   # Log values for accel
            'output_vel': y_vel_train,       # Log values for vel
            'output_arias': y_arias_train  # ORIGINAL values for Arias
        },
        validation_data=(
            X_val, 
            {
                'output_accel': y_accel_val,
                'output_vel': y_vel_val,
                'output_arias': y_arias_val  # ORIGINAL values
            }
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

class MetricsLogger(Callback):
    """Memory-efficient metrics logger to prevent out-of-memory errors"""
    
    def __init__(self, X_val, y_val_accel, y_val_vel, y_val_arias, 
                y_val_accel_orig, y_val_vel_orig, y_val_arias_orig,
                log_interval=1):
        super().__init__()
        self.X_val = X_val
        self.y_val_accel = y_val_accel
        self.y_val_vel = y_val_vel
        self.y_val_arias = y_val_arias
        self.y_val_accel_orig = y_val_accel_orig
        self.y_val_vel_orig = y_val_vel_orig
        self.y_val_arias_orig = y_val_arias_orig
        self.log_interval = log_interval
        self.log_file = os.path.expanduser('~/Desktop/fyp 2_debug1/training_metrics.csv')
        
        # Create log file with header
        with open(self.log_file, 'w') as f:
            f.write("epoch,accel_r2,vel_r2,arias_r2,accel_orig_r2,vel_orig_r2,arias_orig_r2\n")
    
    def on_epoch_end(self, epoch, logs=None):
        # Skip most epochs to save memory
        if epoch % self.log_interval != 0 and epoch != 0:
            return
        
        try:
            log_diagnostic(f"Computing metrics for epoch {epoch}")
            
            # Process in small batches to avoid memory issues
            batch_size = 32
            n_samples = len(self.X_val)
            
            # Use a different random seed each time based on epoch
            # This ensures different subsets for each epoch
            np.random.seed(epoch + int(time.time()) % 10000)
            
            # Limit validation set size
            max_samples = min(10000, n_samples)
            indices = np.random.choice(n_samples, max_samples, replace=False)
            
            X_val_subset = self.X_val[indices]
            y_val_accel_subset = self.y_val_accel[indices]
            y_val_vel_subset = self.y_val_vel[indices]
            y_val_arias_subset = self.y_val_arias[indices]
            y_val_accel_orig_subset = self.y_val_accel_orig[indices]
            y_val_vel_orig_subset = self.y_val_vel_orig[indices]
            y_val_arias_orig_subset = self.y_val_arias_orig[indices]
            log_diagnostic(f"Using validation subset of {max_samples} samples")
            
            # Calculate number of batches
            n_batches = (len(X_val_subset) + batch_size - 1) // batch_size
            
            # Initialize prediction arrays
            all_preds_accel = []
            all_preds_vel = []
            all_preds_arias = []
            
            # Process in batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_val_subset))
                X_batch = X_val_subset[start_idx:end_idx]
                
                # Predict on batch
                preds = self.model.predict(X_batch, verbose=0)
                all_preds_accel.append(preds[0])
                all_preds_vel.append(preds[1])
                all_preds_arias.append(preds[2])
                
                # Force garbage collection every few batches
                if i % 10 == 0:
                    gc.collect()
            
            # Concatenate predictions
            y_pred_accel = np.concatenate(all_preds_accel)
            y_pred_vel = np.concatenate(all_preds_vel)
            y_pred_arias = np.concatenate(all_preds_arias)
            
            # Free memory
            del all_preds_accel, all_preds_vel, all_preds_arias
            gc.collect()
            
            # Calculate only R² metrics (most important)
            accel_r2 = r2_score(y_val_accel_subset, y_pred_accel)
            vel_r2 = r2_score(y_val_vel_subset, y_pred_vel)
            arias_r2 = r2_score(y_val_arias_subset, y_pred_arias)  # Back to log values
            
            # Convert to original space
            y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
            y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
            y_pred_arias_orig = safe_inverse_log10(y_pred_arias)  # Restored this line
            
            # Calculate original space R²
            accel_orig_r2 = r2_score(y_val_accel_orig_subset, y_pred_accel_orig)
            vel_orig_r2 = r2_score(y_val_vel_orig_subset, y_pred_vel_orig)
            arias_orig_r2 = r2_score(y_val_arias_orig_subset, y_pred_arias_orig)
            
            # Free memory
            del y_pred_accel_orig, y_pred_vel_orig, y_pred_arias_orig
            gc.collect()
            
            # Log results
            log_diagnostic(f"Log Space - Accel R²: {accel_r2:.4f}, Vel R²: {vel_r2:.4f}, Arias R²: {arias_r2:.4f}")
            log_diagnostic(f"Orig Space - Accel R²: {accel_orig_r2:.4f}, Vel R²: {vel_orig_r2:.4f}, Arias R²: {arias_orig_r2:.4f}")
            
            # Write to CSV
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch},{accel_r2:.4f},{vel_r2:.4f},{arias_r2:.4f},{accel_orig_r2:.4f},{vel_orig_r2:.4f},{arias_orig_r2:.4f}\n")
                
        except Exception as e:
            log_diagnostic(f"Error in metrics calculation: {str(e)}")
            traceback.print_exc()

def plot_metrics(history):
    """Plot training metrics over epochs"""
    configure_plot_style()
    
    # For testing, check if the history object is valid and has expected keys
    log_diagnostic(f"Available keys in history: {list(history.history.keys())}")
    
    # Plot only if we have valid keys
    try:
        if 'output_accel_loss' in history.history:
            plt.figure(figsize=(10, 6))
            
            # Plot losses
            plt.plot(history.history['output_accel_loss'], label='Acceleration Loss')
            plt.plot(history.history['output_vel_loss'], label='Velocity Loss')
            plt.plot(history.history['output_arias_loss'], label='Arias Loss')
            
            # Plot validation losses if available
            if 'val_output_accel_loss' in history.history:
                plt.plot(history.history['val_output_accel_loss'], '--', label='Val Acceleration Loss')
                plt.plot(history.history['val_output_vel_loss'], '--', label='Val Velocity Loss') 
                plt.plot(history.history['val_output_arias_loss'], '--', label='Val Arias Loss')
            
            plt.title('Training and Validation Loss', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Mean Squared Error', fontsize=14)
            plt.yscale('log')  # Log scale for better visualization
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/training_loss.png'), dpi=300)
            plt.close()
            
        else:
            log_diagnostic("Warning: Expected loss keys not found in history object")
            log_diagnostic(f"Available keys: {list(history.history.keys())}")
    except Exception as e:
        log_diagnostic(f"Error plotting metrics: {str(e)}")

def plot_residual_analysis(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                          y_test_arias, y_pred_arias, y_test_accel_orig=None, 
                          y_test_vel_orig=None, y_test_arias_orig=None, X_test=None):
    """
    Plot comprehensive residual analysis to understand prediction errors and biases
    Now includes analysis against static features to detect model bias
    """
    try:
        configure_plot_style()
        
        # If original values not provided, create them from log values (for dummy data)
        if y_test_accel_orig is None:
            y_test_accel_orig = 10**y_test_accel
        if y_test_vel_orig is None:
            y_test_vel_orig = 10**y_test_vel
        if y_test_arias_orig is None:
            y_test_arias_orig = 10**y_test_arias
        
        # Calculate residuals (in log space)
        accel_residuals = y_test_accel - y_pred_accel
        vel_residuals = y_test_vel - y_pred_vel
        arias_residuals = y_test_arias - y_pred_arias
        
        # Calculate relative errors (in original space)
        accel_orig_pred = 10**y_pred_accel
        vel_orig_pred = 10**y_pred_vel
        arias_orig_pred = 10**y_pred_arias
        
        accel_rel_error = (y_test_accel_orig - accel_orig_pred) / np.maximum(y_test_accel_orig, 1e-10) * 100
        vel_rel_error = (y_test_vel_orig - vel_orig_pred) / np.maximum(y_test_vel_orig, 1e-10) * 100
        arias_rel_error = (y_test_arias_orig - arias_orig_pred) / np.maximum(y_test_arias_orig, 1e-10) * 100
        
        # Figure 1: Residual Histograms (log space)
        plt.figure(figsize=(10, 8))
        
        plt.hist(accel_residuals, bins=50, alpha=0.7, color='#1f77b4', label='Acceleration')
        plt.hist(vel_residuals, bins=50, alpha=0.7, color='#ff7f0e', label='Velocity')
        plt.hist(arias_residuals, bins=50, alpha=0.7, color='#2ca02c', label='Arias')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5, label='Zero Error')
        
        plt.xlabel('Residual (Log Space)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Distribution of Residuals (Log Space)', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/residual_histogram.png'), dpi=300)
        plt.close()
        
        # Only run the static feature analysis if X_test is provided
        if X_test is not None and X_test.shape[0] == y_test_accel.shape[0]:
            # Extract static features from X_test
            static_feature_names = [
                "Magnitude", "PGA", "Depth", "Repi", "Stories",
                "Total_Height", "Base_Length", "Base_Breadth",
                "Typical_Floor_Length", "Typical_Floor_Breadth"
            ]
            
            # For dummy data, just create some random static features for visualization
            if X_test.shape[1] <= 30:  # If X_test doesn't have enough columns
                log_diagnostic("Creating dummy static features for visualization")
                static_features = np.random.rand(X_test.shape[0], len(static_feature_names))
            else:
                # Now plot residuals against each static feature
                static_features = X_test[:, 20:30]  # Adjust these indices to match your data
            
            # Plot residuals vs key static features
            for i, feature_name in enumerate(static_feature_names):
                if i >= static_features.shape[1]:
                    continue  # Skip if index out of bounds
                    
                plt.figure(figsize=(10, 8))
                
                # Create scatter plots with alpha for density visualization
                plt.scatter(static_features[:, i], accel_residuals, alpha=0.5, color='#1f77b4', label='Acceleration')
                plt.scatter(static_features[:, i], vel_residuals, alpha=0.5, color='#ff7f0e', label='Velocity')
                plt.scatter(static_features[:, i], arias_residuals, alpha=0.5, color='#2ca02c', label='Arias')
                
                # Draw a horizontal line at y=0 for reference
                plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
                
                # Calculate and plot trend lines to better visualize bias
                def plot_trend(x, y, color):
                    # Remove NaN values
                    mask = ~np.isnan(x) & ~np.isnan(y)
                    if np.sum(mask) > 2:  # Need at least 3 points for regression
                        z = np.polyfit(x[mask], y[mask], 1)
                        p = np.poly1d(z)
                        x_sorted = np.sort(x[mask])
                        plt.plot(x_sorted, p(x_sorted), color=color, linestyle='-', linewidth=2)
                        # Add correlation text
                        corr = np.corrcoef(x[mask], y[mask])[0,1]
                        plt.text(0.05, 0.95 - 0.05*i, f'Corr: {corr:.3f}', 
                                transform=plt.gca().transAxes, color=color, fontsize=10)
                
                plot_trend(static_features[:, i], accel_residuals, '#1f77b4')
                plot_trend(static_features[:, i], vel_residuals, '#ff7f0e')
                plot_trend(static_features[:, i], arias_residuals, '#2ca02c')
                
                plt.xlabel(feature_name, fontsize=14)
                plt.ylabel('Residual (Log Space)', fontsize=14)
                plt.title(f'Residuals vs {feature_name}', fontsize=16)
                plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
                plt.legend(fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.expanduser(f'~/Desktop/fyp 2_debug1/residual_vs_{feature_name}.png'), dpi=300)
                plt.close()
        
        # Figure: Residuals vs Predicted Values (for all three measures)
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_pred_accel, accel_residuals, alpha=0.5, color='#1f77b4', label='Acceleration')
        plt.scatter(y_pred_vel, vel_residuals, alpha=0.5, color='#ff7f0e', label='Velocity')
        plt.scatter(y_pred_arias, arias_residuals, alpha=0.5, color='#2ca02c', label='Arias')
        
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
        
        plt.xlabel('Predicted Value (Log Space)', fontsize=14)
        plt.ylabel('Residual (Log Space)', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Residuals vs Predicted Values', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/residuals_vs_predicted.png'), dpi=300)
        plt.close()
        
        # Figure: Relative Error Histograms (original space)
        plt.figure(figsize=(10, 8))
        
        # Remove extreme outliers for better visualization
        accel_rel_error_filtered = np.clip(accel_rel_error, -100, 100)
        vel_rel_error_filtered = np.clip(vel_rel_error, -100, 100)
        arias_rel_error_filtered = np.clip(arias_rel_error, -100, 100)
        
        plt.hist(accel_rel_error_filtered, bins=50, alpha=0.7, color='#1f77b4', label='Acceleration')
        plt.hist(vel_rel_error_filtered, bins=50, alpha=0.7, color='#ff7f0e', label='Velocity')
        plt.hist(arias_rel_error_filtered, bins=50, alpha=0.7, color='#2ca02c', label='Arias')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5, label='Zero Error')
        
        plt.xlabel('Relative Error (%)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Distribution of Relative Errors (Original Space)', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/relative_error_histogram.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        log_diagnostic(f"Error in plot_residual_analysis: {str(e)}")
        traceback.print_exc()

def plot_between_within_event_residuals(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                        y_test_arias, y_pred_arias, event_ids=None):
    """Plot between-event and within-event residual distributions"""
    try:
        configure_plot_style()
        
        # Create dummy event_ids if not provided (for testing)
        if event_ids is None:
            log_diagnostic("Creating dummy event IDs for visualization")
            num_samples = len(y_test_accel)
            # Generate random event IDs - 200 events with balanced samples
            num_events = min(200, num_samples // 5)  # 5 samples per event minimum
            event_ids = np.array([i % num_events for i in range(num_samples)])
        
        log_diagnostic(f"Analyzing residuals across {len(np.unique(event_ids))} earthquake events")
        
        # Calculate residuals
        residuals_accel = y_test_accel - y_pred_accel
        residuals_vel = y_test_vel - y_pred_vel
        residuals_arias = y_test_arias - y_pred_arias
        
        # Calculate between-event and within-event terms for each IM
        between_event_accel = []
        within_event_accel = []
        between_event_vel = []
        within_event_vel = []
        between_event_arias = []
        within_event_arias = []
        
        unique_events = np.unique(event_ids)
        
        for event in unique_events:
            event_mask = (event_ids == event)
            
            # For each event, calculate the mean residual (between-event term)
            mean_residual_accel = np.mean(residuals_accel[event_mask])
            mean_residual_vel = np.mean(residuals_vel[event_mask])
            mean_residual_arias = np.mean(residuals_arias[event_mask])
            
            # Store between-event terms
            between_event_accel.append(mean_residual_accel)
            between_event_vel.append(mean_residual_vel)
            between_event_arias.append(mean_residual_arias)
            
            # Calculate within-event residuals (record-to-record variability)
            within_accel = residuals_accel[event_mask] - mean_residual_accel
            within_vel = residuals_vel[event_mask] - mean_residual_vel
            within_arias = residuals_arias[event_mask] - mean_residual_arias
            
            # Store within-event terms
            within_event_accel.extend(within_accel)
            within_event_vel.extend(within_vel)
            within_event_arias.extend(within_arias)
        
        # Convert to numpy arrays for statistical calculations
        between_event_accel = np.array(between_event_accel)
        within_event_accel = np.array(within_event_accel)
        between_event_vel = np.array(between_event_vel)
        within_event_vel = np.array(within_event_vel)
        between_event_arias = np.array(between_event_arias)
        within_event_arias = np.array(within_event_arias)
        
        # Calculate standard deviations (tau and phi)
        tau_accel = np.std(between_event_accel)
        phi_accel = np.std(within_event_accel)
        tau_vel = np.std(between_event_vel)
        phi_vel = np.std(within_event_vel)
        tau_arias = np.std(between_event_arias)
        phi_arias = np.std(within_event_arias)
        
        # Calculate total standard deviation (sigma)
        sigma_accel = np.sqrt(tau_accel**2 + phi_accel**2)
        sigma_vel = np.sqrt(tau_vel**2 + phi_vel**2)
        sigma_arias = np.sqrt(tau_arias**2 + phi_arias**2)
        
        # Plot histograms for acceleration
        plt.figure(figsize=(10, 8))
        
        # For each plot, we'll create a separate histogram rather than trying to plot them all at once
        plt.hist(between_event_accel, bins=30, alpha=0.5, color='#1f77b4', label=f'Between (τ={tau_accel:.3f})')
        plt.hist(within_event_accel, bins=30, alpha=0.5, color='#ff7f0e', label=f'Within (φ={phi_accel:.3f})')
        plt.hist(residuals_accel, bins=30, alpha=0.3, color='#2ca02c', label=f'Total (σ={sigma_accel:.3f})')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
        plt.xlabel('Residual (Log Space)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Between-Event and Within-Event Residuals - Acceleration', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/between_within_acceleration.png'), dpi=300)
        plt.close()
        
        # Plot histograms for velocity
        plt.figure(figsize=(10, 8))
        
        plt.hist(between_event_vel, bins=30, alpha=0.5, color='#1f77b4', label=f'Between (τ={tau_vel:.3f})')
        plt.hist(within_event_vel, bins=30, alpha=0.5, color='#ff7f0e', label=f'Within (φ={phi_vel:.3f})')
        plt.hist(residuals_vel, bins=30, alpha=0.3, color='#2ca02c', label=f'Total (σ={sigma_vel:.3f})')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
        plt.xlabel('Residual (Log Space)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Between-Event and Within-Event Residuals - Velocity', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/between_within_velocity.png'), dpi=300)
        plt.close()
        
        # Plot histograms for arias
        plt.figure(figsize=(10, 8))
        
        plt.hist(between_event_arias, bins=30, alpha=0.5, color='#1f77b4', label=f'Between (τ={tau_arias:.3f})')
        plt.hist(within_event_arias, bins=30, alpha=0.5, color='#ff7f0e', label=f'Within (φ={phi_arias:.3f})')
        plt.hist(residuals_arias, bins=30, alpha=0.3, color='#2ca02c', label=f'Total (σ={sigma_arias:.3f})')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
        plt.xlabel('Residual (Log Space)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Between-Event and Within-Event Residuals - Arias', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/between_within_arias.png'), dpi=300)
        plt.close()
        
        # Log the results
        log_diagnostic("Between-Event and Within-Event Variability:")
        log_diagnostic(f"Acceleration: τ={tau_accel:.3f}, φ={phi_accel:.3f}, σ={sigma_accel:.3f}")
        log_diagnostic(f"Velocity: τ={tau_vel:.3f}, φ={phi_vel:.3f}, σ={sigma_vel:.3f}")
        log_diagnostic(f"Arias: τ={tau_arias:.3f}, φ={phi_arias:.3f}, σ={sigma_arias:.3f}")
        
    except Exception as e:
        log_diagnostic(f"Error in between-within event residual analysis: {str(e)}")
        traceback.print_exc()

def plot_residual_vs_parameters(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                              y_test_arias, y_pred_arias, X_test=None):
    """Plot residuals against parameters to identify potential biases"""
    try:
        configure_plot_style()
        
        # Calculate residuals
        accel_residuals = y_test_accel - y_pred_accel
        vel_residuals = y_test_vel - y_pred_vel
        arias_residuals = y_test_arias - y_pred_arias
        
        # If X_test is not provided or has wrong shape, create dummy data for visualization
        if X_test is None or X_test.shape[0] != len(accel_residuals):
            log_diagnostic("Creating dummy parameter data for visualization")
            num_samples = len(accel_residuals)
            X_test = np.random.rand(num_samples, 10)  # Create 10 dummy parameters
            
            # Generate more interpretable parameters
            param_names = ['Magnitude', 'Distance', 'PGA', 'Depth', 'Height', 
                          'Stories', 'Base Length', 'Base Width', 'Sensor Height', 'VS30']
        else:
            # Use a subset of the actual parameters if available
            param_names = ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5',
                          'Parameter 6', 'Parameter 7', 'Parameter 8', 'Parameter 9', 'Parameter 10']
        
        # Plot residuals vs each parameter
        for i, param_name in enumerate(param_names):
            if i >= X_test.shape[1]:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.scatter(X_test[:, i], accel_residuals, alpha=0.7, color='#3498db', s=60)
            
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
            
            # Add trend line
            mask = ~np.isnan(X_test[:, i]) & ~np.isnan(accel_residuals)
            if np.sum(mask) > 2:
                z = np.polyfit(X_test[mask, i], accel_residuals[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(X_test[mask, i])
                plt.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
                
                # Calculate and display correlation
                corr = np.corrcoef(X_test[mask, i], accel_residuals[mask])[0,1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel(param_name, fontsize=14)
            plt.ylabel('Acceleration Residual (Log Space)', fontsize=14)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.title(f'Acceleration Residuals vs {param_name}', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser(f'~/Desktop/fyp 2_debug1/accel_residual_vs_{param_name.replace(" ", "_")}.png'), dpi=300)
            plt.close()
            
            # Do the same for velocity
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test[:, i], vel_residuals, alpha=0.7, color='#e74c3c', s=60)
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
            
            mask = ~np.isnan(X_test[:, i]) & ~np.isnan(vel_residuals)
            if np.sum(mask) > 2:
                z = np.polyfit(X_test[mask, i], vel_residuals[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(X_test[mask, i])
                plt.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
                
                corr = np.corrcoef(X_test[mask, i], vel_residuals[mask])[0,1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel(param_name, fontsize=14)
            plt.ylabel('Velocity Residual (Log Space)', fontsize=14)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.title(f'Velocity Residuals vs {param_name}', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser(f'~/Desktop/fyp 2_debug1/vel_residual_vs_{param_name.replace(" ", "_")}.png'), dpi=300)
            plt.close()
            
            # And for Arias intensity
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test[:, i], arias_residuals, alpha=0.7, color='#9b59b6', s=60)
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
            
            mask = ~np.isnan(X_test[:, i]) & ~np.isnan(arias_residuals)
            if np.sum(mask) > 2:
                z = np.polyfit(X_test[mask, i], arias_residuals[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(X_test[mask, i])
                plt.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
                
                corr = np.corrcoef(X_test[mask, i], arias_residuals[mask])[0,1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel(param_name, fontsize=14)
            plt.ylabel('Arias Intensity Residual (Log Space)', fontsize=14)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.title(f'Arias Intensity Residuals vs {param_name}', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser(f'~/Desktop/fyp 2_debug1/arias_residual_vs_{param_name.replace(" ", "_")}.png'), dpi=300)
            plt.close()
        
    except Exception as e:
        log_diagnostic(f"Error in plot_residual_vs_parameters: {str(e)}")
        traceback.print_exc()

def plot_metrics(history, X_test, y_test_accel, y_test_vel, y_test_arias, 
                y_test_accel_orig, y_test_vel_orig, y_test_arias_orig, model):
    """Generate and save performance plots for all three predictions comparing log and linear space"""
    try:
        # Predict values
        [y_pred_accel, y_pred_vel, y_pred_arias] = model.predict(X_test, batch_size=1024)
        
        # Transform predictions from log back to original scale
        y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
        # No need to transform y_test_accel_orig, it's already in original space
        
        y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
        # No need to transform y_test_vel_orig, it's already in original space
        
        y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
        # No need to transform y_test_arias_orig, it's already in original space
        
        # Create figure for basic metrics
        plt.figure(figsize=(20, 15))
        
        # 1. Acceleration: Actual vs Predicted (log space)
        plt.subplot(3, 3, 1)
        plt.scatter(y_test_accel, y_pred_accel, alpha=0.5, 
                   color='red', edgecolor='darkred', linewidth=0.5, s=40)
        plt.plot([y_test_accel.min(), y_test_accel.max()], 
                 [y_test_accel.min(), y_test_accel.max()], 'k--', linewidth=2)
        plt.xlabel('Actual Log Acceleration', fontsize=12)
        plt.ylabel('Predicted Log Acceleration', fontsize=12)
        plt.title('Log Space: Acceleration Actual vs Predicted', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 2. Velocity: Actual vs Predicted (log space)
        plt.subplot(3, 3, 2)
        plt.scatter(y_test_vel, y_pred_vel, alpha=0.5,
                   color='orangered', edgecolor='darkred', linewidth=0.5, s=40)
        plt.plot([y_test_vel.min(), y_test_vel.max()], 
                 [y_test_vel.min(), y_test_vel.max()], 'k--', linewidth=2)
        plt.xlabel('Actual Log Velocity', fontsize=12)
        plt.ylabel('Predicted Log Velocity', fontsize=12)
        plt.title('Log Space: Velocity Actual vs Predicted', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 3. Arias: Actual vs Predicted (log space)
        plt.subplot(3, 3, 3)
        plt.scatter(y_test_arias, y_pred_arias, alpha=0.5,
                   color='firebrick', edgecolor='darkred', linewidth=0.5, s=40)
        plt.plot([y_test_arias.min(), y_test_arias.max()], 
                 [y_test_arias.min(), y_test_arias.max()], 'k--', linewidth=2)
        plt.xlabel('Actual Log Arias Intensity', fontsize=12)
        plt.ylabel('Predicted Log Arias Intensity', fontsize=12)
        plt.title('Log Space: Arias Intensity Actual vs Predicted', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 4. Acceleration: Actual vs Predicted (linear space)
        plt.subplot(3, 3, 4)
        plt.scatter(y_test_accel_orig, y_pred_accel_orig, alpha=0.5, 
                   color='red', edgecolor='darkred', linewidth=0.5, s=40)
        plt.plot([y_test_accel_orig.min(), y_test_accel_orig.max()], 
                 [y_test_accel_orig.min(), y_test_accel_orig.max()], 'k--', linewidth=2)
        plt.xlabel('Actual Acceleration (cm/s²)', fontsize=12)
        plt.ylabel('Predicted Acceleration (cm/s²)', fontsize=12)
        plt.title('Linear Space: Acceleration Actual vs Predicted', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 5. Velocity: Actual vs Predicted (linear space)
        plt.subplot(3, 3, 5)
        plt.scatter(y_test_vel_orig, y_pred_vel_orig, alpha=0.5,
                   color='orangered', edgecolor='darkred', linewidth=0.5, s=40)
        plt.plot([y_test_vel_orig.min(), y_test_vel_orig.max()], 
                 [y_test_vel_orig.min(), y_test_vel_orig.max()], 'k--', linewidth=2)
        plt.xlabel('Actual Velocity (cm/s)', fontsize=12)
        plt.ylabel('Predicted Velocity (cm/s)', fontsize=12)
        plt.title('Linear Space: Velocity Actual vs Predicted', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 6. Arias: Actual vs Predicted (linear space)
        plt.subplot(3, 3, 6)
        plt.scatter(y_test_arias_orig, y_pred_arias_orig, alpha=0.5,
                   color='firebrick', edgecolor='darkred', linewidth=0.5, s=40)
        plt.plot([y_test_arias_orig.min(), y_test_arias_orig.max()], 
                 [y_test_arias_orig.min(), y_test_arias_orig.max()], 'k--', linewidth=2)
        plt.xlabel('Actual Arias Intensity (m/s)', fontsize=12)
        plt.ylabel('Predicted Arias Intensity (m/s)', fontsize=12)
        plt.title('Linear Space: Arias Intensity Actual vs Predicted', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 7-9. Training Losses
        plt.subplot(3, 3, 7)
        plt.plot(history.history['output_accel_loss'], label='Acceleration Loss')
        plt.plot(history.history['val_output_accel_loss'], label='Val Acceleration Loss')
        plt.title('Acceleration Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(3, 3, 8)
        plt.plot(history.history['output_vel_loss'], label='Velocity Loss')
        plt.plot(history.history['val_output_vel_loss'], label='Val Velocity Loss')
        plt.title('Velocity Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(3, 3, 9)
        plt.plot(history.history['output_arias_loss'], label='Arias Loss')
        plt.plot(history.history['val_output_arias_loss'], label='Val Arias Loss')
        plt.title('Arias Intensity Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/model_performance.png'), dpi=300)
        plt.close()
        
        # Generate additional comparison plots with error handling for each
        try:
            compare_log_vs_original(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                   y_test_arias, y_pred_arias, 
                                   y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
        except Exception as e:
            log_diagnostic(f"Warning: Failed to create log vs original comparison: {e}")
        
        try:
            compare_train_test_performance(history, X_test, y_test_accel, y_test_vel, y_test_arias, model)
        except Exception as e:
            log_diagnostic(f"Warning: Failed to create train-test performance comparison: {e}")
        
        try:
            plot_floor_predictions(X_test, y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, y_test_arias, y_pred_arias,
                                  y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
        except Exception as e:
            log_diagnostic(f"Warning: Failed to create floor predictions: {e}")
        
        # Generate advanced analysis plots with improved visuals
        try:
            plot_residual_analysis(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                  y_test_arias, y_pred_arias, X_test)
        except Exception as e:
            log_diagnostic(f"Warning: Failed to create residual analysis: {e}")
        
        try:
            plot_im_correlations(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                y_test_arias, y_pred_arias,
                                y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
        except Exception as e:
            log_diagnostic(f"Warning: Failed to create IM correlations: {e}")
        
    except Exception as e:
        log_diagnostic(f"Error in plot_metrics: {e}")
        traceback.print_exc()

def long_training_lr_schedule(initial_lr=0.0003, min_lr=1e-6, epochs=100):
    """
    Create a learning rate schedule for long training runs.
    Includes a short warmup period followed by cosine decay.
    
    Args:
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate at the end of training
        epochs: Total number of training epochs
        
    Returns:
        Schedule function for LearningRateScheduler
    """
    def schedule(epoch):
        # Warmup period for first 5% of training
        warmup_epochs = int(epochs * 0.05)
        
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * ((epoch + 1) / warmup_epochs)
        else:
            # Cosine decay for remaining epochs
            decay_epochs = epochs - warmup_epochs
            decay_epoch = epoch - warmup_epochs
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_epochs))
            return min_lr + (initial_lr - min_lr) * cosine_decay
    
    return schedule

def progressive_train(model, X_train, y_train_accel, y_train_vel, y_train_arias, 
                     X_val, y_val_accel, y_val_vel, y_val_arias,
                     initial_groups=20, final_groups=177, stages=4):
    """Progressive training implementation for earthquake data"""
    # Calculate group sizes for each stage
    group_sizes = np.linspace(initial_groups, final_groups, stages, dtype=int)
    
    for stage, num_groups in enumerate(group_sizes):
        print(f"\nStage {stage + 1}/{stages}: Training on {num_groups} groups")
        
        # Select subset of data for this stage
        train_indices = get_group_indices(X_train, num_groups)
        X_train_subset = X_train[train_indices]
        y_train_accel_subset = y_train_accel[train_indices]
        y_train_vel_subset = y_train_vel[train_indices]
        y_train_arias_subset = y_train_arias[train_indices]
        
        # Calculate learning rate for this stage using square root decay
        lr = 0.0003 * (1 / np.sqrt(stage + 1))
        K.set_value(model.optimizer.learning_rate, lr)
        
        # Create learning rate scheduler for this stage
        lr_scheduler = LearningRateScheduler(
            lambda epoch: lr * (1 - epoch/25)**0.9,  # Polynomial decay within stage
            verbose=1
        )
        
        # Train for this stage
        history = model.fit(
            X_train_subset,
            [y_train_accel_subset, y_train_vel_subset, y_train_arias_subset],
            validation_data=(X_val, [y_val_accel, y_val_vel, y_val_arias]),
            epochs=25,
            batch_size=1024,
            callbacks=[
                EarlyStopping(
                    patience=5, 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                lr_scheduler
            ]
        )
        
        # Save checkpoint after each stage
        model.save(f'model_checkpoint_stage_{stage+1}.keras')
        
        # Log performance for this stage
        val_metrics = model.evaluate(
            X_val, [y_val_accel, y_val_vel, y_val_arias],
            verbose=0
        )
        print(f"Stage {stage + 1} validation metrics:", val_metrics)
        print(f"Current learning rate: {K.get_value(model.optimizer.learning_rate)}")
    
    return history

def get_group_indices(X, num_groups):
    """Helper function to get indices for specified number of groups"""
    # Assuming first column contains group identifiers
    groups = np.unique(X[:, 0])
    selected_groups = groups[:num_groups]
    return np.isin(X[:, 0], selected_groups)

# Modify train_model_main to use progressive training
def train_model_main(data_path, debug_mode=False, epochs=100, batch_size=512, max_groups=None, use_progressive=True):
    """Main function to train the model with option for progressive training"""
    try:
        # Load and prepare data
        X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig = load_and_prepare_data(
            data_path, debug_mode=debug_mode, max_groups=max_groups
        )
        
        # Log data shape information
        log_diagnostic(f"Loaded data shapes: X={X.shape}, y_accel={y_accel.shape}, y_vel={y_vel.shape}, y_arias={y_arias.shape}")
        
        # Extract group information (assuming first few columns contain STID and Earthquake Name)
        # Modify these indices based on your actual data structure
        group_cols = X[:, :2]  # First 2 columns for grouping
        unique_groups = np.unique(group_cols, axis=0)
        num_groups = len(unique_groups)
        
        log_diagnostic(f"Total number of unique building-earthquake groups: {num_groups}")
        
        # Create group-wise splits to preserve data structure
        np.random.seed(42)  # For reproducibility
        group_indices = np.arange(num_groups)
        np.random.shuffle(group_indices)
        
        # New split ratios: 70% train, 15% validation, 15% test
        train_groups = unique_groups[group_indices[:int(0.7 * num_groups)]]
        val_groups = unique_groups[group_indices[int(0.7 * num_groups):int(0.85 * num_groups)]]
        test_groups = unique_groups[group_indices[int(0.85 * num_groups):]]
        
        # Create masks for each split
        def create_mask(groups, group_cols):
            mask = np.zeros(len(group_cols), dtype=bool)
            for group in groups:
                group_mask = np.all(group_cols == group, axis=1)
                mask = mask | group_mask
            return mask
        
        train_mask = create_mask(train_groups, group_cols)
        val_mask = create_mask(val_groups, group_cols)
        test_mask = create_mask(test_groups, group_cols)
        
        # Split data using masks
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        # Acceleration
        y_train_accel = y_accel[train_mask]
        y_val_accel = y_accel[val_mask]
        y_test_accel = y_accel[test_mask]
        
        # Velocity
        y_train_vel = y_vel[train_mask]
        y_val_vel = y_vel[val_mask]
        y_test_vel = y_vel[test_mask]  # Fixed: was incorrectly referencing itself
        
        # Arias
        y_train_arias = y_arias[train_mask]
        y_val_arias = y_arias[val_mask]
        y_test_arias = y_arias[test_mask]
        
        # Original acceleration
        y_train_accel_orig = y_accel_orig[train_mask]
        y_val_accel_orig = y_accel_orig[val_mask]
        y_test_accel_orig = y_accel_orig[test_mask]
        
        # Original velocity
        y_train_vel_orig = y_vel_orig[train_mask]
        y_val_vel_orig = y_vel_orig[val_mask]
        y_test_vel_orig = y_vel_orig[test_mask]  # Fixed: was incorrectly referencing itself
        
        # Original arias
        y_train_arias_orig = y_arias_orig[train_mask]
        y_val_arias_orig = y_arias_orig[val_mask]
        y_test_arias_orig = y_arias_orig[test_mask]
        
        # Log split sizes
        log_diagnostic(f"Training set size: {len(X_train)} samples from {len(train_groups)} groups")
        log_diagnostic(f"Validation set size: {len(X_val)} samples from {len(val_groups)} groups")
        log_diagnostic(f"Test set size: {len(X_test)} samples from {len(test_groups)} groups")
        
        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Save scaler for future use
        scaler_path = os.path.expanduser('~/Desktop/fyp 2_debug1/models/feature_scaler.pkl')
        joblib.dump(scaler_X, scaler_path)
        log_diagnostic(f"Feature scaler saved to {scaler_path}")
        
        # Rest of the function remains the same...
        # Build model with optimized learning rate
        model = build_transformer_model(X_train_scaled.shape[1], learning_rate=0.0005)
        
        if use_progressive:
            # Use progressive training
            log_diagnostic("Starting progressive training...")
            history = progressive_train(
                model=model,
                X_train=X_train_scaled,
                y_train_accel=y_train_accel,
                y_train_vel=y_train_vel,
                y_train_arias=y_train_arias,
                X_val=X_val_scaled,
                y_val_accel=y_val_accel,
                y_val_vel=y_val_vel,
                y_val_arias=y_val_arias,
                initial_groups=20,
                final_groups=177 if not debug_mode else max_groups,
                stages=4
            )
        else:
            # Use regular training
            # Create metrics logger
            metrics_logger = MetricsLogger(
                X_val=X_val_scaled, 
                y_val_accel=y_val_accel, 
                y_val_vel=y_val_vel, 
                y_val_arias=y_val_arias,
                y_val_accel_orig=y_val_accel_orig,
                y_val_vel_orig=y_val_vel_orig,
                y_val_arias_orig=y_val_arias_orig,
                log_interval=1
            )
            
            # Enhanced early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.0001,
                mode='min',
                verbose=1
            )
            
            # Use a fixed learning rate instead of a scheduler
            # Keep ReduceLROnPlateau to reduce learning rate if needed
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
            
            # Model checkpointing
            checkpoint = ModelCheckpoint(
                filepath=os.path.expanduser('~/Desktop/fyp 2_debug1/models/model_epoch_{epoch:02d}_valloss_{val_loss:.4f}.keras'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
            
            log_diagnostic(f"Starting model training for {epochs} epochs with batch size {batch_size}...")
            
            # Train model with all callbacks
            history = model.fit(
                X_train_scaled, 
                [y_train_accel, y_train_vel, y_train_arias],
                validation_data=(X_val_scaled, [y_val_accel, y_val_vel, y_val_arias]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    metrics_logger,
                    early_stopping,
                    reduce_lr,
                    checkpoint
                ],
                verbose=2
            )
            
            log_diagnostic("Model training complete")
        
        # Generate detailed evaluation metrics and plots
        try:
            log_diagnostic("Generating performance metrics and plots...")
            plot_metrics(history, X_test_scaled, 
                        y_test_accel, y_test_vel, y_test_arias,
                        y_test_accel_orig, y_test_vel_orig, y_test_arias_orig,
                        model)
            log_diagnostic("Performance visualization complete")
        except Exception as e:
            log_diagnostic(f"Error generating plots: {e}")
        
        # Calculate final metrics for saving
        [y_pred_accel, y_pred_vel, y_pred_arias] = model.predict(X_test_scaled)
        
        # Calculate R² scores
        r2_accel = r2_score(y_test_accel, y_pred_accel)
        r2_vel = r2_score(y_test_vel, y_pred_vel)
        r2_arias = r2_score(y_test_arias, y_pred_arias)
        
        # Convert predictions to original space
        y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
        y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
        y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
        
        # Calculate R² scores in original space
        r2_accel_orig = r2_score(y_test_accel_orig, y_pred_accel_orig)
        r2_vel_orig = r2_score(y_test_vel_orig, y_pred_vel_orig)
        r2_arias_orig = r2_score(y_test_arias_orig, y_pred_arias_orig)
        
        # Prepare metrics dictionary
        metrics = {
            'r2_scores_log': {
                'acceleration': r2_accel,
                'velocity': r2_vel,
                'arias': r2_arias
            },
            'r2_scores_original': {
                'acceleration': r2_accel_orig,
                'velocity': r2_vel_orig,
                'arias': r2_arias_orig
            },
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'initial_lr': 0.0005,
                'debug_mode': debug_mode,
                'max_groups': max_groups
            }
        }
        
        # Save model with metrics
        model_dir = save_model_with_metrics(
            model, 
            metrics,
            model_name="transformer_model"
        )
        
        log_diagnostic(f"Model and metrics saved to {model_dir}")
        
        return model, history
        
    except Exception as e:
        log_diagnostic(f"Error in train_model_main: {e}")
        traceback.print_exc()
        return None, None

def configure_plot_style():
    """Configure matplotlib plot style according to specifications"""
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'axes.linewidth': 1.1,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.1,
        'ytick.major.width': 1.1,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.minor.width': 1.1,
        'ytick.minor.width': 1.1,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.labelsize': 24,
        'font.family': 'Times New Roman',
        'grid.alpha': 0.6,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })

def compare_log_vs_original(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, y_test_arias, y_pred_arias, y_test_accel_orig, y_test_vel_orig, y_test_arias_orig):
    """Compare predictions in log space vs original space with proper formatting"""
    configure_plot_style()
    
    # Convert log predictions to original scale
    y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
    # No need to transform y_test_accel_orig, it's already in original space
    
    y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
    # No need to transform y_test_vel_orig, it's already in original space
    
    y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
    # No need to transform y_test_arias_orig, it's already in original space
    
    # Figure 1: Acceleration Comparison
    plt.figure(figsize=(8, 6))
    
    # Log space
    plt.scatter(y_test_accel, y_pred_accel, alpha=0.6, 
               color='#1f77b4', edgecolor='#0d3c55', s=50, marker='o', label='Log Scale')
    
    # Get limits for consistent axes
    min_val = min(y_test_accel.min(), y_pred_accel.min())
    max_val = max(y_test_accel.max(), y_pred_accel.max())
    
    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.xlabel(r'$\log_{10}$ Actual Acceleration', fontsize=24)
    plt.ylabel(r'$\log_{10}$ Predicted Acceleration', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    # Set identical limits and 5-7 ticks
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Ensure ticks start and end with labeled values
    plt.locator_params(axis='both', nbins=6)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/accel_log_comparison.png'), dpi=300)
    plt.close()
    
    # Figure 2: Velocity Comparison (log space)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_vel, y_pred_vel, alpha=0.6, 
               color='#ff7f0e', edgecolor='#8B4513', s=50, marker='s', label='Log Scale')
    
    min_val = min(y_test_vel.min(), y_pred_vel.min())
    max_val = max(y_test_vel.max(), y_pred_vel.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.xlabel(r'$\log_{10}$ Actual Velocity', fontsize=24)
    plt.ylabel(r'$\log_{10}$ Predicted Velocity', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.locator_params(axis='both', nbins=6)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/velocity_log_comparison.png'), dpi=300)
    plt.close()
    
    # Figure 3: Arias Intensity Comparison (log space)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_arias, y_pred_arias, alpha=0.6, 
               color='#2ca02c', edgecolor='#1a4314', s=50, marker='^', label='Log Scale')
    
    min_val = min(y_test_arias.min(), y_pred_arias.min())
    max_val = max(y_test_arias.max(), y_pred_arias.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.xlabel(r'$\log_{10}$ Actual Arias Intensity', fontsize=24)
    plt.ylabel(r'$\log_{10}$ Predicted Arias Intensity', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.locator_params(axis='both', nbins=6)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/arias_log_comparison.png'), dpi=300)
    plt.close()
    
    # Figure 4: Original Scale Comparisons (with log axes)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_accel_orig, y_pred_accel_orig, alpha=0.6, 
               color='#1f77b4', edgecolor='#0d3c55', s=50, marker='o', label='Acceleration')
    
    min_val = min(y_test_accel_orig.min(), y_pred_accel_orig.min()) * 0.9
    max_val = max(y_test_accel_orig.max(), y_pred_accel_orig.max()) * 1.1
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.xlabel('Actual Acceleration (cm/s²)', fontsize=24)
    plt.ylabel('Predicted Acceleration (cm/s²)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/accel_original_comparison.png'), dpi=300)
    plt.close()
    
    # Figure 5: Original Scale for Velocity
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_vel_orig, y_pred_vel_orig, alpha=0.6, 
               color='#ff7f0e', edgecolor='#8B4513', s=50, marker='s', label='Velocity')
    
    min_val = min(y_test_vel_orig.min(), y_pred_vel_orig.min()) * 0.9
    max_val = max(y_test_vel_orig.max(), y_pred_vel_orig.max()) * 1.1
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.xlabel('Actual Velocity (cm/s)', fontsize=24)
    plt.ylabel('Predicted Velocity (cm/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/velocity_original_comparison.png'), dpi=300)
    plt.close()
    
    # Figure 6: Original Scale for Arias
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_arias_orig, y_pred_arias_orig, alpha=0.6, 
               color='#2ca02c', edgecolor='#1a4314', s=50, marker='^', label='Arias Intensity')
    
    min_val = min(y_test_arias_orig.min(), y_pred_arias_orig.min()) * 0.9
    max_val = max(y_test_arias_orig.max(), y_pred_arias_orig.max()) * 1.1
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.xlabel('Actual Arias Intensity (m/s)', fontsize=24)
    plt.ylabel('Predicted Arias Intensity (m/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/arias_original_comparison.png'), dpi=300)
    plt.close()

def compare_train_test_performance(history, X_test, y_test_accel, y_test_vel, y_test_arias, model):
    """Compare training and test performance metrics"""
    configure_plot_style()
    
    try:
        # Check if history is valid
        if history is None or not hasattr(history, 'history') or not history.history:
            log_diagnostic("Warning: Training history is empty or invalid")
            return
            
        # Extract training metrics
        train_loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])
        
        # Create epochs array based on the length of loss data
        epochs = np.arange(1, len(train_loss) + 1)
        
        # Make predictions on test set
        [y_pred_accel, y_pred_vel, y_pred_arias] = model.predict(X_test, batch_size=1024)
        
        # Calculate test metrics
        accel_mse = mean_squared_error(y_test_accel, y_pred_accel)
        vel_mse = mean_squared_error(y_test_vel, y_pred_vel)
        arias_mse = mean_squared_error(y_test_arias, y_pred_arias)
        
        accel_r2 = r2_score(y_test_accel, y_pred_accel)
        vel_r2 = r2_score(y_test_vel, y_pred_vel)
        arias_r2 = r2_score(y_test_arias, y_pred_arias)
        
        # Calculate validation metrics from history
        val_accel_mse = history.history.get('val_output_accel_loss', [])
        val_vel_mse = history.history.get('val_output_vel_loss', [])
        val_arias_mse = history.history.get('val_output_arias_loss', [])
        
        # Create performance comparison plot
        plt.figure(figsize=(10, 12))
        
        # Loss plot
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.axhline(y=accel_mse + vel_mse + arias_mse, color='g', linestyle='--', 
                  label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training, Validation, and Test Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Individual metrics
        plt.subplot(2, 1, 2)
        
        if val_accel_mse and val_vel_mse and val_arias_mse:
            plt.plot(epochs, val_accel_mse, 'r-', label='Val Accel MSE')
            plt.plot(epochs, val_vel_mse, 'g-', label='Val Vel MSE')
            plt.plot(epochs, val_arias_mse, 'b-', label='Val Arias MSE')
        
        plt.axhline(y=accel_mse, color='r', linestyle='--', label=f'Test Accel MSE: {accel_mse:.4f}')
        plt.axhline(y=vel_mse, color='g', linestyle='--', label=f'Test Vel MSE: {vel_mse:.4f}')
        plt.axhline(y=arias_mse, color='b', linestyle='--', label=f'Test Arias MSE: {arias_mse:.4f}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('Component-Specific Performance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/train_test_comparison.png'), dpi=300)
        plt.close()
        
        log_diagnostic(f"Test Performance - Accel: R²={accel_r2:.4f}, Vel: R²={vel_r2:.4f}, Arias: R²={arias_r2:.4f}")
        
    except Exception as e:
        log_diagnostic(f"Warning: Failed to create train-test performance comparison: {str(e)}")
        traceback.print_exc()

def plot_floor_predictions(X_test, y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                          y_test_arias, y_pred_arias,
                          y_test_accel_orig, y_test_vel_orig, y_test_arias_orig):
    """Plot predictions by floor level"""
    configure_plot_style()
    
    # Convert to original scale for predictions only
    y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
    y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
    y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
    
    try:
        # If floor info is in X_test (modify column index as needed)
        floor_col_index = 2  # Example: column 2 contains floor information
        floor_values = X_test[:, floor_col_index]
        
        # Create floor quantiles with unique bins and handle duplicates
        try:
            # First attempt: try regular quantiles
            floor_quantiles = pd.qcut(floor_values, 5, labels=False, duplicates='drop')
        except ValueError:
            # If that fails, try linear bins instead
            floor_quantiles = pd.cut(floor_values, 
                                   bins=5, 
                                   labels=False, 
                                   duplicates='drop')
        
        # If we still have issues, create artificial categories
        if floor_quantiles is None or len(np.unique(floor_quantiles)) < 2:
            log_diagnostic("Creating artificial floor categories based on rank")
            floor_ranks = pd.Series(floor_values).rank(method='first')
            floor_quantiles = pd.qcut(floor_ranks, 5, labels=False, duplicates='drop')
        
        # Verify we have valid categories before plotting
        unique_categories = np.unique(floor_quantiles)
        if len(unique_categories) < 2:
            log_diagnostic("Not enough unique floor categories for meaningful plots")
            return
            
        # Plot 1: Acceleration predictions by floor
        try:
            plt.figure(figsize=(8, 6))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            markers = ['o', 's', '^', 'D', 'P']
            
            for floor in unique_categories:
                mask = floor_quantiles == floor
                if np.sum(mask) > 0:
                    plt.scatter(
                        y_test_accel_orig[mask], 
                        y_pred_accel_orig[mask], 
                        alpha=0.6,
                        color=colors[int(floor) % len(colors)], 
                        marker=markers[int(floor) % len(markers)],
                        s=50, 
                        label=f'Floor Group {int(floor)+1}'
                    )
            
            # Add perfect prediction line
            min_val = min(y_test_accel_orig.min(), y_pred_accel_orig.min())
            max_val = max(y_test_accel_orig.max(), y_pred_accel_orig.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Actual Acceleration (cm/s²)', fontsize=24)
            plt.ylabel('Predicted Acceleration (cm/s²)', fontsize=24)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/floor_acceleration_predictions.png'), dpi=300)
            plt.close()
        except Exception as e:
            log_diagnostic(f"Failed to create acceleration floor predictions plot: {str(e)}")
            plt.close()
        
        # Plot 2: Velocity predictions by floor
        try:
            plt.figure(figsize=(8, 6))
            
            for floor in unique_categories:
                mask = floor_quantiles == floor
                if np.sum(mask) > 0:
                    plt.scatter(
                        y_test_vel_orig[mask], 
                        y_pred_vel_orig[mask], 
                        alpha=0.6,
                        color=colors[int(floor) % len(colors)], 
                        marker=markers[int(floor) % len(markers)],
                        s=50, 
                        label=f'Floor Group {int(floor)+1}'
                    )
            
            # Add perfect prediction line
            min_val = min(y_test_vel_orig.min(), y_pred_vel_orig.min())
            max_val = max(y_test_vel_orig.max(), y_pred_vel_orig.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Actual Velocity (cm/s)', fontsize=24)
            plt.ylabel('Predicted Velocity (cm/s)', fontsize=24)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/floor_velocity_predictions.png'), dpi=300)
            plt.close()
        except Exception as e:
            log_diagnostic(f"Failed to create velocity floor predictions plot: {str(e)}")
            plt.close()
        
        # Plot 3: Arias predictions by floor
        try:
            plt.figure(figsize=(8, 6))
            
            for floor in unique_categories:
                mask = floor_quantiles == floor
                if np.sum(mask) > 0:
                    plt.scatter(
                        y_test_arias_orig[mask], 
                        y_pred_arias_orig[mask], 
                        alpha=0.6,
                        color=colors[int(floor) % len(colors)], 
                        marker=markers[int(floor) % len(markers)],
                        s=50, 
                        label=f'Floor Group {int(floor)+1}'
                    )
            
            # Add perfect prediction line
            min_val = min(y_test_arias_orig.min(), y_pred_arias_orig.min())
            max_val = max(y_test_arias_orig.max(), y_pred_arias_orig.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Actual Arias Intensity (m/s)', fontsize=24)
            plt.ylabel('Predicted Arias Intensity (m/s)', fontsize=24)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/floor_arias_predictions.png'), dpi=300)
            plt.close()
        except Exception as e:
            log_diagnostic(f"Failed to create arias floor predictions plot: {str(e)}")
            plt.close()
            
    except Exception as e:
        log_diagnostic(f"Error in floor prediction plots: {str(e)}")
        plt.close()  # Ensure any open figures are closed
        return  # Continue with other plots

def plot_between_within_event_residuals(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                        y_test_arias, y_pred_arias, event_ids=None):
    """Plot between-event and within-event residual distributions"""
    try:
        configure_plot_style()
        
        # Create dummy event_ids if not provided (for testing)
        if event_ids is None:
            log_diagnostic("Creating dummy event IDs for visualization")
            num_samples = len(y_test_accel)
            # Generate random event IDs - 200 events with balanced samples
            num_events = min(200, num_samples // 5)  # 5 samples per event minimum
            event_ids = np.array([i % num_events for i in range(num_samples)])
        
        log_diagnostic(f"Analyzing residuals across {len(np.unique(event_ids))} earthquake events")
        
        # Calculate residuals
        residuals_accel = y_test_accel - y_pred_accel
        residuals_vel = y_test_vel - y_pred_vel
        residuals_arias = y_test_arias - y_pred_arias
        
        # Calculate between-event and within-event terms for each IM
        between_event_accel = []
        within_event_accel = []
        between_event_vel = []
        within_event_vel = []
        between_event_arias = []
        within_event_arias = []
        
        unique_events = np.unique(event_ids)
        
        for event in unique_events:
            event_mask = (event_ids == event)
            
            # For each event, calculate the mean residual (between-event term)
            mean_residual_accel = np.mean(residuals_accel[event_mask])
            mean_residual_vel = np.mean(residuals_vel[event_mask])
            mean_residual_arias = np.mean(residuals_arias[event_mask])
            
            # Store between-event terms
            between_event_accel.append(mean_residual_accel)
            between_event_vel.append(mean_residual_vel)
            between_event_arias.append(mean_residual_arias)
            
            # Calculate within-event residuals (record-to-record variability)
            within_accel = residuals_accel[event_mask] - mean_residual_accel
            within_vel = residuals_vel[event_mask] - mean_residual_vel
            within_arias = residuals_arias[event_mask] - mean_residual_arias
            
            # Store within-event terms
            within_event_accel.extend(within_accel)
            within_event_vel.extend(within_vel)
            within_event_arias.extend(within_arias)
        
        # Convert to numpy arrays for statistical calculations
        between_event_accel = np.array(between_event_accel)
        within_event_accel = np.array(within_event_accel)
        between_event_vel = np.array(between_event_vel)
        within_event_vel = np.array(within_event_vel)
        between_event_arias = np.array(between_event_arias)
        within_event_arias = np.array(within_event_arias)
        
        # Calculate standard deviations (tau and phi)
        tau_accel = np.std(between_event_accel)
        phi_accel = np.std(within_event_accel)
        tau_vel = np.std(between_event_vel)
        phi_vel = np.std(within_event_vel)
        tau_arias = np.std(between_event_arias)
        phi_arias = np.std(within_event_arias)
        
        # Calculate total standard deviation (sigma)
        sigma_accel = np.sqrt(tau_accel**2 + phi_accel**2)
        sigma_vel = np.sqrt(tau_vel**2 + phi_vel**2)
        sigma_arias = np.sqrt(tau_arias**2 + phi_arias**2)
        
        # Plot histograms for acceleration
        plt.figure(figsize=(10, 8))
        
        # For each plot, we'll create a separate histogram rather than trying to plot them all at once
        plt.hist(between_event_accel, bins=30, alpha=0.5, color='#1f77b4', label=f'Between (τ={tau_accel:.3f})')
        plt.hist(within_event_accel, bins=30, alpha=0.5, color='#ff7f0e', label=f'Within (φ={phi_accel:.3f})')
        plt.hist(residuals_accel, bins=30, alpha=0.3, color='#2ca02c', label=f'Total (σ={sigma_accel:.3f})')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
        plt.xlabel('Residual (Log Space)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Between-Event and Within-Event Residuals - Acceleration', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/between_within_acceleration.png'), dpi=300)
        plt.close()
        
        # Plot histograms for velocity
        plt.figure(figsize=(10, 8))
        
        plt.hist(between_event_vel, bins=30, alpha=0.5, color='#1f77b4', label=f'Between (τ={tau_vel:.3f})')
        plt.hist(within_event_vel, bins=30, alpha=0.5, color='#ff7f0e', label=f'Within (φ={phi_vel:.3f})')
        plt.hist(residuals_vel, bins=30, alpha=0.3, color='#2ca02c', label=f'Total (σ={sigma_vel:.3f})')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
        plt.xlabel('Residual (Log Space)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Between-Event and Within-Event Residuals - Velocity', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/between_within_velocity.png'), dpi=300)
        plt.close()
        
        # Plot histograms for arias
        plt.figure(figsize=(10, 8))
        
        plt.hist(between_event_arias, bins=30, alpha=0.5, color='#1f77b4', label=f'Between (τ={tau_arias:.3f})')
        plt.hist(within_event_arias, bins=30, alpha=0.5, color='#ff7f0e', label=f'Within (φ={phi_arias:.3f})')
        plt.hist(residuals_arias, bins=30, alpha=0.3, color='#2ca02c', label=f'Total (σ={sigma_arias:.3f})')
        
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
        plt.xlabel('Residual (Log Space)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.title('Between-Event and Within-Event Residuals - Arias', fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/between_within_arias.png'), dpi=300)
        plt.close()
        
        # Log the results
        log_diagnostic("Between-Event and Within-Event Variability:")
        log_diagnostic(f"Acceleration: τ={tau_accel:.3f}, φ={phi_accel:.3f}, σ={sigma_accel:.3f}")
        log_diagnostic(f"Velocity: τ={tau_vel:.3f}, φ={phi_vel:.3f}, σ={sigma_vel:.3f}")
        log_diagnostic(f"Arias: τ={tau_arias:.3f}, φ={phi_arias:.3f}, σ={sigma_arias:.3f}")
        
    except Exception as e:
        log_diagnostic(f"Error in between-within event residual analysis: {str(e)}")
        traceback.print_exc()

def plot_residual_vs_parameters(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                              y_test_arias, y_pred_arias, X_test=None):
    """Plot residuals against parameters to identify potential biases"""
    try:
        configure_plot_style()
        
        # Calculate residuals
        accel_residuals = y_test_accel - y_pred_accel
        vel_residuals = y_test_vel - y_pred_vel
        arias_residuals = y_test_arias - y_pred_arias
        
        # If X_test is not provided or has wrong shape, create dummy data for visualization
        if X_test is None or X_test.shape[0] != len(accel_residuals):
            log_diagnostic("Creating dummy parameter data for visualization")
            num_samples = len(accel_residuals)
            X_test = np.random.rand(num_samples, 10)  # Create 10 dummy parameters
            
            # Generate more interpretable parameters
            param_names = ['Magnitude', 'Distance', 'PGA', 'Depth', 'Height', 
                          'Stories', 'Base Length', 'Base Width', 'Sensor Height', 'VS30']
        else:
            # Use a subset of the actual parameters if available
            param_names = ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5',
                          'Parameter 6', 'Parameter 7', 'Parameter 8', 'Parameter 9', 'Parameter 10']
        
        # Plot residuals vs each parameter
        for i, param_name in enumerate(param_names):
            if i >= X_test.shape[1]:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.scatter(X_test[:, i], accel_residuals, alpha=0.7, color='#3498db', s=60)
            
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
            
            # Add trend line
            mask = ~np.isnan(X_test[:, i]) & ~np.isnan(accel_residuals)
            if np.sum(mask) > 2:
                z = np.polyfit(X_test[mask, i], accel_residuals[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(X_test[mask, i])
                plt.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
                
                # Calculate and display correlation
                corr = np.corrcoef(X_test[mask, i], accel_residuals[mask])[0,1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel(param_name, fontsize=14)
            plt.ylabel('Acceleration Residual (Log Space)', fontsize=14)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.title(f'Acceleration Residuals vs {param_name}', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser(f'~/Desktop/fyp 2_debug1/accel_residual_vs_{param_name.replace(" ", "_")}.png'), dpi=300)
            plt.close()
            
            # Do the same for velocity
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test[:, i], vel_residuals, alpha=0.7, color='#e74c3c', s=60)
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
            
            mask = ~np.isnan(X_test[:, i]) & ~np.isnan(vel_residuals)
            if np.sum(mask) > 2:
                z = np.polyfit(X_test[mask, i], vel_residuals[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(X_test[mask, i])
                plt.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
                
                corr = np.corrcoef(X_test[mask, i], vel_residuals[mask])[0,1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel(param_name, fontsize=14)
            plt.ylabel('Velocity Residual (Log Space)', fontsize=14)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.title(f'Velocity Residuals vs {param_name}', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser(f'~/Desktop/fyp 2_debug1/vel_residual_vs_{param_name.replace(" ", "_")}.png'), dpi=300)
            plt.close()
            
            # And for Arias intensity
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test[:, i], arias_residuals, alpha=0.7, color='#9b59b6', s=60)
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
            
            mask = ~np.isnan(X_test[:, i]) & ~np.isnan(arias_residuals)
            if np.sum(mask) > 2:
                z = np.polyfit(X_test[mask, i], arias_residuals[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(X_test[mask, i])
                plt.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
                
                corr = np.corrcoef(X_test[mask, i], arias_residuals[mask])[0,1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel(param_name, fontsize=14)
            plt.ylabel('Arias Intensity Residual (Log Space)', fontsize=14)
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.title(f'Arias Intensity Residuals vs {param_name}', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser(f'~/Desktop/fyp 2_debug1/arias_residual_vs_{param_name.replace(" ", "_")}.png'), dpi=300)
            plt.close()
        
    except Exception as e:
        log_diagnostic(f"Error in plot_residual_vs_parameters: {str(e)}")
        traceback.print_exc()

def plot_im_correlations(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel,
                        y_test_arias, y_pred_arias,
                        y_test_accel_orig, y_test_vel_orig, y_test_arias_orig):
    """Plot correlations between different intensity measures"""
    configure_plot_style()
    
    # Convert to original scale for predictions only
    y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
    # No conversion for y_test_accel_orig
    
    y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
    # No conversion for y_test_vel_orig
    
    y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
    # No conversion for y_test_arias_orig
    
    # Figure 1: Acceleration vs Velocity (Actual)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_accel_orig, y_test_vel_orig, alpha=0.6, 
               color='#1f77b4', edgecolor='#0d3c55', s=50, marker='o', label='Actual')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Acceleration (cm/s²)', fontsize=24)
    plt.ylabel('Velocity (cm/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/actual_accel_vs_vel.png'), dpi=300)
    plt.close()
    
    # Figure 2: Acceleration vs Velocity (Predicted)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_pred_accel_orig, y_pred_vel_orig, alpha=0.6, 
               color='#ff7f0e', edgecolor='#8B4513', s=50, marker='s', label='Predicted')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Acceleration (cm/s²)', fontsize=24)
    plt.ylabel('Velocity (cm/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/predicted_accel_vs_vel.png'), dpi=300)
    plt.close()
    
    # Figure 3: Acceleration vs Arias (Actual)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_accel_orig, y_test_arias_orig, alpha=0.6, 
               color='#1f77b4', edgecolor='#0d3c55', s=50, marker='o', label='Actual')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Acceleration (cm/s²)', fontsize=24)
    plt.ylabel('Arias Intensity (m/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/actual_accel_vs_arias.png'), dpi=300)
    plt.close()
    
    # Figure 4: Acceleration vs Arias (Predicted)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_pred_accel_orig, y_pred_arias_orig, alpha=0.6, 
               color='#ff7f0e', edgecolor='#8B4513', s=50, marker='s', label='Predicted')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Acceleration (cm/s²)', fontsize=24)
    plt.ylabel('Arias Intensity (m/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/predicted_accel_vs_arias.png'), dpi=300)
    plt.close()
    
    # Figure 5: Velocity vs Arias (Actual)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test_vel_orig, y_test_arias_orig, alpha=0.6, 
               color='#1f77b4', edgecolor='#0d3c55', s=50, marker='o', label='Actual')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Velocity (cm/s)', fontsize=24)
    plt.ylabel('Arias Intensity (m/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/actual_vel_vs_arias.png'), dpi=300)
    plt.close()
    
    # Figure 6: Velocity vs Arias (Predicted)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_pred_vel_orig, y_pred_arias_orig, alpha=0.6, 
               color='#ff7f0e', edgecolor='#8B4513', s=50, marker='s', label='Predicted')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Velocity (cm/s)', fontsize=24)
    plt.ylabel('Arias Intensity (m/s)', fontsize=24)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/predicted_vel_vs_arias.png'), dpi=300)
    plt.close()

def plot_floor_level_analysis(X_test, y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, y_test_arias, y_pred_arias):
    """Advanced floor-level analysis including event-specific plots"""
    configure_plot_style()
    
    try:
        # Try to extract floor information and building/event IDs
        # Modify these indices based on your actual data structure
        floor_col_index = 2      # Example: column 2 contains floor information
        building_col_index = 3   # Example: column 3 contains building ID
        event_col_index = 1      # Example: column 1 contains earthquake event
        
        floor_values = X_test[:, floor_col_index]
        building_ids = X_test[:, building_col_index]
        event_ids = X_test[:, event_col_index]
        
        # If floor values aren't clear, create quantiles as a proxy
        if len(np.unique(floor_values)) < 3:
            floor_quantiles = pd.qcut(floor_values, 5, labels=False)
            floor_values = floor_quantiles
            log_diagnostic("Using floor quantiles as proxy for actual floor levels")
        
        # Prepare original scale values
        y_test_accel_orig = safe_inverse_log10(y_test_accel)
        y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
        y_test_vel_orig = safe_inverse_log10(y_test_vel)
        y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
        
        # Figure 1: Prediction accuracy by floor level
        plt.figure(figsize=(10, 8))
        
        # Create floor bins for analysis
        unique_floors = np.unique(floor_values)
        num_floors = len(unique_floors)
        
        accel_r2_by_floor = []
        vel_r2_by_floor = []
        accel_mae_by_floor = []
        vel_mae_by_floor = []
        floor_labels = []
        
        for floor in unique_floors:
            mask = (floor_values == floor)
            if np.sum(mask) >= 5:  # Only analyze floors with sufficient data
                # Calculate metrics
                accel_r2 = r2_score(y_test_accel[mask], y_pred_accel[mask])
                vel_r2 = r2_score(y_test_vel[mask], y_pred_vel[mask])
                accel_mae = mean_absolute_error(y_test_accel[mask], y_pred_accel[mask])
                vel_mae = mean_absolute_error(y_test_vel[mask], y_pred_vel[mask])
                
                accel_r2_by_floor.append(accel_r2)
                vel_r2_by_floor.append(vel_r2)
                accel_mae_by_floor.append(accel_mae)
                vel_mae_by_floor.append(vel_mae)
                floor_labels.append(f"Floor {floor}")
        
        # Plot R² by floor
        plt.subplot(2, 1, 1)
        x = np.arange(len(floor_labels))
        width = 0.35
        
        plt.bar(x - width/2, accel_r2_by_floor, width, label='Acceleration', color='#3498db')
        plt.bar(x + width/2, vel_r2_by_floor, width, label='Velocity', color='#e74c3c')
        
        plt.ylim(0, 1.0)
        plt.ylabel('R² Score')
        plt.title('Prediction Accuracy by Floor Level (R²)')
        plt.xticks(x, floor_labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot MAE by floor
        plt.subplot(2, 1, 2)
        plt.bar(x - width/2, accel_mae_by_floor, width, label='Acceleration', color='#3498db')
        plt.bar(x + width/2, vel_mae_by_floor, width, label='Velocity', color='#e74c3c')
        
        plt.ylabel('Mean Absolute Error (Log Scale)')
        plt.title('Prediction Error by Floor Level (MAE)')
        plt.xticks(x, floor_labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/floor_prediction_metrics.png'), dpi=300)
        plt.close()
        
        # Figure 2: Case study - Single building, single event
        # Find a building with multiple floor recordings for the same event
        building_event_combos = [(b, e) for b, e in zip(building_ids, event_ids)]
        unique_combos = set(building_event_combos)
        
        best_combo = None
        max_floor_count = 0
        
        for combo in unique_combos:
            b, e = combo
            mask = (building_ids == b) & (event_ids == e)
            floor_count = len(np.unique(floor_values[mask]))
            if floor_count > max_floor_count:
                max_floor_count = floor_count
                best_combo = combo
        
        # If we found a good case study
        if best_combo and max_floor_count >= 3:
            b, e = best_combo
            case_mask = (building_ids == b) & (event_ids == e)
            case_floors = floor_values[case_mask]
            case_accel_actual = y_test_accel_orig[case_mask]
            case_accel_pred = y_pred_accel_orig[case_mask]
            
            # Sort by floor
            sorted_indices = np.argsort(case_floors)
            case_floors = case_floors[sorted_indices]
            case_accel_actual = case_accel_actual[sorted_indices]
            case_accel_pred = case_accel_pred[sorted_indices]
            
            # Plot profile
            plt.figure(figsize=(8, 10))
            plt.subplot(1, 1, 1)
            
            plt.plot(case_accel_actual, case_floors, 'o-', color='#3498db', 
                    linewidth=2, markersize=10, label='Actual')
            plt.plot(case_accel_pred, case_floors, 's--', color='#e74c3c', 
                    linewidth=2, markersize=10, label='Predicted')
            
            plt.xscale('log')
            plt.xlabel('Peak Ground Acceleration (cm/s²)')
            plt.ylabel('Floor Level')
            plt.title(f'PGA Profile: Building {b}, Event {e}')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/case_study_pga_profile.png'), dpi=300)
            plt.close()
            
            # Calculate floor-to-floor amplification
            if len(case_floors) > 1:
                amplif_actual = []
                amplif_pred = []
                floor_pairs = []
                
                for i in range(1, len(case_floors)):
                    amplif_actual.append(case_accel_actual[i] / case_accel_actual[i-1])
                    amplif_pred.append(case_accel_pred[i] / case_accel_pred[i-1])
                    floor_pairs.append(f"{int(case_floors[i-1])}-{int(case_floors[i])}")
                
                # Plot amplification factors
                plt.figure(figsize=(8, 6))
                x = np.arange(len(floor_pairs))
                width = 0.35
                
                plt.bar(x - width/2, amplif_actual, width, label='Actual', color='#3498db')
                plt.bar(x + width/2, amplif_pred, width, label='Predicted', color='#e74c3c')
                
                plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.8)
                plt.ylabel('Amplification Factor')
                plt.title(f'Floor-to-Floor Amplification: Building {b}, Event {e}')
                plt.xticks(x, floor_pairs)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/case_study_amplification.png'), dpi=300)
                plt.close()
        
    except Exception as e:
        log_diagnostic(f"Error in floor level analysis: {e}")
        traceback.print_exc()

def plot_residual_vs_parameters(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                              y_test_arias, y_pred_arias, 
                              y_test_accel_orig, y_test_vel_orig, y_test_arias_orig, X_test):
    """Plot residual analysis to understand prediction errors"""
    configure_plot_style()
    
    # Calculate residuals (in log space)
    accel_residuals = y_test_accel - y_pred_accel
    vel_residuals = y_test_vel - y_pred_vel
    arias_residuals = y_test_arias - y_pred_arias
    
    # Calculate relative errors (in original space)
    y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
    # No need to transform y_test_accel_orig, it's already in original space
    
    y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
    # No need to transform y_test_vel_orig, it's already in original space
    
    y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
    # No need to transform y_test_arias_orig, it's already in original space
    
    # Figure 1: Residuals vs. Magnitude
    plt.figure(figsize=(10, 12))
    
    # Acceleration residuals
    plt.subplot(3, 1, 1)
    plt.scatter(X_test[:, 4], accel_residuals, alpha=0.7, color='#3498db', s=60)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(X_test[:, 4], accel_residuals, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(X_test[:, 4]), p(np.sort(X_test[:, 4])), 'r--', linewidth=2)
    
    plt.xlabel('Earthquake Magnitude')
    plt.ylabel('Acceleration Residual')
    plt.title('Acceleration Residuals vs. Magnitude')
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    
    # Velocity residuals
    plt.subplot(3, 1, 2)
    plt.scatter(X_test[:, 5], vel_residuals, alpha=0.7, color='#e74c3c', s=60)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(X_test[:, 5], vel_residuals, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(X_test[:, 5]), p(np.sort(X_test[:, 5])), 'r--', linewidth=2)
    
    plt.xlabel('Earthquake Magnitude')
    plt.ylabel('Velocity Residual')
    plt.title('Velocity Residuals vs. Magnitude')
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    
    # Arias residuals
    plt.subplot(3, 1, 3)
    plt.scatter(X_test[:, 6], arias_residuals, alpha=0.7, color='#2ecc71', s=60)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(X_test[:, 6], arias_residuals, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(X_test[:, 6]), p(np.sort(X_test[:, 6])), 'r--', linewidth=2)
    
    plt.xlabel('Earthquake Magnitude')
    plt.ylabel('Arias Intensity Residual')
    plt.title('Arias Intensity Residuals vs. Magnitude')
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/residuals_vs_magnitude.png'), dpi=300)
    plt.close()
    
    # Figure 2: Residuals vs. Distance
    plt.figure(figsize=(10, 12))
    
    # Acceleration residuals
    plt.subplot(3, 1, 1)
    plt.scatter(X_test[:, 7], accel_residuals, alpha=0.7, color='#3498db', s=60)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(X_test[:, 7], accel_residuals, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(X_test[:, 7]), p(np.sort(X_test[:, 7])), 'r--', linewidth=2)
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Acceleration Residual')
    plt.title('Acceleration Residuals vs. Distance')
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    
    # Log x-axis for better visualization of distance effect
    plt.xscale('log')
    
    # Velocity residuals
    plt.subplot(3, 1, 2)
    plt.scatter(X_test[:, 8], vel_residuals, alpha=0.7, color='#e74c3c', s=60)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    
    # Add trend line (for log-transformed x)
    log_distances = np.log10(X_test[:, 7])
    z = np.polyfit(log_distances, vel_residuals, 1)
    p = np.poly1d(z)
    plt.plot(X_test[:, 7], p(log_distances), 'r--', linewidth=2)
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Velocity Residual')
    plt.title('Velocity Residuals vs. Distance')
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Log x-axis for better visualization of distance effect
    plt.xscale('log')
    
    # Arias residuals
    plt.subplot(3, 1, 3)
    plt.scatter(X_test[:, 9], arias_residuals, alpha=0.7, color='#2ecc71', s=60)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    
    # Add trend line (for log-transformed x)
    z = np.polyfit(log_distances, arias_residuals, 1)
    p = np.poly1d(z)
    plt.plot(X_test[:, 7], p(log_distances), 'r--', linewidth=2)
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Arias Intensity Residual')
    plt.title('Arias Intensity Residuals vs. Distance')
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/residuals_vs_distance.png'), dpi=300)
    plt.close()
    
    # Figure 3: Residuals vs. Floor Level
    plt.figure(figsize=(10, 12))
    
    # Create floor categories (quantiles if continuous)
    if len(np.unique(X_test[:, 2])) < 10:  # If floors are already categorical
        floor_categories = X_test[:, 2]
    else:  # Create quantiles if continuous
        floor_categories = pd.qcut(X_test[:, 2], 5, labels=False)
    
    # Acceleration residuals
    plt.subplot(3, 1, 1)
    plt.boxplot([accel_residuals[floor_categories == i] for i in range(np.max(floor_categories)+1)],
              patch_artist=True,
              boxprops=dict(facecolor='#3498db', alpha=0.7),
              medianprops=dict(color='red'),
              flierprops=dict(marker='o', markerfacecolor='red', markersize=8))
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    plt.xlabel('Floor Level (Quantile)')
    plt.ylabel('Acceleration Residual')
    plt.title('Acceleration Residuals vs. Floor Level')
    plt.grid(True, which='major', linestyle='--', alpha=0.6, axis='y')
    
    # Velocity residuals
    plt.subplot(3, 1, 2)
    plt.boxplot([vel_residuals[floor_categories == i] for i in range(np.max(floor_categories)+1)],
              patch_artist=True,
              boxprops=dict(facecolor='#e74c3c', alpha=0.7),
              medianprops=dict(color='blue'),
              flierprops=dict(marker='o', markerfacecolor='blue', markersize=8))
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    plt.xlabel('Floor Level (Quantile)')
    plt.ylabel('Velocity Residual')
    plt.title('Velocity Residuals vs. Floor Level')
    plt.grid(True, which='major', linestyle='--', alpha=0.6, axis='y')
    
    # Arias residuals
    plt.subplot(3, 1, 3)
    plt.boxplot([arias_residuals[floor_categories == i] for i in range(np.max(floor_categories)+1)],
              patch_artist=True,
              boxprops=dict(facecolor='#2ecc71', alpha=0.7),
              medianprops=dict(color='purple'),
              flierprops=dict(marker='o', markerfacecolor='purple', markersize=8))
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    plt.xlabel('Floor Level (Quantile)')
    plt.ylabel('Arias Intensity Residual')
    plt.title('Arias Intensity Residuals vs. Floor Level')
    plt.grid(True, which='major', linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/fyp 2_debug1/residuals_vs_floor.png'), dpi=300)
    plt.close()

def save_model_with_metrics(model, metrics, model_name, base_dir=None):
    """Save model and metrics to disk"""
    if base_dir is None:
        base_dir = os.path.expanduser('~/Desktop/fyp 2_debug1/models')
    
    # Create unique timestamped directory for this model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{base_dir}/{model_name}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model in Keras format (.keras extension)
    model.save(f"{model_dir}/model.keras")  # Changed from save_format="tf" to .keras extension
    
    # Save metrics as JSON
    with open(f"{model_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    log_diagnostic(f"Model and metrics saved to {model_dir}")
    return model_dir

# -------------------------
# ✅ Step 5: Main Execution
# -------------------------
if __name__ == "__main__":
    try:
        # Define debug parameters
        DEBUG_MODE = True
        MAX_GROUPS = 10

        # Initialize diagnostic log
        initialize_diagnostic_log()
        
        # Define data path
        data_path = os.path.expanduser("~/Desktop/fyp 2_debug1/combined_seismic_data.csv")
        
        # Call train_model_main with progressive training enabled
        model, history = train_model_main(
            data_path=data_path, 
            debug_mode=DEBUG_MODE, 
            max_groups=MAX_GROUPS,
            use_progressive=False  # Enable progressive training
        )

        # Load the data again for plotting if we have a valid model
        if model is not None:
            log_diagnostic("Generating all visualization plots...")
            
            # Load the same data used for training with debug settings
            X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig = load_and_prepare_data(
                data_path, 
                debug_mode=DEBUG_MODE,
                max_groups=MAX_GROUPS
            )
            
            # Split the data the same way as in training
            X_train, X_test, y_train_accel, y_test_accel, y_train_vel, y_test_vel, y_train_arias, y_test_arias, y_train_accel_orig, y_test_accel_orig, y_train_vel_orig, y_test_vel_orig, y_train_arias_orig, y_test_arias_orig = train_test_split(
                X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig, 
                test_size=0.2, random_state=42
            )
            
            # Apply the same scaling
            scaler_X = StandardScaler()
            scaler_X.fit(X_train)  # Fit on training data
            X_test_scaled = scaler_X.transform(X_test)  # Transform test data
            
            # Get predictions for test data
            [y_pred_accel, y_pred_vel, y_pred_arias] = model.predict(X_test_scaled)
            
            # Calculate original space predictions
            y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
            y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
            y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
            
            # Generate all plots with explicit error handling
            try:
                log_diagnostic("Plotting model metrics...")
                plot_metrics(history, X_test_scaled, 
                            y_test_accel, y_test_vel, y_test_arias,
                            y_test_accel_orig, y_test_vel_orig, y_test_arias_orig,
                            model)
            except Exception as e:
                log_diagnostic(f"Error in plot_metrics: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting log vs original comparison...")
                compare_log_vs_original(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                        y_test_arias, y_pred_arias, 
                                        y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
            except Exception as e:
                log_diagnostic(f"Error in compare_log_vs_original: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting residual analysis...")
                plot_residual_analysis(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                      y_test_arias, y_pred_arias, X_test)
            except Exception as e:
                log_diagnostic(f"Error in plot_residual_analysis: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting IM correlations...")
                plot_im_correlations(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                     y_test_arias, y_pred_arias,
                                     y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
            except Exception as e:
                log_diagnostic(f"Error in plot_im_correlations: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting floor predictions...")
                plot_floor_predictions(X_test, y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                      y_test_arias, y_pred_arias,
                                      y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
            except Exception as e:
                log_diagnostic(f"Error in plot_floor_predictions: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting between-within event residuals...")
                plot_between_within_event_residuals(X_test, y_test_accel, y_pred_accel, 
                                                   y_test_vel, y_pred_vel, 
                                                   y_test_arias, y_pred_arias)
            except Exception as e:
                log_diagnostic(f"Error in plot_between_within_event_residuals: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting residual vs parameters...")
                plot_residual_vs_parameters(y_test_accel, y_pred_accel, 
                                          y_test_vel, y_pred_vel, 
                                          y_test_arias, y_pred_arias,
                                          y_test_accel_orig, y_test_vel_orig, y_test_arias_orig, 
                                          X_test)
            except Exception as e:
                log_diagnostic(f"Error in plot_residual_vs_parameters: {e}")
                traceback.print_exc()
            
            # Save the model
            model_path = os.path.expanduser("~/Desktop/fyp 2_debug1/final_model.keras")
            model.save(model_path)
            log_diagnostic(f"Model saved to {model_path}")
        
    except Exception as e:
        log_diagnostic(f"Error during model training: {str(e)}")
        traceback.print_exc()
        
        try:
            # When loading a saved model, make sure the custom objects are defined
            model_path = os.path.expanduser("~/Desktop/fyp 2_debug1/final_model.keras")
            log_diagnostic(f"Loading saved model from {model_path}...")
            
            # Define custom functions needed for model loading
            def huber_loss(y_true, y_pred, delta=1.0):
                """Huber loss - a more robust loss function for regression."""
                error = y_true - y_pred
                abs_error = tf.abs(error)
                quadratic = tf.minimum(abs_error, delta)
                linear = abs_error - quadratic
                return 0.5 * tf.square(quadratic) + delta * linear
            
            def feature_cross_layer(x, hidden_dim=32):
                """
                Create pairwise feature interactions to model complex feature relationships.
                Uses a more efficient implementation to avoid memory issues.
                
                This version is compatible with Keras functional API and doesn't use tf.shape()
                """
                # Create feature crosses more efficiently
                # Instead of explicit cross products, use factorized approach
                x1 = Dense(hidden_dim, kernel_regularizer=l2(0.001))(x)
                x2 = Dense(hidden_dim, kernel_regularizer=l2(0.001))(x)
                
                # Element-wise multiplication for interaction
                cross = Multiply()([x1, x2])
                
                # Combine original features with cross features
                combined = Concatenate()([x, cross])
                
                return combined
            
            # Load model with custom objects
            model = tf.keras.models.load_model(model_path, 
                                              custom_objects={
                                                  'huber_loss': huber_loss,
                                                  'feature_cross_layer': feature_cross_layer
                                              })
            log_diagnostic("Model loaded successfully")
            
            # Load data and generate plots
            log_diagnostic("Generating plots for loaded model...")
            data_path = os.path.expanduser("~/Desktop/fyp 2_debug1/combined_seismic_data.csv")
            
            # Same data loading code as above
            X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig = load_and_prepare_data(data_path)
            
            X_train, X_test, y_train_accel, y_test_accel, y_train_vel, y_test_vel, y_train_arias, y_test_arias, y_train_accel_orig, y_test_accel_orig, y_train_vel_orig, y_test_vel_orig, y_train_arias_orig, y_test_arias_orig = train_test_split(
                X, y_accel, y_vel, y_arias, y_accel_orig, y_vel_orig, y_arias_orig, 
                test_size=0.2, random_state=42
            )
            
            scaler_X = StandardScaler()
            scaler_X.fit(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            [y_pred_accel, y_pred_vel, y_pred_arias] = model.predict(X_test_scaled)
            
            y_pred_accel_orig = safe_inverse_log10(y_pred_accel)
            y_pred_vel_orig = safe_inverse_log10(y_pred_vel)
            y_pred_arias_orig = safe_inverse_log10(y_pred_arias)
            
            # Same plot generation code with try/except blocks
            # (Add all the same plotting function calls as in the training section)
            try:
                log_diagnostic("Plotting residual analysis...")
                plot_residual_analysis(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                      y_test_arias, y_pred_arias, X_test)
            except Exception as e:
                log_diagnostic(f"Error in plot_residual_analysis: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting IM correlations...")
                plot_im_correlations(y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                     y_test_arias, y_pred_arias,
                                     y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
            except Exception as e:
                log_diagnostic(f"Error in plot_im_correlations: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting floor predictions...")
                plot_floor_predictions(X_test, y_test_accel, y_pred_accel, y_test_vel, y_pred_vel, 
                                      y_test_arias, y_pred_arias,
                                      y_test_accel_orig, y_test_vel_orig, y_test_arias_orig)
            except Exception as e:
                log_diagnostic(f"Error in plot_floor_predictions: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting between-within event residuals...")
                plot_between_within_event_residuals(X_test, y_test_accel, y_pred_accel, 
                                                   y_test_vel, y_pred_vel, 
                                                   y_test_arias, y_pred_arias)
            except Exception as e:
                log_diagnostic(f"Error in plot_between_within_event_residuals: {e}")
                traceback.print_exc()
            
            try:
                log_diagnostic("Plotting residual vs parameters...")
                plot_residual_vs_parameters(y_test_accel, y_pred_accel, 
                                          y_test_vel, y_pred_vel, 
                                          y_test_arias, y_pred_arias,
                                          y_test_accel_orig, y_test_vel_orig, y_test_arias_orig, 
                                          X_test)
            except Exception as e:
                log_diagnostic(f"Error in plot_residual_vs_parameters: {e}")
                traceback.print_exc()
            
            # Note: we can't plot metrics that require history when loading a saved model
            log_diagnostic("All available plots generated for loaded model")
            
        except Exception as e:
            log_diagnostic(f"Could not load saved model: {str(e)}")
            traceback.print_exc()





import os
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, Layer, Add # Add others if needed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2 # Import if used in original build
from sklearn.preprocessing import StandardScaler # Import if needed for input_dim
import traceback

# --- IMPORTANT: Import necessary components from your training script ---
# Make sure these definitions are identical to those in transformer_1.py
from transformer_1 import (
    build_transformer_model,
    PositionEmbedding,
    reshape_lambda_func,
    slice_lambda_func,
    # --- Add any other custom losses/metrics used during ORIGINAL training ---
    # e.g., huber_loss (if it was custom)
    # If Huber was standard tf.keras.losses.Huber(), you might not need it here,
    # but include it if you defined it manually.
    SENSOR_FEATURES, # Need these for build_transformer_model
    NUM_SENSORS,
)

def determine_input_dim(base_dir):
    """Determine the expected input dimension from the saved scaler."""
    scaler_path = os.path.join(base_dir, "model_data", "scaler_X.joblib")
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}. Cannot determine input dimension.")
        return None
    try:
        scaler = joblib.load(scaler_path)
        # Ensure scaler is fitted before accessing n_features_in_
        if not hasattr(scaler, 'n_features_in_'):
             print(f"Error: Scaler loaded from {scaler_path} does not seem to be fitted.")
             # Attempt fallback if possible (e.g., inspect mean_ or scale_)
             if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                 dim = len(scaler.mean_)
                 print(f"Determined input dimension from scaler mean_: {dim}")
                 return dim
             else:
                 return None
        dim = scaler.n_features_in_
        print(f"Determined input dimension from scaler: {dim}")
        return dim
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

def main_converter():
    print("--- Starting Model Conversion Process ---")
    base_dir = os.path.expanduser("~/Desktop/fyp 2_debug1/")
    # --- Path to the ORIGINAL model file (with potentially bad Lambda config) ---
    # Let's stick with the models/ path as per your last run's output
    original_model_path = os.path.join(base_dir, "models", "final_model.keras")
    # --- Path for the NEW, corrected model file ---
    new_model_path = os.path.join(base_dir, "models", "final_model_loadable.keras")
    scaler_path = os.path.join(base_dir, "model_data", "scaler_X.joblib")

    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
    print(f"Original model weights path: {original_model_path}") # Modified print statement
    print(f"Target new model path: {new_model_path}")

    try:
        # Determine input dimension
        input_dim = determine_input_dim(base_dir)
        if input_dim is None:
             raise ValueError("Could not determine input dimension from scaler.")
        print(f"\nDetermined input dimension: {input_dim}")

        # --- Build New Model Structure FIRST ---
        # Uses the corrected build_transformer_model from transformer_1.py
        print("\nBuilding new model structure (with output_shape)...")
        new_model = build_transformer_model(input_dim=input_dim)
        print("New model structure built.")
        # Optional: print summary to check
        # print("\nNew Model Summary:")
        # new_model.summary()

        # --- Load Weights from Original Model ---
        print(f"\nLoading weights from original file: {original_model_path}...")
        if not os.path.exists(original_model_path):
             raise FileNotFoundError(f"Original model file not found at: {original_model_path}")

        # Use load_weights instead of load_model
        new_model.load_weights(original_model_path)
        print("Weights loaded successfully into the new structure.")

        # --- Save New Model (Structure + Weights) ---
        print(f"\nSaving new loadable model to: {new_model_path}...")
        # Keras v3 format handles registered/serializable custom objects automatically.
        new_model.save(new_model_path)
        print("New model saved successfully.")
        print("\n--- Model Conversion Complete ---")

    except FileNotFoundError as e:
        print(f"\n--- Error: Required file not found ---")
        print(e)
        traceback.print_exc()
    except Exception as e:
        print(f"\n--- An unexpected error occurred during conversion ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        # --- FIX Typo ---
        # We didn't enable it, but good practice to ensure it's disabled if logic changes
        # tf.keras.config.disable_unsafe_deserialization()
        # print("Unsafe deserialization ensured disabled on error.")
        # --- End fix ---

if __name__ == "__main__":
    main_converter()
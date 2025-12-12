import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontManager
import argparse
import os
import json
import joblib
import tensorflow as tf
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split # Needed for replicating the split
import scipy.stats as stats
from itertools import combinations, product # Added product
import traceback # Added for better error printing
import time
import pickle

# --- FIX: Import necessary components from transformer_1.py ---
try:
    from transformer_1 import (
        PositionEmbedding,
        reshape_lambda_func,
        slice_lambda_func,
        prepare_features as prepare_features_t1, # Import original function to get X shape
        # Add huber loss if it was custom, otherwise TF handles standard 'huber'
    )
    custom_objects = {
        'PositionEmbedding': PositionEmbedding,
        "reshape_lambda_func": reshape_lambda_func,
        "slice_lambda_func": slice_lambda_func,
    }
except ImportError:
    print("Warning: Could not import from transformer_1.py. Model loading might fail.")
    custom_objects = {}
    prepare_features_t1 = None # Mark as unavailable

# --- Helper Functions (safe_inverse_log10, configure_plot_style) ---
def safe_inverse_log10(y_log, y_orig=None):
    result = np.power(10.0, np.clip(y_log, -15, 15))
    if y_orig is not None:
        y_orig = np.asarray(y_orig)
        result = np.where(y_orig == 0, 0.0, result)
    return result

def configure_plot_style():
    plt.rcParams.update({
        'figure.figsize': (8, 6), 'axes.linewidth': 1.1, 'xtick.direction': 'out',
        'ytick.direction': 'out', 'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.major.width': 1.1, 'ytick.major.width': 1.1, 'xtick.minor.size': 3,
        'ytick.minor.size': 3, 'xtick.minor.width': 1.1, 'ytick.minor.width': 1.1,
        'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 24,
        'font.family': 'Times New Roman', 'axes.grid': True, 'grid.alpha': 0.6,
        'grid.linestyle': '--', 'grid.linewidth': 0.5, 'axes.edgecolor': 'black',
        'axes.facecolor': 'white', 'legend.fontsize': 16,
    })
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True

# --- Data Loading (MODIFIED to load BOTH Train and Test data) ---
def load_processed_data(data_dir):
    """Load scaled/unscaled X train/test data, targets (y), group labels, and original static features."""
    print(f"Loading processed data from: {data_dir}")
    data = {}
    # MODIFIED: Add train files back
    files_to_load = {
        # Test Data
        "X_test_scaled": "X_test_scaled.npy",
        "X_test": "X_test.npy", # Unscaled engineered test data
        "y_accel_log_test": "y_test_accel.npy", "y_vel_log_test": "y_test_vel.npy", "y_arias_log_test": "y_test_arias.npy",
        "y_accel_orig_test": "y_test_accel_orig.npy", "y_vel_orig_test": "y_test_vel_orig.npy", "y_arias_orig_test": "y_test_arias_orig.npy",
        "group_labels_test": "group_labels_test.npy",
        "X_test_static_original": "X_test_static_original.npy",
        # Train Data (Add these back)
        "X_train_scaled": "X_train_scaled.npy",
        "y_accel_log_train": "y_train_accel.npy", "y_vel_log_train": "y_train_vel.npy", "y_arias_log_train": "y_train_arias.npy",
        "y_accel_orig_train": "y_train_accel_orig.npy", "y_vel_orig_train": "y_train_vel_orig.npy", "y_arias_orig_train": "y_train_arias_orig.npy",
    }
    try:
        for name, filename in files_to_load.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                data[name] = np.load(filepath, allow_pickle=True)
                # Flatten only target arrays (y_)
                if name.startswith('y_') and data[name].ndim > 1:
                    data[name] = data[name].flatten()
                print(f"  Loaded {name}: {data[name].shape if hasattr(data[name], 'shape') else type(data[name])}")
            else:
                # Make static original and X_test (unscaled) optional
                is_optional = name in ["X_test", "X_test_static_original"]
                # Training data is needed for comparison plots
                is_required_for_comparison = name.startswith("X_train") or (name.startswith("y_") and "train" in name)
                # Group labels are required for the between/within plot
                is_required_for_between_within = name == "group_labels_test"

                # Handle missing files based on requirements
                if not is_optional and not is_required_for_comparison and not is_required_for_between_within:
                     raise FileNotFoundError(f"Required core file not found: {filepath}")
                elif is_required_for_comparison and not os.path.exists(filepath):
                     print(f"Warning: Required file for train/test comparison plots not found: {filepath}. Comparison plots will be skipped.")
                     data[name] = None # Mark as None if missing
                elif is_required_for_between_within and not os.path.exists(filepath):
                     print(f"Warning: Required file for Between/Within plot not found: {filepath}. Between/Within plot will be skipped.")
                     data[name] = None
                elif is_optional and not os.path.exists(filepath): # Optional files
                    print(f"Warning: Optional file not found: {filepath}. Some plots/axes might be skipped or look incorrect.")
                    data[name] = None

        # Organize into dictionaries
        test_targets = {
            'acceleration': {'log': data.get('y_accel_log_test'), 'orig': data.get('y_accel_orig_test')},
            'velocity': {'log': data.get('y_vel_log_test'), 'orig': data.get('y_vel_orig_test')},
            'arias': {'log': data.get('y_arias_log_test'), 'orig': data.get('y_arias_orig_test')}
        }
        # ADDED: Train targets dictionary
        train_targets = {
            'acceleration': {'log': data.get('y_accel_log_train'), 'orig': data.get('y_accel_orig_train')},
            'velocity': {'log': data.get('y_vel_log_train'), 'orig': data.get('y_vel_orig_train')},
            'arias': {'log': data.get('y_arias_log_train'), 'orig': data.get('y_arias_orig_train')}
        }

        # Check if essential test data is missing
        if data.get("X_test_scaled") is None or any(v is None for v in test_targets['acceleration'].values()):
             raise FileNotFoundError("Essential test data (X_test_scaled.npy or target files) is missing.")

        # Return all loaded data
        return (data.get("X_train_scaled"), data.get("X_test_scaled"),
                train_targets, test_targets,
                data.get("X_test"), data.get("group_labels_test"),
                data.get("X_test_static_original"))

    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred loading processed data: {e}")
        raise

# --- Model Loading (As before) ---
def load_model(model_filepath):
    print(f"Loading model from: {model_filepath}")
    if not os.path.isfile(model_filepath): raise FileNotFoundError(f"No such file: '{model_filepath}'")
    try: model = tf.keras.models.load_model(model_filepath, custom_objects=custom_objects); print("Model loaded."); return model
    except Exception as e: print(f"Model loading error: {e}"); raise

# --- Residual Calculation (As before) ---
def calculate_all_residuals_and_metrics(model, X_test_scaled, y_tests):
    print(f"Calculating residuals... Predicting on X_test_scaled (Shape: {X_test_scaled.shape})...")
    start_time = time.time()
    try:
        predictions_all_log = model.predict(X_test_scaled, batch_size=4096, verbose=1)
        end_time = time.time()
        print(f"Prediction finished in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"ERROR during model.predict: {e}")
        traceback.print_exc()
        raise
        
    print("Processing predictions and calculating metrics...")
    results = {}; target_names = ['acceleration', 'velocity', 'arias']
    if not isinstance(predictions_all_log, list) or len(predictions_all_log) != len(target_names):
         print("ERROR: Model output format mismatch. Expected a list of 3 arrays.")
         raise ValueError("Model output format mismatch")

    for i, target_name in enumerate(target_names):
        print(f"  Processing target: {target_name}...")
        if target_name not in y_tests or 'log' not in y_tests[target_name] or 'orig' not in y_tests[target_name]:
             print(f"ERROR: Missing target data for '{target_name}' in y_tests dictionary.")
             raise KeyError(f"Missing data for target '{target_name}'")

        y_test_log = y_tests[target_name]['log']; y_test_orig = y_tests[target_name]['orig']; y_pred_log = predictions_all_log[i].flatten()

        if y_test_log.shape != y_pred_log.shape:
             print(f"ERROR: Shape mismatch for {target_name}. True shape: {y_test_log.shape}, Pred shape: {y_pred_log.shape}")
             raise ValueError(f"Shape mismatch for {target_name}")

        if np.isnan(y_pred_log).any():
            print(f"WARN: NaNs detected in predictions for {target_name}. Residuals/R2 might be affected.")

        residuals_log = y_test_log - y_pred_log; r2_log = r2_score(y_test_log, y_pred_log)
        y_pred_orig = safe_inverse_log10(y_pred_log, y_orig=y_test_orig); r2_orig = r2_score(y_test_orig, y_pred_orig); residuals_orig = y_test_orig - y_pred_orig
        print(f"  {target_name.capitalize()}: R² Log={r2_log:.4f}, R² Orig={r2_orig:.4f}")
        results[target_name] = {'true_log': y_test_log, 'predicted_log': y_pred_log, 'residuals_log': residuals_log, 'r2_log': r2_log,
                                'true_orig': y_test_orig, 'predicted_orig': y_pred_orig, 'r2_orig': r2_orig, 'residuals_orig': residuals_orig}
    print("Finished calculating all residuals and metrics.")
    return results

# --- Plotting Functions (plot_single_parameter, plot_residuals_vs_parameters, etc., as previously defined) ---
# --- MODIFIED Plotting Function: Residuals vs. True Value ---
def plot_single_parameter(target_name, target_results, output_dir):
    """ Plots Residuals vs. TRUE Value (Log & Original) + Log Residual Distribution """
    print(f"Generating True vs Residual plot for {target_name.capitalize()}...")
    configure_plot_style()
    true_log = target_results['true_log']; true_orig = target_results['true_orig']
    residuals_log = target_results['residuals_log']; residuals_orig = target_results['residuals_orig']
    r2_log = target_results['r2_log']; r2_orig = target_results['r2_orig']
    titles = {"acceleration": "Acceleration", "velocity": "Velocity", "arias": "Arias Intensity"}
    units = {"acceleration": r'$(cm/s^2)$', "velocity": r'$(cm/s)$', "arias": r'$(m/s)$'}
    full_name = titles.get(target_name, target_name.capitalize()); unit = units.get(target_name, '')
    scatter_log_color = '#1f77b4'; scatter_log_edge = '#0d3c55'; scatter_log_marker = 'o'
    scatter_orig_color = '#ff7f0e'; scatter_orig_edge = '#8B4513'; scatter_orig_marker = 's'
    hist_log_color = scatter_log_color; hist_log_edge = scatter_log_edge
    label_fontsize = 18; tick_fontsize = 16; r2_fontsize = 14; hist_label_fontsize = 14
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.15)
    ax_res_log = fig.add_subplot(gs[0, 0]); ax_res_orig = fig.add_subplot(gs[1, 0]); ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_res_log)

    # Top Scatter: Log Residuals vs TRUE Log Value
    ax_res_log.scatter(true_log, residuals_log, alpha=0.5, s=30, edgecolor=scatter_log_edge, color=scatter_log_color, marker=scatter_log_marker)
    ax_res_log.axhline(0, color='black', linestyle='-', linewidth=1.1); ax_res_log.tick_params(axis='x', labelbottom=False)
    ax_res_log.set_ylabel("Residuals (Log Space)", fontsize=label_fontsize, family='Times New Roman'); ax_res_log.tick_params(axis='y', labelsize=tick_fontsize); ax_res_log.locator_params(axis='y', nbins=6)
    res_log_min, res_log_max = np.percentile(residuals_log[np.isfinite(residuals_log)], [1, 99])
    true_log_min, true_log_max = np.percentile(true_log[np.isfinite(true_log)], [1, 99])
    ax_res_log.set_ylim(res_log_min - 0.1*(res_log_max-res_log_min), res_log_max + 0.1*(res_log_max-res_log_min))
    ax_res_log.set_xlim(true_log_min - 0.05*(true_log_max-true_log_min), true_log_max + 0.05*(true_log_max-true_log_min))
    common_ylim = ax_res_log.get_ylim()
    # Ensure nbins is only set for linear scale on top plot if necessary
    if ax_res_log.get_xaxis().get_scale() == 'linear':
         ax_res_log.locator_params(axis='x', nbins=6)
    if ax_res_log.get_yaxis().get_scale() == 'linear':
         ax_res_log.locator_params(axis='y', nbins=6)
    ax_res_log.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)
    r2_log_text = f'Log $R^2$={r2_log:.3f}'; ax_res_log.text(0.04, 0.95, r2_log_text, transform=ax_res_log.transAxes, fontsize=r2_fontsize, va='top', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='grey'))

    # Bottom Scatter: Original Residuals vs TRUE Original Value
    ax_res_orig.scatter(true_orig, residuals_orig, alpha=0.5, s=30, edgecolor=scatter_orig_edge, color=scatter_orig_color, marker=scatter_orig_marker)
    # Always use the original label, don't add "(Log Scale)"
    ax_res_orig.set_xlabel(f"Actual Value ({full_name}, {unit})", fontsize=label_fontsize, family='Times New Roman')
    ax_res_orig.axhline(0, color='black', linestyle='-', linewidth=1.1);
    ax_res_orig.set_ylabel("Residuals (Orig. Space)", fontsize=label_fontsize, family='Times New Roman'); ax_res_orig.tick_params(axis='both', labelsize=tick_fontsize)

    # Set axis limits based on percentiles (still useful even with linear scale)
    q_low_true, q_high_true = np.percentile(true_orig[np.isfinite(true_orig)], [1, 99]); true_range = q_high_true - q_low_true
    # Ensure lower limit is not negative, especially important for linear scale
    ax_res_orig.set_xlim(max(0, q_low_true - 0.05*true_range), q_high_true + 0.05*true_range)
    q_low_res, q_high_res = np.percentile(residuals_orig[np.isfinite(residuals_orig)], [1, 99]); res_range = q_high_res - q_low_res
    ax_res_orig.set_ylim(q_low_res - 0.1 * res_range, q_high_res + 0.1 * res_range)

    # Always assume linear scale now for setting nbins
    ax_res_orig.locator_params(axis='x', nbins=6)
    # Check Y axis scale (less likely to be log, but check anyway)
    if ax_res_orig.get_yaxis().get_scale() == 'linear':
        ax_res_orig.locator_params(axis='y', nbins=6)

    ax_res_orig.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)
    r2_orig_text = f'Orig. $R^2$={r2_orig:.3f}'; ax_res_orig.text(0.04, 0.95, r2_orig_text, transform=ax_res_orig.transAxes, fontsize=r2_fontsize, va='top', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='grey'))

    # Residual Histogram
    ax_hist.hist(residuals_log, bins=50, orientation='horizontal', density=True, color=hist_log_color, edgecolor=hist_log_edge, alpha=0.7)
    ax_hist.axhline(0, color='black', linestyle='-', linewidth=1.1); ax_hist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax_hist.xaxis.tick_top(); ax_hist.xaxis.set_label_position('top'); ax_hist.set_xlabel("Distribution", fontsize=hist_label_fontsize, family='Times New Roman')
    ax_hist.tick_params(axis='x', labelsize=tick_fontsize * 0.8); ax_hist.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, alpha=0.6); ax_hist.locator_params(axis='x', nbins=4); ax_hist.set_ylim(common_ylim)

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95)
    output_filename = os.path.join(output_dir, f"{target_name}_true_vs_residual_plot.png")
    print(f"Attempting to save plot to: {output_filename}") # Print path before saving
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Saved plot successfully: {output_filename}") # Confirmation
    except FileNotFoundError: # Catch specific error
        print(f"ERROR: Directory not found when trying to save '{output_filename}'. Please check directory existence and permissions.")
        traceback.print_exc() # Show where the error occurred
    except Exception as e: # Catch other potential save errors
        print(f"ERROR saving plot '{output_filename}': {type(e).__name__} - {e}")
        traceback.print_exc()
    plt.close(fig) # Close figure even if save fails

# --- Plotting Function 2: Residuals vs. Parameters ---
def plot_residuals_vs_parameters(results, X_test_unscaled, X_test_static_original, output_dir):
    # X_test_unscaled is the engineered data BEFORE scaling
    # X_test_static_original is the raw static data BEFORE engineering
    print("Generating Residuals vs Parameters plots...")
    configure_plot_style()
    # --- UPDATED param_config ---
    param_config = {
        # Use original static data
        'Magnitude':    {'data': X_test_static_original, 'idx': 0, 'log_axis': False},
        'PGA':          {'data': X_test_static_original, 'idx': 1, 'log_axis': True}, # Use log axis for PGA
        'Depth':        {'data': X_test_static_original, 'idx': 2, 'log_axis': False},
        'Distance (Repi)': {'data': X_test_static_original, 'idx': 3, 'log_axis': True}, # Use log axis for Distance
        'Stories':      {'data': X_test_static_original, 'idx': 4, 'log_axis': False},
        'Total Height': {'data': X_test_static_original, 'idx': 5, 'log_axis': False},
        # Use engineered data (X_test_unscaled) for first input sensor height
        'Input Sensor 1 Height': {'data': X_test_unscaled, 'idx': 0, 'log_axis': False}
        # REMOVED 'Floor (Index 2)' as it was incorrect
    }
    target_names = list(results.keys()); colors = {'acceleration': '#1f77b4', 'velocity': '#ff7f0e', 'arias': '#2ca02c'}

    for param_name, config in param_config.items():
        data_array = config['data']
        param_idx = config['idx']
        use_log_axis = config['log_axis']

        # Check if the required data array is available
        if data_array is None:
            print(f"Warn: Data array for '{param_name}' not loaded. Skipping plot.")
            continue
        # Check index bounds
        # Ensure data_array is a numpy array before checking shape
        if not isinstance(data_array, np.ndarray) or data_array.ndim < 2:
             print(f"Warn: Data array for '{param_name}' is not a valid 2D numpy array. Skipping plot.")
             continue
        if param_idx >= data_array.shape[1]:
            print(f"Warn: Index {param_idx} for '{param_name}' out of bounds for its data array ({data_array.shape[1]} cols). Skipping plot.")
            continue

        param_values = data_array[:, param_idx]
        data_source_name = 'static_original' if np.array_equal(data_array, X_test_static_original) else ('engineered_unscaled' if np.array_equal(data_array, X_test_unscaled) else 'unknown')
        print(f"  Plotting residuals vs {param_name} (using index {param_idx} from {data_source_name} array, log_axis={use_log_axis})")


        fig, axes = plt.subplots(len(target_names), 1, figsize=(8, 6 * len(target_names)), sharex=True)
        axes = [axes] if len(target_names) == 1 else axes
        # Make title slightly shorter if needed
        plot_title = f'Log Residuals vs. {param_name}'
        if len(plot_title) > 60: # Arbitrary limit
             plot_title = f'Log Residuals vs.\n{param_name}' # Add newline
        fig.suptitle(plot_title, fontsize=24, y=1.02, family='Times New Roman')


        for i, target_name in enumerate(target_names):
            ax = axes[i]; residuals_log = results[target_name]['residuals_log']; color = colors[target_name]
            # Ensure consistent length for masking
            if len(param_values) != len(residuals_log):
                 print(f"FATAL WARNING: Length mismatch for '{param_name}' ({len(param_values)}) and residuals ({len(residuals_log)}). Skipping subplot.")
                 continue # Skip this subplot

            # Filter out non-finite values before plotting
            mask = np.isfinite(param_values) & np.isfinite(residuals_log)
            if use_log_axis:
                # Ensure values are positive for log axis
                mask &= (param_values > 1e-9) # Use small positive threshold

            param_vals_finite = param_values[mask]
            res_log_finite = residuals_log[mask]

            if len(param_vals_finite) < 10:
                ax.text(0.5, 0.5, "Not enough finite data", ha='center', va='center', transform=ax.transAxes)
                print(f"    -> Not enough finite data points for {target_name} subplot.")
                continue # Skip this subplot

            ax.scatter(param_vals_finite, res_log_finite, alpha=0.4, s=25, color=color, edgecolors='k', linewidths=0.5)
            ax.axhline(0, color='black', linestyle='-', linewidth=1.1)

            # Trend line (calculate based on appropriate scale)
            try:
                # Use finite values for trend calculation
                x_for_trend = np.log10(param_vals_finite) if use_log_axis else param_vals_finite
                if len(x_for_trend) < 2 or len(res_log_finite) < 2:
                    print(f"Warn: Not enough points ({len(x_for_trend)}) for trend line calculation for {target_name} vs {param_name}.")
                    continue # Skip trend line if not enough points

                slope, intercept, r_value, p_value, std_err = stats.linregress(x_for_trend, res_log_finite)

                # Plot trend line using the min/max of the *finite* parameter values
                x_plot_trend = np.array([np.min(param_vals_finite), np.max(param_vals_finite)])
                # Handle potential division by zero or log of non-positive if param_vals_finite min/max are bad, though mask should prevent this
                if use_log_axis and np.any(x_plot_trend <= 0):
                     print(f"Warn: Skipping trend line plotting for {target_name} vs {param_name} due to non-positive x values for log axis.")
                else:
                     y_plot_trend = intercept + slope * (np.log10(x_plot_trend) if use_log_axis else x_plot_trend)
                     ax.plot(x_plot_trend, y_plot_trend, color='red', linestyle='--', linewidth=1.5)
                     corr_text = f'Corr: {r_value:.2f}\nSlope: {slope:.3f}'; ax.text(0.04, 0.95, corr_text, transform=ax.transAxes, fontsize=14, va='top', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='grey'))

            except ValueError as e:
                print(f"Warn: Trend line failed for {target_name} vs {param_name}: {e}")
            except Exception as e_gen: # Catch other potential errors
                 print(f"Warn: Unexpected error during trend line calculation for {target_name} vs {param_name}: {e_gen}")


            ax.set_ylabel(f"{target_name.capitalize()}\nLog Residual", fontsize=18, family='Times New Roman'); ax.tick_params(axis='y', labelsize=16); ax.locator_params(axis='y', nbins=5); ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)

            # Add units if relevant (e.g., for height)
            units_dict = {'Total Height': '(in)', 'Input Sensor 1 Height': '(in)', 'Distance (Repi)': '(km)'} # Add more as needed
            unit_str = units_dict.get(param_name, '')
            ax_xlabel = f"{param_name} {unit_str}".strip()

            if use_log_axis:
                ax.set_xscale('log')
                # Adjust x-axis limits for log scale if necessary, ensuring positive values
                min_x_plot = np.min(param_vals_finite)
                max_x_plot = np.max(param_vals_finite)
                if min_x_plot > 0:
                    ax.set_xlim(min_x_plot * 0.9, max_x_plot * 1.1)
                ax_xlabel += " (Log Scale)"

            if i == len(target_names) - 1:
                ax.set_xlabel(ax_xlabel, fontsize=20, family='Times New Roman')
                ax.tick_params(axis='x', labelsize=16)
                # Set nbins ONLY if axis is linear scale
                if not use_log_axis:
                     ax.locator_params(axis='x', nbins=6)
            else: # Use else, not elif True
                ax.tick_params(axis='x', labelbottom=False)


        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to prevent title overlap
        # Sanitize filename
        safe_param_name = "".join(c if c.isalnum() else "_" for c in param_name)
        output_filename = os.path.join(output_dir, f"residuals_vs_{safe_param_name}.png")
        plt.savefig(output_filename, dpi=300)
        print(f"  Saved plot to: {output_filename}")
        plt.close(fig)


# --- Plotting Function 3: Performance vs Input Sensor Height --- MODIFIED ---
def plot_performance_vs_input_sensor_height(results, X_test_unscaled, output_dir, min_samples_per_bin=10):
    """Plots R2 and MAE against the height of the first input sensor (proxy for vertical location)."""
    if X_test_unscaled is None: print("Skipping Performance vs Input Sensor Height: Unscaled X_test unavailable."); return
    print("Generating Performance vs Input Sensor Height plots...")
    configure_plot_style()
    HEIGHT_COL_IDX = 0 # Use index 0 for the height of the first input sensor
    if X_test_unscaled.ndim < 2 or HEIGHT_COL_IDX >= X_test_unscaled.shape[1]:
        print(f"Warn: Height index {HEIGHT_COL_IDX} out of bounds or X_test_unscaled is not 2D ({X_test_unscaled.shape}). Skipping height plot.")
        return

    height_values = X_test_unscaled[:, HEIGHT_COL_IDX]
    target_names = list(results.keys())

    # Bin the height values for smoother plotting
    finite_height_mask = np.isfinite(height_values)
    if np.sum(finite_height_mask) < min_samples_per_bin * 2: # Need enough data to bin
         print(f"Warn: Not enough finite height values ({np.sum(finite_height_mask)}) to generate plot. Skipping.")
         return

    try:
        # Use quantile bins for robustness against skewed distributions
        height_bins, bin_edges = pd.qcut(height_values[finite_height_mask], q=10, labels=False, retbins=True, duplicates='drop')
        # Create labels like "0-100 in", "100-250 in", etc.
        bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f} in" for i in range(len(bin_edges)-1)]
        print(f"Using quantile bins for Input Sensor 1 Height ({len(bin_labels)} bins).")
        # Assign bin index - use -1 for values outside the finite mask
        height_bins_full = np.full(len(height_values), -1, dtype=int)
        height_bins_full[finite_height_mask] = height_bins

    except Exception as e:
        print(f"Warn: Binning failed for Input Sensor 1 Height (Error: {e}). Trying numeric bins.")
        try:
             # Fallback to numeric bins
             height_bins, bin_edges = pd.cut(height_values[finite_height_mask], bins=10, labels=False, retbins=True, duplicates='drop')
             bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f} in" for i in range(len(bin_edges)-1)]
             print(f"Using numeric bins for Input Sensor 1 Height ({len(bin_labels)} bins).")
             height_bins_full = np.full(len(height_values), -1, dtype=int)
             height_bins_full[finite_height_mask] = height_bins
        except Exception as e2:
             print(f"Warn: Numeric binning also failed (Error: {e2}). Skipping height plot.")
             return

    metrics_by_height = {t: {'r2_log': [], 'mae_orig': [], 'count': []} for t in target_names}
    valid_bin_labels = []
    # Iterate through the bins generated by qcut/cut
    num_bins_generated = len(bin_labels)
    for i in range(num_bins_generated):
        # Use the height_bins_full array which aligns with results
        mask = (height_bins_full == i)
        count = np.sum(mask)
        if count < min_samples_per_bin:
            # print(f"  Skipping bin {i} ('{bin_labels[i]}'): only {count} samples.") # Optional verbosity
            continue

        valid_bin_labels.append(bin_labels[i]) # Add the label for this valid bin

        for target_name in target_names:
            res_t = results[target_name]
            # Ensure all required arrays have the same length as the mask
            required_lengths = [len(res_t['true_log']), len(res_t['predicted_log']), len(res_t['true_orig']), len(res_t['predicted_orig'])]
            if len(mask) != required_lengths[0]: # Check against one, assuming others are same
                print(f"FATAL WARNING: Length mismatch between height mask ({len(mask)}) and target results ({required_lengths[0]}) for {target_name}. Skipping bin.")
                # Reset metrics for this bin if length mismatch occurs
                r2_log_val = np.nan
                mae_orig_val = np.nan
                metrics_by_height[target_name]['r2_log'].append(r2_log_val)
                metrics_by_height[target_name]['mae_orig'].append(mae_orig_val)
                metrics_by_height[target_name]['count'].append(0)
                continue # Skip to next target or bin if fatal mismatch

            # Combine height mask with validity masks for targets
            target_mask = mask & np.isfinite(res_t['true_log']) & np.isfinite(res_t['predicted_log']) & \
                          np.isfinite(res_t['true_orig']) & np.isfinite(res_t['predicted_orig'])
            current_samples = np.sum(target_mask)

            if current_samples < 2: # Need at least 2 samples for R2/MAE
                r2_log_val = np.nan
                mae_orig_val = np.nan
            else:
                # Calculate metrics only on valid, finite data within the bin
                r2_log_val = r2_score(res_t['true_log'][target_mask], res_t['predicted_log'][target_mask])
                mae_orig_val = mean_absolute_error(res_t['true_orig'][target_mask], res_t['predicted_orig'][target_mask])

            metrics_by_height[target_name]['r2_log'].append(r2_log_val)
            metrics_by_height[target_name]['mae_orig'].append(mae_orig_val)
            metrics_by_height[target_name]['count'].append(current_samples)

    if not valid_bin_labels:
        print("Warn: No height bins met min samples threshold. Skipping height performance plot.")
        return

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(valid_bin_labels)*0.8), 10))
    num_valid_bins = len(valid_bin_labels)
    x = np.arange(num_valid_bins) # X-axis positions for bars
    width = 0.25 # Width of bars
    colors = {'acceleration': '#1f77b4', 'velocity': '#ff7f0e', 'arias': '#2ca02c'}
    offsets = [-width, 0, width] # Offsets for grouped bars

    # Plot R² Scores
    ax = axes[0]
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Line at R²=0
    for i, target_name in enumerate(target_names):
        # Get R² values ensuring they align with valid_bin_labels
        r2_vals = metrics_by_height[target_name]['r2_log']
        if len(r2_vals) != num_valid_bins:
             print(f"Warn: R2 value count mismatch for {target_name}. Expected {num_valid_bins}, got {len(r2_vals)}")
             continue # Skip plotting this target if mismatch
        # Handle NaN values gracefully for plotting
        r2_vals_plot = [val if np.isfinite(val) else 0 for val in r2_vals] # Replace NaN with 0 for bar height
        ax.bar(x + offsets[i], r2_vals_plot, width, label=target_name.capitalize(), color=colors[target_name], alpha=0.8)

    ax.set_ylabel('$R^2$ Score (Log Space)', fontsize=18, family='Times New Roman')
    ax.set_title('Prediction Accuracy vs. Input Sensor Height', fontsize=20, family='Times New Roman') # Updated title
    ax.set_xticks(x)
    ax.set_xticklabels(valid_bin_labels, rotation=30, ha='right')
    ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7); ax.legend(fontsize=14)
    # Set y-axis limits appropriately
    all_r2_logs = [item for sublist in [metrics_by_height[t]['r2_log'] for t in target_names] for item in sublist if np.isfinite(item)]
    min_r2_log = np.min(all_r2_logs) if all_r2_logs else 0
    ax.set_ylim(bottom=min(0, min_r2_log - 0.1))

    # Plot MAE Scores
    ax = axes[1]
    for i, target_name in enumerate(target_names):
        mae_vals = metrics_by_height[target_name]['mae_orig']
        if len(mae_vals) != num_valid_bins:
             print(f"Warn: MAE value count mismatch for {target_name}. Expected {num_valid_bins}, got {len(mae_vals)}")
             continue # Skip plotting this target if mismatch
        # Handle NaN values gracefully for plotting
        mae_vals_plot = [val if np.isfinite(val) else 0 for val in mae_vals] # Replace NaN with 0 for bar height
        ax.bar(x + offsets[i], mae_vals_plot, width, label=target_name.capitalize(), color=colors[target_name], alpha=0.8)

    ax.set_ylabel('MAE (Original Space)', fontsize=18, family='Times New Roman')
    ax.set_title('Prediction Error vs. Input Sensor Height', fontsize=20, family='Times New Roman') # Updated title
    ax.set_xticks(x)
    ax.set_xticklabels(valid_bin_labels, rotation=30, ha='right')
    ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7); ax.legend(fontsize=14)
    ax.set_yscale('log') # Keep log scale for MAE as errors can vary widely
    # Adjust y-limits for log scale, ensuring bottom > 0
    all_maes = [item for sublist in [metrics_by_height[t]['mae_orig'] for t in target_names] for item in sublist if np.isfinite(item) and item > 0]
    min_mae = np.min(all_maes) if all_maes else 1e-9 # Default minimum if no valid MAEs
    ax.set_ylim(bottom=max(1e-9, min_mae * 0.5))

    plt.tight_layout()
    output_filename = os.path.join(output_dir, "performance_vs_input_sensor_height.png") # Updated filename
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot to: {output_filename}")
    plt.close(fig)

# --- Plotting Function 4: Floor-Level Performance ---
# ... (Keep implementation - VERIFY INDEX) ...
def plot_floor_level_performance(results, X_test_unscaled, output_dir, min_samples_per_floor=10):
    if X_test_unscaled is None: print("Skipping Floor Level Performance: Unscaled X_test unavailable."); return
    print("Generating Floor Level Performance plots...")
    configure_plot_style(); FLOOR_COL_IDX = 2 # --- VERIFY/ADJUST ---
    if FLOOR_COL_IDX >= X_test_unscaled.shape[1]: print(f"Warn: Floor index {FLOOR_COL_IDX} out of bounds for X_test_unscaled ({X_test_unscaled.shape[1]} cols). Skipping floor plot."); return
    floor_values = X_test_unscaled[:, FLOOR_COL_IDX]; target_names = list(results.keys())
    unique_floors = np.unique(floor_values[np.isfinite(floor_values)])
    if len(unique_floors) > 15:
        try: floor_bins = pd.qcut(floor_values, q=10, labels=False, duplicates='drop'); _, bin_edges = pd.qcut(floor_values, q=10, retbins=True, duplicates='drop'); bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]; print(f"Using quantile bins ({len(bin_labels)} bins).")
        except Exception as e:
            try: floor_bins = pd.cut(floor_values, bins=10, labels=False, duplicates='drop'); _, bin_edges = pd.cut(floor_values, bins=10, retbins=True, duplicates='drop'); bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]; print(f"Using numeric bins ({len(bin_labels)} bins).")
            except Exception as e2: print(f"Warn: Binning failed (Error: {e2}). Skipping floor plot."); return
    elif len(unique_floors) < 2: print(f"Warn: Not enough unique floor values ({len(unique_floors)}). Skipping floor plot."); return
    else: sorted_unique_floors = np.sort(unique_floors); bin_labels = [f"Fl {int(f)}" if f == int(f) else f"Fl {f:.1f}" for f in sorted_unique_floors]; bin_map = {val: i for i, val in enumerate(sorted_unique_floors)}; floor_bins = np.array([bin_map.get(f, -1) for f in floor_values]); print(f"Using discrete floors: {bin_labels}")
    metrics_by_floor = {t: {'r2_log': [], 'mae_orig': [], 'count': []} for t in target_names}; valid_bin_labels = []; max_bin_index = int(np.nanmax(floor_bins)) if not np.all(np.isnan(floor_bins)) else -1
    for i in range(max_bin_index + 1):
        mask = (floor_bins == i) & np.isfinite(floor_bins); count = np.sum(mask)
        if count < min_samples_per_floor: continue
        if i < len(bin_labels): valid_bin_labels.append(bin_labels[i])
        else: print(f"Warn: Bin index {i} label mismatch. Skipping."); continue
        for target_name in target_names:
            res_t = results[target_name]; target_mask = mask & np.isfinite(res_t['true_log']) & np.isfinite(res_t['predicted_log']) & np.isfinite(res_t['true_orig']) & np.isfinite(res_t['predicted_orig'])
            if np.sum(target_mask) < 2: r2_log_val = np.nan; mae_orig_val = np.nan
            else: r2_log_val = r2_score(res_t['true_log'][target_mask], res_t['predicted_log'][target_mask]); mae_orig_val = mean_absolute_error(res_t['true_orig'][target_mask], res_t['predicted_orig'][target_mask])
            metrics_by_floor[target_name]['r2_log'].append(r2_log_val); metrics_by_floor[target_name]['mae_orig'].append(mae_orig_val); metrics_by_floor[target_name]['count'].append(np.sum(target_mask))
    if not valid_bin_labels: print("Warn: No floor bins met min samples. Skipping floor plot."); return
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(valid_bin_labels)*0.8), 10)); num_valid_bins = len(valid_bin_labels); x = np.arange(num_valid_bins); width = 0.25; colors = {'acceleration': '#1f77b4', 'velocity': '#ff7f0e', 'arias': '#2ca02c'}; offsets = [-width, 0, width]
    ax = axes[0]; ax.axhline(0, color='grey', linestyle='--', linewidth=0.8);
    for i, target_name in enumerate(target_names): r2_vals = metrics_by_floor[target_name]['r2_log']; ax.bar(x + offsets[i], r2_vals, width, label=target_name.capitalize(), color=colors[target_name], alpha=0.8)
    ax.set_ylabel('$R^2$ Score (Log Space)', fontsize=18, family='Times New Roman'); ax.set_title('Prediction Accuracy by Floor Level', fontsize=20, family='Times New Roman'); ax.set_xticks(x); ax.set_xticklabels(valid_bin_labels, rotation=30, ha='right'); ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=16); ax.grid(axis='y', linestyle='--', alpha=0.7); ax.legend(fontsize=14); ax.set_ylim(bottom=min(0, np.nanmin([item for sublist in [metrics_by_floor[t]['r2_log'] for t in target_names] for item in sublist])-0.1 if metrics_by_floor else 0))
    ax = axes[1];
    for i, target_name in enumerate(target_names): mae_vals = metrics_by_floor[target_name]['mae_orig']; ax.bar(x + offsets[i], mae_vals, width, label=target_name.capitalize(), color=colors[target_name], alpha=0.8)
    ax.set_ylabel('MAE (Original Space)', fontsize=18, family='Times New Roman'); ax.set_title('Prediction Error by Floor Level', fontsize=20, family='Times New Roman'); ax.set_xticks(x); ax.set_xticklabels(valid_bin_labels, rotation=30, ha='right'); ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=16); ax.grid(axis='y', linestyle='--', alpha=0.7); ax.legend(fontsize=14); ax.set_yscale('log')
    all_maes = [item for sublist in [metrics_by_floor[t]['mae_orig'] for t in target_names] for item in sublist if np.isfinite(item)]
    min_mae = np.min(all_maes) if all_maes else 1e-9; ax.set_ylim(bottom=max(1e-9, min_mae*0.5))
    plt.tight_layout(); output_filename = os.path.join(output_dir, "floor_level_performance.png"); plt.savefig(output_filename, dpi=300); print(f"Saved plot to: {output_filename}"); plt.close(fig)


# --- Plotting Function 5: Stress Case Performance ---
# ... (Keep implementation - VERIFY INDICES/THRESHOLDS) ...
def plot_stress_case_performance(results, X_test_unscaled, output_dir, stress_cases=None):
    if X_test_unscaled is None: print("Skipping Stress Case plots: Unscaled X_test unavailable."); return
    if stress_cases is None: # Define Default Stress Cases
        stress_cases = { "High_Magnitude": {'col_idx': 20, 'threshold': 6.5, 'condition': 'greater'}, "Low_Distance": {'col_idx': 23, 'threshold': 20, 'condition': 'less'}, "High_PGA": {'col_idx': 21, 'threshold': 0.3, 'condition': 'greater'}, "Tall_Building": {'col_idx': 24, 'threshold': 15, 'condition': 'greater'} } # NOTE: Indices refer to X_test_unscaled (engineered data)
    print("Generating Stress Case Performance plots..."); configure_plot_style(); target_names = list(results.keys()); colors = {'acceleration': '#1f77b4', 'velocity': '#ff7f0e', 'arias': '#2ca02c'}
    for case_name, criteria in stress_cases.items():
        col_idx = criteria['col_idx']; threshold = criteria['threshold']; condition = criteria['condition']
        if X_test_unscaled is None or col_idx >= X_test_unscaled.shape[1]: print(f"Warn: Data or Index {col_idx} for '{case_name}' invalid. Skipping."); continue
        param_values = X_test_unscaled[:, col_idx]; valid_param_mask = np.isfinite(param_values)
        if condition == 'greater': mask = valid_param_mask & (param_values > threshold)
        elif condition == 'less': mask = valid_param_mask & (param_values < threshold)
        else: print(f"Warn: Invalid condition '{condition}' for '{case_name}'. Skipping."); continue
        num_samples = np.sum(mask);
        if num_samples < 20: print(f"Warn: Insufficient samples ({num_samples}) for '{case_name}'. Skipping."); continue
        print(f"  Plotting stress case: {case_name} ({num_samples} samples)")
        try: # Add try-except around each case plot
            fig, axes = plt.subplots(1, len(target_names), figsize=(8 * len(target_names), 7)); axes = [axes] if len(target_names) == 1 else axes
            fig.suptitle(f'Performance for Stress Case: {case_name}', fontsize=24, y=1.03, family='Times New Roman')
            case_metrics = {} # Store R2 for display

            for i, target_name in enumerate(target_names):
                ax = axes[i]; res_t = results[target_name]
                # Ensure mask length matches target length
                if len(mask) != len(res_t['true_orig']):
                     print(f"FATAL WARNING: Mask length ({len(mask)}) mismatch for target {target_name} ({len(res_t['true_orig'])}). Skipping subplot.")
                     continue

                target_mask = mask & np.isfinite(res_t['true_orig']) & np.isfinite(res_t['predicted_orig']);
                y_true_orig_finite = res_t['true_orig'][target_mask];
                y_pred_orig_finite = res_t['predicted_orig'][target_mask];
                color = colors[target_name]
                current_num_samples = len(y_true_orig_finite) # Samples for this specific target

                if current_num_samples < 2:
                     ax.text(0.5, 0.5, "Not enough data", ha='center', va='center', transform=ax.transAxes)
                     case_metrics[f'{target_name}_orig_r2'] = np.nan # Store NaN R2
                     continue

                r2_stress = r2_score(y_true_orig_finite, y_pred_orig_finite)
                case_metrics[f'{target_name}_orig_r2'] = r2_stress # Store R2

                ax.scatter(y_true_orig_finite, y_pred_orig_finite, alpha=0.5, s=30, color=color, edgecolors='k', linewidths=0.5)
                min_val_data = max(1e-9, np.min([y_true_orig_finite.min(), y_pred_orig_finite.min()]) * 0.8); max_val_data = np.max([y_true_orig_finite.max(), y_pred_orig_finite.max()]) * 1.2; ax.plot([min_val_data, max_val_data], [min_val_data, max_val_data], 'k--', linewidth=1.5)
                ax.set_xlabel(f"Actual {target_name.capitalize()} (Orig.)", fontsize=18, family='Times New Roman'); ax.set_ylabel(f"Predicted {target_name.capitalize()} (Orig.)", fontsize=18, family='Times New Roman'); ax.tick_params(axis='both', labelsize=16); ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)

                use_log_scale = False
                if max_val_data / min_val_data > 100:
                    ax.set_xscale('log'); ax.set_yscale('log'); use_log_scale = True

                ax.set_xlim(min_val_data, max_val_data); ax.set_ylim(min_val_data, max_val_data);

                # FIX: Apply nbins ONLY if scale is linear
                if ax.get_xscale() == 'linear':
                    ax.locator_params(axis='x', nbins=5)
                if ax.get_yscale() == 'linear':
                    ax.locator_params(axis='y', nbins=5)

                r2_text = f'$R^2$={r2_stress:.3f} (N={current_num_samples})'; ax.text(0.04, 0.95, r2_text, transform=ax.transAxes, fontsize=14, va='top', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='grey'))

            plt.tight_layout(rect=[0, 0.03, 1, 0.97]);
            # Sanitize filename
            safe_case_name = "".join(c if c.isalnum() else "_" for c in case_name)
            output_filename = os.path.join(output_dir, f"stress_case_{safe_case_name}_performance.png"); plt.savefig(output_filename, dpi=300); print(f"  Saved plot to: {output_filename}"); plt.close(fig)
        except Exception as e:
             print(f"  Error plotting stress case {case_name}: {e}")
             traceback.print_exc() # Print traceback for debugging
             continue # Continue to next stress case


# --- Plotting Function 6: IM Correlation Comparison ---
# ... (Keep implementation) ...
def plot_im_correlations(results, output_dir):
    print("Generating IM Correlation Comparison plots..."); configure_plot_style(); target_names = list(results.keys()); units = {"acceleration": r'$(cm/s^2)$', "velocity": r'$(cm/s)$', "arias": r'$(m/s)$'}; pairs = list(combinations(target_names, 2)); n_pairs = len(pairs); n_cols = 2; n_rows = n_pairs; fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), squeeze=False)
    fig.suptitle('Intensity Measure Correlations: Actual vs. Predicted (Original Space)', fontsize=26, y=1.0, family='Times New Roman')
    for i, (im1_name, im2_name) in enumerate(pairs):
        ax_actual = axes[i, 0]; im1_true = results[im1_name]['true_orig']; im2_true = results[im2_name]['true_orig']; mask_actual = np.isfinite(im1_true) & np.isfinite(im2_true) & (im1_true > 0) & (im2_true > 0); im1_true_f = im1_true[mask_actual]; im2_true_f = im2_true[mask_actual]
        if len(im1_true_f) > 1: corr_actual = np.corrcoef(np.log10(im1_true_f), np.log10(im2_true_f))[0, 1]; ax_actual.scatter(im1_true_f, im2_true_f, alpha=0.4, s=20, color='#1f77b4', edgecolors='k', linewidths=0.3); corr_text = f'Actual Corr($\\log_{{10}}$): {corr_actual:.3f}'; ax_actual.text(0.04, 0.95, corr_text, transform=ax_actual.transAxes, fontsize=14, va='top', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='grey'))
        else: ax_actual.text(0.5, 0.5, "Not enough data", ha='center', va='center', transform=ax_actual.transAxes)
        ax_actual.set_xlabel(f"Actual {im1_name.capitalize()} {units.get(im1_name, '')}", fontsize=18, family='Times New Roman'); ax_actual.set_ylabel(f"Actual {im2_name.capitalize()} {units.get(im2_name, '')}", fontsize=18, family='Times New Roman'); ax_actual.tick_params(axis='both', labelsize=16); ax_actual.set_xscale('log'); ax_actual.set_yscale('log'); ax_actual.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6); ax_actual.locator_params(axis='x', numticks=5); ax_actual.locator_params(axis='y', numticks=5)
        ax_pred = axes[i, 1]; im1_pred = results[im1_name]['predicted_orig']; im2_pred = results[im2_name]['predicted_orig']; mask_pred = np.isfinite(im1_pred) & np.isfinite(im2_pred) & (im1_pred > 0) & (im2_pred > 0); im1_pred_f = im1_pred[mask_pred]; im2_pred_f = im2_pred[mask_pred]
        if len(im1_pred_f) > 1: corr_pred = np.corrcoef(np.log10(im1_pred_f), np.log10(im2_pred_f))[0, 1]; ax_pred.scatter(im1_pred_f, im2_pred_f, alpha=0.4, s=20, color='#ff7f0e', edgecolors='k', linewidths=0.3); corr_text = f'Predicted Corr($\\log_{{10}}$): {corr_pred:.3f}'; ax_pred.text(0.04, 0.95, corr_text, transform=ax_pred.transAxes, fontsize=14, va='top', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='grey'))
        else: ax_pred.text(0.5, 0.5, "Not enough data", ha='center', va='center', transform=ax_pred.transAxes)
        ax_pred.set_xlabel(f"Predicted {im1_name.capitalize()} {units.get(im1_name, '')}", fontsize=18, family='Times New Roman'); ax_pred.set_ylabel(f"Predicted {im2_name.capitalize()} {units.get(im2_name, '')}", fontsize=18, family='Times New Roman'); ax_pred.tick_params(axis='both', labelsize=16); ax_pred.set_xscale('log'); ax_pred.set_yscale('log'); ax_pred.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6); ax_pred.locator_params(axis='x', numticks=5); ax_pred.locator_params(axis='y', numticks=5)
        if len(im1_true_f)>0 and len(im1_pred_f)>0: xlim = (min(im1_true_f.min(), im1_pred_f.min()), max(im1_true_f.max(), im1_pred_f.max())); ax_actual.set_xlim(xlim); ax_pred.set_xlim(xlim)
        if len(im2_true_f)>0 and len(im2_pred_f)>0: ylim = (min(im2_true_f.min(), im2_pred_f.min()), max(im2_true_f.max(), im2_pred_f.max())); ax_actual.set_ylim(ylim); ax_pred.set_ylim(ylim)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); output_filename = os.path.join(output_dir, "im_correlation_comparison.png"); plt.savefig(output_filename, dpi=300); print(f"Saved plot to: {output_filename}"); plt.close(fig)


# --- Plotting Function 6: Between-Within Event Residuals ---
# ... (Keep implementation) ...
# --- MODIFIED: Plot histograms instead of scatter ---
def plot_between_within_event_residuals(results, group_labels_test, output_dir):
    if group_labels_test is None:
        print("Skipping Between/Within Event Residual plot: Group labels (event IDs) unavailable.")
        return
    print("Generating Between-Within Event Residual plots...")
    configure_plot_style()
    target_names = list(results.keys())
    colors = {'between': '#1f77b4', 'within': '#ff7f0e', 'total': '#2ca02c'}
    fig, axes = plt.subplots(len(target_names), 1, figsize=(8, 6 * len(target_names)), sharex=True)
    axes = [axes] if len(target_names) == 1 else axes # Ensure axes is always iterable

    fig.suptitle('Between-Event and Within-Event Residual Distributions (Log Space)', fontsize=24, y=1.02, family='Times New Roman')

    # Convert group labels to a unique identifier for each event (assuming tuple format)
    try:
        # Handle potential different structures in group_labels_test
        if group_labels_test.ndim == 2 and group_labels_test.shape[1] == 2:
             # If it's Nx2 array of (STID, QuakeName)
             event_ids = np.array(['_'.join(map(str, pair)) for pair in group_labels_test])
        elif group_labels_test.ndim == 1:
             # If it's already a 1D array of identifiers
             event_ids = group_labels_test
        else:
            raise ValueError(f"Unexpected shape for group_labels_test: {group_labels_test.shape}") # Corrected indentation
    except Exception as e: # Corrected indentation
        print(f"Error processing group_labels_test: {e}. Cannot generate Between/Within plots.")
        return # Corrected indentation

    for i, target_name in enumerate(target_names):
        ax = axes[i]
        residuals_log = results[target_name]['residuals_log']

        # Ensure event_ids and residuals_log have the same length
        if len(event_ids) != len(residuals_log):
            print(f"FATAL WARNING: Length mismatch between event_ids ({len(event_ids)}) and residuals ({len(residuals_log)}) for {target_name}. Skipping.")
            continue

        finite_mask = np.isfinite(residuals_log)
        residuals_log_finite = residuals_log[finite_mask]
        event_ids_finite = event_ids[finite_mask]
        unique_events, event_counts = np.unique(event_ids_finite, return_counts=True)

        print(f"  Analyzing {target_name.capitalize()} across {len(unique_events)} unique events.")

        if len(unique_events) < 2 or len(residuals_log_finite) < 10:
            print(f"Warn: Not enough events/data for {target_name}. Skipping.")
            ax.text(0.5, 0.5, "Not enough data/events", ha='center', va='center', transform=ax.transAxes)
            continue

        between_event_residuals = []
        within_event_residuals = []

        for event in unique_events:
            event_mask = (event_ids_finite == event)
            res_event = residuals_log_finite[event_mask]
            if len(res_event) > 0:
                mean_residual_event = np.mean(res_event)
                between_event_residuals.append(mean_residual_event)
                if len(res_event) > 1: # Need more than 1 record for within-event variability
                    within_event_residuals.extend(res_event - mean_residual_event)

        between_event_residuals = np.array(between_event_residuals)
        within_event_residuals = np.array(within_event_residuals)

        tau = np.std(between_event_residuals) if len(between_event_residuals) > 1 else np.nan
        phi = np.std(within_event_residuals) if len(within_event_residuals) > 1 else np.nan
        sigma = np.std(residuals_log_finite) if len(residuals_log_finite) > 1 else np.nan

        # Determine common range for histograms
        all_res = np.concatenate([between_event_residuals, within_event_residuals, residuals_log_finite])
        q_low, q_high = np.percentile(all_res[np.isfinite(all_res)], [1, 99])
        hist_range = (q_low - 0.1*(q_high-q_low), q_high + 0.1*(q_high-q_low))
        bins = 50

        # Plot Histograms
        ax.hist(between_event_residuals, bins=bins, range=hist_range, density=True, alpha=0.6, color=colors['between'], label=f'Between ($\\tau={tau:.3f}$)')
        ax.hist(within_event_residuals, bins=bins, range=hist_range, density=True, alpha=0.6, color=colors['within'], label=f'Within ($\\phi={phi:.3f}$)')
        ax.hist(residuals_log_finite, bins=bins, range=hist_range, density=True, alpha=0.3, color=colors['total'], label=f'Total ($\\sigma={sigma:.3f}$)')

        ax.axvline(0, color='black', linestyle='-', linewidth=1.1)
        ax.set_xlabel(f"Log Residual", fontsize=18, family='Times New Roman')
        ax.set_ylabel("Density", fontsize=18, family='Times New Roman')
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=5)
        ax.set_title(f"{target_name.capitalize()}", fontsize=20, family='Times New Roman')
        ax.legend(fontsize=14)
    # --- END MODIFICATION ---

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect to prevent title overlap
    output_filename = os.path.join(output_dir, "between_within_event_residuals_hist.png") # Changed filename
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot to: {output_filename}")
    plt.close(fig)

# --- RE-INTRODUCED: Plot Actual vs Predicted Comparison ---
def plot_actual_vs_predicted_comparison(target_name,
                                        y_true_log_train, y_pred_log_train,
                                        y_true_orig_train, y_pred_orig_train,
                                        y_true_log_test, y_pred_log_test,
                                        y_true_orig_test, y_pred_orig_test,
                                        output_dir):
    """Plots Actual vs Predicted for Train and Test sets (Log and Original space)."""
    # Check if data for this target is available for both train and test
    comparison_possible = (y_true_log_train is not None and y_pred_log_train is not None and
                           y_true_orig_train is not None and y_pred_orig_train is not None and
                           y_true_log_test is not None and y_pred_log_test is not None and
                           y_true_orig_test is not None and y_pred_orig_test is not None)

    if not comparison_possible:
        print(f"Skipping Actual vs Predicted Comparison plot for {target_name.capitalize()}: Missing train or test data/predictions.")
        return

    print(f"Generating Actual vs Predicted Comparison plot for {target_name.capitalize()}...")
    configure_plot_style()

    titles = {"acceleration": "Acceleration", "velocity": "Velocity", "arias": "Arias Intensity"}
    units = {"acceleration": r'$(cm/s^2)$', "velocity": r'$(cm/s)$', "arias": r'$(m/s)$'}
    full_name = titles.get(target_name, target_name.capitalize()); unit = units.get(target_name, '')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # 2x2 layout
    fig.suptitle(f'Actual vs. Predicted: {full_name}', fontsize=26, y=1.0, family='Times New Roman') # Removed unit from main title

    plot_configs = [
        {'ax': axes[0, 0], 'title': 'Train Set (Log Space)', 'x': y_true_log_train, 'y': y_pred_log_train, 'color': '#1f77b4', 'log_scale': False, 'xlabel': f'Actual Log {full_name}', 'ylabel': f'Predicted Log {full_name}'},
        {'ax': axes[0, 1], 'title': 'Test Set (Log Space)', 'x': y_true_log_test, 'y': y_pred_log_test, 'color': '#1f77b4', 'log_scale': False, 'xlabel': f'Actual Log {full_name}', 'ylabel': f'Predicted Log {full_name}'},
        {'ax': axes[1, 0], 'title': 'Train Set (Original Space)', 'x': y_true_orig_train, 'y': y_pred_orig_train, 'color': '#ff7f0e', 'log_scale': True, 'xlabel': f'Actual {full_name} {unit}', 'ylabel': f'Predicted {full_name} {unit}'},
        {'ax': axes[1, 1], 'title': 'Test Set (Original Space)', 'x': y_true_orig_test, 'y': y_pred_orig_test, 'color': '#ff7f0e', 'log_scale': True, 'xlabel': f'Actual {full_name} {unit}', 'ylabel': f'Predicted {full_name} {unit}'},
    ]

    for cfg in plot_configs:
        ax = cfg['ax']; x_data = cfg['x']; y_data = cfg['y']

        # Data already checked at function start, but double-check just in case
        if x_data is None or y_data is None:
             print(f"  Internal Warning: Missing data for subplot '{cfg['title']}'. Skipping.")
             ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center', transform=ax.transAxes)
             continue

        # Filter finite values for plotting and R2 calculation
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_plot = x_data[finite_mask]
        y_plot = y_data[finite_mask]

        if len(x_plot) < 2:
            print(f"  Warning: Not enough finite data points ({len(x_plot)}) for subplot '{cfg['title']}'. Skipping.")
            ax.text(0.5, 0.5, "Not enough data", ha='center', va='center', transform=ax.transAxes)
            continue

        # Scatter plot
        ax.scatter(x_plot, y_plot, alpha=0.4, s=20, color=cfg['color'], edgecolors='k', linewidths=0.3)

        # Calculate R2 score
        try:
            r2 = r2_score(x_plot, y_plot)
            r2_text = f'$R^2$={r2:.3f}';
        except ValueError: # Handle cases like constant input
            r2_text = '$R^2$=Undef.'
        ax.text(0.04, 0.95, r2_text, transform=ax.transAxes, fontsize=14, va='top', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='grey'))

        # Determine plot limits and add 1:1 line
        min_val_data = np.min([x_plot.min(), y_plot.min()])
        max_val_data = np.max([x_plot.max(), y_plot.max()])

        # Handle case where min == max
        if np.isclose(min_val_data, max_val_data):
             range_val = abs(min_val_data) * 0.2 if not np.isclose(min_val_data, 0) else 1.0 # Add small range if constant
        else:
             range_val = max_val_data - min_val_data


        # Add padding to limits
        plot_min = min_val_data - 0.05 * range_val
        plot_max = max_val_data + 0.05 * range_val

        # Ensure positive limits for log scale
        if cfg['log_scale']:
             plot_min_log = max(1e-9, x_plot[x_plot > 1e-9].min() if np.any(x_plot > 1e-9) else 1e-9,
                                y_plot[y_plot > 1e-9].min() if np.any(y_plot > 1e-9) else 1e-9)
             plot_max_log = max(x_plot.max(), y_plot.max())
             # Use log-based padding if range is large
             if plot_max_log / plot_min_log > 10:
                 plot_min = plot_min_log / 1.5
                 plot_max = plot_max_log * 1.5
             else: # Use linear padding for smaller log ranges
                 plot_min = max(1e-9, plot_min_log * 0.9)
                 plot_max = plot_max_log * 1.1

             # Final check to ensure min < max
             if plot_min >= plot_max:
                 plot_max = plot_min * 10
             plot_min = max(1e-9, plot_min) # Ensure positivity after adjustments

        # Ensure 1:1 line uses appropriate limits
        line_min = max(plot_min, 1e-9) if cfg['log_scale'] else plot_min
        line_max = plot_max
        ax.plot([line_min, line_max], [line_min, line_max], 'k--', linewidth=1.2)

        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        ax.set_title(cfg['title'], fontsize=18, family='Times New Roman')
        ax.set_xlabel(cfg['xlabel'], fontsize=16, family='Times New Roman')
        ax.set_ylabel(cfg['ylabel'], fontsize=16, family='Times New Roman')
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)

        if cfg['log_scale']:
            ax.set_xscale('log')
            ax.set_yscale('log')
            # Auto minor ticks are usually good for log
            ax.minorticks_on()
        else:
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap
    output_filename = os.path.join(output_dir, f"{target_name}_actual_vs_predicted_comparison.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot to: {output_filename}")
    plt.close(fig)

def main():
    # No arguments needed as paths are hardcoded
    # --- Define Base Project Directory ---
    base_dir = os.path.expanduser("~/Desktop/fyp 2_debug1")

    # --- HARDCODED Paths ---
    # Data directory is fixed to 'model_data' under base_dir
    model_data_dir = os.path.join(base_dir, "model_data")
    # --- MODIFIED: Path to the model FILE ---
    specific_model_filepath = os.path.join(base_dir, "models/final_model_loadable.keras")
    # Output directory (e.g., save plots in the base directory or a 'plots' folder)
    output_dir = os.path.join(base_dir, "plots_critical_assessment") # New dedicated folder

    # --- Validate Paths ---
    if not os.path.isfile(specific_model_filepath):
        print(f"Error: Specified model FILE does not exist: {specific_model_filepath}")
        return
    if not os.path.isdir(model_data_dir):
        print(f"Error: Model data directory does not exist: {model_data_dir}")
        return

    # Create output directory if it doesn't exist with specific error handling
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f" Ensured output directory exists: {output_dir}") # ADDED Confirmation print
        # Verify write permissions (optional but helpful for debugging)
        if not os.access(output_dir, os.W_OK):
             print(f"WARNING: No write permissions for output directory: {output_dir}")
    except OSError as e:
        print(f"ERROR: Could not create output directory '{output_dir}': {e}")
        traceback.print_exc() # Print traceback for directory error
        return # Exit if directory creation fails

    print(f"Using Model File: {specific_model_filepath}")
    print(f"Using Data Directory : {model_data_dir}")
    print(f"Output Directory for Assessment Plots: {output_dir}")


    try: # Start of main try block
        # Load data including original static features
        X_train_scaled, X_test_scaled, train_targets, test_targets, \
         X_test_unscaled, group_labels_test, X_test_static_original = load_processed_data(model_data_dir)

        # Basic check if data loaded
        if X_test_scaled is None or test_targets['acceleration']['log'] is None:
             print("Error: Essential test data files (.npy) could not be loaded. Cannot proceed.")
             return

        # Optional: Check if loaded labels match X_test length
        if group_labels_test is not None and len(group_labels_test) != X_test_scaled.shape[0]:
            print(f"FATAL WARNING: Loaded group_labels_test count ({len(group_labels_test)}) does not match X_test count ({X_test_scaled.shape[0]}). Ensure corresponding files were saved correctly.")
            # Depending on severity, you might want to exit:
            # return


        # Load model
        model = load_model(specific_model_filepath)

        # Calculate residuals and metrics
        test_results = calculate_all_residuals_and_metrics(model, X_test_scaled, test_targets)

        # Plot 1: True vs Residuals
        for target_name in test_results.keys():
            print(f"\nAttempting to generate Test True vs Residual plot for {target_name}...")
            plot_single_parameter(target_name=target_name, target_results=test_results[target_name], output_dir=output_dir)

        # Plot 2: Residuals vs Parameters (Test Set Only)
        print("\nAttempting to generate Test Residuals vs Parameters plots...")
        plot_residuals_vs_parameters(test_results, X_test_unscaled, X_test_static_original, output_dir)

        # Plot 3: Performance vs Input Sensor Height (Test Set Only)
        print("\nAttempting to generate Test Performance vs Input Sensor Height plot...")
        plot_performance_vs_input_sensor_height(test_results, X_test_unscaled, output_dir)

        # Plot 4: Stress Case Performance (Test Set Only)
        print("\nAttempting to generate Test Stress Case plots...")
        required_stress_indices = [20, 21, 23, 24]
        if X_test_unscaled is not None and X_test_unscaled.ndim == 2 and all(idx < X_test_unscaled.shape[1] for idx in required_stress_indices):
            plot_stress_case_performance(test_results, X_test_unscaled, output_dir)
        else:
            print(f"Skipping Stress Case plots: X_test_unscaled condition not met.")


        # Plot 5: IM Correlation Comparison (Test Set Only)
        print("\nAttempting to generate Test IM Correlation plot...")
        plot_im_correlations(test_results, output_dir)

        # Plot 6: Between/Within Event Residuals (Test Set Only)
        print("\nAttempting to generate Test Between/Within Residuals plot...")
        if group_labels_test is not None:
            plot_between_within_event_residuals(test_results, group_labels_test, output_dir)
        else:
            print("Skipping Between/Within Event Residual plot: Group labels (group_labels_test.npy) not found.")

        # --- Check if data needed for comparison plots is available ---
        comparison_data_available = (X_train_scaled is not None and
                                     train_targets['acceleration']['log'] is not None) # Check one essential train target
        if not comparison_data_available:
            print("Warning: Training data files missing. Comparison plots cannot be generated.")
            # Set train_predictions to empty if data is missing
            train_predictions = {}
        else:
            print("Training data found. Proceeding with comparison plot generation...")


        # --- Calculate TRAIN predictions (if data available) ---
        train_predictions = {} # Initialize dictionary
        if comparison_data_available:
            print(f"\nPredicting on X_train_scaled (Shape: {X_train_scaled.shape})...") # Added newline
            start_time = time.time()
            try: # Added try block for prediction
                predictions_train_log_list = model.predict(X_train_scaled, batch_size=4096, verbose=1)
                end_time = time.time()
                print(f"Training prediction finished in {end_time - start_time:.2f} seconds.")

                target_names = ['acceleration', 'velocity', 'arias']
                if not isinstance(predictions_train_log_list, list) or len(predictions_train_log_list) != len(target_names):
                     print("ERROR: Model output format mismatch on training data.")
                     train_predictions = {} # Reset predictions
                     comparison_data_available = False # Mark as failed
                     # raise ValueError("Model output format mismatch on train data") # Optional: stop execution
                else:
                    # Store train predictions (log and original)
                    for i, target_name in enumerate(target_names):
                         y_train_log = train_targets.get(target_name, {}).get('log') # Use .get for safety
                         y_train_orig = train_targets.get(target_name, {}).get('orig')
                         pred_train_log = predictions_train_log_list[i].flatten()

                         if y_train_log is None or y_train_orig is None:
                             print(f"WARN: Missing actual train data for {target_name}. Comparison plot will fail.")
                             train_predictions[target_name] = {'predicted_log': None, 'predicted_orig': None}
                             comparison_data_available = False
                             continue

                         if y_train_log.shape != pred_train_log.shape:
                             print(f"ERROR: Shape mismatch for TRAIN {target_name}. True: {y_train_log.shape}, Pred: {pred_train_log.shape}")
                             train_predictions[target_name] = {'predicted_log': None, 'predicted_orig': None}
                             comparison_data_available = False
                             continue

                         pred_train_orig = safe_inverse_log10(pred_train_log, y_orig=y_train_orig)
                         train_predictions[target_name] = {'predicted_log': pred_train_log, 'predicted_orig': pred_train_orig}
                         print(f"  Stored train predictions for {target_name}.") # Added confirmation
            except Exception as e: # Added except block
                print(f"ERROR during training data prediction: {e}")
                traceback.print_exc()
                train_predictions = {} # Reset predictions
                comparison_data_available = False # Cannot do comparison if prediction fails

        # =================== PLOTTING ===================

        # --- Plot 1: True vs Residuals (Test Set Only) ---
        # ... (this part should be present as it ran) ...

        # --- *** ADD THIS SECTION BACK *** ---
        # --- Plot Actual vs Predicted Comparison (Train vs Test) ---
        print("\nAttempting to generate Actual vs Predicted Comparison plots...")
        if comparison_data_available: # Correct indentation
            print("Sufficient data available for comparison plots.")
            for target_name in test_results.keys(): # Corrected loop structure
                # Check if predictions for this specific target exist
                if target_name in train_predictions and train_predictions[target_name].get('predicted_log') is not None: # Correct indentation
                    plot_actual_vs_predicted_comparison( # Correct indentation
                        target_name=target_name,
                        y_true_log_train=train_targets.get(target_name, {}).get('log'),
                        y_pred_log_train=train_predictions.get(target_name, {}).get('predicted_log'),
                        y_true_orig_train=train_targets.get(target_name, {}).get('orig'),
                        y_pred_orig_train=train_predictions.get(target_name, {}).get('predicted_orig'),
                        y_true_log_test=test_results[target_name]['true_log'],
                        y_pred_log_test=test_results[target_name]['predicted_log'],
                        y_true_orig_test=test_results[target_name]['true_orig'],
                        y_pred_orig_test=test_results[target_name]['predicted_orig'],
                        output_dir=output_dir
                    )
                else: # Correct indentation
                    print(f"Skipping comparison plot for {target_name} due to missing train predictions for this target.") # Correct indentation
        else: # Correct indentation
             print("Skipping Actual vs Predicted Comparison plots due to missing/failed training data or predictions check earlier.") # Correct indentation
        # --- *** END OF SECTION TO ADD BACK *** ---


        # --- Plot 2: Residuals vs Parameters (Test Set Only) ---
        # ... (rest of the plotting calls for test set) ...

        print("\nAnalysis complete.") # Added newline after all plotting


    except FileNotFoundError as e: # Correct indentation for main except
        print(f"\n--- File Not Found Error During Data Loading ---") # Correct indentation
        print(f"Could not find a required data file (.npy): {e}") # Correct indentation # Removed pkl mention
        print("Please ensure 'transformer_1.py' was run successfully and saved all necessary files in the 'model_data' directory.") # Correct indentation
    except Exception as e: # Correct indentation for main except # Catch other errors (like potential savefig issues if not handled within function)
        print(f"\n--- An error occurred during the analysis ---") # Correct indentation
        print(f"Error Type: {type(e).__name__}") # Correct indentation
        print(f"Error Details: {e}") # Correct indentation
        print("\n--- Traceback ---") # Correct indentation
        traceback.print_exc() # Correct indentation
        print("-----------------\n") # Correct indentation

if __name__ == "__main__":
    main() 
# Seismic Response Prediction Models

This repository contains transformer-based deep learning models for predicting seismic responses in buildings.

## File Structure

- `transformer_1.py`: Main transformer model for seismic response prediction
- `transformer_3.py`: Testing variant of the transformer model
- `plot.py`: Generates visualization plots from model predictions
- `plot_2.py`: Alternative plotting script with additional visualizations
- `convert.py`: Utility script to convert raw Keras models for use with plot_2.py

## Data Directories

- `combined_seismic_data`: Main dataset used by both transformer models
- `plots`: Contains model performance graphs and visualizations
- `clean_data_visualizations_FULL`: Event data visualizations from the full dataset
- `cleaned_data_visualizations_DEBUG_10_GROUPS`: Event data visualizations from debug mode with limited groups
- `model_data`: Directory where trained models and training history are saved

## Running the Models

### Running the Main Model (transformer_1.py)

```bash
python transformer_1.py
```

This will:
1. Load the dataset from `combined_seismic_data`
2. Train the transformer model on the full dataset
3. Save the model and metrics to the `model_data` directory
4. Generate performance plots in the `plots` directory

### Running the Testing Model (transformer_3.py)

```bash
python transformer_3.py --debug True --max_groups 10
```

Parameters:
- `--debug`: Set to `True` to enable debug mode (default: False)
- `--max_groups`: Number of groups to use for testing (only active in debug mode)

The testing model is primarily used for development and debugging purposes.

#
```bash
python plot_2.py
```

## Model Data

- Training history is saved to `model_data/training_history.pkl`
- Model weights are saved to `model_data/transformer_model.keras`
- Model metrics are saved along with the model

## convert.py

convert.py is a utility script that reconstructs and converts a raw Keras model (often with Lambda layers issues) into a loadable format for advanced plotting. It:

- Determines the input dimension from the saved feature scaler (`model_data/scaler_X.joblib`).
- Rebuilds the model architecture using `build_transformer_model()` from `transformer_1.py`.
- Loads weights from the original saved Keras model (`models/final_model.keras`).
- Saves a new model file (`models/final_model_loadable.keras`) that can be loaded without deserialization errors.

Usage(optional):
```bash
python convert.py
```

You can edit the `main_converter()` function in `convert.py` to specify a different model filename:
```python
original_model_path = os.path.join(base_dir, "models", "final_model.keras")
    # --- Path for the NEW, corrected model file ---
new_model_path = os.path.join(base_dir, "models", "final_model_loadable.keras")
```

### plot_2.py

`plot_2.py` is an advanced visualization script for detailed model assessment. It:

- Loads the converted model (`models/final_model_loadable.keras`).
- Reads processed data arrays (scaled/unscaled features and targets) from `model_data/`.
- Computes residuals, R² scores, and other metrics.
- Generates multiple plot types:
  - True vs Residual (log & original)
  - Residuals vs Parameters
  - Performance vs Input Sensor Height
  - Floor-Level Performance
  - Stress-Case Performance
  - Intensity Measure Correlation Comparisons
  - Between/Within Event Residual Distributions
  - Actual vs Predicted Comparisons for Train & Test

Usage:
```bash
python plot_2.py
```

All figures are saved to `plots_critical_assessment/` under the project directory.

import os
import pandas as pd
import re
import numpy as np

# File paths setup
desktop_path = os.path.expanduser("~/Desktop")
fyp_folder = os.path.join(desktop_path, "fyp")
peak_acceleration_file = os.path.join(fyp_folder, "peak_acceleration_data.csv")
gm_metadata_file = os.path.join(fyp_folder, "GM_METADATA.csv")
station_info_file = os.path.join(fyp_folder, "STATION_INFO.csv")
output_file = os.path.join(fyp_folder, "combined_seismic_data.csv")

# Load Peak Acceleration Data
peak_accel_df = pd.read_csv(peak_acceleration_file, encoding="ISO-8859-1")

# Print peak_accel_df columns for debugging
print("\n🔍 Columns in peak_accel_df:")
print(peak_accel_df.columns.tolist())

# Fix encoding issue with column names
if any('cm/s' in col for col in peak_accel_df.columns):
    # Find the actual acceleration column name that contains "cm/s"
    accel_col = [col for col in peak_accel_df.columns if 'cm/s' in col and 'Velocity' not in col][0]
    velocity_col = [col for col in peak_accel_df.columns if 'Velocity' in col][0]
    arias_col = [col for col in peak_accel_df.columns if 'Arias' in col][0]
    
    # Rename to standardized names
    peak_accel_df.rename(columns={
        accel_col: "Peak Acceleration (cm/s²)",
        velocity_col: "Peak Velocity (cm/s)",
        arias_col: "Arias Intensity (m/s)"
    }, inplace=True)

# Rename columns in Peak Acceleration file
if "Event" in peak_accel_df.columns:
    peak_accel_df.rename(columns={"Event": "Earthquake Name"}, inplace=True)
if "Station" in peak_accel_df.columns:
    peak_accel_df.rename(columns={"Station": "STID"}, inplace=True)

# After renaming, print again to verify
print("\n🔍 Columns in peak_accel_df after renaming:")
print(peak_accel_df.columns.tolist())

# Load GM_METADATA
gm_metadata_raw = pd.read_csv(gm_metadata_file, encoding="ISO-8859-1", skiprows=1, header=None)
gm_metadata_df = gm_metadata_raw.iloc[1:].reset_index(drop=True)
gm_metadata_df.columns = gm_metadata_raw.iloc[0].dropna()

# Modified STATION_INFO loading and processing
station_info_raw = pd.read_csv(station_info_file, encoding="ISO-8859-1", header=None)

# Extract headers from row 3 (index 2)
headers_row = station_info_raw.iloc[2]
# Extract sensor IDs from row 2 (index 1)
sensor_ids_row = station_info_raw.iloc[1]
# Get the data starting from row 4 (index 3)
station_data = station_info_raw.iloc[3:].reset_index(drop=True)

# Define the base columns we want to capture
base_column_names = [
    "STID", "Station Name", "Stories Above Ground", "No. of Stories Below Ground",
    "Base Length (in)", "Base Breadth (in)", "Typical Floor Length (in)", 
    "Typical Floor Breadth (in)", "Vs30 (m/s)", "Total_Height (in)", "Stories", 
    "BDType"
]

# Find indices of base columns
base_column_indices = []
base_column_found = []
for col_name in base_column_names:
    for i, header in enumerate(headers_row):
        if header == col_name:
            base_column_indices.append(i)
            base_column_found.append(col_name)
            break

# Create station_info_df with found base columns
station_info_df = pd.DataFrame(
    station_data.iloc[:, base_column_indices].values,
    columns=base_column_found
)

# Create sensor properties DataFrames
sensor_properties = {}
current_col = len(base_column_found)  # Start after base columns

# Process sensor columns in blocks of 3
while current_col < len(headers_row):
    sensor_id = str(sensor_ids_row[current_col]).strip()
    if sensor_id.startswith('S'):
        sensor_num = int(sensor_id[1:])
        # Create DataFrame for this sensor's properties
        sensor_df = pd.DataFrame({
            'Height': station_data.iloc[:, current_col],
            'Breadth': station_data.iloc[:, current_col + 1],
            'Length': station_data.iloc[:, current_col + 2]
        })
        sensor_properties[sensor_num] = sensor_df
        current_col += 3
    else:
        current_col += 1

# Extract Sensor ID from peak acceleration data
def extract_sensor_id(sensor_filename):
    match = re.search(r"GM_[XYZ]_\d+_(\d+)_", str(sensor_filename))
    return int(match.group(1)) if match else None

# Extract sensor direction (X, Y, or Z) from filename
def extract_sensor_direction(sensor_filename):
    match = re.search(r"GM_([XYZ])_", str(sensor_filename))
    return match.group(1) if match else None

if "Sensor File" in peak_accel_df.columns:
    peak_accel_df["Sensor_ID"] = peak_accel_df["Sensor File"].apply(extract_sensor_id)
    peak_accel_df["Sensor_Direction"] = peak_accel_df["Sensor File"].apply(extract_sensor_direction)

# Convert STID format
for df in [peak_accel_df, gm_metadata_df, station_info_df]:
    df["STID"] = df["STID"].astype(str).str.strip().apply(lambda x: x if x.startswith("CE") else "CE" + x)

# Merge the base station info
combined_df = peak_accel_df.merge(
    station_info_df,
    on="STID",
    how="left"
)

# Function to get sensor properties
def get_sensor_properties(stid, sensor_id):
    if pd.isna(sensor_id):
        return pd.Series([None, None, None])
    
    sensor_num = int(sensor_id)
    if sensor_num in sensor_properties:
        # Get the row index from station_info_df for this STID
        station_idx = station_info_df[station_info_df['STID'] == stid].index
        if len(station_idx) > 0:
            idx = station_idx[0]
            sensor_data = sensor_properties[sensor_num].iloc[idx]
            return pd.Series([
                sensor_data['Height'],
                sensor_data['Breadth'],
                sensor_data['Length']
            ])
    return pd.Series([None, None, None])

# Add sensor properties
sensor_props = combined_df.apply(
    lambda row: get_sensor_properties(row['STID'], row['Sensor_ID']), 
    axis=1
)
combined_df["Height (in)"] = sensor_props[0]
combined_df["Breadth (in)"] = sensor_props[1]
combined_df["Length (in)"] = sensor_props[2]

# Merge GM_METADATA
combined_df = combined_df.merge(
    gm_metadata_df[["STID", "Earthquake Name", "Magnitude", "PGA", "Depth", "Repi"]],
    on=["STID", "Earthquake Name"],
    how="left"
)

# Add log-transformed peak acceleration
combined_df["Log_Peak_Acceleration"] = np.log10(combined_df["Peak Acceleration (cm/s²)"].abs())

# Add log-transformed peak velocity
combined_df["Log_Peak_Velocity"] = np.log10(combined_df["Peak Velocity (cm/s)"].abs())

# Add log-transformed Arias intensity
combined_df["Log_Arias_Intensity"] = np.log10(combined_df["Arias Intensity (m/s)"].abs())

# Format Sensor_ID as 'SXX'
combined_df["Sensor_ID"] = combined_df["Sensor_ID"].apply(lambda x: f"S{int(x)}" if pd.notna(x) else None)

# Update keep_columns list to include the new columns
keep_columns = [
    "STID", "Sensor_ID", "Sensor_Direction", "Earthquake Name", 
    "Peak Acceleration (cm/s²)", "Log_Peak_Acceleration", 
    "Peak Velocity (cm/s)", "Log_Peak_Velocity",
    "Arias Intensity (m/s)", "Log_Arias_Intensity",
    "Magnitude", "PGA", "Depth", "Repi", 
    "Height (in)", "Breadth (in)", "Length (in)", "Stories", 
    "Total_Height (in)", "BDType", "Base Length (in)", 
    "Base Breadth (in)", "Typical Floor Length (in)", "Typical Floor Breadth (in)"
]

# Filter keep_columns to only include columns that exist
keep_columns = [col for col in keep_columns if col in combined_df.columns]

# Keep only available columns
combined_df = combined_df[keep_columns]

# Save the final output
combined_df.to_csv(output_file, index=False)

print(f"\n✅ Processed data saved at: {output_file}")
print("\n🔍 Columns in final output:")
print(combined_df.columns.tolist())
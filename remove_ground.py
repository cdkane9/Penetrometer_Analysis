import pandas as pd
import os
import ast
import re



def read_scope(file_path):
    scope_df = pd.read_csv(file_path, skiprows=1, usecols=[0, 1])
    idx_init = scope_df[scope_df.iloc[:, 0] == 'depth (mm)'].index[0]
    scope_raw = scope_df.iloc[idx_init + 1:].reset_index(drop=True).astype(float)
    scope_raw.columns = ['depth (mm)', 'hardness (kPa)']

    return scope_raw

all_profs = pd.read_csv('/Users/colemankane_1/Documents/BSU/Penetrometer_Analysis/all_profiles_thinned.csv')
base_dir = '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports'

clean_scopes_25 = pd.read_csv('scope_cleaning_filled.csv')
clean_scopes_24 = pd.read_csv('scope_24_cleaning_filled.csv')
clean_scopes = pd.concat([clean_scopes_25, clean_scopes_24])

ground_scopes = clean_scopes[clean_scopes['ground'] == 1]['profile']
ground_scope_paths = pd.Series([os.path.join(base_dir, 'Snow_Scope', g) for g in ground_scopes])



clean_smps = pd.read_csv('smp_cleaning_filled.csv')
ground_smps = clean_smps[clean_smps['ground'] == 1]['profile']
ground_smp_paths = pd.Series([os.path.join(base_dir, 'smp_profiles_exports' ,g) for g in ground_smps])


def trim_bottom_5cm(file_path, type:str=None):
    """Opens a single profile CSV, locates depth/distance column,
    drops the bottom 50mm of data, and rewrites the file in place."""
    # set a cutoff for WY24, WY25
    wy24_cutoff = pd.to_datetime('2024-10-01')
    wy25_cutoff = pd.to_datetime('2025-10-01')

    date_pattern = r'\d{4}[-_]\d{1,2}[-_]\d{1,2}'
    if type:
        profile_id = os.path.basename(file_path)
        print(profile_id)
        match =  re.search(date_pattern, profile_id)
        date_str = match.group().replace('_', '-')
        date = pd.to_datetime(date_str)
        print(date)

        if date < wy24_cutoff:
            file_path = os.path.join('/Users/colemankane_1/Desktop/fraser_23_24_exports/Snow_Scopes', profile_id)

    if not os.path.exists(file_path):
        print(f"Warning: File not found for trimming: {file_path}")
        return

    try:
        # Load the profile dataset
        if type:
            profile_df = read_scope(file_path)
        else:
            profile_df = pd.read_csv(file_path)
            print(profile_df)

        # Identify which column tracking depth/distance is present
        depth_col = None
        for col in ['depth (mm)', 'distance']:
            if col in profile_df.columns:
                depth_col = col
                break

        if depth_col is None:
            print(f"Skipping {os.path.basename(file_path)}: Could not find depth/distance columns.")
            return

            # Calculate max depth to locate where it hit the ground
            max_depth = profile_df[depth_col].max()

            # Keep only the measurements that are above the bottom 50 mm (5 cm)
            trimmed_df = profile_df[profile_df[depth_col] <= (max_depth - 49)]

            # Overwrite the original profile data file directly in its parent directory
            #trimmed_df.to_csv(file_path, index=False)
            print(f"Successfully trimmed 5cm off the bottom of: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"Error processing profile {file_path}: {e}")


ground_scope_paths.apply(
    lambda x: trim_bottom_5cm(x, type='scope')
)
ground_smp_paths.apply(
    lambda x: trim_bottom_5cm(x)
)
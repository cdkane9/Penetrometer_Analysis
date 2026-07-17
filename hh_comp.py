import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from profile_matching import resample, same_depth
import os
import argparse
import ast

# pull out depth and force columns from ram profile
def ram_head(df):
    df['depth'] = df['l_cm'] * 10
    df['force'] = df['rr_N']
    return df

def smp_head(path_in):
    try:
        prof_raw = pd.read_csv(path_in, low_memory=False, skiprows=0, usecols=[1, 2])
        prof_raw.columns = ['depth', 'force']
    except pd.errors.ParserError:
        prof_raw = pd.read_csv(path_in, low_memory=False)
        prof_raw.columns = ['depth', 'force']
    return prof_raw
    
def scope_head(df):
    idx_init = df[df.iloc[:, 0] == 'depth (mm)'].index[0]
    scope_prof = df.iloc[idx_init + 1:].reset_index(drop=True).astype(float)
    scope_prof.columns = ['depth', 'force']
    # convert from kPa to N
    scope_prof['force'] *= 1000 * 0.0000554
    return scope_prof
    
    
    
def set_dtw(prof, ref_prof, window_cm=10, penalty=0.05, use_derivative=False):
    s1 = ref_prof['force'].values
    s2 = prof['force'].values
    depths = ref_prof['depth'].values

    s1max, s1min = np.max(s1), np.min(s1)
    s2max, s2min = np.max(s2), np.min(s2)
    s1_norm = (s1 - s1min) / (s1max - s1min + 1e-6)
    s2_norm = (s2 - s2min) / (s2max - s2min + 1e-6)

    if use_derivative:
        grad1 = np.gradient(s1_norm)
        grad2 = np.gradient(s2_norm)
        s1_input = (grad1 - np.min(grad1)) / (np.max(grad1) - np.min(grad1) + 1e-6)
        s2_input = (grad2 - np.min(grad2)) / (np.max(grad2) - np.min(grad2) + 1e-6)
    else:
        s1_input = s1_norm
        s2_input = s2_norm

    N, M = len(s1_input), len(s2_input)
    window = int(window_cm / 0.1)

    cost_mat = np.full((N, M), np.inf)
    cost_mat[0, 0] = 0

    for i in range(N):
        j_start = max(0, i - window)
        j_end = min(M, i + window)
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue
            v1 = cost_mat[i - 1, j] + penalty if i > 0 else np.inf
            v2 = cost_mat[i, j - 1] + penalty if j > 0 else np.inf
            v3 = cost_mat[i - 1, j - 1] if (i > 0 and j > 0) else np.inf
            cost_mat[i, j] = abs(s1_input[i] - s2_input[j]) + min(v1, v2, v3)

    path = []
    i, j = N - 1, M - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
            continue
        elif j == 0:
            i -= 1
            continue
        choices = [cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1]]
        best = np.argmin(choices)
        if best == 0: i -= 1
        elif best == 1: j -= 1
        else:
            i -= 1
            j -= 1
    path.append((0, 0))
    path.reverse()

    # Fixed coordinate reconstruction mapping to avoid peak attenuation
    ref_path_indices = [p[0] for p in path]
    prof_path_indices = [p[1] for p in path]
    matched_prof_indices = np.interp(np.arange(N), ref_path_indices, prof_path_indices)
    final_forces = s2[np.round(matched_prof_indices).astype(int)]

    return pd.DataFrame({'depth': depths, 'force': final_forces}), cost_mat, path


all_profs = pd.read_csv('all_profiles_thinned.csv')

prefixes_to_exclude = ('JPLMet_20240409', 'JPLMet_20240410', 'JPLMet_20250407_TS', 'JPLMet_20240405')

all_profs = all_profs[~all_profs['id'].str.startswith(prefixes_to_exclude)]

hh_categories = ['F', '4F', '1F', 'P', 'K', 'I']

def get_hardness(pen_df:pd.DataFrame, top:pd.Series, bottom:pd.Series, hs:float, grain_type, penetrometer, actual_depths=None):
    '''
    Function for determining force across stratigraphic layers
    :param type: penetrometer type
    :param pen_df: data from penetrometer with columns ['depth', 'force']
    :param top: top of layers [cm]
    :param bottom: bottom of layers [cm]
    :param hs: HS as measured from pitwall [cm]
    :return:
    '''
    
    if pen_df is None or pen_df.empty:
        return np.nan
    
    if actual_depths is not None:
        t_depth, b_depth = actual_depths
    
    else:

        delta = top - bottom
        
        t_depth = ((hs - top) + (0.15 * delta)) * 10
        b_depth = ((hs - bottom) - (0.15 * delta)) * 10

    layer_ix = pen_df.index[(pen_df['depth'] >= t_depth) &
                            (pen_df['depth'] <= b_depth)]
                            
    if layer_ix.empty: return np.nan
    else:
        mean = np.nanmean(pen_df.loc[layer_ix, 'force'])

    return mean
    
    
def compile_layers():
    '''
    Create a main dataframe that has all layers (top, bottom, HH, etc.), then add a column with average force
    '''
    all_layers = []
    strat_files = '/bsuhome/colemankane/Documents/crrel_exports/' + all_profs['id'] + '_strat.csv'
    
    for filename in strat_files:
        try:
            strat = pd.read_csv(filename)
            hs = strat['top_cm'].iloc[0]
            profile_id = os.path.basename(filename).replace('_strat.csv', '')
            strat['hs'] = hs
            strat['profile_id'] = profile_id
            all_layers.append(strat)
        except Exception as e:
            print(e)
            continue
    
    all_layers = pd.concat(all_layers, ignore_index=True)
    all_layers = all_layers.dropna(subset=['bottom_cm'])
    return all_layers
    
    
def get_pen_paths(profile_id, penetrometer):
    pen_map = {'smp': 'smp_paths', 'scope': 'scope_paths', 'ram': 'ram'}
    column_name = pen_map[penetrometer]
    
    match = all_profs[all_profs['id'] == profile_id]
    if match.empty:
        return []
    
    path_val = match.iloc[0][column_name]
    if pd.isna(path_val) or path_val == '[]':
        return []
        
    if penetrometer in ["smp", "scope"]:
        try:
            return ast.literal_eval(path_val)
        except (ValueError, SyntaxError):
            return []
    else:
        # Single string path (ram)
        return [path_val]

def layer_average_force(layers_df, penetrometer, ms_df=None):
    
    mean_forces_column = []
    loaded_cache = {}
    
    id_col = 'id' if ms_df is not None and 'id' in ms_df.columns else None
    
    
    for idx, row in layers_df.iterrows():
        pid = row['profile_id']
        top = row['top_cm']
        bottom = row['bottom_cm']
        hs = row['hs']
        grain_type = row['type']
        
        paths = get_pen_paths(pid, penetrometer)
        
        
        if not paths:
            mean_forces_column.append(np.nan)
            continue
        
        # to take average across all profiles made at given pit
        file_level_averages = []
        
        for path in paths:
            if path not in loaded_cache:
                try:
                    if penetrometer == 'smp':
                        loaded_cache[path] = smp_head(path)
                    elif penetrometer == 'scope':
                        loaded_cache[path] = scope_head(pd.read_csv(path, skiprows=1, usecols=[0, 1]))
                    elif penetrometer == 'ram':
                        loaded_cache[path] = ram_head(pd.read_csv(path))
                except Exception as e:
                    print(e)
                    loaded_cache[path] = None
            
            pen_df = loaded_cache[path]
            if pen_df is None or pen_df.empty:
                continue
            
            # use manually identified depth of crusts and lenses
            actual_depths = None
            
            is_crust = grain_type in ['MFcr', 'IF', 'IFil']
            
            if ms_df is not None and is_crust and id_col is not None:
                bot_col = 'bot_cm' if 'bot_cm' in ms_df.columns else 'bottom_cm'
                
                profile_filename = os.path.basename(path)
                prof_col = None
                if penetrometer == 'smp' and 'smp_id' in ms_df.columns: 
                    prof_col = 'smp_id'
                elif penetrometer == 'scope' and 'scope_id' in ms_df.columns: 
                    prof_col = 'scope_id'
                elif penetrometer == 'ram':
                    prof_col = 'sram_id'
                
                # find the row of pen_cleaning_filled that matches id and layer boundaries
                match = ms_df[
                    (ms_df[id_col] == pid) &
                    (ms_df[prof_col] == profile_filename) &
                    (np.isclose(ms_df['top_cm'], top, atol=0.1)) &
                    (np.isclose(ms_df[bot_col], bottom, atol=0.1))
                ]
                
                print(match)
        
                
                if (penetrometer == 'ram') & (not match.empty):
                    t_mm = match['top_depth_cm'].iloc[0] * 10
                    b_mm = match['bot_depth_mm'].iloc[0] * 10
                    if not pd.isna(t_mm) and not pd.isna(b_mm):
                        actual_depths = (t_mm, b_mm)
                
                elif (penetrometer in ['smp', 'scope']) & (not match.empty):
                    t_mm = match['top_depth_mm'].iloc[0]
                    b_mm = match['bot_depth_mm'].iloc[0]
                    if not pd.isna(t_mm) and not pd.isna(b_mm):
                        actual_depths = (t_mm, b_mm)
                        
                
                    
            
            # calculate hardness for given profile
            file_mean = get_hardness(pen_df, top, bottom, hs, grain_type, penetrometer, actual_depths)
            if not pd.isna(file_mean):
                file_level_averages.append(file_mean)
                
        if file_level_averages:
            mean_forces_column.append(np.nanmean(file_level_averages))
        else:
            mean_forces_column.append(np.nan)
            
    layers_df[f'avg_force_{penetrometer}'] = mean_forces_column
    return layers_df
    
    
'''   

def layer_average_force(layers_df, penetrometer):
    # Create a clean series tracking index modifications
    output_series = pd.Series(np.nan, index=layers_df.index)
    
    # Group by pit ID so we process all layers in a pit together
    for pid, group in layers_df.groupby('profile_id'):
        paths = get_pen_paths(pid, penetrometer)
        if not paths:
            continue
            
        # 1. Load and resample all raw profiles for this pit
        raw_profiles = []
        for path in paths:
            try:
                if penetrometer == 'smp':
                    df = smp_head(path)
                elif penetrometer == 'scope':
                    df = scope_head(pd.read_csv(path, skiprows=1, usecols=[0, 1]))
                elif penetrometer == 'ram':
                    df = ram_head(pd.read_csv(path))
                
                if df is not None and not df.empty:
                    raw_profiles.append(resample(df))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        if not raw_profiles:
            continue
            
        # 2. Align all profiles in this pit to the first profile (Master Reference)
        aligned_profiles = [raw_profiles[0]]  # Reference profile doesn't need self-warping
        ref_prof = raw_profiles[0]
        
        for prof in raw_profiles[1:]:
            try:
                matched_prof, _, _ = set_dtw(prof, ref_prof, window_cm=10, use_derivative=True)
                aligned_profiles.append(matched_prof)
            except Exception as e:
                print(f"DTW failed for a profile in pit {pid}: {e}")
                aligned_profiles.append(prof) # Fallback to unaligned if DTW errors out

        # 3. Calculate layer averages using the pre-aligned profile ensemble
        for idx, row in group.iterrows():
            top = row['top_cm']
            bottom = row['bottom_cm']
            hs = row['hs']
            
            file_level_averages = []
            for pen_df in aligned_profiles:
                file_mean = get_hardness(pen_df, top, bottom, hs)
                if not pd.isna(file_mean):
                    file_level_averages.append(file_mean)
                    
            if file_level_averages:
                output_series.loc[idx] = np.nanmean(file_level_averages)
                
    layers_df[f'avg_force_{penetrometer}'] = output_series
    return layers_df
'''


if __name__ == '__main__':
    layers = compile_layers()
    
    
    smp_crust = pd.read_csv('smp_mastersheet_fill.csv')
    scope_crust = pd.read_csv('scope_mastersheet_fill.csv')
    sram_crust = pd.read_csv('sram_mastersheet_fill.csv')

    
    ram_layers = layer_average_force(layers, 'ram', sram_crust)
    scope_layers = layer_average_force(layers, 'scope', scope_crust)
    smp_layers = layer_average_force(layers, 'smp', smp_crust)
    
    ram_layers.to_csv('/bsuhome/colemankane/Documents/penetrometer_analysis/all_ram_layers.csv')
    scope_layers.to_csv('/bsuhome/colemankane/Documents/penetrometer_analysis/all_scope_layers.csv')
    smp_layers.to_csv('/bsuhome/colemankane/Documents/penetrometer_analysis/all_smp_layers.csv')
    
    
    
    
    
    
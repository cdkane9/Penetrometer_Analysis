import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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


all_profs = pd.read_csv('all_profiles_thinned.csv')

prefixes_to_exclude = ('JPLMet_20240409', 'JPLMet_20240410', 'JPLMet_20250407_TS', 'JPLMet_20240405')

all_profs = all_profs[~all_profs['id'].str.startswith(prefixes_to_exclude)]

hh_categories = ['F', '4F', '1F', 'P', 'K', 'I']

def get_hardness(pen_df:pd.DataFrame, top:pd.Series, bottom:pd.Series, hs:float):
    '''
    Function for determining force across stratigraphic layers
    :param type: penetrometer type
    :param pen_df: data from penetrometer with columns ['depth', 'force']
    :param top: top of layers [cm]
    :param bottom: bottom of layers [cm]
    :param hs: HS as measured from pitwall [cm]
    :return:
    '''
    
    

    delta = top - bottom
    
    t_depth = ((hs - top) + (0.15 * delta)) * 10
    #print(f'top {t_depth}')
    b_depth = ((hs - bottom) - (0.15 * delta)) * 10
    #print(f'bot {b_depth}')

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
        
def layer_average_force(layers_df, penetrometer):
    
    mean_forces_column = []
    loaded_cache = {}
    
    for idx, row in layers_df.iterrows():
        pid = row['profile_id']
        top = row['top_cm']
        bottom = row['bottom_cm']
        hs = row['hs']
        
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
            
            # calculate hardness for given profile
            file_mean = get_hardness(pen_df, top, bottom, hs)
            if not pd.isna(file_mean):
                file_level_averages.append(file_mean)
                
        if file_level_averages:
            mean_forces_column.append(np.nanmean(file_level_averages))
        else:
            mean_forces_column.append(np.nan)
            
    layers_df[f'avg_force_{penetrometer}'] = mean_forces_column
    return layers_df
    

if __name__ == '__main__':
    layers = compile_layers()
    
    ram_layers = layer_average_force(layers, 'ram')
    scope_layers = layer_average_force(layers, 'scope')
    smp_layers = layer_average_force(layers, 'smp')
    
    ram_layers.to_csv('/bsuhome/colemankane/Documents/penetrometer_analysis/all_ram_layers.csv')
    scope_layers.to_csv('/bsuhome/colemankane/Documents/penetrometer_analysis/all_scope_layers.csv')
    smp_layers.to_csv('/bsuhome/colemankane/Documents/penetrometer_analysis/all_smp_layers.csv')
    
    
    
    
    
    
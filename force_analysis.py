import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import random
import ast
from profile_matching import *

'''
Outline:
    Compare force across stratigraphic layers between two different penetrometers.
    In each iteration of a M.C. simulation, choose random pen_a, pen_b profiles from
    each pit, match pen_a to pen_b (using profile_matching.py).  Then calculate max/
    min/average force across each layer.  
'''




pit_path = '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports'

caca = pd.read_csv('all_profiles2.csv')

# values in these two columns were constructed as lists
# Read in as strings, re-convert back to list
caca['scope_paths'] = caca['scope_paths'].dropna().apply(ast.literal_eval)
caca['smp_paths'] = caca['smp_paths'].dropna().apply(ast.literal_eval)

# filter df based on instruments to compare


# establish which comparison df to use
def get_pen_paths(pens:str) -> pd.DataFrame:
    if pens in ['smp_scope', 'scope_smp']:
        df = caca[
            (caca['scope_paths'].str.len() > 0) &
            (caca['smp_paths'].str.len() > 0)
            ].reset_index()
        return df
    elif pens in ['ram_scope', 'scope_ram']:
        df = caca[
            (caca['scope_paths'].str.len() > 0) &
            (caca['ram'].str.len() > 0)
            ].reset_index()
        return df
    elif pens in ['ram_smp', 'smp_ram']:
        df = caca[
            (caca['ram'].str.len() > 0) &
            (caca['smp_paths'].str.len() > 0)
            ].reset_index()
        return df
    else:
        raise ValueError('Invalid pens')

# calculate max/min/avg force across given stratigraphy boundaries
def get_hardness(type:str, pen_df:pd.DataFrame, top:pd.Series, bottom:pd.Series, hs:float):
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

    if type in ['smp', 'scope']:
        # convert top and bottom (cm above ground) to mm below surface
        # add in buffer of 20% of layer thickness
        t_depth = ((hs - top) + (0.1 * delta)) * 10
        b_depth = ((hs - bottom) - (0.1 * delta)) * 10

    elif type == 'ram':
        # convert top and bottom (cm above ground) to cm below surface
        # add in buffer of 20% of layer thickness
        t_depth = ((hs - top) + (0.1 * delta))
        b_depth = ((hs - bottom) - (0.1 * delta))

    else:
        raise ValueError('Invalid type')

    layer_ix = pen_df.index[(pen_df['depth'] >= t_depth) &
                            (pen_df['depth'] <= b_depth)]
    if layer_ix.empty: return np.nan, np.nan, np.nan
    else:
        mean = np.nanmean(pen_df.loc[layer_ix, 'force'])
        max = np.nanmax(pen_df.loc[layer_ix, 'force'])
        min = np.nanmin(pen_df.loc[layer_ix, 'force'])

    return mean, max, min

# comparison for smp and snow scope
def scope_smp_comp(df:pd.DataFrame):
    '''
    Iterates through entire dataframe of SMP and Scope profiles, selects
    random profile of each from each pit, matches Scope to SMP, compares max/min/average
    force across stratigraphic layers, returns tuple of R^2, slope, and intercept
    :param df: DF containing paths to SMP and Scope profiles
    :return: list of R^2, slope, and intercept
    '''
    # initialize lists for layer-wise values across all pits
    all_scope_vals = []
    all_smp_vals = []

    for i in df.index:
        smp_paths = df['smp_paths'].iloc[i]
        scope_paths = df['scope_paths'].iloc[i]

        # read in stratigraphy file
        pit_id = df['id'].iloc[i] + '_strat.csv'
        pit = pd.read_csv(os.path.join(pit_path, pit_id))
        top = pit['top_cm']
        bot = pit['bottom_cm']
        grain_type = pit['type']
        size = pit['grain avg_mm']
        hs = top.max()


        # chose random profile from each
        smp_path = random.choice(smp_paths)
        scope_path = random.choice(scope_paths)

        # pass smp and scope paths to profile_matching wrapper function
        smp, match_scope, _, og_scope, _ = wrapper(
            smp_path, scope_path,
            'smp', 'scope',
            distance_cosine
        )

        for idx in range(len(pit)):
            # set top and bottom of layer
            t_val = top.iloc[idx]
            b_val = bot.iloc[idx]

            # calculate hardness across layer
            smp_mean, smp_max, smp_min = get_hardness('smp', smp, t_val, b_val, hs)
            scope_mean, scope_max, scope_min = get_hardness('scope', match_scope, t_val, b_val, hs)

            # store values
            all_smp_vals.append([smp_mean, smp_max, smp_min])
            all_scope_vals.append([scope_mean, scope_max, scope_min])

    # conver to array for calculating regression
    all_scope_vals = np.array(all_scope_vals)
    all_smp_vals = np.array(all_smp_vals)

    res_of_mean = stats.linregress(all_smp_vals[:, 0], all_scope_vals[:, 0])
    res_of_max = stats.linregress(all_smp_vals[:, 1], all_scope_vals[:, 1])
    res_of_min = stats.linregress(all_smp_vals[:, 2], all_scope_vals[:, 2])

    arr_of_mean = np.array(
        [res_of_mean.slope, res_of_mean.intercept, res_of_mean.rvalue ** 2, res_of_mean.pvalue, res_of_mean.stderr])
    arr_of_max = np.array(
        [res_of_max.slope, res_of_max.intercept, res_of_max.rvalue ** 2, res_of_max.pvalue, res_of_max.stderr])
    arr_of_min = np.array(
        [res_of_min.slope, res_of_min.intercept, res_of_min.rvalue ** 2, res_of_min.pvalue, res_of_min.stderr])

    return all_scope_vals, all_smp_vals, arr_of_mean, arr_of_max, arr_of_min


if __name__ == '__main__':
    layer_scope, layer_smp, metrics, _, _ = scope_smp_comp(get_pen_paths('scope_smp'))
    plt.scatter(layer_scope[:, 0], layer_smp[: 0])
    plt.show()
    pass
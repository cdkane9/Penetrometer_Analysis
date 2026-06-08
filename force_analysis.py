import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import random
import ast
from profile_matching import *
from scipy.optimize import curve_fit

'''
Outline:
    Compare force across stratigraphic layers between two different penetrometers.
    In each iteration of a M.C. simulation, choose random pen_a, pen_b profiles from
    each pit, match pen_a to pen_b (using profile_matching.py).  Then calculate max/
    min/average force across each layer.  
'''




pit_path = '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports'

caca = pd.read_csv('all_profiles_thinned.csv')

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
        smp_id = os.path.basename(smp_path)

        scope_path = random.choice(scope_paths)
        scope_id = os.path.basename(scope_path)

        # pass smp and scope paths to profile_matching wrapper function
        smp, match_scope, _, og_scope, _ = wrapper(
            smp_path, scope_path,
            'smp', 'scope',
            distance_cosine
        )

        print(f'Matching Scope {scope_id} to SMP {smp_id}\n')

        for idx in range(len(pit) - 1):
            # set top and bottom of layer
            t_val = top.iloc[idx]
            b_val = bot.iloc[idx + 1]

            # calculate hardness across layer
            smp_mean, smp_max, smp_min = get_hardness('smp', smp, t_val, b_val, hs)
            scope_mean, scope_max, scope_min = get_hardness('scope', match_scope, t_val, b_val, hs)

            # catch instances where stratigraphy is deeper than smp or scope
            if (np.isnan(smp_mean) or np.isnan(scope_mean) or
                    np.isnan(smp_max) or np.isnan(scope_max) or
                    np.isnan(smp_min) or np.isnan(scope_min)):
                continue

            # store values
            all_smp_vals.append([smp_mean, smp_max, smp_min])
            all_scope_vals.append([scope_mean, scope_max, scope_min])



    # convert to array for calculating regression
    all_scope_vals = np.array(all_scope_vals)
    all_smp_vals = np.array(all_smp_vals)

    if len(all_smp_vals) < 2:
        arr_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        return all_scope_vals, all_smp_vals, arr_nan, arr_nan, arr_nan
    ############ NEW ##########
    # try other slope and r^2 calculation
    x = all_smp_vals[:, 0]
    y = all_scope_vals[:, 0]

    def target_func(x, m, b):
        return m * x + b

    popt, _ = curve_fit(target_func, x, y)
    slope_mean = popt[0]
    int_mean = popt[1]

    # Calculate uncentered R^2 manually
    residuals = y - target_func(x, slope_mean, int_mean)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(y ** 2)
    r2_mean = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # calculate correlation statistics
    #res_of_mean = stats.linregress(all_smp_vals[:, 0], all_scope_vals[:, 0])
    #res_of_max = stats.linregress(all_smp_vals[:, 1], all_scope_vals[:, 1])
    #res_of_min = stats.linregress(all_smp_vals[:, 2], all_scope_vals[:, 2])

    # unpack statistics into an array
    #arr_of_mean = np.array(
    #    [res_of_mean.slope, res_of_mean.intercept, res_of_mean.rvalue ** 2, res_of_mean.pvalue, res_of_mean.stderr])
    #arr_of_max = np.array(
    #    [res_of_max.slope, res_of_max.intercept, res_of_max.rvalue ** 2, res_of_max.pvalue, res_of_max.stderr])
    #arr_of_min = np.array(
    #    [res_of_min.slope, res_of_min.intercept, res_of_min.rvalue ** 2, res_of_min.pvalue, res_of_min.stderr])
    arr_of_max = None
    arr_of_min = None

    return all_scope_vals, all_smp_vals, slope_mean, int_mean, r2_mean#arr_of_mean, arr_of_max, arr_of_min


if __name__ == '__main__':
    num_iters = 100

    r2s = np.zeros(num_iters)
    slopes = np.zeros(num_iters)
    intercepts = np.zeros(num_iters)

    master_scope_means = []
    master_smp_means = []

    pen_df = get_pen_paths('scope_smp')

    for iter_idx in range(num_iters):
        print('############################################')
        print(f'Iteration: {iter_idx + 1}')
        print('############################################')
        # scope_smp_comp randomly pairs profiles and performs the linear regression
        #scope_vals, smp_vals, arr_of_mean, _, _ = scope_smp_comp(pen_df)
        scope_vals, smp_vals, slope, intercept, r2 = scope_smp_comp(pen_df)

        # Store iteration-specific tracking metrics (Slope, Intercept, R^2)
        #slopes[iter_idx] = arr_of_mean[0]
        #intercepts[iter_idx] = arr_of_mean[1]
        #r2s[iter_idx] = arr_of_mean[2]
        slopes[iter_idx] = slope
        intercepts[iter_idx] = intercept
        r2s[iter_idx] = r2




        # Extract the column index 0 (Mean Force) from the returned arrays
        # Use clean filtering to skip any NaN layers caused by depth/buffer mismatches
        valid_mask = ~np.isnan(scope_vals[:, 0]) & ~np.isnan(smp_vals[:, 0])

        # Extend the master list with the valid layers found in this specific iteration
        master_scope_means.extend(scope_vals[valid_mask, 0])
        master_smp_means.extend(smp_vals[valid_mask, 0])

        if (iter_idx + 1) % 100 == 0:
            print(
                f"  Iteration {iter_idx + 1}/{num_iters} complete. Current mean R²: {np.mean(r2s[:iter_idx + 1]):.4f}")

        # Convert master lists into numpy arrays for plotting
    master_scope_means = np.array(master_scope_means)
    master_smp_means = np.array(master_smp_means)

    x = master_smp_means
    y = master_scope_means

    def target_func(x, m, b):
        return m * x + b

    popt, _ = curve_fit(target_func, x, y)
    slope_mean = popt[0]
    int_mean = popt[1]

    # Calculate uncentered R^2 manually
    residuals = y - target_func(x, slope_mean, int_mean)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(y ** 2)
    r2_mean = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    print(f'ALL R2: {r2_mean:.4f}')
    print(f'SLOPE: {slope_mean:.4f}')

    print("\nSimulation Finished!")
    print(f"Total layer data points collected: {len(master_scope_means)}")
    print(f"Overall Monte Carlo Mean R²: {np.mean(r2s):.4f} (±{np.std(r2s):.4f})")
    print(f"Overall Monte Carlo Mean Slope: {np.mean(slopes):.4f} (±{np.std(r2s):.4f}")

    # 3. Create the plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Left Plot: Master Scatter Plot of all data points across all iterations
    # Since 1000 iterations will produce thousands of overlapping points,
    # using a low alpha (transparency) helps visualize data density.
    ax[0].scatter(master_smp_means, master_scope_means, alpha=0.5, color='teal', marker='x')

    # plot trend line calculated over all values at once
    ax[0].plot(np.arange(np.min(x), np.max(x), 1), np.arange(np.min(x), np.max(x), 1) * slope_mean + int_mean,
               color='black', label=f'Line from all data, (R^2={r2_mean:.4f})')

    # Plot an average trendline over the scatter plot
    x_vals = np.array([0, np.max(master_smp_means)])
    y_vals = np.mean(slopes) * x_vals + np.mean(intercepts)
    ax[0].plot(x_vals, y_vals, color='crimson', linestyle='--', linewidth=2,
               label=f'Avg Line (R²={np.mean(r2s):.3f})')

    ax[0].set_title('Aggregated Layer Mean Force (All Iterations)')
    ax[0].set_xlabel('Reference SMP Mean Force (N)')
    ax[0].set_ylabel('Matched Snow Scope Mean Force (N)')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Right Plot: Histogram showing the distribution of R^2 across your simulation runs
    ax[1].hist(r2s, bins=25, color='royalblue', edgecolor='black', alpha=0.7)
    ax[1].axvline(np.mean(r2s), color='crimson', linestyle='--', linewidth=2,
                  label=f'Mean R² = {np.mean(r2s):.3f}')
    ax[1].set_title('Distribution of Alignment $R^2$ Scores')
    ax[1].set_xlabel('$R^2$ Variance')
    ax[1].set_ylabel('Frequency Count')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


    pass
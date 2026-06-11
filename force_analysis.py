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

prefixes_to_exclude = ('JPLMet_20250407_TS', 'JPLMet_20250408_TS')

caca = caca[~caca['id'].str.startswith(prefixes_to_exclude, na=False)]


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
def scope_smp_comp(df:pd.DataFrame, score_threshold:float=1):
    '''
    Iterates through entire dataframe of SMP and Scope profiles, selects
    random profile of each from each pit, matches Scope to SMP, compares max/min/average
    force across stratigraphic layers, returns tuple of R^2, slope, and intercept
    :param df: DF containing paths to SMP and Scope profiles
    :param score_threshold: maximum allowable cosine-difference score to be included in analysis
    :return: list of R^2, slope, and intercept
    '''
    # initialize lists for layer-wise values across all pits
    all_scope_vals = []
    all_smp_vals = []
    all_scores = []
    removed_profiles = 0

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
        smp, match_scope, _, og_scope, score = wrapper(
            smp_path, scope_path,
            'smp', 'scope',
            distance_cosine
        )

        print(f'Matched Scope {scope_id} to SMP {smp_id}\n with score {score:.4f}')

        if score_threshold is not None and score > score_threshold:
            print(f'\nPoor cosine score: {score:.4f}\n')
            removed_profiles += 1
            continue

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
            all_scores.append(score)



    # convert to array for calculating regression
    all_scope_vals = np.array(all_scope_vals)
    all_smp_vals = np.array(all_smp_vals)
    all_scores = np.array(all_scores)

    if len(all_smp_vals) < 2:
        arr_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        return all_scope_vals, all_smp_vals, arr_nan, arr_nan, arr_nan, all_scores
    ############ NEW ##########
    # try other slope and r^2 calculation
    x = all_smp_vals[:, 0]
    y = all_scope_vals[:, 0]

    def mae_loss(params, x, y):
        m, b = params
        return np.mean(np.abs(y - (m* x + b)))

    res = minimize(mae_loss, x0=[1, 0], args=(x, y))
    slope_mean = res.x[0]
    int_mean = res.x[1]


    # Calculate R_MAE manually
    residuals = np.abs(y - (slope_mean * x + int_mean))
    sad_res = np.sum(residuals)
    sad_tot = np.sum(np.abs(y - np.median(y)))
    r2_mean = 1.0 - (sad_res / sad_tot) if sad_tot != 0 else 0.0

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

    return all_scope_vals, all_smp_vals, slope_mean, int_mean, r2_mean, all_scores, removed_profiles#arr_of_mean, arr_of_max, arr_of_min

def ram_smp_comp(df: pd.DataFrame, score_threshold:float=0.3):
    '''
    Iterate through entire dataframe of SMP and standard ram profiles, seletts
    a random SMP profile from each pit, matches the ram profile to the SMP, compares max/min/average
    force across stratigraphic layer, return tuple of R^2, slope, and intercept of regression line
    :param df: DF containing paths to SMP and ram profiles
    :return: lsit of R^2, slope and intercept of regression line
    '''

    # initialize lists for layer-wise values across pits
    all_ram_vals = []
    all_smp_vals = []
    all_scores = []
    removed_profiles = 0
    total_attempted = 0

    for i in df.index:
        smp_paths = df['smp_paths'].iloc[i]
        ram_path = df['ram'].iloc[i]
        ram_id = os.path.basename(ram_path)




        # read in stratigraphy file
        pit_id = df['id'].iloc[i] + '_strat.csv'
        pit = pd.read_csv(os.path.join(pit_path, pit_id))
        top = pit['top_cm']
        bot = pit['bottom_cm']
        grain_type = pit['type']
        size = pit['grain avg_mm']
        hs = top.max().astype(float)

        # choose random SMP profile
        smp_path = random.choice(smp_paths)
        smp_id = os.path.basename(smp_path)


        # match the ram to the smp with profile_matching.py
        smp, match_ram, _, og_ram, score = wrapper(
            smp_path, ram_path,
            'smp', 'ram',
            distance_cosine
        )

        print(f'Matched ram {ram_id} to SMP {smp_id} with score {score:.4f}')
        if score_threshold is not None and score > score_threshold:
            print(f'\nSkipping match due to poor matching - score: {score:.4f}\n')
            removed_profiles += 1
            continue

        # pull out average force for matched profiles
        for idx in range(len(pit) - 1):
            # set top and botom of layer
            t_val = top.iloc[idx]
            b_val = bot.iloc[idx + 1]

            # calculate hardness across layer
            smp_mean, smp_max, smp_min = get_hardness('smp', smp, t_val, b_val, hs)
            ram_mean, ram_max, ram_min = get_hardness('ram', match_ram, t_val, b_val, hs)

            # catch instances where stratigraphy is deeper than the smp or ram
            if (np.isnan(smp_mean) or np.isnan(ram_mean) or
                    np.isnan(smp_max) or np.isnan(ram_max) or
                    np.isnan(smp_min) or np.isnan(ram_min)):
                continue

            # store values
            all_smp_vals.append([smp_mean, smp_max, smp_min])
            all_ram_vals.append([ram_mean, ram_max, ram_min])
            all_scores.append(score)

    # conver to an array for calculating regression
    all_smp_vals = np.array(all_smp_vals)
    all_ram_vals = np.array(all_ram_vals)
    all_scores = np.array(all_scores)

    # set variables for regression
    x = all_smp_vals[:, 0]
    y = all_ram_vals[:, 0]

    # calculate regression
    def mae_loss(params, x, y):
        m, b = params
        return np.mean(np.abs(y - (m * x + b)))

    res = minimize(mae_loss, x0=[1, 0], args=(x, y))
    slope_mean = res.x[0]
    int_mean = res.x[1]

    # Calculate R_MAE manually
    residuals = np.abs(y - (slope_mean * x + int_mean))
    sad_res = np.sum(residuals)
    sad_tot = np.sum(np.abs(y - np.median(y)))
    r2_mean = 1.0 - (sad_res / sad_tot) if sad_tot != 0 else 0.0

    return all_ram_vals, all_smp_vals, slope_mean, int_mean, r2_mean, all_scores, removed_profiles

def MC_pen_comp(comp:str, num_iters:int, score_threshold:float=1):
    '''
    Monte-Carlo simulation on all profiles.  For each iteration, randomly select profiles from each pit,
    match one profile to the other, compare the hardness across stratigraphic layers, fit a regression
    for all data in that iteration.  Return mean slope, intercept, and r^2 for entire simulation
    :param pena: Reference profile - treat as independent variable, penetrometer with most confidence
    :param penb: Evaluation profile - dependent variable, penetrometer that will be matched
    :param comp: String determining which instruments to compare
    :param num_iters: # of iterations in simulation
    :return: mean_pena      ==> mean hardness across each stratigraphic layers for pena
             mean_penb      ==> mean hardness across each stratigraphic layers for penb
             slopes         ==> slopes of regression lines from each iteration
             intercepts     ==> intercepts of regression lines from each iteration
             r2s            ==> r^2 values of regression lines from each iteration
    '''
    # determines which dataframe and which comparison function to use
    COMP_MAP = {
        'ram_smp': ram_smp_comp,
        'smp_ram': ram_smp_comp,

        'scope_smp': scope_smp_comp,
        'smp_scope': scope_smp_comp,

        #'ram_scope': ram_scope_comp,
        #'scope_ram': ram_scope_comp
    }
    if comp not in COMP_MAP:
        raise ValueError(f"Invalid comparison string '{comp}'. Must be one of {list(COMP_MAP.keys())}")

    comp_function = COMP_MAP[comp]
    pen_df = get_pen_paths(comp)


    # initialize array for statistics
    r2s = np.zeros(num_iters)
    slopes = np.zeros(num_iters)
    intercepts = np.zeros(num_iters)

    master_pena_means = []
    master_penb_means = []
    master_scores = []

    # track how many match scores were removed
    total_removed = 0
    total_attempted = 0

    for iter_idx in range(num_iters):
        print('############################################')
        print(f'Iteration: {iter_idx + 1}')
        print('############################################')

        # comp_function randomly selects two profiles from same pit, matches them, calculates linear regression
        penb_vals, pena_vals, slope, intercept, r2, scores, removed_count = comp_function(pen_df, score_threshold=score_threshold)


        # store metrics for each iteration
        slopes[iter_idx] = slope
        intercepts[iter_idx] = intercept
        r2s[iter_idx] = r2

        # add number of profiles removed
        total_removed += removed_count
        total_attempted += len(pena_vals)

        # Skip over any NaN layers, and add valid layers to master lists
        valid_mask = ~np.isnan(penb_vals[:, 0]) & ~np.isnan(pena_vals[:, 0])
        master_pena_means.extend(pena_vals[valid_mask])
        master_penb_means.extend(penb_vals[valid_mask])
        master_scores.extend(scores[valid_mask])

        if (iter_idx + 1) % 10 == 0:
            print(
                f'Iteration {iter_idx + 1}/{num_iters} complete. Current mean R²: {np.mean(r2s[:iter_idx + 1]):.4f}")'
            )

    master_pena_means = np.array(master_pena_means)
    master_penb_means = np.array(master_penb_means)
    master_scores = np.array(master_scores)

    # calculate percentage of profiles removed
    pct_removed = (total_removed / total_attempted * 100) if total_attempted > 0 else 0
    print(f'# of profiles removed: {total_removed}')
    print(f'# of profiles attempted: {total_attempted}')
    print(f'% of profiles removed: {pct_removed}')
    print(f'N data points: {len(master_pena_means)}')
    print(f'Overall Mean R2: {np.mean(r2s):.4f} (+/- {np.std(r2s):.4f})')
    print(f'Overall Mean Slope: {np.mean(slopes):.4f} (±{np.std(slopes):.4f}')

    return master_pena_means, master_penb_means, slopes, intercepts, r2s, master_scores, pct_removed

if __name__ == '__main__':
    #master_smp_means, master_scope_means, slopes, intercepts, r2s, cos_scores, nfg_pct = MC_pen_comp(
    #    'smp_scope', 25, score_threshold=0.25
    #)

    master_smp_means, master_scope_means, slopes, intercepts, r2s, cos_scores, nfg_pct = MC_pen_comp(
        'smp_scope', 500, score_threshold=1
    )

    print(f'Median cosine distance score: {np.nanmedian(cos_scores):.4f}')

    # 3. Create the plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Left Plot: Master Scatter Plot of all data points across all iterations
    # Since 1000 iterations will produce thousands of overlapping points,
    # using a low alpha (transparency) helps visualize data density.
    sc = ax[0].scatter(
        master_smp_means[:, 0],
        master_scope_means[:, 0],
        c=cos_scores,
        cmap='viridis',
        alpha=0.6,
        marker='x'
    )

    cbar = fig.colorbar(sc, ax=ax[0])
    cbar.set_label('Cosine distance score')

    # Plot an average trendline over the scatter plot
    x_vals = np.array([0, np.max(master_smp_means[:, 0])])
    y_vals = np.mean(slopes) * x_vals + np.mean(intercepts)
    ax[0].plot(x_vals, y_vals, color='crimson', linestyle='--', linewidth=2,
               label=f'Avg Line (R²={np.mean(r2s):.3f})')

    ax[0].set_title('Aggregated Layer Mean Force (500 Iterations)')
    ax[0].set_xlabel('Reference SMP Mean Force (N)')
    ax[0].set_ylabel('Matched Ram Mean Force (N)')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Right Plot: Histogram showing the distribution of R^2 across your simulation runs
    ax[1].hist(r2s, bins=40, color='royalblue', edgecolor='black', alpha=0.7)
    ax[1].axvline(np.mean(r2s), color='crimson', linestyle='--', linewidth=2,
                  label=f'Mean R² = {np.mean(r2s):.3f}')
    #ax[1].axvline(np.median(cos_scores), color='green', label='Median Cosine Distance')
    #ax[1].set_title('Distribution of Alignment Cosine-Distance Scores')
    ax[1].set_xlabel(r'$r^2$ distribution')
    ax[1].set_ylabel('Frequency Count')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.hist(cos_scores, bins=40, color='royalblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Cosine Distance score distribution')
    ax1.set_xlabel('Cosine Distance score')
    ax1.set_ylabel('Frequency Count')

    plt.tight_layout()
    plt.show()


    pass
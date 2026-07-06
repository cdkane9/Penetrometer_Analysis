import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cosine
from pyts.metrics import dtw


'''
Outline:
    Read in profiles and resample to matching 0.1cm 1D grid
    Assign 'artificial layers' ==> chunk into 4cm layers
    Find best alpha values to transform each artifical layer
    Resample and linearly interpolate back to original depth grid.
    Code was based on Hagenmuller and Pilloix 2016  
'''

# remove header info from scope profiles
def scope_head(df):
    idx_init = df[df.iloc[:, 0] == 'depth (mm)'].index[0]
    scope_prof = df.iloc[idx_init + 1:].reset_index(drop=True).astype(float)
    scope_prof.columns = ['depth', 'force']
    # convert from kPa to N
    scope_prof['force'] *= 1000 * 0.0000554
    return scope_prof

# pull out depth and force columns from ram profile
def ram_head(df):

    df['depth'] = df['l_cm'] * 10
    df['force'] = df['rr_N']
    return df

# for profiles of drastically different depths, remove excess of deeper profile
def same_depth(prof_a, prof_b):
    max_a = np.max(prof_a['depth'])
    max_b = np.max(prof_b['depth'])
    max_depth = np.min([max_a, max_b])

    prof_a_trim = prof_a[prof_a['depth'] < max_depth]

    prof_b_trim = prof_b[prof_b['depth'] < max_depth]


    return prof_a_trim, prof_b_trim

# STEP 1: Resample and smooth profile
def resample(df, delta_h=1):
    ''' resample a profile with gaussian kernel and linear interpolation'''
    og_res = df['depth'].iloc[1] - df['depth'].iloc[0]
    target_resolution = delta_h

    if og_res >= target_resolution:
        og_depths = df['depth'].values
        target_depths = np.arange(0, og_depths.max(), target_resolution)
        resamp_values = np.interp(target_depths, og_depths, df['force'].values)

    else:
        window_size = target_resolution // og_res

        sigma = window_size / 2
        smoothed = gaussian_filter1d(df['force'], sigma=sigma)

        og_depths = np.arange(len(df)) * og_res
        target_depths = np.arange(0, og_depths.max(), target_resolution)

        resamp_values = np.interp(target_depths, og_depths, smoothed)

    # resamples with 1mm resolution, returns dataframe in centimeteres
    resamp_df = pd.DataFrame({'depth': target_depths / 10, 'force': resamp_values})

    return resamp_df


# STEP 2: Divide into arbitrary layers with thickness delta_L
def delta_L(profile, delta_l=4):
    '''
    decompse profile onto specified 1-D grid with layer thickness l
    :param profile:
    :param delta_l: minimum layer thickness to preserve
    :return: profile with indices of layers
    '''
    depth = profile['depth']
    profile['layer_ix'] = (depth // delta_l).astype(int)
    return profile

# STEP 3: Multiply delta_L from STEP 2 by some alpha to stretch/shrink
#         original arbitrary layer thickness (\alpha_j * \delta_L)
def transform(depths, force, alphas, delta_L=4):
    '''
    stretches/thins layer j by factor of alpha_j
    essentially multiply the depth values by some factor
    if alpha < 1 ==> thinning
    if alpha > 1 ==> stretching
    if alpha = 1 ==> no change
    :param profile: profile with indices of layers
    :return:
    '''
    original_d = np.asanyarray(depths)
    original_f = np.asanyarray(force)
    alphas = np.asanyarray(alphas)

    layer_thickness = alphas * delta_L
    offsets = np.zeros(len(alphas) + 1)
    offsets[1:] = np.cumsum(layer_thickness)

    # transformed depths
    h_T = np.empty(len(original_d))

    for i in range(len(original_d)):
        h = original_d[i]

        j = int(h // delta_L)

        if j >= len(alphas):
            last_layer_ix = len(alphas) - 1
            base_offset = offsets[last_layer_ix]

            h_from_last_layer_start = h - (last_layer_ix * delta_L)

            h_T[i] = base_offset + (alphas[last_layer_ix] * h_from_last_layer_start)

        else:
            h_in_layer = h % delta_L
            h_T[i] = offsets[j] + (alphas[j] * h_in_layer)

    return h_T

# STEP 4a: Define cost function - MSE of difference in log-hardness between
#          two profiles
def distance_mse(ref_hard, ref_d, prof_hard):
    '''
    calculate mean squared difference of hardness between two profiles on log-scale
    adapted from Hagenmuller and Pilloix 2016
    :param reference: reference profile (pd.DataFrame)
    :param profile: profile to be evaluated (pd.DataFrame)
    :return: mean-squared difference on log-scale
    '''

    dx = ref_d.iloc[1] - ref_d.iloc[0]
    h_tot = ref_d.iloc[-1]

    ref_sigma = np.log(ref_hard + 1e-6)
    prof_sigma = np.log(prof_hard + 1e-6)

    diff_sq = dx * (ref_sigma - prof_sigma)**2
    total_cost = np.sum(diff_sq)

    return total_cost / h_tot


# STEP 4a: Define cost function - minimize (1 - pearson correlation)
def distance_correlation(ref_hard, ref_d, prof_hard):
    corr_matrix = np.corrcoef(ref_hard, prof_hard)
    if np.isnan(corr_matrix[0, 1]):
        return 1
    return 1 - corr_matrix[0, 1]


# STEP 4a: Define cost function - minimize cosine distance
def distance_cosine(ref_hard, ref_d, prof_hard):
    # SciPy does 1 - cosine() under the hood, so minimize this function
    return cosine(ref_hard + 1e-6, prof_hard + 1e-6)

def cost_dtw(prof, ref_prof, method='sakoechiba', window_cm=15):
    pena_orig = ref_prof['force'].values
    penb_orig = prof['force'].values
    depths = ref_prof['depth'].values

    N = len(pena_orig)

    # normalize the force values
    pena_min, pena_max = np.min(pena_orig), np.max(pena_orig)
    penb_min, penb_max = np.min(penb_orig), np.max(penb_orig)

    pena_norm = (pena_orig - pena_min) / (pena_max - pena_min + 1e-6)
    penb_norm = (penb_orig - penb_min) / (penb_max - penb_min + 1e-6)

    if method == 'sakoechiba':
        window_ix = int(window_cm / 0.1)
        dtw_options = {'window_size': window_ix}
    elif method == 'fast':
        dtw_options = {'radius': 4}
    else: dtw_options = None

    score, path = dtw(
        x=pena_norm,
        y=penb_norm,
        dist='absolute',
        method=method,
        options=dtw_options,
        return_path=True
    )

    ref_indices = path[0]
    prof_indices = path[1]

    warped_forces = np.zeros(N)
    counts = np.zeros(N)

    for idx_ref, idx_prof in zip(ref_indices, prof_indices):
        warped_forces[idx_ref] += penb_orig[idx_prof]
        counts[idx_ref] += 1

    counts[counts == 0] = 1
    final_forces = warped_forces / counts

    matched_profile = pd.DataFrame({
        'depth': depths,
        'force': final_forces
    })

    normalized_score = score / N

    return matched_profile, ref_prof, None, normalized_score


# STEP 4b: Optimize values of alpha to minimize distance function
def objective_function(alphas, ref, prof, optimize_func, delta_L=4):
    '''
    Objective function to minimize: for an array of alpha values,
    transform profile depth, interpolate back to compare with reference
    profile and returns distance cost.
    :param alphas:
    :param ref_df:
    :param prof_df:
    :param delta_L:
    :return:
    '''
    # do first profile transformation
    transformed_depths = transform(prof['depth'], prof['force'], alphas, delta_L)

    # interpolate transformed profile back onto 1mm grid
    interp_prof_force = np.interp(ref['depth'], transformed_depths, prof['force'])

    # calculate distance between transformed profile and reference profile
    cost = optimize_func(ref['force'], ref['depth'], interp_prof_force)


    return cost

# STEP 4c: Intra-set variability - compare set of profiles without designating one as a reference
def variability():
    pass

# STEP 5: Find best values for alpha
def minimize_cost(prof, ref_prof, optimize_func, delta_L, lower_bound=0.1, upper_bound=1.9, global_epsilon=0.2):
    '''
    minimize objective function to find best values for alpha
    :param prof: profile to be stretched/thinned
    :param ref_prof: reference profile (usually SMP)
    :param lower_bound: lower bound for alphas
    :param upper_bound: upper bounds for alphas
    :return: transformed profile that matches ref_prof
    '''

    # determine number of layers of thickness \delta_L
    num_layers = int(ref_prof['depth'].max() / delta_L)

    # intitial guess is that alpha = 1
    alpha_0 = np.ones(num_layers)

    # sets bounds for amount of stretch/thin that can occur for any one layer
    bounds = [(lower_bound, upper_bound) for _ in range(num_layers)]

    # set bounds to limit total depth change of profile
    low_global = (1 - global_epsilon) * num_layers
    high_global = (1 + global_epsilon) * num_layers

    # build a placeholder matrix to pass into LinearConstraint
    matrix_to_sum = np.ones((1, num_layers))
    global_depth_constraint = LinearConstraint(matrix_to_sum, low_global, high_global)

    result = minimize(
        objective_function,
        alpha_0,
        args=(ref_prof, prof, optimize_func, delta_L),
        method='SLSQP',
        bounds=bounds,
        constraints=global_depth_constraint
    )
    best_alphas = result.x
    score = result.fun

    matched_depth = transform(prof['depth'], prof['force'], best_alphas, delta_L=delta_L)
    matched_profile = pd.DataFrame({
        'depth': matched_depth,
        'force': prof['force']
    })

    return matched_profile, ref_prof, best_alphas, score

######################################################################################################


def wrapper(reference:str, profile:str,
            type_ref:str, type_prof:str,
            optimizing_function,
            delta_L:int=4,
            delta_h:int=1,
            lower_bound:float=0.3,
            upper_bound:float=1.7,):
    '''
    Calls all necessary functions for vertically shifting a profile to match a provided reference
    :param reference: path to reference profile
    :param profile: path to profile to be stretched/thinned
    :param type_ref: type of reference profile (usually SMP)
    :param type_prof: type of profile to be stretched/thinned (Snow Scope, ram)
    :param optimizing_function: optimizing function to be used on profile
    :param delta_L: arbitrary chunk thickness that will be stretched/thinned [cm]
    :param delta_h: std. of gaussian used to smooth profiles
    :param lower/upper_bound: bounds for alphas (set at +/-70%) from Hagenmuller and Pilloix 2016
    :return: ref_prof    ==>  the unaltered reference profile
             best_alphas ==>  best values to stretch/thin profile
             match_prof  ==>  stretched/thinned profile
    '''
    # TODO: capabilities intraset variability, option for passing multiple profiles?
    print(f'Reference: {reference}')
    print(f'Prof: {profile}')
    
    try:
        # read in each file based on the type
        if type_ref == 'scope':
            ref_raw = scope_head(pd.read_csv(reference, skiprows=1, usecols=[0, 1]))
        elif type_ref == 'ram':
            ref_raw = ram_head(pd.read_csv(reference))
        elif type_ref == 'smp':
            # Assumes SMP has already been converted to a .csv of raw sample data
            ref_raw =  pd.read_csv(reference, low_memory=False, skiprows=0, usecols=[1, 2])
            ref_raw.columns = ['depth', 'force']
        
        else:
            raise ValueError('Unknown reference profile type\n'
                             'Must be [scope, ram, smp]')
        
        if type_prof == 'scope':
            prof_raw = scope_head(pd.read_csv(profile, skiprows=1, usecols=[0, 1]))
        
        elif type_prof == 'ram':
            prof_raw = ram_head(pd.read_csv(profile))
        
        elif type_prof == 'smp':
            # Assumes SMP has already been converted to a .csv of raw sample data
            try:
                prof_raw = pd.read_csv(profile, low_memory=False, skiprows=0, usecols=[1,2])
                prof_raw.columns = ['depth', 'force']
            except pd.errors.ParserError:
                prof_raw = pd.read_csv(profile, low_memory=False)
                prof_raw.columns = ['depth', 'force']
        else:
            raise ValueError('Unkown profile type\n'
                             'Must be [scope, ram, smp]')
        
        # ensure profiles are same depth by trimming off excess from deeper profile
        ref, prof = same_depth(ref_raw, prof_raw)
        
        # smooth profiles w/ gaussian filter kernel (std. dev. = delta_h)
        ref_resamp = resample(ref, delta_h)
        prof_resamp = resample(prof.copy(), delta_h)

        # set window size to 10 % of profile depth
        depth = ref_resamp['depth'].iloc[-1]
        win_sz = 0.2 * depth

        if optimizing_function == 'dtw':
            match_prof, ref_prof, best_alphas, score = cost_dtw(
                prof_resamp, ref_resamp,
                method='multiscale', window_cm=win_sz
            )
        else:
            # Do the profile matching
            match_prof, ref_prof, best_alphas, score = minimize_cost(prof_resamp, ref_resamp,
                                                              optimizing_function,
                                                              delta_L=delta_L,
                                                              lower_bound=lower_bound,
                                                              upper_bound=upper_bound)
        
        
        return ref_prof, match_prof, best_alphas, prof_resamp, score
    
    except IndexError:
        print('Warning: profile alignment failed due to insufficient data')
        return None, None, None, None, None


######################################################################################################




if __name__ == '__main__':

    data_path = '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports'
    scopes = os.path.join(data_path, 'Snow_Scope')
    scope_lst = os.listdir(scopes)

    smps = os.path.join(data_path, 'smp_profiles_exports')
    smp_lst = os.listdir(smps)

    # scope profile: SN340 PN 33, 34, 35, 36, 38
    # smp profiles: SN19, PN 552 - 561

    scope_pns = ['33', '34', '35', '36', '38']
    smp_pns = list(range(552, 562))
    smp_pns = [str(i) for i in smp_pns]
    smp_paths = [os.path.join(smps, f'S19M{i.zfill(4)}.PNT_samples.csv') for i in smp_pns]

    scope_ids = [f'Profile{i}_SN00340' for i in scope_pns]
    scope_paths = [next((f for f in scope_lst if i in f), None) for i in scope_ids]
    scope_paths = [os.path.join(scopes, i) for i in scope_paths]

    # read in snow scopes, function also renames columns to 'depth' and 'force'
    scope_a = scope_head(pd.read_csv(scope_paths[2], skiprows=1, usecols=[0, 1]))
    scope_b = scope_head(pd.read_csv(scope_paths[-2], skiprows=1, usecols=[0, 1]))

    # read in smps, rename columns
    smp_a = pd.read_csv(smp_paths[1], low_memory=False, skiprows=0, usecols=[1, 2])
    smp_b = pd.read_csv(smp_paths[1], low_memory=False, skiprows=0, usecols=[1, 2])
    smp_a.columns = ['depth', 'force']
    smp_b.columns = ['depth', 'force']

    ram_path = os.path.join(data_path, 'BDG_20250115_sram.csv')
    ram = pd.read_csv(ram_path)
    print(ram)


    #smp_a['force'] *= 61.9
    #smp_b['force'] *= 61.9

    # resample to 1mm grid, but depth in units of cm
    scope_a = resample(scope_a, 1)
    scope_b = resample(scope_b, 1)

    smp_a = resample(smp_a, 1)
    #smp_a['force'] *= 2.98
    smp_b = resample(smp_b, 1)
    #smp_b['force'] *= 2.98

    scope_a, smp_a = same_depth(scope_a, smp_a)
    scope_b, smp_b = same_depth(scope_b, smp_b)



    matched_scope_a, smp_a, best_alphas, cost = minimize_cost(scope_a, smp_a, distance_cosine, 4)
    matched_scope_b, smp_b, _, _ = minimize_cost(scope_b, smp_b, distance_mse, 4)

    matched_scope_a, smp_a = same_depth(matched_scope_a, smp_a)

    matched_scope_b, smp_b = same_depth(matched_scope_b, smp_b)

    fig, ax = plt.subplots(1, 3, figsize=(14, 7), sharey=True)

    # Subplot 1: Profile Alignment
    ax[0].plot(scope_a['force'], scope_a['depth'], label='Original Scope', alpha=0.5, color='gray')
    ax[0].plot(matched_scope_a['force'], matched_scope_a['depth'], label='Matched Scope', color='blue')
    ax[0].plot(smp_a['force'], smp_a['depth'], label='Reference SMP', color='green', alpha=0.7)
    ax[0].set_title('Profile Matching Alignment')
    ax[0].set_xlabel('Force/Hardness')
    ax[0].set_ylabel('Depth (cm)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Subplot 2: Alpha Scaling per layer
    num_layers = len(best_alphas)
    layer_midpoints = np.arange(num_layers) * 4 + 2  # 4cm thickness, midpoint is +2
    ax[1].barh(layer_midpoints, best_alphas - 1.0, height=4 * 0.8, left=1.0,
               color=['crimson' if a < 1 else 'teal' for a in best_alphas], alpha=0.7)
    ax[1].axvline(1.0, color='black', linestyle='--')
    ax[1].set_title('Layer Expansion/Compression ($\\alpha$)')
    ax[1].set_xlabel('Scale Factor')
    ax[1].grid(axis='x', alpha=0.3)

    # Subplot 3: Absolute Displacement Path
    displacement = matched_scope_a['depth'] - scope_a['depth']
    ax[2].plot(displacement, scope_a['depth'], color='purple', linewidth=2)
    ax[2].axvline(0, color='black', linestyle='--')
    ax[2].set_title('Net Physical Shift')
    ax[2].set_xlabel('Displacement (cm)')
    ax[2].grid(True, alpha=0.3)

    # Invert y-axis for the whole shared figure view
    ax[0].invert_yaxis()

    plt.tight_layout()
    plt.show()




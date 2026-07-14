

from pyts.metrics import dtw
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import N_A

from profile_matching import *

'''
A quick script for testing different methods of dynamic time warping
'''

data_path = '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports'
scopes = os.path.join(data_path, 'Snow_Scope')
scope_lst = os.listdir(scopes)

smps = os.path.join(data_path, 'smp_profiles_exports')
smp_lst = os.listdir(smps)

# scope profile: SN340 PN 33, 34, 35, 36, 38
# smp profiles: SN19, PN 552 - 561

scope_pns = ['12', '13', '15', '16', '17']
smp_pns = list(range(461, 471))

smp_pns = [str(i) for i in smp_pns]
smp_paths = [os.path.join(smps, f'S19M{i.zfill(4)}.PNT_samples.csv') for i in smp_pns]

scope_ids = [f'Profile{i}_SN00308' for i in scope_pns]
scope_paths = [next((f for f in scope_lst if i in f), None) for i in scope_ids]
scope_paths = [os.path.join(scopes, i) for i in scope_paths]

# read in snow scopes, function also renames columns to 'depth' and 'force'
scope_a = scope_head(pd.read_csv(scope_paths[2], skiprows=1, usecols=[0, 1]))
scope_b = scope_head(pd.read_csv(scope_paths[-2], skiprows=1, usecols=[0, 1]))

# read in smps, rename columns
smp_a = pd.read_csv(smp_paths[1], low_memory=False, skiprows=0, usecols=[1, 2])
smp_b = pd.read_csv(smp_paths[2], low_memory=False, skiprows=0, usecols=[1, 2])
smp_a.columns = ['depth', 'force']
smp_b.columns = ['depth', 'force']

ram_path = os.path.join(data_path, 'BDG_20250115_sram.csv')
ram = ram_head(pd.read_csv(ram_path))




'''
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

    return matched_profile, ref_prof, None, normalized_score'''

def dtw(prof, ref_prof, window_cm=10, penalty=0.05, use_derivative=False):
    '''
    Uses dynamic time warping to align prof and ref_prof with a locality
    window constraint
    :param prof: Profile to be matched
    :param ref_prof: Reference profile, considered truth
    :param window_cm: when calculating cost matrix, how many adjacent points to consider
    :return:
    '''

    s1 = ref_prof['force'].values
    s2 = prof['force'].values

    depths = ref_prof['depth'].values

    # because cost matrix is defined on absolute difference, normalize each profile
    s1max, s1min = np.max(s1), np.min(s1)
    s2max, s2min = np.max(s2), np.min(s2)

    s1_norm = (s1 - s1min) / (s1max - s1min + 1e-6)
    s2_norm = (s2 - s2min) / (s2max - s2min + 1e-6)

    if use_derivative:
        grad1 = np.gradient(s1_norm)
        grad2 = np.gradient(s2_norm)
        # Re-normalize gradients to [0, 1] range
        s1_input = (grad1 - np.min(grad1)) / (np.max(grad1) - np.min(grad1) + 1e-6)
        s2_input = (grad2 - np.min(grad2)) / (np.max(grad2) - np.min(grad2) + 1e-6)
    else:
        s1_input = s1_norm
        s2_input = s2_norm

    N, M = len(s1_input), len(s2_input)

    # convert window [cm] to index units (i.e. mm resolution dz steps)
    window = int(window_cm / 0.1)

    # initialize cost matrix with infinities, set cost at 0, 0 = 0
    cost_mat = np.full((N, M), np.inf)
    cost_mat[0, 0] = 0

    # fill cost matrix within bounded window
    for i in range(N):
        j_start = max(0, i - window)
        j_end = min(M, i + window)
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue

            # add a penalty for moving horizontally or vertically along cost matrix
            v1 = cost_mat[i - 1, j] + penalty if i > 0 else np.inf
            v2 = cost_mat[i, j - 1] + penalty if j > 0 else np.inf
            v3 = cost_mat[i - 1, j - 1] if (i > 0 and j > 0) else np.inf

            cost_mat [i, j] = abs(s1_input[i] - s2_input[j]) + min(v1, v2, v3)

    # now find best path through cost matrix
    path = []
    i, j = N - 1, M - 1 # start at bottom of profile
    while i > 0 or j > 0:
        path.append((i, j))
        # if reached bottom of one profile, continue on with other profile
        if i == 0 :
            j -= 1
            continue
        elif j == 0:
            i -= 1
            continue
        # choose the direction (E, S, SE) with minimum cost value
        choices = [cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1]]
        best = np.argmin(choices)
        if best == 0:
            i -= 1
        elif best == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.append((0, 0))
    path.reverse()

    # path is a list of ordered pairs (ref_ix, prof_ix)
    #   which points to which profile index to match to reference index
    shifted_forces = np.zeros(N)
    counts = np.zeros(N)

    for idx_ref, idx_prof in path:
        shifted_forces[idx_ref] += s2[idx_prof]
        counts[idx_ref] += 1

    # in instances where multiple points in prof line up with single point
    #   in ref_prof, take the average of all forces
    counts[counts == 0] = 1
    final_forces = shifted_forces / counts
    matched_profile = pd.DataFrame({
        'depth': depths,
        'force': final_forces
    })

    return matched_profile, cost_mat, path



scope_a = resample(scope_a, 5)
scope_b = resample(scope_b, 5)

smp_a = resample(smp_a, 5)
smp_b = resample(smp_b, 5)


#scope_a, smp_b = same_depth(scope_a, smp_b)

matched_scope_b, mat, path = dtw(prof=scope_a, ref_prof=smp_b)
path = np.array(path)

plt.matshow(mat)
plt.plot(path[:, 0], path[:, 1], alpha=1, color='red')
plt.colorbar()
plt.grid()
plt.show()
plt.figure(1)


plt.plot(smp_b['depth'], smp_b['force'], marker='o', markersize=2, color='red', label='Ref SMP')
plt.plot(scope_a['depth'], scope_a['force'], marker='x', markersize=2, color='grey', label='OG Scope')
plt.plot(matched_scope_b['depth'], matched_scope_b['force'], marker='o', markersize=2, color='blue', label='Matched Scope')
plt.grid()
plt.legend()
plt.show()
plt.grid()
plt.figure(2)

shift = path[:, 1] - path[:, 0]
plt.plot(shift, path[:, 0])
plt.gca().invert_yaxis()
plt.grid()
plt.show()
plt.figure(3)




'''
matched_scope_a, smp_a, _, score = cost_dtw(smp_b, smp_a)
fig, ax = plt.subplots(1, 3, figsize=(14, 7), sharey=True)

# Subplot 1: Profile Alignment
#ax[0].plot(scope_a['force'], scope_a['depth'], label='Original Scope', alpha=0.5, color='gray')
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


'''
import numpy as np
import pandas as pd
import os
import random as rand
import ast
from profile_matching import *
from scipy.spatial.distance import cosine


'''
Quick script that selects random snow scope and SMP from given pit
and matches the scope to SMP for all data collected.
'''


caca = pd.read_csv('/Users/colemankane_1/Documents/BSU/Penetrometer_Analysis/all_profiles_thinned.csv')

caca['scope_paths'] = caca['scope_paths'].dropna().apply(ast.literal_eval)
caca['smp_paths'] = caca['smp_paths'].dropna().apply(ast.literal_eval)
caca['ram'] = caca['ram'].dropna()


caca_filt = caca[
    (caca['ram'].str.len() > 0) &
    (caca['smp_paths'].str.len()>0)
].reset_index()

rand_pits = rand.sample(list(caca_filt.index), len(caca_filt))

for i in rand_pits:

    smp_paths = caca_filt['smp_paths'].iloc[i]


    ram_path = caca_filt['ram'].iloc[i]

    ram = os.path.basename(ram_path)
    ram_path = ram_path.replace('/bsuhome/colemankane/Documents/crrel_exports', '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports')
    print('ram path:', ram_path)
    og_ram = pd.read_csv(ram_path)
    og_ram_prof = {
        'depth': og_ram['l_cm'],
        'force': og_ram['rr_N']
    }
    print(og_ram.columns)

    scope_paths = caca_filt['scope_paths'].iloc[i]
    print(scope_paths)
    try:
        scope_path = scope_paths[0]
        scope_path = scope_path.replace('/bsuhome/colemankane/Documents/crrel_exports/Snow_Scope',
                                    '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports/Snow_Scope')
    except: scope_path = None



    smp_path = rand.choice(smp_paths)
    smp_path1 = rand.choice(smp_paths)
    smp_path = smp_path.replace('/bsuhome/colemankane/Documents/crrel_exports/smp_profiles_exports', '/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports/smp_profiles_exports')
    smp_base = os.path.basename(smp_path)


    if scope_path:
        smp, matched_scope, best_alphas, og_scope, score = wrapper(
            smp_path, scope_path,
            'smp', 'scope',
            'dtw'
        )

        #/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports/smp_profiles_exports
        #/Users/colemankane_1/Library/CloudStorage/GoogleDrive-ColemanKane@boisestate.edu/Shared drives/2024-2025 CRREL Snow Strength/Data/Scrubbed pit_strength_transect data/crrel_exports/smp_profile_exports/S06M0327.PNT_samples.csv
        print('score:', score)



        fig, ax = plt.subplots(1, 3, figsize=(14, 7), sharey=True)

        # Subplot 1: Profile Alignment
        ax[0].step(og_scope['force'], og_scope['depth'], label='Original Scope', alpha=0.5, color='gray')
        ax[0].step(matched_scope['force'], matched_scope['depth'], label='Matched Scope', color='blue')
        ax[0].step(smp['force'], smp['depth'], label='Reference ram', color='green', alpha=0.7)
        ax[0].set_title('Profile Matching Alignment')
        ax[0].set_xlabel('Force/Hardness')
        ax[0].set_ylabel('Depth (cm)')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        '''
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
        displacement = matched_ram['depth'] - og_ram['depth']
        ax[2].plot(displacement, matched_ram['depth'], color='purple', linewidth=2)
        ax[2].axvline(0, color='black', linestyle='--')
        ax[2].set_title('Net Physical Shift')
        ax[2].set_xlabel('Displacement (cm)')
        ax[2].grid(True, alpha=0.3)
    '''
    # Invert y-axis for the whole shared figure view
        ax[0].invert_yaxis()

        plt.tight_layout()
        plt.show()

    smp = None
    match_scope = None
    best_alphas = None
    og_scope = None
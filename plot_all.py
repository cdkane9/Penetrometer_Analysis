import numpy as np
import pandas as pd
import os
import random as rand
import ast
from profile_matching import *


caca = pd.read_csv('/Users/colemankane_1/Documents/BSU/Penetrometer_Analysis/all_profiles.csv')
caca = caca.replace('/Users/colemankane/',
                    '/Users/colemankane_1/', regex=True)
#caca.to_csv('/Users/colemankane_1/Documents/BSU/Penetrometer_Analysis/all_profiles2.csv')
caca['scope_paths'] = caca['scope_paths'].dropna().apply(ast.literal_eval)
caca['smp_paths'] = caca['smp_paths'].dropna().apply(ast.literal_eval)


caca_filt = caca[
    (caca['scope_paths'].str.len() > 0) &
    (caca['smp_paths'].str.len() > 0)
].reset_index()
rand_pits = rand.sample(list(caca_filt.index), len(caca_filt))
print(len(caca_filt))
for i in rand_pits:

    smp_paths = caca_filt['smp_paths'].iloc[i]
    scope_paths = caca_filt['scope_paths'].iloc[i]

    smp_path = rand.choice(smp_paths)
    scope_path = rand.choice(scope_paths)

    print(smp_path)

    smp, match_scope, best_alphas, og_scope = wrapper(
        smp_path, scope_path,
        'smp', 'scope',
        distance_cosine
    )





    fig, ax = plt.subplots(1, 3, figsize=(14, 7), sharey=True)

    # Subplot 1: Profile Alignment
    ax[0].plot(og_scope['force'], og_scope['depth'], label='Original Scope', alpha=0.5, color='gray')
    ax[0].plot(match_scope['force'], match_scope['depth'], label='Matched Scope', color='blue')
    ax[0].plot(smp['force'], smp['depth'], label='Reference SMP', color='green', alpha=0.7)
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
    displacement = match_scope['depth'] - og_scope['depth']
    ax[2].plot(displacement, match_scope['depth'], color='purple', linewidth=2)
    ax[2].axvline(0, color='black', linestyle='--')
    ax[2].set_title('Net Physical Shift')
    ax[2].set_xlabel('Displacement (cm)')
    ax[2].grid(True, alpha=0.3)

    # Invert y-axis for the whole shared figure view
    ax[0].invert_yaxis()

    plt.tight_layout()
    plt.show()

    smp = None
    match_scope = None
    best_alphas = None
    og_scope = None
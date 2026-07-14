import pandas as pd
import os
import ast

'''
Outline: 
    Take the spreadsheet that was created after manually inspecting all profiles
    and use that information to thin out 'all_profiles2.csv'
'''


all_profs = pd.read_csv('/Users/colemankane_1/Documents/BSU/Penetrometer_Analysis/all_profiles2.csv')

clean_scopes_25 = pd.read_csv('/Users/colemankane_1/Documents/BSU/Crrel Snow Strength/pen_analysis/scope_cleaning_filled.csv')
clean_scopes_24 = pd.read_csv('/Users/colemankane_1/Documents/BSU/Crrel Snow Strength/pen_analysis/scope_24_cleaning_filled.csv')


clean_scopes = pd.concat([clean_scopes_25, clean_scopes_24])
tossed_scopes = set(clean_scopes[(clean_scopes['toss'] == 1)]['profile'])



clean_smps = pd.read_csv('/Users/colemankane_1/Documents/BSU/Crrel Snow Strength/pen_analysis/smp_cleaning_filled.csv')
tossed_smps = set(clean_smps[(clean_smps['toss'] == 1) | (clean_smps['overload'] == 1)]['profile'])





def filter_tossed_profiles(paths_str, tossed_set):
    if pd.isna(paths_str):
        return paths_str

    try:
        paths_list = ast.literal_eval(paths_str)

        if isinstance(paths_list, list):
            filtered_list = [
                path for path in paths_list if os.path.basename(path) not in tossed_set
            ]

            return str(filtered_list)
    except (ValueError, SyntaxError):
        return paths_str

    return paths_str

all_profs['scope_paths'] = all_profs['scope_paths'].apply(
    lambda x: filter_tossed_profiles(x, tossed_scopes)
)

all_profs['smp_paths'] = all_profs['smp_paths'].apply(
    lambda x: filter_tossed_profiles(x, tossed_smps)
)

def trim_bottom_5cm(file_path, depth_column, resolution):
    if not os.path.exists(file_path):
        print('File not found')
        return
    try:
        profile_df = pd.read_csv(file_path)
        for col in ['depth (mm)', 'distance [mm]']:
            if col in profile_df.columns:
                depth_col = col
                break
        if depth_col is None:
            print(f'skipping {os.path.basename(file_path)}, could not find depth column')



        if resolution == 'mm':
            profile_df[depth_column] /= 10

        max_depth = profile_df[depth_col].max()
        trimmed_df = profile_df[profile_df[depth_col]<= (max_depth - 5)]

        if resolution == 'mm':
            trimmed_df[depth_column] *= 10
        #trimmed_df.to_csv(file_path, index=False)
    except:
        print(f'error {file_path}')









all_profs.to_csv('/Users/colemankane_1/Documents/BSU/Penetrometer_Analysis/all_profiles_thinned.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def group_force(layer_df_path, penetrometer):
    layer_df = pd.read_csv(layer_df_path)
    col_name = f'avg_force_{penetrometer}'

    valid = layer_df .dropna(subset=[col_name]).copy()
    valid['HH_clean'] = (
        valid['HH'].astype(str).str.replace(r'[+-]', '', regex=False)
    )

    categories = ['F', '4F', '1F', 'P', 'K', 'I']

    grouped = valid.groupby('HH_clean')[col_name].apply(list).to_dict()

    hardness_dict = {cat: grouped.get(cat, []) for cat in categories}
    return hardness_dict

def plot_hardness(pen_dict, Penetrometer):
    plot_data = []
    for category, forces in pen_dict.items():
        for force in forces:
            plot_data.append({'Hand Hardness': category, 'Penetration Resistance [N]': force})

    df_plot = pd.DataFrame(plot_data)

    fig, ax = plt.subplots()
    ax.set_yscale('symlog', linthresh=0.1)


    #sns.set_theme(style='whitegrid')

    order = ['F', '4F', '1F', 'P', 'K', 'I']

    sns.violinplot(
        data = df_plot,
        x='Hand Hardness',
        y='Penetration Resistance [N]',
        order=order,
        palette='viridis',
        #notch=False,
        #fliersize=0,
        width=0.6
    )

    sns.stripplot(
        data=df_plot,
        x="Hand Hardness",
        y="Penetration Resistance [N]",
        order=order,
        color="black",
        alpha=0.4,
        size=4,
        jitter=0.2,
    )

    max_overall_force = df_plot['Penetration Resistance [N]'].max()
    y_limit_upper = max_overall_force * 1.18
    ax.set_ylim(0, y_limit_upper)

    intervals = {}

    for i, cat in enumerate(order):
        category_forces = pen_dict.get(cat, [])
        n = len(category_forces)
        mean_force = np.mean(category_forces)
        std_force = np.std(category_forces)
        max_force = np.max(category_forces)

        lower = round(np.percentile(category_forces, 20), 2)
        median = round(np.percentile(category_forces, 50), 2)
        upper = round(np.percentile(category_forces, 80), 2)

        intervals[cat] = {
            "Lower Bound": round(lower, 2),
            "Center/Median": round(median, 2),
            "Upper Bound": round(upper, 2),
        }

        #label_text = f"n={n}\nμ={mean_force:.1f}\nσ={std_force:.1f}"
        label_text = f'n={n}\nMedian={median}\n20th={lower}\n80th={upper}'
        y_position = max_force + (y_limit_upper * 0.01)

        ax.text(
            i,
            y_position,
            label_text,
            ha='center',
            va='bottom',
            fontsize=11,
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                ec='gray'
            ),
        )

    plt.title(
        f'{Penetrometer} vs. Hand Hardness'
    )
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'figures/hh_{Penetrometer}')
    plt.show()

if __name__ == '__main__':
    ram = group_force('all_ram_layers.csv', 'ram')
    smp = group_force('all_smp_layers.csv', 'smp')
    scope = group_force('all_scope_layers.csv', 'scope')


    plot_hardness(ram, 'Ram')
    plot_hardness(smp, 'SMP')
    plot_hardness(scope, 'Scope')
    pass
"""
Plots for Forecast Combination with Simulated AR(1) Losses

Figures J.1 and J.2: mean regret, confidence intervals and regret bounds.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES = {
    (1, 'ftl'): 'AR1_loss_gaps_mix_exercise_vary_gap1_fix_q_FTL',
    (1, 'decreasing_hedge'): 'AR1_loss_gaps_mix_exercise_vary_gap1_fix_q_DecHedge',
    (2, 'ftl'): 'AR1_loss_gaps_mix_exercis_vary_q_fix_gap1_FTL',
    (2, 'decreasing_hedge'): 'AR1_loss_gaps_mix_exercise_vary_q_fix_gap1_DecHedge',
}

#region --- Manuscript Figures


def make_figures(summary, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 10, 'axes.labelsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': .8, 'axes.edgecolor': 'black',
        'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    for (figure, method), name in FIGURES.items():
        fig, (upper, lower) = plt.subplots(2, 1, figsize=(5.7, 7.2), sharex=True)
        fig.subplots_adjust(left=.14, right=.98, bottom=.12, top=.98, hspace=.20)
        choices = [0., .2, .45] if figure == 1 else [.01, .05, .2]
        fixed, varying = ('q', 'delta1') if figure == 1 else ('delta1', 'q')
        for value, color, style, marker in zip(choices, ['C0', 'C1', 'C2'], ['-', '--', '-.'], ['o', 's', '^']):
            rows = summary.loc[(summary.method == method) & (summary[fixed] == value)].sort_values(varying)
            x = rows[varying].to_numpy()
            label = rf'$q={value:.2f}$' if figure == 1 else rf'$\Delta_1={value:.2f}$'
            upper.fill_between(x, rows.ci_low.to_numpy(), rows.ci_high.to_numpy(),
                               color=color, alpha=.12, linewidth=0)
            upper.plot(x, rows.mean_regret, color=color, linestyle=style, marker=marker,
                       markersize=4, linewidth=1.6, label=label)
            for kind, bound_style in [('H', '-'), ('B', '--')]:
                lower.plot(x, rows['bound_'+kind], color=color, linestyle=bound_style,
                           marker=marker, markersize=3.5, linewidth=1.5)
            if figure == 2:
                iid = rows.loc[rows.q == 0].iloc[0]
                for key, symbol in [('bound_iid_H', 'o'), ('bound_iid_B', 'D')]:
                    lower.plot([0], [iid[key]], marker=symbol, linestyle='none',
                               markerfacecolor='white', markeredgecolor=color,
                               markersize=6, markeredgewidth=1.2, zorder=6)
        for ax in [upper, lower]:
            ax.set_axisbelow(True)
            ax.grid(which='major', color='0.85')
        upper.set_ylabel('Mean cumulative regret')
        upper.set_ylim(bottom=0)
        lower.set_ylabel('Theoretical upper bound')
        lower.set_yscale('log')
        legend = upper.legend(loc='lower right', fontsize='small', framealpha=1, fancybox=False)
        legend.get_frame().set_edgecolor('0.85')
        if figure == 1:
            lower.set_xlim(0, .31)
            lower.set_xticks([.01, .025, .05, .1, .2, .3], ['0.01', '0.025', '0.05', '0.10', '0.20', '0.30'])
            lower.set_xlabel(r'Population gap $\Delta_1$')
        else:
            lower.set_xlim(-.01, .46)
            lower.set_xticks([0, .1, .2, .3, .4, .45], ['0', '0.10', '0.20', '0.30', '0.40', '0.45'])
            lower.set_xlabel(r'AR persistence $q$')
        lower.tick_params(axis='x', labelrotation=45)
        fig.savefig(destination / (name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f'Figure J.{figure}: {name}.pdf')

#endregion

import argparse
from plot_scripts.calculate_avg_performance import calculate_avg_performance
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

font_dict=dict(family='Arial',
            size=18,
            color='black'
            )

agent_paths = {
        'level0_agent': r"..\logs\VariableWindLevel0-v17\1618915245ppo",
        'level1_agent': r"..\logs\VariableWindLevel1-v17\1618921752ppo",
        'level2_agent': r"..\logs\VariableWindLevel2-v17\1618928488ppo",
        'level3_agent': r"..\logs\VariableWindLevel3-v17\1618934307ppo",
        'level4_agent': r"..\logs\VariableWindLevel4-v17\1618940658ppo",
        'level5_agent': r"..\logs\VariableWindLevel5-v17\1618946704ppo",
        'levelPSFtest_agent': r"..\logs\VariableWindPSFtest-v17\1619770390ppo"
    }
agent_paths_psf = {
        'level0_agent_psf': r"..\logs\VariableWindLevel0-v17\1619804074ppo",
        'level1_agent_psf': r"..\logs\VariableWindLevel1-v17\1619817419ppo",
        'level2_agent_psf': r"..\logs\VariableWindLevel2-v17\1619805498ppo",
        'level3_agent_psf': r"..\logs\VariableWindLevel3-v17\1619826109ppo",
        'level4_agent_psf': r"..\logs\VariableWindLevel4-v17\1619817419ppo",
        'level5_agent_psf': r"..\logs\VariableWindLevel5-v17\1619809322ppo",
        'levelPSFtest_agent_psf': r"..\logs\VariableWindPSFtest-v17\1619696400ppo"
    }

def get_data(paths_dict):

    performance_data = []
    crash_data = []
    for _, path in agent_paths.items():
        agent_folder = os.path.join(path,'test_data')
        agent_performances = []
        agent_crashes = []
        for filename in os.listdir(agent_folder):
            if filename.endswith(".csv") and '6000' in filename and 'PSFtest' not in filename:
                file = os.path.join(agent_folder, filename)
                perf, crash = calculate_avg_performance(file)
                agent_performances.append(perf)
                agent_crashes.append(crash)
        performance_data.append(agent_performances)
        crash_data.append(agent_crashes)
    return performance_data, crash_data

#                           Test Lvl 0, Test Lvl 1, Test Lvl 2, Test Lvl 3, Test Lvl 4, Test Lvl 5 
performance_data_3000 = np.array([  [6193.86, 6073.30, 5867.92, 6108.16, 5855.94, 5815.67], # Agent level 0
                                    [6184.24, 6043.18, 5736.11, 6060.25, 5791.88, 5681.88], # Agent level 1
                                    [6041.56,5952.68,5163.92,5953.56,5783.19,5598.68], # Agent level 2
                                    [6156.46,6098.61,5772.30,6100.28,5850.98,5837.25], # Agent level 3
                                    [6163.32,6055.70,5835.18,6089.63,5883.88,5760.50], # Agent level 4
                                    [5995.40,5738.16,5138.83,5651.29,5126.57,4954.20] # Agent level 5
                                    ])

crash_data_3000 = np.array([[0,0,22,0,0,23], 
                            [0,0,25,0,0,19],
                            [0,6,14,1,0,17],
                            [0,0,22,0,0,31],
                            [0,0,23,0,0,26],
                            [0,1,25,1,1,30]
                            ])

# 6000 timesteps
performance_data_6000 = np.array([   [12427.57,12262.73,11765.28,12241.83,11820.11,11743.90], 
                                [12395.82,12149.57,11720.24,12149.67,11544.79,11651.32],
                                [12136.35,11956.19,10221.33,12006.35,11600.21,10912.29],
                                [12384.29,12141.83,11686.95,12241.95,11795.42,11775.06],
                                [12409.96,12340.44,11774.98,12190.39,11764.91,11684.04],
                                [12009.11,11416.92,9759.69,11434.15,10335.79,9938.75]
                                ])
performance_data_6000 = 100*performance_data_6000/(3*6000)

crash_data_6000 = np.array([ [0,0,24,0,0,27], 
                        [0,1,36,0,0,27],
                        [0,1,10,2,1,22],
                        [0,0,31,0,1,28],
                        [0,0,28,0,0,32],
                        [0,4,20,2,1,22]
                        ])


def plot_gen_performance(save=False, group_by_test_level=False):
    global performance_data
    xlabels = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    
    if group_by_test_level:
        performance_data = performance_data.transpose()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels,
        y=performance_data[:,0],
        name='Agent Level 0' if group_by_test_level else 'Test Level 0',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=performance_data[:,1],
        name='Agent Level 1' if group_by_test_level else 'Test Level 1',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=performance_data[:,2],
        name='Agent Level 2' if group_by_test_level else 'Test Level 2',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=performance_data[:,3],
        name='Agent Level 3' if group_by_test_level else 'Test Level 3',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=performance_data[:,4],
        name='Agent Level 4' if group_by_test_level else 'Test Level 4',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=performance_data[:,5],
        name='Agent Level 5' if group_by_test_level else 'Test Level 5',
    ))

    fig.update_layout(  barmode='group', 
                        margin=dict(l=0, r=0, t=0, b=0),
                        font=font_dict,  # font formatting
                        plot_bgcolor='white',
                        template="simple_white")
    # x and y-axis formatting
    fig.update_yaxes(title_text='Performance',  # axis label
                    showline=True,  # add line at x=0
                    linecolor='black',  # line color
                    linewidth=2.4, # line size
                    ticks='outside',  # ticks outside axis
                    tickfont=font_dict, # tick label font
                    mirror='allticks',  # add ticks to top/right axes
                    tickwidth=2.4,  # tick width
                    tickcolor='black',  # tick color
                    range=[4000, 6300]
                    )
    fig.update_xaxes(title_text='Test Level' if group_by_test_level else 'Agent Train Level',
                    showline=True,
                    showticklabels=True,
                    linecolor='black',
                    linewidth=2.4,
                    ticks='outside',
                    tickson="boundaries",
                    tickfont=font_dict,
                    mirror='allticks',
                    tickwidth=2.4,
                    tickcolor='black',
                    )
    fig.show()

    if save:
        if group_by_test_level:
            fig.write_image("plots/generalization_performance_bar_group_by_test_level.pdf")
        else:
            fig.write_image("plots/generalization_performance_bar_group_by_agent_level.pdf")

def plot_gen_crash(save=False, group_by_test_level=False):
    global crash_data
    xlabels = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']


    if group_by_test_level:
        crash_data = crash_data.transpose()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels,
        y=crash_data[:,0],
        name='Agent Level 0' if group_by_test_level else 'Test Level 0',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=crash_data[:,1],
        name='Agent Level 1' if group_by_test_level else 'Test Level 1',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=crash_data[:,2],
        name='Agent Level 2' if group_by_test_level else 'Test Level 2',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=crash_data[:,3],
        name='Agent Level 3' if group_by_test_level else 'Test Level 3',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=crash_data[:,4],
        name='Agent Level 4' if group_by_test_level else 'Test Level 4',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=crash_data[:,5],
        name='Agent Level 5' if group_by_test_level else 'Test Level 5',
    ))

    fig.update_layout(  barmode='group', 
                        margin=dict(l=0, r=0, t=0, b=0),
                        font=font_dict,  # font formatting
                        plot_bgcolor='white',
                        template="simple_white")
    # x and y-axis formatting
    fig.update_yaxes(title_text='Crashes [%]',  # axis label
                    showline=True,  # add line at x=0
                    linecolor='black',  # line color
                    linewidth=2.4, # line size
                    ticks='outside',  # ticks outside axis
                    tickfont=font_dict, # tick label font
                    mirror='allticks',  # add ticks to top/right axes
                    tickwidth=2.4,  # tick width
                    tickcolor='black',  # tick color
                    range=[0, 100],
                    )
    fig.update_xaxes(title_text='Test Level' if group_by_test_level else 'Agent Train Level',
                    showline=True,
                    showticklabels=True,
                    linecolor='black',
                    linewidth=2.4,
                    ticks='outside',
                    tickson="boundaries",
                    tickfont=font_dict,
                    mirror='allticks',
                    tickwidth=2.4,
                    tickcolor='black',
                    )
    fig.show()

    if save:
        if group_by_test_level:
            fig.write_image("plots/generalization_crash_bar_group_by_test_level.pdf")
        else:
            fig.write_image("plots/generalization_crash_bar_group_by_agent_level.pdf")

def plot_gen_heatmap(performance_data, crash_data, save=False):
    textcolors=["white","black"]

    data = performance_data
    ylabel = "Performance"
    filename = "plots/generalization_performance_heatmap_6000.pdf"
    format = "{:.2f}".format
    cmap = 'autumn'
    
    # data = crash_data
    # ylabel = "Crash rate"
    # filename = "plots/generalization_crash_heatmap_6000.pdf"
    # format = mtick.PercentFormatter(decimals=0)
    # cmap = 'autumn_r'
    # textcolors.reverse()
    
    labels = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, format=format)
    cbar.ax.set_ylabel(ylabel, rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("Agent Train Level")
    ax.set_xlabel("Test Level")

    # Normalize the threshold to the images color range.
    threshold = data.min()+(data.max()-data.min())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j]),ha="center", va="center", color=textcolors[int(data[i, j] > threshold)])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    fig.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_training_performance(performance_data, crash_data, save=False):

    labels = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    fig, ax = plt.subplots()
    new_performance_data = [performance_data[i][i] for i in range(len(performance_data[0]))]
    new_crash_data = [crash_data[i][i] for i in range(len(crash_data[0]))]

    lns1 = ax.bar(labels, new_performance_data, width=0.5, label='Performance')
    ax.set_ylabel('Performance')
    ax.set_ylim([0,100])
    ax.set_axisbelow(True)
    ax.grid(True, color='gray', axis='y', linestyle='dashed')
    ax.set_yticks(np.arange(0,101,5))
    # ax.set_ylim([0,100])
    # ax.tick_params(axis='y', colors='darkorange')

    ax2 = ax.twinx()
    lns2 = ax2.bar(labels, new_crash_data, width=0.5, color="red", label='Crash Rate')
    ax2.set_ylabel('Crash rate')
    ax2.set_ylim([0,100])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.tick_params(axis='y', colors='red')
    ax2.set_yticks(np.arange(0,101,5))
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    fig.tight_layout()
    if save:
        plt.savefig("plots/training_performance_bar_6000.pdf", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    parser.add_argument(
        '--psf',
        help='Plot PSF results',
        action='store_true'
    )
    parser.add_argument(
        '--group_by_test_level',
        help='Group bars by test level instead of agent level',
        action='store_true'
    )
    args = parser.parse_args()

    if args.psf:
        performance_data, crash_data = get_data(agent_paths_psf)
    else:
        performance_data, crash_data = get_data(agent_paths)
    
    performance_data = 100*np.array(performance_data)/(3*6000)

    

    # plot_gen_performance(save=args.save,group_by_test_level=args.group_by_test_level)
    # plot_gen_crash(save=args.save,group_by_test_level=args.group_by_test_level)
    # plot_gen_heatmap(performance_data, crash_data, save=args.save)
    plot_training_performance(performance_data, crash_data, save=args.save)
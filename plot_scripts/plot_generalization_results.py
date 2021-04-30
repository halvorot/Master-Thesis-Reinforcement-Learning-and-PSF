import argparse

import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

font_dict=dict(family='Arial',
            size=18,
            color='black'
            )

performance_data_3000 = np.array([  [6193.86, 6073.30, 5867.92, 6108.16, 5855.94, 5815.67], 
                                    [6184.24, 6043.18, 5736.11, 6060.25, 5791.88, 5681.88],
                                    [6041.56,5952.68,5163.92,5953.56,5783.19,5598.68],
                                    [6156.46,6098.61,5772.30,6100.28,5850.98,5837.25],
                                    [6163.32,6055.70,5835.18,6089.63,5883.88,5760.50],
                                    [5995.40,5738.16,5138.83,5651.29,5126.57,4954.20]
                                    ])

crash_data_3000 = np.array([[0,0,22,0,0,23], 
                            [0,0,25,0,0,19],
                            [0,6,14,1,0,17],
                            [0,0,22,0,0,31],
                            [0,0,23,0,0,26],
                            [0,1,25,1,1,30]
                            ])

# 6000 timesteps
performance_data = np.array([   [], 
                                [],
                                [],
                                [],
                                [],
                                []
                                ])
performance_data = performance_data/6000

crash_data = np.array([ [], 
                        [],
                        [],
                        [],
                        [],
                        []
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

def plot_gen_heatmap(save=False):
    global performance_data, crash_data
    textcolors=["white","black"]

    data = performance_data
    ylabel = "Performance"
    filename = "plots/generalization_performance_heatmap_6000.pdf"
    format = "{:.3f}".format
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    parser.add_argument(
        '--group_by_test_level',
        help='Group bars by test level instead of agent level',
        action='store_true'
    )
    args = parser.parse_args()

    # plot_gen_performance(save=args.save,group_by_test_level=args.group_by_test_level)
    # plot_gen_crash(save=args.save,group_by_test_level=args.group_by_test_level)
    plot_gen_heatmap(save=args.save)
import argparse

import plotly.io as pio
import plotly.graph_objects as go

font_dict=dict(family='Arial',
            size=18,
            color='black'
            )

def plot_gen_performance(save=False):

    xlabels = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[6193.86,6184.24,6041.56,6156.46,1,1],
        name='Level 0',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[6073.30,6043.18,5952.68,6098.61,1,1],
        name='Level 1',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[5867.92,5736.11,5163.92,5772.30,1,1],
        name='Level 2',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[6108.16,6060.25,5953.56,6100.28,1,1],
        name='Level 3',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[5855.94,5791.88,5783.19,5850.98,0,1],
        name='Level 4',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[5815.67,5681.88,5598.68,5837.25,1,0],
        name='Level 5',
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
                    )
    fig.update_xaxes(title_text='Agent Train Level',
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
        fig.write_image("plots/generalization_performance_bar.pdf")

def plot_gen_crash(save=False):

    xlabels = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[0,0,0,0,0,0],
        name='Level 0',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[0,0,6,0,0,0],
        name='Level 1',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[22,25,14,22,0,0],
        name='Level 2',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[0,0,1,0,0,0],
        name='Level 3',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[0,0,0,0,0,0],
        name='Level 4',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[23,19,17,31,0,0],
        name='Level 5',
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
    fig.update_xaxes(title_text='Agent Train Level',
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
        fig.write_image("plots/generalization_crash_bar.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    plot_gen_performance(save=args.save)
    plot_gen_crash(save=args.save)
import argparse

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_gen_performance(save=False):

    xlabels = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[0,1,1,1,1,1],
        name='Level 0',
        marker_color='lightblue',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[1,0,1,1,1,1],
        name='Level 1',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[1,1,0,1,1,1],
        name='Level 2',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[1,1,1,0,1,1],
        name='Level 3',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[1,1,1,1,0,1],
        name='Level 4',
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[1,1,1,1,1,0],
        name='Level 5',
    ))

    font_dict=dict(family='Arial',
            size=26,
            color='black'
            )

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
                    tickfont=font_dict,
                    mirror='allticks',
                    tickwidth=2.4,
                    tickcolor='black',
                    )
    fig.show()

    if save:
        fig.write_image("plots/generalization_performance_bar.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    plot_gen_performance(save=args.save)
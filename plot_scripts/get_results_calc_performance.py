from plot_scripts.calculate_avg_performance import calculate_avg_performance
from plot_scripts.plot_generalization_results import plot_gen_heatmap
import argparse
import numpy as np
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()
    
    dir = os.path.dirname(__file__)
    agent_paths = {
        'level0_agent': r"..\logs\VariableWindLevel0-v17\1618915245ppo",
        'level1_agent': r"..\logs\VariableWindLevel1-v17\1618921752ppo",
        'level2_agent': r"..\logs\VariableWindLevel2-v17\1618928488ppo",
        'level3_agent': r"..\logs\VariableWindLevel3-v17\1618934307ppo",
        'level4_agent': r"..\logs\VariableWindLevel4-v17\1618940658ppo",
        'level5_agent': r"..\logs\VariableWindLevel5-v17\1618946704ppo",
    }
    agent_paths_psf = {
        'level0_agent_psf': r"..\logs\VariableWindLevel0-v17\1618915245ppo",
        'level1_agent_psf': r"..\logs\VariableWindLevel1-v17\1618921752ppo",
        'level2_agent_psf': r"..\logs\VariableWindLevel2-v17\1618928488ppo",
        'level3_agent_psf': r"..\logs\VariableWindLevel3-v17\1618934307ppo",
        'level4_agent_psf': r"..\logs\VariableWindLevel4-v17\1618940658ppo",
        'level5_agent_psf': r"..\logs\VariableWindLevel5-v17\1618946704ppo",
    }

    performance_data = []
    crash_data = []
    for _, path in agent_paths.items():
        agent_folder = os.path.join(path,'test_data')
        agent_performances = []
        agent_crashes = []
        for filename in os.listdir(agent_folder):
            if filename.endswith(".csv") and '6000' in filename:
                file = os.path.join(agent_folder, filename)
                perf, crash = calculate_avg_performance(file)
                agent_performances.append(perf)
                agent_crashes.append(crash)
        performance_data.append(agent_performances)
        crash_data.append(agent_crashes)

    performance_data = 100*np.array(performance_data)/(3*6000)

    plot_gen_heatmap(performance_data,crash_data,args.save)
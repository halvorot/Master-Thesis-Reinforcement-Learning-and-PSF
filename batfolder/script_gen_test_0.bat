@echo off
python test_agent.py --env VariableWindPSFtest-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindLevel0-v17\1618915245ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindPSFtest-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindLevel1-v17\1618921752ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindPSFtest-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindLevel2-v17\1618928488ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindPSFtest-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindLevel3-v17\1618934307ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindPSFtest-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindLevel4-v17\1618940658ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindPSFtest-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindLevel5-v17\1618946704ppo\agents\last_model_10000000.zip"
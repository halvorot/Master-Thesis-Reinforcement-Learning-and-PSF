@echo off
python test_agent.py --env VariableWindLevel0-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindPSFtest-v17\1619770390ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindLevel1-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindPSFtest-v17\1619770390ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindLevel2-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindPSFtest-v17\1619770390ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindLevel3-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindPSFtest-v17\1619770390ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindLevel4-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindPSFtest-v17\1619770390ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindLevel5-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindPSFtest-v17\1619770390ppo\agents\last_model_10000000.zip"
python test_agent.py --env VariableWindPSFtest-v17 --num_episodes 100 --time 600 --agent "logs\VariableWindPSFtest-v17\1619770390ppo\agents\last_model_10000000.zip"
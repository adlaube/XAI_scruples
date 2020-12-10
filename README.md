# Explain Scruples

https://github.com/allenai/scruples

## How to install?

- clone
- create venv and activate venv from requirements.txt
- install scruples in venv:             https://github.com/allenai/scruples#setup
- download scruples model & config:     https://github.com/allenai/scruples/blob/master/docs/demos.md#norms
- download scruples data:               https://github.com/allenai/scruples#data
- move model (.json & .bin) to ./model/anecdotes or ./models/dilemmas
- move data to ./data/anecdotes or ./data/dilemmas
- update path and config: start_server.sh (for GPU; CPU: start_server_CPU.sh)

## How to run?

- run start_server.sh
- configure anecdotes.py 
- run anecdotes.py
- results in root



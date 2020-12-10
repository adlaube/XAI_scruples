# Explain Scruples

https://github.com/allenai/scruples

## Features

- explain anecdotes with [LIME](https://github.com/marcotcr/lime)
- report generation with statistics on explanation features


## Findings

- model considers connotated verbs and personal pronouns

## Run scruples

- clone XAI_scruples 
- create venv from requirements.txt
- clone and install scruples in venv from requirements.txt:             https://github.com/allenai/scruples#setup
- download scruples model & config:                                     https://github.com/allenai/scruples/blob/master/docs/demos.md#norms
- move model (.json & .bin) to ./model/anecdotes or ./models/dilemmas

- update path and config: start_server.sh (for GPU; CPU: start_server_CPU.sh)

        ./start_server.sh


## Run scruples in Docker

- clone XAI_scruples
- create venv from requirements.txt
- clone scruples:       https://github.com/allenai/scruples
- build and run docker container from Dockerfile

        docker build --tag scruples .
        docker run --name scruples_instance -publish 5050:8000 scruples

## Run 

- download scruples data:                                               https://github.com/allenai/scruples#data
- move data to XAI_scruples/data/anecdotes and/or XAI_scruples/data/dilemmas
- configure params in ancedotes.py 
- run ancedotes.py
- html and pngs will be created in root dir


## Runtime LIME on sample anecdote
- single explanation with 10 pertubations:              14s CPU
- single explanation with 5000 pertubations: 6532s CPU, 348s GPU
- CPU i7, GPU GTX1080


## Other explanation techniques
- [Anchors](https://github.com/marcotcr/anchor):      runtime extremly high, ~3hours per anecdote on GPU                       


- [SHAP](https://github.com/slundberg/shap)   not compatible with embedding layers:         https://github.com/slundberg/shap/issues/595





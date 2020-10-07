source venv/bin/activate
rm log/scruples.log 
killall scruples
export SCRUPLES_NORMS_ACTIONS_BASELINE=roberta
export SCRUPLES_NORMS_ACTIONS_MODEL='/home/adlaube/XAI_scruples/models/dilemmas'
export SCRUPLES_NORMS_CORPUS_BASELINE=roberta
export SCRUPLES_NORMS_CORPUS_MODEL='/home/adlaube/XAI_scruples/models/anecdotes'
export SCRUPLES_NORMS_PREDICT_BATCH_SIZE=20
export SCRUPLES_NORMS_GPU_IDS='0'
setsid scruples demo norms > log/scruples.log &
sleep 30


rm scruples.log 
killall scruples
export SCRUPLES_NORMS_ACTIONS_BASELINE=roberta
export SCRUPLES_NORMS_ACTIONS_MODEL='/home/adlaube/XAI_scruples/models/dilemmas'
export SCRUPLES_NORMS_CORPUS_BASELINE=roberta
export SCRUPLES_NORMS_CORPUS_MODEL='/home/adlaube/XAI_scruples/models/anecdotes'
export SCRUPLES_NORMS_PREDICT_BATCH_SIZE=20
export SCRUPLES_NORMS_GPU_IDS=''
setsid scruples demo norms > scruples.log &
sleep 30


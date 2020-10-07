rm anec.log
killall anecdotes
setsid python anecdotes_single.py > anec.log &


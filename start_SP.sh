rm log/anec.log
killall anecdotes
setsid python anecdotes_SP.py > log/anec.log &

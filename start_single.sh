rm log/anec.log
killall anecdotes
setsid python anecdotes_single.py > log/anec.log &




N_CONTEXT=2048
N_PREDICT=512

chat() {
    ./bin/llama-cli -c ${N_CONTEXT} -n ${N_PREDICT} --color -cnv --repeat-penalty 1.15 --repeat-last-n 128 -m $1 
}

ask() {
    ./bin/llama-cli -c ${N_CONTEXT} -n ${N_PREDICT} --repeat-penalty 1.15 --repeat-last-n 128 -m $1 -p $2 
}

srv() {
    ./bin/llama-server -c ${N_CONTEXT} -n ${N_PREDICT} --repeat-penalty 1.15 --repeat-last-n 128 -m $1
}


alias i="ipython -i scripts/start.py"

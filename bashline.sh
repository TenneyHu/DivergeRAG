
#echo "Starting rag..."
#python ./src/baselines/rag.py --option "rag"

#echo "Starting mmr..."
#python ./src/baselines/rag.py --option "mmr"

#echo "Starting shuffle..."
#python ./src/baselines/rag.py --option "shuffle"

echo "Starting expand..."
python ./src/baselines/rag.py --option "expand"

echo "Starting all..."
python ./src/baselines/rag.py --option "all"

echo "All jobs finished."
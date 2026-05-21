#!/usr/bin/env bash
model="resnet"
batch_size=32
n_epochs=4000
patience=12800
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}
model="d4equivariant"
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}
model="nco"
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}

model="resnet"
batch_size=4096
n_epochs=51200
patience=12800
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}
model="d4equivariant"
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}
model="nco"
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}

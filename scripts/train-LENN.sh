#!/usr/bin/env bash
# LENN — lattice-equivariant neural network (Ortali & Gabbana et al. 2025)
model="lenn"
batch_size=32
n_epochs=4000
patience=12800
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}

model="lenn"
batch_size=4096
n_epochs=51200
patience=12800
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}

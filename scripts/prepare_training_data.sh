#!/usr/bin/env bash

mkdir -p ./data/vibe_db
export PYTHONPATH="./:$PYTHONPATH"

# 3DPW
python lib/data_utils/threedpw_utils.py --dir ./data/3dpw

# MPI-INF-3D-HP
python lib/data_utils/mpii3d_utils.py --dir ./data/mpi_inf_3dhp

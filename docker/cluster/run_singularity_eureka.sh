#!/usr/bin/env bash
# Called on compute node by submit_job_slurm_eureka.sh
# $1 = path to timestamped IsaacLabEureka dir on cluster shared fs
# $2 = container profile name (e.g. isaac-lab-base)
# $3+ = args forwarded to python script

echo "(run_singularity_eureka.sh): eureka=$1  profile=$2  args=${@:3}"

setup_directories() {
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        [ ! -d "$dir" ] && mkdir -p "$dir" && echo "Created: $dir"
    done
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $SCRIPT_DIR/.env.cluster.bsc
source $SCRIPT_DIR/../.env.base

module load singularity

echo "=== HOST GPU STATUS ==="
nvidia-smi || echo "nvidia-smi not available on host"
echo "======================="

setup_directories
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# Permanent logs dir
mkdir -p "$CLUSTER_EUREKA_DIR/logs"
mkdir -p "$CLUSTER_EUREKA_DIR/llm_requests"

# Copy full IsaacLabEureka tree to compute node local storage
cp -r $1 $TMPDIR/isaaclab_eureka

# Create bind-mount destinations inside the copied tree.
# These must exist in $TMPDIR/isaaclab_eureka because bind #1 overlays
# /workspace/isaaclab_eureka with this dir; subsequent nested binds
# (logs, llm_requests) resolve against it, not the sandbox rootfs.
mkdir -p $TMPDIR/isaaclab_eureka/logs
mkdir -p $TMPDIR/isaaclab_eureka/llm_requests

# Node-local writable /tmp for the container. --containall isolates /tmp and the
# --writable-tmpfs overlay is small, so tempfile.gettempdir() finds nothing usable
# (isaaclab's configure_logging and torch.save both need real temp space). Bind a
# large node-local dir over /tmp instead.
mkdir -p $TMPDIR/container_tmp

# Load secrets (gitignored, rsynced from local machine)
if [ -f "$TMPDIR/isaaclab_eureka/.env.secret" ]; then
    source $TMPDIR/isaaclab_eureka/.env.secret
fi

# Unpack container and create missing bind destinations
tar -xf $CLUSTER_SIF_PATH/$2.tar -C $TMPDIR
mkdir -p $TMPDIR/$2.sif/workspace/isaaclab_eureka
mkdir -p $TMPDIR/$2.sif/workspace/isaaclab_eureka/llm_requests

singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/isaaclab_eureka:/workspace/isaaclab_eureka:rw \
    -B $TMPDIR/isaaclab_eureka/IsaacLab/source:/workspace/isaaclab/source:rw \
    -B $CLUSTER_EUREKA_DIR/logs:/workspace/isaaclab_eureka/logs:rw \
    -B $CLUSTER_EUREKA_DIR/llm_requests:/workspace/isaaclab_eureka/llm_requests:rw \
    -B $TMPDIR/container_tmp:/tmp:rw \
    -B $CLUSTER_HF_CACHE_DIR:${DOCKER_USER_HOME}/.cache/huggingface:ro \
    --env OPENAI_API_KEY=$OPENAI_API_KEY \
    --env FILE_LLM_REQUEST_DIR=/workspace/isaaclab_eureka/llm_requests \
    --env FILE_LLM_POLL_INTERVAL=60 \
    --env PYTHONPATH=/workspace/isaaclab_eureka/source/isaaclab_eureka \
    --env TMPDIR=/tmp \
    --env HF_HOME=${DOCKER_USER_HOME}/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --pwd /workspace/isaaclab_eureka \
    --nv --writable-tmpfs --containall $TMPDIR/$2.sif \
    bash -c "
        export LD_LIBRARY_PATH=/usr/local/cuda-compat:\$LD_LIBRARY_PATH
        /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}
    "

rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity_eureka.sh): Done"

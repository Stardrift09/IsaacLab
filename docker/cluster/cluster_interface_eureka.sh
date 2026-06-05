#!/usr/bin/env bash

# Like cluster_interface.sh but rsyncs the full IsaacLabEureka tree.
# push: same as original (converts Docker image to .sif)
# job:  rsyncs IsaacLabEureka/ → cluster, submits via submit_job_slurm_eureka.sh

set -e
tabs 4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
EUREKA_ROOT="$SCRIPT_DIR/../../.."   # IsaacLabEureka/

#==
# Helper functions (copied from cluster_interface.sh)
#==

display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

version_gte() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

check_docker_version() {
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed!" >&2; exit 1
    fi
    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')
    if [ "$docker_version" = "24.0.7" ] && [ "$apptainer_version" = "1.2.5" ]; then
        echo "[INFO]: Docker ${docker_version} and Apptainer ${apptainer_version} are tested and compatible."
    elif version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO]: Docker ${docker_version} and Apptainer ${apptainer_version} are tested and compatible."
    else
        display_warning "Non-tested versions: Docker ${docker_version}, Apptainer ${apptainer_version}."
    fi
}

check_image_exists() {
    if ! docker image inspect "$1" &> /dev/null; then
        echo "[Error] Image '$1' does not exist!" >&2; exit 1
    fi
}

check_singularity_image_exists() {
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$1.tar ]"; then
        echo "[Error] Image '$1' not found on $CLUSTER_LOGIN!" >&2; exit 1
    fi
}

help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<profile>] [<job_args>...]"
    echo -e "\ncommands:"
    echo -e "  push [<profile>]              Push the docker image to the cluster."
    echo -e "  job  [<profile>] [<job_args>] Rsync IsaacLabEureka and submit a job.\n"
}

#==
# Main
#==

while getopts ":h" opt; do
    case ${opt} in
        h) help; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; help; exit 1 ;;
    esac
done
shift $((OPTIND -1))

[ $# -lt 1 ] && { echo "Error: command required." >&2; help; exit 1; }

command=$1; shift
profile="eureka"

source $SCRIPT_DIR/.env.cluster.bsc

case $command in
    push)
        [ $# -gt 1 ] && { echo "Error: too many args for push." >&2; exit 1; }
        [ $# -eq 1 ] && profile=$1
        if ! command -v apptainer &> /dev/null; then
            echo "[INFO] apptainer not found. Install from https://apptainer.org"; exit 1
        fi
        check_docker_version
        echo "[INFO] Building isaac-lab-$profile image from Dockerfile.eureka..."
        docker build -f "$SCRIPT_DIR/../Dockerfile.eureka" -t "isaac-lab-$profile:latest" "$EUREKA_ROOT"
        # BSC overlay: layer the cu126 torch downgrade on top, retag as the shipped image.
        echo "[INFO] Applying BSC overlay from Dockerfile.eureka.bsc..."
        docker build -f "$SCRIPT_DIR/../Dockerfile.eureka.bsc" -t "isaac-lab-$profile:latest" "$EUREKA_ROOT"
        mkdir -p $SCRIPT_DIR/exports
        rm -rf $SCRIPT_DIR/exports/isaac-lab-$profile*
        cd $SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --sandbox --fakeroot isaac-lab-$profile.sif docker-daemon://isaac-lab-$profile:latest
        tar -cvf $SCRIPT_DIR/exports/isaac-lab-$profile.tar isaac-lab-$profile.sif
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
        scp $SCRIPT_DIR/exports/isaac-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar
        ;;
    job)
        if [ $# -ge 1 ]; then
            passed_profile=$1
            if [ -f "$SCRIPT_DIR/../.env.$passed_profile" ]; then
                profile=$passed_profile; shift
            fi
        fi
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$profile" ]   && echo -e "\tProfile: $profile"
        [ -n "$job_args" ]  && echo -e "\tJob args: $job_args"

        current_datetime=$(date +"%Y%m%d_%H%M%S")
        CLUSTER_EUREKA_TS_DIR="${CLUSTER_EUREKA_DIR}/runs/${current_datetime}"

        check_singularity_image_exists isaac-lab-$profile

        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_EUREKA_TS_DIR"
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_EUREKA_DIR/logs"

        echo "[INFO] Syncing IsaacLabEureka..."
        # trailing slash = sync contents (not the dir itself) into CLUSTER_EUREKA_TS_DIR
        rsync -rh --progress \
            --exclude=".git" \
            --include=".env.secret" \
            --exclude="IsaacLab/docker/cluster/exports/" \
            --exclude="Master_thesis/" \
            --exclude="logs_history/" \
            --exclude=".cache/" \
            --exclude="vlm_comparision/" \
            --filter="+ /libero/***" \
            --filter="+ /offline_assets/***" \
            --filter="dir-merge,- .gitignore" \
            "$EUREKA_ROOT/" $CLUSTER_LOGIN:$CLUSTER_EUREKA_TS_DIR

        echo "[INFO] Submitting job..."
        ssh $CLUSTER_LOGIN \
            "cd $CLUSTER_EUREKA_TS_DIR && \
             bash $CLUSTER_EUREKA_TS_DIR/IsaacLab/docker/cluster/submit_job_slurm_eureka.sh \
             \"$CLUSTER_EUREKA_TS_DIR\" \"isaac-lab-$profile\" $job_args"
        ;;
    *)
        echo "Error: invalid command '$command'" >&2; help; exit 1 ;;
esac

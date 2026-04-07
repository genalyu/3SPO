#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLURM_SCRIPT="${1:-$SCRIPT_DIR/run_alfworld.slurm}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
BAD_MIG_EXIT_CODE="${BAD_MIG_EXIT_CODE:-42}"
POLL_SECONDS="${POLL_SECONDS:-20}"

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: Slurm script not found: $SLURM_SCRIPT"
    exit 1
fi

echo "INFO: Submitting $SLURM_SCRIPT with up to $MAX_ATTEMPTS attempts"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    submit_output=$(sbatch "$SLURM_SCRIPT")
    job_id=$(echo "$submit_output" | awk '{print $4}')

    if [ -z "$job_id" ]; then
        echo "ERROR: Failed to parse job id from sbatch output: $submit_output"
        exit 1
    fi

    echo "INFO: Attempt $attempt/$MAX_ATTEMPTS submitted job $job_id"

    while true; do
        sacct_output=$(sacct -j "$job_id" --parsable2 --noheader --format=JobIDRaw,State,ExitCode | awk -F'|' -v id="$job_id" '$1 == id {print $0; exit}')

        if [ -z "$sacct_output" ]; then
            sleep "$POLL_SECONDS"
            continue
        fi

        state=$(echo "$sacct_output" | awk -F'|' '{print $2}')
        exit_code_pair=$(echo "$sacct_output" | awk -F'|' '{print $3}')
        exit_code=${exit_code_pair%%:*}

        case "$state" in
            PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING|SUSPENDED)
                sleep "$POLL_SECONDS"
                ;;
            COMPLETED)
                echo "INFO: Job $job_id completed successfully on attempt $attempt"
                exit 0
                ;;
            FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
                if [ "$exit_code" = "$BAD_MIG_EXIT_CODE" ]; then
                    echo "INFO: Job $job_id exited with bad-MIG code $BAD_MIG_EXIT_CODE; retrying"
                    break
                fi
                echo "ERROR: Job $job_id ended in state $state with exit code $exit_code_pair"
                exit 1
                ;;
            *)
                echo "INFO: Job $job_id entered state $state; continuing to poll"
                sleep "$POLL_SECONDS"
                ;;
        esac
    done

    attempt=$((attempt + 1))
done

echo "ERROR: Exhausted $MAX_ATTEMPTS attempts without getting a valid MIG placement"
exit 1

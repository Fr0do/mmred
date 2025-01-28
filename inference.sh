#!/bin/bash
# Start the script in its own process group
set -m

(
    python scripts/openai_server_inference.py --port 8000 &
    # python scripts/openai_server_inference.py --port 8001 &
    # python scripts/openai_server_inference.py --port 8002 &
    # python scripts/openai_server_inference.py --port 8003 &
    # python scripts/openai_server_inference.py --port 8004 &
    # python scripts/openai_server_inference.py --port 8005 &
    # python scripts/openai_server_inference.py --port 8006 &
    # python scripts/openai_server_inference.py --port 8007 &
    # python scripts/openai_server_inference.py --port 8008 &
    # python scripts/openai_server_inference.py --port 8009 &
    wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT
wait

echo "Inference complete."
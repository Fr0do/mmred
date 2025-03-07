#!/bin/bash
# Start the script in its own process group
set -m
BASE_URL="--base_url https://us-central1-aiplatform.googleapis.com/v1/projects/celtic-bazaar-451314-u6/locations/us-central1/endpoints/openapi"
API_KEY=$(cat gemini_creds.txt)
ARGS="--data_path /workspace-SR004.nfs2/data/long_vqa_synth/ --exp_name main_1mv --semaphore_limit 32 --batch_size 32"
TEXT1_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main_1mv/all_text_serialized_questions.json --exp_name main_1mv  --semaphore_limit 32 --batch_size 128"
TEXT_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main/all_text_serialized_questions.json  --exp_name main --semaphore_limit 32 --batch_size 128"
TEXT_NLG_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main_1mv/text_description_serialized.json  --exp_name nlg_1mv --semaphore_limit 32 --batch_size 128"
(
    python scripts/openai_server_inference.py $BASE_URL --api_key $API_KEY $TEXT1_ARGS --model_name google/gemini-2.0-flash-001 --output_csv data/main_1mv/qa_pairs_answers_gemini-2.0-flash-001_text.csv &
    wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT
wait

echo "Inference complete."
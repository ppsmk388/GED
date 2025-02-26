output_folder="./output/llama-2-7b-chat"
model_path="./DAG/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
num_gpus=4



# gen ultrafeedback
python vllm_generate.py \
--input_json_path="./reference/ultrafeedback_sample_5k_reference.json" \
--output_json_path="${output_folder}/reference_output.json" \
--model="${model_path}" \
--gpu=${num_gpus}

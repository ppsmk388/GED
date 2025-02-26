output_folder="./output/llama-2-7b-chat"
model_path="./xxx/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
num_gpus=4



# gen alpaca
python vllm_generate.py \
--input_json_path="./reference/alpaca_reference.json" \
--output_json_path="${output_folder}/alpaca_output.json" \
--model="${model_path}" \
--gpu=${num_gpus}

# gen human 
python vllm_generate.py \
--input_json_path="./reference/human_eval_reference.json" \
--output_json_path="${output_folder}/human_eval_output.json" \
--model="${model_path}" \
--gpu=${num_gpus}

# gen math
python vllm_generate.py \
--input_json_path="./reference/math_reference.json" \
--output_json_path="${output_folder}/math_output.json" \
--model="${model_path}" \
--gpu=${num_gpus}

# gen gsm8k
python vllm_generate.py \
--input_json_path="./reference/gsm8k_reference.json" \
--output_json_path="${output_folder}/gsm8k_output.json" \
--model="${model_path}" \
--gpu=${num_gpus}

# gen gaia
python vllm_generate.py \
--input_json_path="./reference/gaia_reference.json" \
--output_json_path="${output_folder}/gaia_output.json" \
--model="${model_path}" \
--gpu=${num_gpus}


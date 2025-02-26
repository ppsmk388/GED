import os
import subprocess

def find_json_files(input_folder):
    return [f for f in os.listdir(input_folder) if f.endswith('output.json')]

def get_prefix(file_name):
    return file_name.split('_')[0]

def run_evaluation_script(prefix, json_path, eval_folder):
    script_name = f"eval_{prefix}.py"
    script_path = os.path.join(eval_folder, script_name)
    if os.path.exists(script_path):
        subprocess.run(["python", script_path, "--j", json_path])
    else:
        print(f"Script {script_name} not found in evaluation folder.")

def main(input_folder, eval_folder):
    json_files = find_json_files(input_folder)
    for json_file in json_files:
        prefix = get_prefix(json_file)
        json_path = os.path.join(input_folder, json_file)
        run_evaluation_script(prefix, json_path, eval_folder)

if __name__ == "__main__":
    input_folder = "./eval_result/evaluation_folder/mistral-7b-instruct-v0.1"
    eval_folder = "./eval_folder_0811/0927_all_eval_result/evaluation_script"
    main(input_folder, eval_folder)
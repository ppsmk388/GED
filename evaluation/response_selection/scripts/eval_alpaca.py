import json
import argparse
import os
import re
from typing import Optional
from openai import OpenAI
import time
import multiprocessing
import tqdm
import random


# Get the current script folder
current_script_folder = os.path.dirname(os.path.realpath(__file__))



# Initialize the OpenAI client with the custom base URL and API key
client = OpenAI(api_key='xxx',)

def replace_user_prompt(instruction, output_1, output_2, switch):
    template = """
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{
    "instruction": "{instruction}",
}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{
    {
        "model_identifier": "model1",
        "output": "{output_1}"
    },
    {
        "model_identifier": "model2",
        "output": "{output_2}"
    }
}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): model1 or model2.

## Best Model Identifier
"""
    template = template.replace("{instruction}", instruction)
    if switch:
        template = template.replace("{output_1}", output_2)
        template = template.replace("{output_2}", output_1)
    else:
        template = template.replace("{output_1}", output_1)
        template = template.replace("{output_2}", output_2)
    return template


def extract_model_identifier(output, switch):
    import re
    
    change_back = {
        "model1": "model2",
        "model2": "model1"
    }

    # Convert to lowercase and slice the last 50 characters of the output
    output_segment = output[-50:].lower()

    # This regex will match 'model1' or 'model2'
    pattern = r'\b(model1|model2)\b'
    matches = re.findall(pattern, output_segment)

    # Check which model appears last in the segment (which means first from the back to front)
    if matches:
        last_match = matches[-1]  # Get the last match found, which is the first from the end
        if switch:
            last_match = change_back[last_match]
        
        return True, last_match
    else:
        # No valid identifier was found
        return False, None

    
def get_eval_result(args):
    item, formatted_prompt, switch = args
    result = False
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":"You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."},
                    {"role": "user", "content": formatted_prompt}
                ]
            )

            
            output_text = response.choices[0].message.content


            print(f'Output: {output_text} \n\n\n')
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)
            continue
        


        success, result = extract_model_identifier(output_text, switch)
        # covert to bool
        result = result == "model1"
        if success:
            print(f'Got a valid result: {result} at iteration {i}')
            break
    else:
        # print(response.json())
        print('Failed to get a valid result after 5 iterations')
        result = None

    return item, output_text, result

def get_reference_output(instruction, ref_json):
    for item in ref_json:
        if instruction == item["instruction"]:
            return item["reference_output"]
    raise ValueError(f"Cannot find reference output for instruction: {instruction}")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--j", type=str, default="/home/xxxx/_Code/Research_repo/tony_eval_folder_0811/0904_new_evaluation/refined_dataset/alpaca-llama3-70b-output.json", help="The file to read from")
    args = argparse.parse_args()
    file_name = os.path.basename(args.j)
    file_folder = os.path.dirname(args.j)
    output_file_name = file_name.replace("output", "result")
    output_file_path = os.path.join(file_folder, output_file_name)
    

    with open(f"../reference/alpaca_reference.json", 'r') as f:
        ref_json = json.loads(f.read())
        
        

    # Construct the input file path
    # Determine the correct input file path.
    input_file_path = args.j
    with open(input_file_path, 'r') as f:
        input_json = json.loads(f.read())

    
    final_result_list = []
    process_args = []
    for input_item in input_json:
        current_instruction = input_item["instruction"]
        current_output = input_item["output"]
        current_reference_output = get_reference_output(current_instruction, ref_json)
        switch = random.choice([True, False])
        formatted_prompt = replace_user_prompt(current_instruction, current_output, current_reference_output, switch)
        final_result_list.append((current_instruction, formatted_prompt, switch))
        
        process_args.append((input_item, formatted_prompt, switch))

        
    
    num_procs = 20
    
    with multiprocessing.Pool(num_procs) as p:
        results = list(
            tqdm.tqdm(
                p.imap(get_eval_result, process_args),
                total=len(process_args),
                desc="Processing summarizations"
            )
        )
        
    final_result_list = []
    for item, output_text, result in results:
        json_item_to_modify = item
        json_item_to_modify["result"] = result
        json_item_to_modify["result_output_text"] = output_text
        final_result_list.append(json_item_to_modify)
        
        
    try:
        with open(output_file_path, 'w') as file:
            json.dump(final_result_list, file, indent=4)
        print(f"Data written successfully")
    except Exception as e:
        print(f"Failed to write output: {e}")
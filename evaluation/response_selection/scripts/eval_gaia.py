import json
import argparse
import os
import re
from typing import Optional
from openai import OpenAI
import time
import multiprocessing
import tqdm


# Get the current script folder
current_script_folder = os.path.dirname(os.path.realpath(__file__))


def scorer(response):
    response = response.lower()
    if "the answer is correct" in response or "the answer is approximated but should be correct" in response:
        return True
    else:
        return False

# Initialize the OpenAI client with the custom base URL and API key
client = OpenAI(api_key='xxx',)

check_sys_msg = """You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
You are given:
    1. A problem.
    2. A reply with the answer to the problem.
    3. A ground truth answer.
Please do the following:
1. Extract the answer in the reply: "FINAL ANSWER:  <answer extracted>".
2. Check whether the answer in the reply matches the ground truth answer. When comparison is not obvious (for example, 3*\\sqrt(6) and 7.348), you may write code to check the answer and wait for the user to execute the code.
3. After everything is done, please choose a reply from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." """
    
def get_eval_result(args):
    item, messages = args
    result = False
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            
            output_text = response.choices[0].message.content

            print(f'Output: {output_text} \n\n\n')
            
            if scorer(output_text):
                # win_counter += 1
                result = True
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f'Input message is: {messages}')
            if e.status_code == 400:
                output_text = "The reply doesn't contain an answer.\n" 
                error_msg = """Error code: 400 - {'error': {'message': 'Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt.', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'invalid_prompt'}}'"""
                output_text += error_msg
                result = False
                break
            else:
                print(f"Retrying in 3 seconds")
                output_text = str(e)
                result = False
            # time.sleep(3)
            continue
    else:
        # print(response.json())
        print('Failed to get a valid result after 5 iterations')
        result = False

    return item, output_text, result



if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--j", type=str, default="", help="The file to read from")
    args = argparse.parse_args()
        
    
    file_name = os.path.basename(args.j)
    file_folder = os.path.dirname(args.j)
    # output_file_name = file_name.replace("output", "result")
    output_file_path = os.path.join(file_folder, file_name)
    

    # Construct the input file path
    # Determine the correct input file path.
    input_file_path = args.j
    with open(input_file_path, 'r') as f:
        input_json = json.loads(f.read())


    final_result_list = []
    process_args = []
    for input_item in input_json:
        user_prompt  = "Problem: " + input_item["instruction"] + f"\n\nReply: {input_item.get('output')}\n\nGround truth answer: " + input_item["ground_truth"]

        messages = [
            {"role": "system", "content": check_sys_msg},
            {"role": "user", "content": user_prompt}
        ]
        
        # Randomly decide whether to switch outputs
        process_args.append((input_item, messages))

   
        
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
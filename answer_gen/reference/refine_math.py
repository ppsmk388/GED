import json



input_json = json.load(open("math_reference.json", "r"))


for item in input_json:
    ori_prompt = item["prompt"]
    
    
    new_prompt = f'You are an Expert Mathematician. Your task is to provide a response that is thorough and accurate. Ensure that your answer is precise and complete, covering all important aspects of the question: {ori_prompt}'
    
    item["prompt"] = new_prompt
    
json.dump(input_json, open("math_reference.json", "w"), indent=4)
    
    
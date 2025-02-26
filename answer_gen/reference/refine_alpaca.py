import json



input_json = json.load(open("alpaca_reference.json", "r"))


for item in input_json:
    ori_prompt = item["instruction"]
    
    
    new_prompt = f'{ori_prompt}'
    
    item["prompt"] = new_prompt
    
    # delte output key
    del item["output"]
    
json.dump(input_json, open("alpaca_reference.json", "w"), indent=4)
    
    
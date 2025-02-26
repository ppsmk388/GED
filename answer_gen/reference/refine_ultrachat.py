import json



input_json = json.load(open("ultrafeedback_sample_5k_reference.json", "r"))


for item in input_json:
    ori_prompt = item["instruction"]
    
    
    new_prompt = "You are a helpful assistant.\n" + ori_prompt
    
    item["prompt"] = new_prompt
    
json.dump(input_json, open("ultrafeedback_sample_5k_reference.json", "w"), indent=4)
    
    
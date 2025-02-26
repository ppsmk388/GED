import os
import json
import pandas as pd

def find_json_files(input_folder):
    return [f for f in os.listdir(input_folder) if f.endswith('result.json')]

def get_prefix(file_name):
    return file_name.split('_')[0]

def run_evaluation_script(prefix, json_path):
    total_count, correct_count = 0, 0
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check each item in the JSON file
    for item in data:
        total_count += 1
        if item.get("result") == 1 or item.get("result") is True:
            correct_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy * 100

def main(input_folder):
    # Initialize a dictionary to store results for each category
    results = {
        "alpaca": None,
        "gsm8k": None,
        "math": None,
        "gaia": None,
        "human": None
    }

    json_files = find_json_files(input_folder)

    # Process each JSON file and update the results dictionary
    for json_file in json_files:
        prefix = get_prefix(json_file)
        json_path = os.path.join(input_folder, json_file)
        accuracy = run_evaluation_script(prefix, json_path)

        # Map prefix to the corresponding column in the results dictionary
        if prefix.lower() in results:
            results[prefix] = accuracy

    # Convert the results dictionary to a DataFrame with one row
    df = pd.DataFrame([results])

    # Save the DataFrame to an Excel file
    excel_path = os.path.join(input_folder, "acc.xlsx")
    df.to_excel(excel_path, index=False)

if __name__ == "__main__":
    
    input_folder = ""
    main(input_folder)

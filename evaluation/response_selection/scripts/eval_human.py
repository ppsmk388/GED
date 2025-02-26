import argparse
import json
import os
import tempfile
import subprocess  # Import subprocess for better control over external commands
import re
# Custom exception for handling timeouts
class TimeoutException(Exception):
    pass

# Get the current script folder
current_script_folder = os.path.dirname(os.path.realpath(__file__))

def extract_python_code(text: str):
    """
    Extract the Python code block enclosed by triple backticks, accounting for different capitalizations
    and extra newlines or spaces.
    
    Args:
        text (str): The text containing the code block.
        
    Returns:
        Optional[str]: Extracted Python code block, or None if no code block is found.
    """
    # Regex to find the python code block, case-insensitive match for 'python'
    pattern = r"from my_tests\s*(.*?)\s*```"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # Extract code block
        return "from my_tests " + match.group(1).strip()
    
    return None


test_part2 = """
def run_tests(candidate):
    try:
        check(candidate)
        # We can search for this string in the output
        print("PASSED")
    except:
        print("FAILED")
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--j", type=str, default="", help="The file to read from")
    
    args = parser.parse_args()
    
    input_json = json.loads(open(args.j, 'r').read())
    

    
    file_name = os.path.basename(args.j)
    file_folder = os.path.dirname(args.j)
    output_file_name = file_name.replace("output", "result")
    output_file_path = os.path.join(file_folder, output_file_name)
        

    for item in input_json:
        my_tests_code = item["test"]
        test_code = item["output"]
        
        test_part1 = item["test"]
        test_part1 = test_part1.replace("\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\n", "")
        my_tests_code = test_part1 + test_part2
        
        test_code = extract_python_code(item["output"])



        if test_code is None:
            item["result"] = result
            print(f'Cannot find valid test code')
            
            continue
        
        # make temp folder for testing
        temp_folder = tempfile.mkdtemp()
        my_tests_path = os.path.join(temp_folder, "my_tests.py")
        test_path = os.path.join(temp_folder, "test.py")
        
        # write my_tests.py and test.py
        with open(my_tests_path, 'w') as f:
            f.write(my_tests_code)
        with open(test_path, 'w') as f:
            f.write(test_code)
        
        try:  
            # Run the test with timeout and capture output
            output = subprocess.check_output(
                ["python3", test_path], 
                cwd=temp_folder, 
                timeout=10,  # Timeout in seconds
                stderr=subprocess.STDOUT
            ).decode('utf-8')
            
            if "PASSED" in output:
                result = 1
            else:
                result = 0
        except subprocess.TimeoutExpired:
            print(f'Timeout occurred for test ID')
            result = 0
        except Exception as e:
            print(f'Exception for test ID')
            result = 0
        finally:
            item["result"] = result
            os.system(f"rm -rf {temp_folder}")  # Clean up the temp folder
    
    with open(output_file_path, 'w') as f:
        json.dump(input_json, f, indent=4)

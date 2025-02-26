import sys
import copy
import argparse
import os
import json

# Add paths for custom utilities
sys.path.append('../rank_utility')
sys.path.append('../')

from collections import defaultdict
from mallows import *
from scipy.stats import spearmanr, kendalltau
from tool import *  # Assumes all required functions are defined here

def show_combine_rankings(rankings, method_type='weight_score'):

    if method_type == 'weight_score':
        # Example return value might look like: [[23], [15], [2], ... [29]]
        return weighted_scoring_method(rankings)[0]
    elif method_type in ['kemeny', 'weighted_kemeny', 'pairwise_majority', 'weighted_pairwise_majority']:
        return universalizing_weak_supervision(method_type=method_type, rank_list=rankings)



def main():
    """
    Main function to parse arguments, read ranking data, combine rankings, and
    compute correlation metrics (Spearman and Kendall).
    """
    parser = argparse.ArgumentParser(description="Example script with arguments")
    args = parser.parse_args()
    print(args)

    method = 'SelfControl'
    task_name = 'AlpacEval'

    # Define the folder path where rank files might reside
    folder_path = 'Denose rank path'

    # Initialize a dictionary to store data for each evaluation model
    answer_model_data_collect = {}

    # This index list is used later to compare combined rankings with an existing "gold" or reference ranking
    model_rank_alpac_index = list(range(30))

    # Predefine the evaluation models and collect ranking data
    for eval_model in ['llama3_70b', 'qwen1.5_72b', 'mistral87b', 'all']:
        # Initialize data structure
        rank_list_data = {
            'llama3_70b': {"baseline": {}},
            'qwen1.5_72b': {"baseline": {}},
            'mistral87b': {"baseline": {}},
            'all': {"baseline": {}},
        }

        for answer_model in ['GPT4']:
            # Go through each rank type and read the relevant JSON
            for rank_type in [
                'weight_score',
                'kemeny',
                'weighted_kemeny',
                'pairwise_majority',
                'weighted_pairwise_majority'
            ]:
                file_path = os.path.join(
                    folder_path,
                    f'{task_name}/{eval_model}/{answer_model}/{rank_type}/{method}/{rank_type}_rank_list.json'
                )

                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        rank_list_result_save = json.load(file)
                else:
                    print(f"File {file_path} does not exist.")
                    rank_list_result_save = {}

                # Safely fill in the nested dictionaries
                if answer_model not in rank_list_data[eval_model]['baseline']:
                    rank_list_data[eval_model]['baseline'][answer_model] = {}

                rank_list_data[eval_model]['baseline'][answer_model][rank_type] = \
                    rank_list_result_save.get(eval_model, {}).get('baseline', {})

                if answer_model not in rank_list_data[eval_model]:
                    rank_list_data[eval_model][answer_model] = {}

                rank_list_data[eval_model][answer_model][rank_type] = \
                    rank_list_result_save.get(eval_model, {}).get(rank_type, {})

        answer_model_data_collect[eval_model] = rank_list_data[eval_model]

    # Path for storing combined ranking results
    combine_json_path = (
        f'{"your result save path"}/rank_ensemble/universalizing-weak-supervision'
        f'{"your result save path"}//synthetic/ModelRanking/baseline_combine_rank_save.json'
    )

    # Ensure the folder for storing combined results exists
    combine_folder = os.path.dirname(combine_json_path)
    if not os.path.exists(combine_folder):
        print(f"Folder {combine_folder} does not exist, creating it...")
        os.makedirs(combine_folder)
        print(f"Folder {combine_folder} successfully created.")
    else:
        print(f"Folder {combine_folder} already exists.")

    # If the combined results file exists, load it
    if os.path.exists(combine_json_path):
        print(f"File {combine_json_path} already exists, reading it...")
        with open(combine_json_path, 'r') as json_file:
            combine_rank_save = json.load(json_file)
        compute_flag = "Done"
    else:
        # If it does not exist, prepare a new dictionary structure for later storage
        combine_rank_save = {
            'llama3_70b': {"baseline": {}, "origin": {}},
            'qwen1.5_72b': {"baseline": {}, "origin": {}},
            'mistral87b': {"baseline": {}, "origin": {}},
            'all': {"baseline": {}, "origin": {}},
            'graph_ensemble': {"baseline": {}, "origin": {}},
        }
        compute_flag = "NotDone"

    # Process graph_ensemble for 'all' model
    for e_model in ['all']:
        for a_model in ['GPT4']:
            for r_ens_m in [
                'weight_score', 'kemeny', 'weighted_kemeny',
                'pairwise_majority', 'weighted_pairwise_majority'
            ]:
                separator = "=" * 106
                print(separator)
                print(f"Rank1_type: {r_ens_m}")
                print(separator)

                # If compute_flag is not "Done", we need to compute and store the data
                if compute_flag != "Done":
                    all_ans_list = answer_model_data_collect[e_model]['baseline'][a_model]['weight_score']['graph_ensemble']

                    # In case each element is a nested list of lists, flatten them
                    if isinstance(all_ans_list[0][0], list):
                        flattened = []
                        for single_rank in all_ans_list:
                            tmp = []
                            for sub_list in single_rank:
                                tmp.extend(sub_list)
                            flattened.append(tmp)
                        all_ans_list = flattened

                    # Combine the rankings using the current method
                    combine_rank = show_combine_rankings(rankings=all_ans_list, method_type=r_ens_m)
                    # Store the combined results
                    combine_rank_save['graph_ensemble']["baseline"][r_ens_m] = combine_rank
                else:
                    # If already computed, just load the saved result
                    combine_rank = combine_rank_save['graph_ensemble']["baseline"][r_ens_m]

                # Calculate and display correlation metrics
                spearman_corr, kendall_corr = calculate_similarities(
                    rank1=model_rank_alpac_index,
                    rank2=combine_rank
                )
                print(f"{r_ens_m}:  (Spearman: {spearman_corr * 100:.2f}%, Kendall: {kendall_corr * 100:.2f}%)")

    # Process other models
    for e_model in ['all', 'llama3_70b', 'qwen1.5_72b', 'mistral87b']:
        print(f"\nProcessing model: {e_model}")
        for a_model in ['GPT4']:
            for r_ens_m in [
                'weight_score', 'kemeny', 'weighted_kemeny',
                'pairwise_majority', 'weighted_pairwise_majority'
            ]:
                separator = "=" * 106
                print(separator)
                print(f"Rank1_type: {r_ens_m}")
                print(separator)

                if compute_flag != "Done":
                    all_ans_list = answer_model_data_collect[e_model]['baseline'][a_model][r_ens_m]['rank_ensemble']
                    # Combine the rankings
                    combine_rank = show_combine_rankings(rankings=all_ans_list, method_type=r_ens_m)
                    combine_rank_save[e_model]["baseline"][r_ens_m] = combine_rank
                else:
                    combine_rank = combine_rank_save[e_model]["baseline"][r_ens_m]

                # Calculate and display correlation metrics
                spearman_corr, kendall_corr = calculate_similarities(
                    rank1=model_rank_alpac_index,
                    rank2=combine_rank
                )
                print(f"{r_ens_m}:  (Spearman: {spearman_corr * 100:.2f}%, Kendall: {kendall_corr * 100:.2f}%)")

    # If the JSON file did not exist, create and save the combined rank data
    if not os.path.exists(combine_json_path):
        with open(combine_json_path, 'w') as json_file:
            json.dump(combine_rank_save, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

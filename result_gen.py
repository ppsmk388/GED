import argparse
import os
import os
import json


parser = argparse.ArgumentParser(description="Example script with arguments")
parser.add_argument("--answer_model", type=str, default='qwen1.5-14b', help="eval_model, ['llama3-70b', 'qwen2-72b', 'mixtral-8x7b']")
parser.add_argument("--task_name", type=str, default='human_eval', help="task_name, ['gsm8k', 'human_eval', 'alpaca', 'math', 'truthful_qa', 'ultra']")
parser.add_argument("--rank_type", type=str, default='weight_score')
args = parser.parse_args()
print(args)
task_name = args.task_name
rank_type = args.rank_type
answer_model = args.answer_model
folder_path = 'xxx'
save_path = 'xxx'
answer_model_data_collect={}
eval_model_list = ['xxx']
eval_model_list = ['xxx']


for eval_model in eval_model_list+ [ 'all']:
    rank_list_data = {}
    for mm in eval_model_list+ [ 'all']:
        rank_list_data[mm] = {
            "baseline":{
            },}
        file_path = os.path.join(folder_path, f'{task_name}/{eval_model}/{answer_model}/{rank_type}_rank_list.json')
        with open(file_path, 'r', encoding='utf-8') as file:
            rank_list_result_save = json.load(file)
        if answer_model not in rank_list_data[eval_model]['baseline']:
            rank_list_data[eval_model]['baseline'][answer_model] = {}
        if rank_type not in rank_list_data[eval_model]['baseline'][answer_model]:
            rank_list_data[eval_model]['baseline'][answer_model][rank_type] = {}
            rank_list_data[eval_model]['baseline'][answer_model][rank_type] = rank_list_result_save[eval_model]['baseline']
        if answer_model not in rank_list_data[eval_model]:
            rank_list_data[eval_model][answer_model] = {}

        if rank_type not in rank_list_data[eval_model][answer_model]:
            rank_list_data[eval_model][answer_model][rank_type] = rank_list_result_save[eval_model][rank_type]
    answer_model_data_collect[eval_model] = rank_list_data[eval_model]


for e_model in eval_model_list+ [ 'all']:
    a_model = answer_model
    data_list = []
    if rank_type == 'graph_ensemble':
        all_ans_list = answer_model_data_collect[e_model][a_model]['weight_score']['graph_ensemble']
    else:
        all_ans_list = answer_model_data_collect[e_model][a_model][rank_type]['rank_ensemble']

    for question_ans_list in all_ans_list:
        data_list.append(question_ans_list)
    r_save_path = f'{save_path}'
    f_save_path = r_save_path + f'/{task_name}/{e_model}/{a_model}/{rank_type}/data.json'
    os.makedirs(os.path.dirname(f_save_path), exist_ok=True)
    with open(f_save_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)



for e_model in eval_model_list+ [ 'all']:
    a_model = answer_model
    data_list = []
    if rank_type == 'graph_ensemble':
        all_ans_list = answer_model_data_collect[e_model]['baseline'][a_model]['weight_score']['graph_ensemble']
    else:
        all_ans_list = answer_model_data_collect[e_model]['baseline'][a_model][rank_type]['rank_ensemble']
    for question_ans_list in all_ans_list:
        data_list.append(question_ans_list)
    r_save_path_baseline = r_save_path + f'/baseline'
    f_save_path = r_save_path_baseline + f'/{task_name}/{e_model}/{a_model}/{rank_type}/data.json'
    os.makedirs(os.path.dirname(f_save_path), exist_ok=True)
    with open(f_save_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)
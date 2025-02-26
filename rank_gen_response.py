import copy
import sys
sys.path.append('./rank_utility')
sys.path.append('./model-rank')
import argparse
import os
from mallows import *
from tool import *



def list_update(method, ensemble_type, eval_model=None, all_model_list=None, rank_type=None):
    """
    Refactored version of 'list_update' for improved readability.
    Retains the original logic and relies on globally available
    functions / variables (e.g., 'read_data', 'DGtoDAG', 'super_method', etc.).
    """

    data_save_path = f'Your Raw Preference Graph Save Path'

    # ----------------------------------------------------------------
    # 1. HELPER: Load model data from disk
    # ----------------------------------------------------------------
    def _load_data_for_model(model_name):
        """
        Loads graph_list and raw_graph_list from disk for the specified model.
        """
        g_list = read_data(file_path=f'{data_save_path}/{model_name}/{method}/graph_list.pkl')
        r_list = read_data(file_path=f'{data_save_path}/{model_name}/{method}/raw_graph_list.pkl')
        return g_list, r_list

    # ----------------------------------------------------------------
    # 2. HELPER: For 'rank_ensemble', combine multiple models' ranks
    # ----------------------------------------------------------------
    def _combine_rank_ensemble(all_e_model):
        """
        Combines the rank lists across multiple models using 'combine_rankings'.
        Returns a single merged rank_list.
        """
        # Compute rank lists per model first
        tmp_rank_dict_save = {}
        for e_model in all_model_list:
            tmp_rank_dict_save[e_model] = rank_ensemble_graph_to_rank(
                graph_list=all_e_model[e_model]['graph_list'],
                w_type=w_type
            )

        # Combine them across models
        merged_rank_list = []
        first_model = all_model_list[0]
        for i in range(len(tmp_rank_dict_save[first_model])):
            per_model_ranks = [tmp_rank_dict_save[m][i] for m in all_model_list]
            merged_rank_list.append(combine_rankings(rankings=per_model_ranks, type=rank_type))
        return merged_rank_list

    # ----------------------------------------------------------------
    # 3. HELPER: For 'rank_ensemble', process the raw_graph_list to DAG
    # ----------------------------------------------------------------
    def _process_raw_graph_list_rank_ensemble(raw_graph_list_):
        """
        Takes a raw_graph_list, potentially applies 'SelfControl' transformations,
        and then applies 'rank_ensemble_raw_graph_to_rank'.
        """
        if super_method != 'SelfControl':
            return rank_ensemble_raw_graph_to_rank(raw_graph_list=raw_graph_list_)
        else:
            # SelfControl transformation
            trans_graphs = []
            for tmp_graph in raw_graph_list_:
                dag_graph, _ = DGtoDAG(method='ContraSolver').get_dag(tmp_graph)
                trans_graphs.append(dag_graph)
            return rank_ensemble_raw_graph_to_rank(raw_graph_list=trans_graphs)

    # ----------------------------------------------------------------
    # 4. HELPER: For 'rank_ensemble', combine raw DAG results across models
    # ----------------------------------------------------------------
    def _combine_raw_dag_rank_ensemble(all_e_model):
        """
        Combines the raw DAG rank lists across multiple models,
        returning a final merged list.
        """
        tmp_rank_dict_save = {}

        if super_method != 'SelfControl':
            for e_model in all_model_list:
                tmp_rank_dict_save[e_model] = rank_ensemble_raw_graph_to_rank(
                    raw_graph_list=all_e_model[e_model]['raw_graph_list']
                )
        else:
            # SelfControl transformation for all models
            sc_dict = copy.deepcopy(all_e_model)
            for e_model in all_model_list:
                trans_graphs = []
                for tmp_graph in sc_dict[e_model]['raw_graph_list']:
                    dag_graph, _ = DGtoDAG(method='ContraSolver').get_dag(tmp_graph)
                    trans_graphs.append(dag_graph)
                sc_dict[e_model]['raw_graph_list'] = trans_graphs

            for e_model in all_model_list:
                tmp_rank_dict_save[e_model] = rank_ensemble_raw_graph_to_rank(
                    raw_graph_list=sc_dict[e_model]['raw_graph_list']
                )

        # Merge the per-model raw DAG ranks
        merged_raw_list = []
        first_model = all_model_list[0]
        for i in range(len(tmp_rank_dict_save[first_model])):
            per_model_ranks = [tmp_rank_dict_save[m][i] for m in all_model_list]
            merged_raw_list.append(combine_rankings(rankings=per_model_ranks, type=rank_type))
        return merged_raw_list

    # ----------------------------------------------------------------
    # 5. HELPER: For 'graph_ensemble', combine multiple graphs
    # ----------------------------------------------------------------
    def _combine_graph_ensemble(all_e_model):
        """
        Performs the DAG_graph_ensemble across multiple models' graph_list,
        then calls 'weighted_descendants_count_sort' on the combined result.
        Returns a final rank_list.
        """
        merged_graphs = []
        merged_ranks = []

        first_model = all_model_list[0]
        num_graphs = len(all_e_model[first_model]['graph_list'])
        for i in range(num_graphs):
            g_tmp_list = []
            for e_model in all_model_list:
                g_tmp_list.append(all_e_model[e_model]['graph_list'][i])
            g_ens = DAG_graph_ensemble(graph_list=g_tmp_list)
            merged_graphs.append(g_ens)
            merged_ranks.append(weighted_descendants_count_sort(g_ens, w_type=w_type))

        return merged_graphs, merged_ranks

    # ----------------------------------------------------------------
    # 6. HELPER: For 'graph_ensemble', combine raw graphs
    # ----------------------------------------------------------------
    def _combine_raw_graph_ensemble(all_e_model):
        """
        Similar to _combine_graph_ensemble, but for raw_graph_list.
        Also transforms the final DAG via 'Greedy' before sorting.
        Returns a final raw_rank_list.
        """
        merged_raw_graphs = []
        merged_raw_ranks = []

        first_model = all_model_list[0]
        num_graphs = len(all_e_model[first_model]['raw_graph_list'])
        for i in range(num_graphs):
            g_tmp_list = []
            for e_model in all_model_list:
                g_tmp_list.append(all_e_model[e_model]['raw_graph_list'][i])
            g_ens = DAG_graph_ensemble(graph_list=g_tmp_list)
            g_ens_dag, _ = DGtoDAG(method='Greedy').get_dag(g_ens)

            merged_raw_graphs.append(g_ens_dag)
            merged_raw_ranks.append(weighted_descendants_count_sort(g_ens_dag, w_type=w_type))

        return merged_raw_graphs, merged_raw_ranks

    # ----------------------------------------------------------------
    # 7. LOAD DATA
    # ----------------------------------------------------------------
    # Depending on eval_model, either load a single model or multiple
    if eval_model != 'all':
        graph_list, raw_graph_list = _load_data_for_model(eval_model)
        all_model_data = None
    else:
        all_model_data = {}
        for tmp_eval_model in all_model_list:
            g_list, r_list = _load_data_for_model(tmp_eval_model)
            all_model_data[tmp_eval_model] = {
                'graph_list': g_list,
                'raw_graph_list': r_list
            }

    # ----------------------------------------------------------------
    # 8. MAIN LOGIC: rank_ensemble vs graph_ensemble
    # ----------------------------------------------------------------
    if ensemble_type == 'rank_ensemble':
        # ------------------------------
        # (a) Rank list for graph_list
        # ------------------------------
        if eval_model != 'all':
            # Single model
            rank_list = rank_ensemble_graph_to_rank(graph_list=graph_list, w_type=w_type)
        else:
            # Multiple models combined
            rank_list = _combine_rank_ensemble(all_model_data)

        # If top-level items are lists-of-lists, flatten them
        if isinstance(rank_list[0][0], list):
            rank_list = update_rank_list(rank_list=rank_list)

        # Store the final rank_list
        rank_list_result_save[eval_model][f"{rank_type}"][ensemble_type] = rank_list

        # ------------------------------
        # (b) Baseline (raw_graph_list)
        # ------------------------------
        if eval_model != 'all':
            # Single model
            all_dag_rank_list = _process_raw_graph_list_rank_ensemble(raw_graph_list)
        else:
            # Multiple models combined
            all_dag_rank_list = _combine_raw_dag_rank_ensemble(all_model_data)

        # Flatten the final baseline list if needed, then store
        rank_list_result_save[eval_model]["baseline"][ensemble_type] = update_rank_list(
            rank_list=all_dag_rank_list
        )

    elif ensemble_type == 'graph_ensemble':
        # ------------------------------
        # (a) Combine or directly rank the existing graph_list
        # ------------------------------
        if eval_model != 'all':
            # Single model
            rank_list = [weighted_descendants_count_sort(g, w_type=w_type) for g in graph_list]
        else:
            # Multiple models
            merged_graphs, rank_list = _combine_graph_ensemble(all_model_data)
            graph_list = merged_graphs  # updated combined graph_list if needed

        # ------------------------------
        # (b) Baseline from raw graphs
        # ------------------------------
        if eval_model != 'all':
            raw_rank_list = []
            for g1 in raw_graph_list:
                g1_dag, _ = DGtoDAG(method='Greedy').get_dag(g1)
                raw_rank_list.append(weighted_descendants_count_sort(g1_dag, w_type=w_type))
        else:
            merged_raw_list, raw_rank_list = _combine_raw_graph_ensemble(all_model_data)
            raw_graph_list = merged_raw_list  # updated combined raw graphs if needed

        # Save results
        rank_list_result_save[eval_model]["baseline"][ensemble_type] = raw_rank_list
        rank_list_result_save[eval_model][f"{rank_type}"][ensemble_type] = rank_list

    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")




def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--method", type=str, default='Greedy', help="Sort method, ['Greedy', 'SelfControl']")
    parser.add_argument("--eval_model", type=str, default='llama3-8b', help="Evaluation model, ['llama3-8b', 'qwen2-7b', 'mistral-7b', 'all']")
    parser.add_argument("--answer_model", type=str, default='qwen1.5-32b', help="Answer model, ['llama3-70b', 'qwen2-72b', 'mixtral-8x7b', 'qwen1.5-32b']")
    parser.add_argument("--task_name", type=str, default='10k-ultra', help="Task name, ['gsm8k', 'human_eval', 'alpaca', 'math', 'truthful_qa', 'ultra']")
    parser.add_argument("--w_type", type=str, default='noWeight', help="Weight type, ['add', 'multiple', 'noWeight']")
    parser.add_argument("--rank_type", type=str, default='pairwise_majority', help="Ranking method, ['weight_score', 'kemeny', 'weighted_kemeny', 'pairwise_majority', 'weighted_pairwise_majority']")
    
    return parser.parse_args()



def main():
    """Main function to execute the ranking process."""
    # Parse arguments
    args = parse_arguments()

    # List of evaluator models
    # "In the response selection setting, we adopt [llama3-8b, qwen2-7b, mistral-7b] as the evaluator set."
    all_model_list = ["llama3-8b", "qwen2-7b", "mistral-7b"]

    # List of ensemble methods
    ensemble_list = ['graph_ensemble', 'rank_ensemble']

    # Define the result storage path
    folder_path = f'{"Your denoise rank save path"}/{args.w_type}/{args.task_name}/{args.eval_model}/{args.answer_model}/{args.rank_type}/{args.method}'

    # Initialize the ranking results dictionary
    rank_list_result_save = {
        evaluator_model: {
            "baseline": {"rank_ensemble": [], "graph_ensemble": []},
            f"{args.rank_type}": {"rank_ensemble": [], "graph_ensemble": []},
        }
        for evaluator_model in all_model_list + ["all"]
    }

    # Execute the list_update function for each ensemble type
    for ensemble_type in ensemble_list:
        list_update(
            method=args.method, ensemble_type=ensemble_type, 
            eval_model=args.eval_model, all_model_list=all_model_list,
            rank_type=args.rank_type,
        )

    # Define the file path for saving results
    file_path = os.path.join(folder_path, f'{args.rank_type}_rank_list.json')

    # Save ranking results
    save_dict_to_json(rank_list_result_save, file_path)

# Run the script when executed
if __name__ == "__main__":
    main()

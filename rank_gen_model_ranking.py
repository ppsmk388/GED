import sys
sys.path.append('./rank_utility') 
sys.path.append('./model-rank') 
import argparse
import os
import argparse
from mallows import *
import pickle
from tool import *





def list_update(method, ensemble_type,  eval_model=None, all_model_list=None, rank_type=None, super_method=None, w_type=None, rank_list_result_save=None):

    data_save_path = '/data/huzhengyu/DGG_t/model_rank_raw_save/same_id_same_question/processed_data'

    # ----------------------------------------------------------------
    # 1. HELPER: Load model data from disk
    # ----------------------------------------------------------------
    def read_data(file_path):
        """Reads and returns pickled data from the given file path."""
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def load_model_data(model_name):
        """Loads graph_list and raw_graph_list for a given evaluator model."""
        model_path = f"{data_save_path}/{model_name}/{method}"
        return {
            'graph_list': read_data(f"{model_path}/graph_list.pkl"),
            'raw_graph_list': read_data(f"{model_path}/raw_graph_list.pkl")
        }

    def load_all_models_data():
        """Loads graph_list and raw_graph_list for all models in all_model_list."""
        return {m: load_model_data(m) for m in all_model_list}

    # ----------------------------------------------------------------
    # 2. HELPER: Utility functions for flattening and transformation
    # ----------------------------------------------------------------
    def flatten_rank_list(rank_list_in):
        """
        Flattens a rank list if any items are nested lists.
        E.g., [[['a'], ['b']], ...] => [['a', 'b'], ...].
        """
        flattened = []
        for ranks in rank_list_in:
            if any(isinstance(r, list) for r in ranks):
                merged = []
                for group in ranks:
                    merged.extend(group)
                flattened.append(merged)
            else:
                flattened.append(ranks)
        return flattened

    def maybe_transform_selfcontrol(graphs):
        """
        If super_method == 'SelfControl', transforms each graph into a DAG using 'ContraSolver';
        otherwise, returns graphs unchanged.
        """
        if super_method == 'SelfControl':
            transformed = []
            for g in graphs:
                dag_graph, _ = DGtoDAG(method='ContraSolver').get_dag(g)
                transformed.append(dag_graph)
            return transformed
        return graphs

    # ----------------------------------------------------------------
    # 3. HELPER: For 'rank_ensemble', combine multiple models' ranks
    # ----------------------------------------------------------------
    def single_model_rank_ensemble(model_name):
        """
        Processes a single model using the 'rank_ensemble' approach.
        Updates rank_list_result_save in place for the given model_name.
        """
        model_data = load_model_data(model_name)

        # Main rank
        rank_list = rank_ensemble_graph_to_rank(
            graph_list=model_data['graph_list'],
            w_type=w_type
        )
        rank_list_result_save[model_name][f"{rank_type}"]['rank_ensemble'] = flatten_rank_list(rank_list)

        # Baseline rank
        raw_graphs = maybe_transform_selfcontrol(model_data['raw_graph_list'])
        dag_rank_list = rank_ensemble_raw_graph_to_rank(raw_graphs)
        rank_list_result_save[model_name]["baseline"]['rank_ensemble'] = flatten_rank_list(dag_rank_list)

    def multi_model_rank_ensemble():
        """
        Processes multiple models using the 'rank_ensemble' approach,
        aggregating rank lists across all evaluator models in all_model_list.
        """
        all_data = load_all_models_data()

        # Step A: rank_ensemble for each model
        partial_ranks = {
            m: rank_ensemble_graph_to_rank(all_data[m]['graph_list'], w_type=w_type)
            for m in all_model_list
        }

        # Step B: combine partial ranks for each question/set index
        combined_rank_list = []
        total_sets = len(partial_ranks[all_model_list[0]])
        for idx in range(total_sets):
            local_ranks = [partial_ranks[m][idx] for m in all_model_list]
            combined = combine_rankings(local_ranks, type=rank_type)
            combined_rank_list.append(combined)

        rank_list_result_save['all'][f"{rank_type}"]['rank_ensemble'] = flatten_rank_list(combined_rank_list)

        # Step C: baseline from raw_graphs
        partial_dag_ranks = {}
        for m in all_model_list:
            raw_graphs = maybe_transform_selfcontrol(all_data[m]['raw_graph_list'])
            partial_dag_ranks[m] = rank_ensemble_raw_graph_to_rank(raw_graphs)

        # Step D: combine partial DAG ranks
        combined_dag_rank_list = []
        for idx in range(total_sets):
            local_dag_ranks = [partial_dag_ranks[m][idx] for m in all_model_list]
            combined_dag = combine_rankings(local_dag_ranks, type=rank_type)
            combined_dag_rank_list.append(combined_dag)

        rank_list_result_save['all']["baseline"]['rank_ensemble'] = flatten_rank_list(combined_dag_rank_list)

    # ----------------------------------------------------------------
    # 4. HELPER: For 'graph_ensemble', combine multiple models' graphs
    # ----------------------------------------------------------------
    def single_model_graph_ensemble(model_name):
        """
        Processes a single model using the 'graph_ensemble' approach.
        Updates rank_list_result_save in place for the given model_name.
        """
        model_data = load_model_data(model_name)

        # Main rank for each graph
        rank_list = [
            weighted_descendants_count_sort(g, w_type=w_type)
            for g in model_data['graph_list']
        ]
        rank_list_result_save[model_name][f"{rank_type}"]['graph_ensemble'] = flatten_rank_list(rank_list)

        # Baseline rank: transform raw graphs to DAG (Greedy) then rank
        baseline_list = []
        for g in model_data['raw_graph_list']:
            dag_graph, _ = DGtoDAG(method='Greedy').get_dag(g)
            baseline_list.append(weighted_descendants_count_sort(dag_graph, w_type=w_type))
        rank_list_result_save[model_name]["baseline"]['graph_ensemble'] = flatten_rank_list(baseline_list)

    def multi_model_graph_ensemble():
        """
        Processes multiple models using the 'graph_ensemble' approach,
        building an ensemble graph across all models per question/set index.
        """
        all_data = load_all_models_data()

        total_sets = len(all_data[all_model_list[0]]['graph_list'])
        combined_graph_list = []

        # Step A: build an ensemble graph per question
        for idx in range(total_sets):
            graphs_to_ensemble = [all_data[m]['graph_list'][idx] for m in all_model_list]
            g_ens = DAG_graph_ensemble(graph_list=graphs_to_ensemble)
            combined_graph_list.append(g_ens)

        # Step B: rank the ensemble graph for each question
        combined_rank_list = [
            weighted_descendants_count_sort(g_ens, w_type=w_type)
            for g_ens in combined_graph_list
        ]
        rank_list_result_save['all'][f"{rank_type}"]['graph_ensemble'] = flatten_rank_list(combined_rank_list)

        # Step C: baseline from raw graphs (transform to DAG via 'Greedy')
        combined_baseline_list = []
        for idx in range(total_sets):
            raw_to_ensemble = [all_data[m]['raw_graph_list'][idx] for m in all_model_list]
            g_ens = DAG_graph_ensemble(graph_list=raw_to_ensemble)
            dag_graph, _ = DGtoDAG(method='Greedy').get_dag(g_ens)
            combined_baseline_list.append(
                weighted_descendants_count_sort(dag_graph, w_type=w_type)
            )

        rank_list_result_save['all']["baseline"]['graph_ensemble'] = flatten_rank_list(combined_baseline_list)

    # ----------------------------------------------------------------
    # 5. Dispatcher: Choose the appropriate flow based on arguments
    # ----------------------------------------------------------------
    single_or_multi = 'single' if eval_model != 'all' else 'multi'
    dispatch_map = {
        'rank_ensemble': {
            'single': single_model_rank_ensemble,
            'multi': multi_model_rank_ensemble
        },
        'graph_ensemble': {
            'single': single_model_graph_ensemble,
            'multi': multi_model_graph_ensemble
        }
    }

    if ensemble_type not in dispatch_map:
        raise ValueError(f"Invalid ensemble_type: {ensemble_type}")

    # If single model, pass the model name; otherwise call the multi-model function
    if single_or_multi == 'single':
        dispatch_map[ensemble_type]['single'](eval_model)
    else:
        dispatch_map[ensemble_type]['multi']()







def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--method", type=str, default='SelfControl', help="sort_method,  ['Exact', 'Greedy', 'Sort', 'PageRank', 'SelfControl']")
    parser.add_argument("--eval_model", type=str, default='llama3_70b', help="eval_model, ['llama3_70b', 'qwen2_72b', 'mistral87b', 'all']")
    parser.add_argument("--w_type", type=str, default='add', help="w_type, ['add' , 'multiple', 'noWeight']")
    parser.add_argument("--rank_type", type=str, default='weight_score', help="eval_model, ['weight_score', 'kemeny', 'weighted_kemeny', 'pairwise_majority', 'weighted_pairwise_majority']")
    
    return parser.parse_args()




def main():
    """Main function to execute the ranking process."""
    # Parse arguments
    args = parse_arguments()
    # List of evaluator models
    # "In the model ranking  setting, we adopt ['llama3_70b', 'mistral87b', 'qwen2_72b', 'qwen1.5_72b']  as the evaluator set."
    all_model_list = ['llama3_70b', 'mistral87b', 'qwen2_72b', 'qwen1.5_72b']

    # List of ensemble methods
    ensemble_list = ['graph_ensemble', 'rank_ensemble']

    # Define the result storage path
    folder_path = f'{"Your denoise rank save path"}/{args.w_type}/{args.eval_model}/{args.rank_type}/{args.method}'


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



if __name__ == "__main__":
    main()

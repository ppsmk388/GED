
import sys
sys.path.append('./rank_utility') 
import copy
import numpy as np
import networkx as nx
from collections import defaultdict
from ranking_utils import RankingUtils, Ranking
from mallows import *
from ws_ranking import WeakSupRanking


class DGtoDAG:
    def __init__(self, method='Greedy'):
        """
        Input: method - a string in ['ContraSolver', 'Greedy']
        """
        self.G = None
        self.weighted = False
        self.method = method
        assert method in ['ContraSolver', 'Greedy']

    def out_weight_sum(self, S, u):
        """
        Input: S - a subset of nodes in G
               u - a node
        """
        valid_edges = [e for e in self.G.out_edges(u, data=True) if e[1] in S]
        if self.weighted:
            return sum([e[2]['weight'] for e in valid_edges])
        else:
            return len(valid_edges)

    def order_to_fas(self, nodes_order):
        """
        Input: nodes_order - an ordered list of nodes in G
        """
        nodes_index = {node: i for i, node in enumerate(nodes_order)}
        feedback_arcs = []
        for node in nodes_order:
            for e in self.G.out_edges(node, data=False):
                if e[1] in nodes_index:
                    if nodes_index[e[1]] < nodes_index[node]:
                        feedback_arcs.append((node, e[1]))
        return feedback_arcs


    def contrasolver_fas(self, component):
        """
        Implement the ContraSolver algorithm to compute feedback arc set.
        Input: component - a set of nodes in G that form a strongly connected component
        """

        subG = self.G.subgraph(component).copy()
        if self.weighted:
            G_prime = nx.maximum_branching(subG)
        else:
            temp_subG = subG.copy()
            for u, v in temp_subG.edges():
                temp_subG.edges[u, v]['weight'] = 1
            G_prime = nx.maximum_branching(temp_subG)

        E_prime = set(G_prime.edges())
        E_r = set(subG.edges()) - E_prime
        E_h = set()
        E_c = set()
        while E_r:
            edges_in_er = list(E_r)
            if self.weighted:
                edges_in_er.sort(key=lambda e: subG.edges[e]['weight'])
            else:
                edges_in_er.sort()

            for (yi, yj) in edges_in_er:
                if (yi, yj) not in E_r:
                    continue  
                if nx.has_path(G_prime, yj, yi):
                    E_c.add((yi, yj))
                    path = nx.shortest_path(G_prime, yj, yi)
                    cycle_edges = list(zip(path, path[1:]))
                    E_h.update(cycle_edges)
                    E_r.remove((yi, yj))
            edges_in_er = list(E_r)
            if self.weighted:
                edges_in_er.sort(key=lambda e: -subG.edges[e]['weight'])
            else:
                edges_in_er.sort(reverse=True)

            for (yi, yj) in edges_in_er:
                E_r.remove((yi, yj))
                if not nx.has_path(G_prime, yj, yi):
                    G_prime.add_edge(yi, yj)
                    if self.weighted:
                        G_prime.edges[yi, yj]['weight'] = subG.edges[yi, yj]['weight']
                    break
        feedback_arcs = list(set(subG.edges()) - set(G_prime.edges()))
        return feedback_arcs
    
    
    def greedy_fas(self, component):
        """
        Reference: Eades, Peter, Xuemin Lin, and William F. Smyth. "A fast and effective heuristic for the feedback arc set problem."
            Information processing letters 47.6 (1993): 319-323.

        Input: component - a set of nodes in G that form a strongly connected component
        """

        subG = self.G.subgraph(component).copy()
        ending_sequence = []
        starting_sequence = []

        while len(subG.nodes) > 0:
            # Remove all current sinks and sources
            while (cur_sinks := [node for node in subG.nodes if subG.out_degree(node) == 0]):
                ending_sequence += cur_sinks
                subG.remove_nodes_from(cur_sinks)
            while (cur_sources := [node for node in subG.nodes if subG.in_degree(node) == 0]):
                starting_sequence += cur_sources
                subG.remove_nodes_from(cur_sources)
            if len(subG.nodes) == 0:
                break

            # Find the node with the maximum in and out degree difference and remove it
            cur_node_list = list(subG.nodes)
            deltas = []
            for node in cur_node_list:
                if self.weighted:
                    delta = sum([e[2]['weight'] for e in subG.out_edges(node, data=True)]) \
                            - sum([e[2]['weight'] for e in subG.in_edges(node, data=True)])
                else:
                    delta = subG.out_degree(node) - subG.in_degree(node)
                deltas.append(delta)
            greedy_node = cur_node_list[np.argmax(deltas)]

            starting_sequence.append(greedy_node)
            subG.remove_node(greedy_node)

        ending_sequence.reverse()
        nodes_order = starting_sequence + ending_sequence

        # Get the feedback arc set
        return self.order_to_fas(nodes_order)


    def get_dag(self, G):
        self.G = G
        self.weighted = nx.is_weighted(self.G)
        feedback_arcs = []

        scc = list(nx.strongly_connected_components(self.G))
        for component in scc:
            if self.method == 'Greedy':
                feedback_arcs += self.greedy_fas(component)
            elif self.method == 'ContraSolver':
                feedback_arcs += self.contrasolver_fas(component)
            else:
                pass
        for arc in feedback_arcs:
            self.G.remove_edge(*arc)
        return self.G, len(feedback_arcs)



def update_rank_list(rank_list):
    final_r_l = []
    for r in rank_list:
        tmp_r = []
        for r_i in r:
            tmp_r.extend(r_i)
        final_r_l.append(tmp_r)
    return final_r_l

def save_dict_to_json(data, file_path):
    """Create directory if it does not exist and save dictionary as a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Dictionary saved to {file_path}")

# Combine rankings using different methods
def combine_rankings(rankings, type='weight_score'):
    # Flatten nested lists of rankings
    if isinstance(rankings[0][0], list):
        new_rankings = []
        for r_l in rankings:
            r_ll = []
            for kkk in r_l:
                r_ll.extend(kkk)
            new_rankings.append(r_ll)
    else:
        new_rankings = copy.deepcopy(rankings)
        tmp_rank = []
        for r_l in rankings:
            r_ll = []
            for kkk in r_l:
                r_ll.append([kkk])
            tmp_rank.append(r_ll)

    # Combine rankings using weighted scoring method or weak supervision
    if type == 'weight_score':
        return weighted_scoring_method(new_rankings)
    elif type in ['kemeny', 'weighted_kemeny', 'pairwise_majority', 'weighted_pairwise_majority']:
        return universalizing_weak_supervision(method_type=type, rank_list=new_rankings)




# Convert raw graph ensemble to ranking based on weighted out-degree
def rank_ensemble_raw_graph_to_rank(raw_graph_list):
    all_dag_rank_list = []
    for G in raw_graph_list:
        try:
            # Sort nodes by weighted out-degree
            sorted_nodes = sort_nodes_by_weighted_out_degree(G)
            sorted_nodes = [[i] for i in sorted_nodes]
            all_dag_rank_list.append(sorted_nodes)
        except nx.NetworkXUnfeasible:
            print("Graph is not a DAG, cannot perform topological sort")
    return all_dag_rank_list

# Raw graph ensemble to ranking by weighted out-degree
def raw_graph_ensemble_graph_to_rank(raw_graph_list):
    baseline_withoutGraph_G_ensemble = nx.DiGraph()
    baseline_withoutGraph_G_ensemble.add_nodes_from(range(len(model_dict)))

    for G in raw_graph_list:
        for u, v, data in G.edges(data=True):
            if baseline_withoutGraph_G_ensemble.has_edge(u, v):
                baseline_withoutGraph_G_ensemble[u][v]['weight'] += data['weight']
            else:
                baseline_withoutGraph_G_ensemble.add_edge(u, v, weight=data['weight'])
    rank = sort_nodes_by_weighted_out_degree(baseline_withoutGraph_G_ensemble)
    return rank




# Universalizing weak supervision method with different ranking strategies
def universalizing_weak_supervision(method_type='kemeny', rank_list=[]):
    rank_data = copy.deepcopy(rank_list)

    # Convert ranking data into Ranking objects
    L = [[Ranking(rank) for rank in rank_data]]

    # Create RankingUtils object
    d = len(rank_data[0])  # Number of items
    r_utils = RankingUtils(d)

    # Define and train the weak supervision model
    num_lfs = len(rank_data)  # Number of annotators
    label_model = WeakSupRanking(r_utils)
    conf = {"train_method": "median_triplet_opt"}
    label_model.train(conf, L, num_lfs)

    if method_type == 'kemeny':
        mv_conf = {"train_method": "median_triplet_opt", "inference_rule": "kemeny"}
        Y = label_model.infer_ranking(mv_conf, L)
    elif method_type == 'weighted_kemeny':
        uws_conf = {"train_method": "median_triplet_opt", "inference_rule": "weighted_kemeny"}
        Y = label_model.infer_ranking(uws_conf, L)
    elif method_type == 'pairwise_majority':
        mv_conf = {"train_method": "median_triplet_opt", "inference_rule": "pairwise_majority"}
        Y = label_model.infer_ranking(mv_conf, L)
    elif method_type == 'weighted_pairwise_majority':
        uws_conf = {"train_method": "median_triplet_opt", "inference_rule": "weighted_pairwise_majority"}
        Y = label_model.infer_ranking(uws_conf, L)

    aggregated_ranking = [rank.permutation for rank in Y][0]
    final_ranking_result = [[i] for i in aggregated_ranking]

    return final_ranking_result

# Final ranking generation function, processes rankings with nested lists
def final_ranking_gen(ranking):
    if all(isinstance(sublist, int) for sublist in ranking):
        return ranking

    new_ranking = []
    for sublist in ranking:
        if isinstance(sublist, int):
            new_ranking.append([sublist])
        elif len(sublist) > 1:
            # Shuffle the sublist before splitting
            random.shuffle(sublist)
            for item in sublist:
                new_ranking.append([item])
        else:
            new_ranking.append(sublist)
    return new_ranking





# Calculate the similarity between two rankings using Spearman and Kendall rank correlation coefficients
def calculate_similarities(rank1, rank2):
    def split_nested_lists(nested_list):
        result = []
        for sublist in nested_list:
            if len(sublist) > 1:
                for item in sublist:
                    result.append([item])
            else:
                result.append(sublist)
        return result

    def all_integers(input_list):
        return all(isinstance(item, int) for item in input_list)

    if not all_integers(rank2):
        rank2 = split_nested_lists(rank2)
    if len(rank1) != len(rank2):
        raise ValueError("Rankings must have the same length.")

    spearman_correlation, _ = spearmanr(rank1, rank2)
    kendall_correlation, _ = kendalltau(rank1, rank2)
    return spearman_correlation, kendall_correlation


# Weighted scoring method for combining rankings
def weighted_scoring_method(rankings):
    node_scores = defaultdict(int)
    for ranking in rankings:
        for rank, nodes in enumerate(ranking):
            score = len(ranking) - rank
            for node in nodes:
                node_scores[node] += score
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    combined_ranking = []
    current_score = None
    current_group = []
    for node, score in sorted_nodes:
        if score != current_score:
            if current_group:
                combined_ranking.append(current_group)
            current_group = [node]
            current_score = score
        else:
            current_group.append(node)
    if current_group:
        combined_ranking.append(current_group)
    return combined_ranking



def Multi_weighted_descendants_count_sort(G):

    weighted_descendants_count = {}

    for node in G.nodes():
        descendants = nx.descendants(G, node)
        weighted_count = 0
        for desc in descendants:
            path_length = nx.shortest_path_length(G, source=node, target=desc, weight='weight')
            weighted_count += 1 / path_length  # Inverse weighting based on path length
        weighted_descendants_count[node] = weighted_count

    # Group nodes by weighted descendant count
    count_to_nodes = defaultdict(list)
    for node, count in weighted_descendants_count.items():
        count_to_nodes[count].append(node)

    # Sort counts in descending order
    sorted_counts = sorted(count_to_nodes.keys(), reverse=True)

    # Construct the sorted output list
    sorted_nodes = [sorted(count_to_nodes[count]) for count in sorted_counts]

    return sorted_nodes


def Add_weighted_descendants_count_sort(G):

    weighted_descendants_count = {}

    for node in G.nodes():
        descendants = nx.descendants(G, node)
        weighted_count = 0
        for desc in descendants:
            # Find the shortest path from node to descendant
            path = nx.shortest_path(G, source=node, target=desc, weight='weight')
            # Compute the total weight of the path
            path_weight = sum(G.edges[path[i], path[i + 1]]['weight'] for i in range(len(path) - 1))
            weighted_count += path_weight
        weighted_descendants_count[node] = weighted_count

    # Group nodes by weighted descendant count
    count_to_nodes = defaultdict(list)
    for node, count in weighted_descendants_count.items():
        count_to_nodes[count].append(node)

    # Sort counts in descending order
    sorted_counts = sorted(count_to_nodes.keys(), reverse=True)

    # Construct the sorted output list
    sorted_nodes = []
    for count in sorted_counts:
        sorted_nodes.extend(sorted(count_to_nodes[count]))

    return sorted_nodes


def No_weight_descendants_count_sort(G):

    # Compute the number of descendants for each node
    descendants_count = {node: len(nx.descendants(G, node)) for node in G.nodes()}

    # Group nodes by descendant count
    count_to_nodes = defaultdict(list)
    for node, count in descendants_count.items():
        count_to_nodes[count].append(node)

    # Sort counts in descending order
    sorted_counts = sorted(count_to_nodes.keys(), reverse=True)

    # Construct the sorted output list
    sorted_nodes = [sorted(count_to_nodes[count]) for count in sorted_counts]

    return sorted_nodes



# Main method for sorting nodes based on different weighting types
def weighted_descendants_count_sort(G=None, w_type=None):
    if w_type == 'add':
        return Add_weighted_descendants_count_sort(G)
    elif w_type == 'multiple':
        return Multi_weighted_descendants_count_sort(G)
    else:
        return No_weight_descendants_count_sort(G)

# Rank ensemble graph to rank conversion method
def rank_ensemble_graph_to_rank(graph_list, w_type):
    rank_list = []
    for G in graph_list:
        try:
            # Get nodes sorted by weighted descendant count
            sorted_nodes = weighted_descendants_count_sort(G, w_type)
            rank_list.append(sorted_nodes)
        except nx.NetworkXUnfeasible:
            print("Graph is not a DAG, cannot perform topological sort")
    return rank_list



# Sort nodes by weighted out-degree for DAGs
def sort_nodes_by_weighted_out_degree(G):
    # Calculate the weighted out-degree for each node
    weighted_out_degrees = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}

    # Sort nodes by weighted out-degree in descending order
    sorted_nodes = sorted(weighted_out_degrees.keys(), key=lambda node: weighted_out_degrees[node], reverse=True)

    return sorted_nodes


# Graph ensemble construction for DAGs
def DAG_graph_ensemble(graph_list):
    G_ensemble = nx.DiGraph()
    G_ensemble.add_nodes_from(range(len(model_dict)))

    for G in graph_list:
        for u, v, data in G.edges(data=True):
            if G_ensemble.has_edge(u, v):
                G_ensemble[u][v]['weight'] += data['weight']
            else:
                G_ensemble.add_edge(u, v, weight=data['weight'])

    # Process edges where A points to B and B points to A
    try:
        edges_to_remove = []

        for u, v in list(G_ensemble.edges()):
            if (u, v) in edges_to_remove or (v, u) in edges_to_remove:
                continue
            if G_ensemble.has_edge(v, u):
                weight_uv = G_ensemble[u][v]['weight']
                weight_vu = G_ensemble[v][u]['weight']

                if weight_uv > weight_vu:
                    G_ensemble[u][v]['weight'] -= weight_vu
                    edges_to_remove.append((v, u))
                elif weight_uv < weight_vu:
                    G_ensemble[v][u]['weight'] -= weight_uv
                    edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((u, v))
                    edges_to_remove.append((v, u))

        # Remove edges after processing
        G_ensemble.remove_edges_from(edges_to_remove)
    except:
        pass

    return G_ensemble


def read_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
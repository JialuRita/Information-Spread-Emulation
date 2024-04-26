import networkx as nx
import csv
import random
import pandas as pd


random.seed(3407)

def readGraph(graph_path):
    G = nx.Graph()
    elist = []  #边集
    # 读取csv文件
    df = pd.read_csv(graph_path, header=0)
    print(df)
    for index, row in df.iterrows():
        node_1 = row['node_1']
        node_2 = row['node_2']
        elist.append((node_1,node_2))   
    #print(elist)
    G.add_edges_from(elist)
    return G

def extract_connected_subgraph(graph, target_size=300, sparsity=0.8):
    """Extract a sparse and unevenly connected subgraph from the provided graph."""
    print("Extracting connected subgraph...")
    nodes = list(graph.nodes())
    weights = [deg for _, deg in graph.degree()]  #节点度
    selected_nodes = random.choices(nodes, weights=weights, k=target_size)  #选取300个节点
    subgraph = nx.Graph(graph.subgraph(selected_nodes))  #生成子网络
    edges_to_remove = random.sample(list(subgraph.edges()), int(len(subgraph.edges()) * (1 - sparsity)))
    subgraph.remove_edges_from(edges_to_remove)
    ensure_graph_connectivity(subgraph)  #保证连通性
    print("Subgraph extraction complete.")
    return subgraph

def ensure_graph_connectivity(graph):
    """Ensure that the graph is connected, adding edges if necessary."""
    if not nx.is_connected(graph):
        print("Subgraph is not connected, adding edges...")
        for component in nx.connected_components(graph):
            connect_components(graph, component)
def connect_components(graph, component):
    """Connect disconnected components in the graph."""
    connected_components = list(nx.connected_components(graph))
    while len(connected_components) > 1:
        first_component = connected_components.pop()
        second_component = connected_components[0]
        graph.add_edge(random.choice(list(first_component)), random.choice(list(second_component)))

#建立Threshold Model
def simulate_threshold_model(graph, commitment_threshold, memory_size, num_rounds=1000):
    """Simulate the threshold model on the graph."""
    print(f"Model parameters: C={commitment_threshold}, M={memory_size}, T={num_rounds}")
    total_nodes = len(graph.nodes())
    memory_state = {node: ['A'] * memory_size for node in graph.nodes()}
    committed_nodes = set(random.sample(list(graph.nodes()), int(commitment_threshold * total_nodes)))
    simulate_information_spread(graph, committed_nodes, memory_state, memory_size, num_rounds)
    return calculate_conversion_rate(graph, committed_nodes, memory_state, memory_size)

def simulate_information_spread(graph, committed_nodes, memory_state, memory_size, num_rounds):
    """Simulate the spread of information across the graph."""
    num_edges = len(graph.edges())
    num_nodes = len(graph.nodes())
    sample_size = min(num_edges, num_nodes)  #Ensure the sample size does not exceed the number of edges
    for _ in range(num_rounds):
        for edge in random.sample(list(graph.edges()), sample_size):
            speaker, hearer = random.choice([(edge[0], edge[1]), (edge[1], edge[0])])
            propagate_information(speaker, hearer, memory_state, memory_size, committed_nodes)

def propagate_information(speaker, hearer, memory_state, memory_size, committed_nodes):
    """Propagate information from speaker to hearer based on their memory state."""
    message = 'A' if memory_state[speaker].count('B') <= memory_size / 2 else 'B'
    if hearer not in committed_nodes:
        memory_state[hearer].pop(0)
        memory_state[hearer].append(message)

def calculate_conversion_rate(graph, committed_nodes, memory_state, memory_size):
    """Calculate the conversion rate of non-committed nodes."""
    non_committed_nodes = set(graph.nodes()) - committed_nodes
    converted_count = sum(1 for node in non_committed_nodes if memory_state[node].count('B') > memory_size / 2)
    return converted_count / len(non_committed_nodes)

def process_m_value(params):
    graph, m = params
    print("对子网络进行仿真：")
    for c in range(10, 100):
        p = simulate_threshold_model(graph, c/100, m)  #建立阈值模型
        if p == 1:
            print(f"Threshold model parameter found: C={c/100}, M={m}")
            return m, c/100
    print("No suitable threshold model parameter found.")
    return None

#保存仿真结果
def save_results_to_file(filename, results):
    """Save the simulation results to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['M', 'C'])
        for result in results:
            writer.writerow(result)

if __name__ == '__main__':
    filePath = "./deezer_clean_data/RO_edges.csv"  #数据
    G = readGraph(filePath)
    M = 21  #阈值范围
    subgraph = extract_connected_subgraph(G)
    results = []
    print("Starting parameter search simulation...")
    for m in range(1, M):
        result = process_m_value((subgraph.copy(), m))
        if result:
            results.append(result)
    print("Simulation parameter search complete.")
    print("Saving results to file...")
    save_results_to_file('result.csv', results)  #保存结果
    print("Results saved.")


import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def build_graph(cve_items, features):
    G = nx.Graph()
    
    for i, item in enumerate(cve_items):
        node_id = item['cve']['CVE_data_meta']['ID']
        G.add_node(node_id, x=features[i])

        # Add dummy edges (for prototype) between nodes with shared keywords
        for j in range(i):
            if set(item['cve']['description']['description_data'][0]['value'].split()) & \
               set(cve_items[j]['cve']['description']['description_data'][0]['value'].split()):
                G.add_edge(node_id, cve_items[j]['cve']['CVE_data_meta']['ID'])

    pyg_graph = from_networkx(G)
    return pyg_graph

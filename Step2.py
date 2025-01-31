#### Extract the disease information, plot the network graoh, visualize the X-ray image for a specific patient from the graohic database 

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from textwrap import wrap
from PIL import Image
import copy
import os

# Define the base directory (update with the correct path)
base_dir = os.path.abspath(os.path.join(current_working_dir, "..", ".."))
xray_image_dir = os.path.join(base_dir, "ProgrammingTest_Data/png")

def load_patient_data(graphs_dict, patient_id):
    """Extract patient data (disease, report, image file name) from the graph."""
    if patient_id not in graphs_dict:
        raise ValueError(f"Patient {patient_id} not found in the data.")

    patient_graph = graphs_dict[patient_id]
    nodes = patient_graph.get("nodes", [])
    edges = patient_graph.get("links", [])

    disease = None
    report_text = None
    image_file_name = None

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        relationship = edge.get("relationship")

        if relationship == "HAS_REPORT" and source == patient_id:
            report_text = target
        elif relationship == "HAS_DIAGNOSIS" and source == patient_id:
            disease = target
        elif relationship == "HAS_IMAGE" and source == patient_id:
            image_file_name = target

    if not report_text or not disease or not image_file_name:
        raise ValueError(f"Incomplete data for patient {patient_id}.")

    return disease, report_text, image_file_name

def plot_network_graph(subgraph, ax):
    """Plot the network graph, excluding nodes with type='FULL_REPORT' and edges with relationship='HAS_REPORT'."""
    # Create a deep copy of the subgraph to avoid modifying the original
    filtered_graph = copy.deepcopy(subgraph)

    # Remove nodes with type='FULL_REPORT'
    nodes_to_remove = [
        node for node, attr in filtered_graph.nodes(data=True) if attr.get("type") == "FULL_REPORT"
    ]
    filtered_graph.remove_nodes_from(nodes_to_remove)

    # Remove all edges with the relationship 'HAS_REPORT'
    edges_to_remove = [
        (u, v) for u, v, attr in filtered_graph.edges(data=True) if attr.get("relationship") == "HAS_REPORT"
    ]
    filtered_graph.remove_edges_from(edges_to_remove)

    # Generate positions for the nodes
    pos = nx.spring_layout(filtered_graph)
    edge_labels = nx.get_edge_attributes(filtered_graph, 'relationship')

    # Draw the filtered graph
    nx.draw(filtered_graph, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, ax=ax)
    nx.draw_networkx_edge_labels(filtered_graph, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title("Network Graph (Filtered)")

def display_xray_image(image_path, ax):
    """Display the X-ray image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"X-ray image not found at {image_path}")

    xray_image = Image.open(image_path)
    ax.imshow(xray_image, cmap='gray')
    ax.set_title("X-ray Image")
    ax.axis('off')

def main():
    # Load the JSON file containing the graphs
    with open('all_graphs.json', 'r') as f:
        graphs_dict = json.load(f)

    # Specify the patient ID
    specified_patient_id = "Patient_0472"

    try:
        # Extract patient data
        disease, report_text, image_file_name = load_patient_data(graphs_dict, specified_patient_id)

        # Load the network graph
        patient_graph = graphs_dict[specified_patient_id]
        subgraph = json_graph.node_link_graph(patient_graph)

        # Load the X-ray image path
        xray_image_path = os.path.join(xray_image_dir, image_file_name)

        # Create a figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns

        # Print the FULL REPORT
        print(f"{specified_patient_id}, Disease: {disease}, Report: {report_text}")

        # Set the super title of the figure
        figure_title = f"Network Graph and X-ray Image for {specified_patient_id}"
        wrapped_figure_title = "\n".join(wrap(figure_title, 100))
        fig.suptitle(wrapped_figure_title, fontsize=14)

        # Plot the network graph
        plot_network_graph(subgraph, axs[0])

        # Display the X-ray image
        display_xray_image(xray_image_path, axs[1])

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
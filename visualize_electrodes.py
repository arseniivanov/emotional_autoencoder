#This file should contain the code to visualize EEG positions from different datasets.
#This should be done to be able to compare signals from different datasets in order to map them to the same space.

import networkx as nx
import matplotlib.pyplot as plt

# Define the electrode locations and their neighbors
electrode_positions = {
    "Nz": ["Fpz"],
    "Fpz": ["Nz", "AFz"],
    "AFz": ["Fpz", "Fz"],
    "Fz": ["AFz", "FCz"],
    "FCz": ["Fz", "Cz"],
    "Cz": ["FCz", "CPz"],
    "CPz": ["Cz", "Pz"],
    "Pz": ["CPz", "POz"],
    "POz": ["Pz", "Oz"],
    "Oz": ["POz", "Iz"],
    "Iz": ["Oz"]
}

# Create a graph
eeg_graph = nx.Graph(electrode_positions)

# Add positions for the graph visualization
pos = {
    "Nz": (0, 10),
    "Fpz": (0, 8),
    "AFz": (0, 6),
    "Fz": (0, 4),
    "FCz": (0, 2),
    "Cz": (0, 0),
    "CPz": (0, -2),
    "Pz": (0, -4),
    "POz": (0, -6),
    "Oz": (0, -8),
    "Iz": (0, -10)
}

# Visualize the graph
nx.draw(eeg_graph, pos, with_labels=True, node_size=1000, node_color='cyan', font_size=12, font_weight='bold')
plt.show()
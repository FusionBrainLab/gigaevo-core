from __future__ import annotations

"""Lightweight immutable specification of a DAG.

`DAGSpec` records factory callables that produce fresh `Stage` instances 
every time a DAG is materialized. This avoids costly `deepcopy` operations 
and guarantees no shared mutable state leaks between program runs.
"""

from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel, Field, field_validator

from src.programs.automata import ExecutionOrderDependency, DataFlowEdge
from src.programs.stages.base import Stage


# DataFlowEdge is now imported from automata.py


class DAGSpec(BaseModel):
    """An immutable blueprint used by `DagFactory` to build `DAG` instances."""

    nodes: Dict[str, Callable[[], Stage]] = Field(
        ..., description="Stage factories by name"
    )
    data_flow_edges: List[DataFlowEdge] = Field(
        ..., description="Data flow edges with semantic input names"
    )
    exec_order_deps: Optional[Dict[str, List[ExecutionOrderDependency]]] = (
        Field(None, description="Execution order dependencies by stage name")
    )
    max_parallel_stages: int = Field(
        8, description="Maximum parallel stages allowed"
    )
    dag_timeout: float = Field(2400.0, description="Timeout for DAG execution")

    @field_validator("data_flow_edges")
    @classmethod
    def validate_data_flow_edges(
        cls, v: List[DataFlowEdge], info
    ) -> List[DataFlowEdge]:
        """Validate that all referenced stages in data_flow_edges exist in nodes."""
        if "nodes" not in info.data:
            return v

        nodes = info.data["nodes"]
        unknown_sources = {
            edge.source_stage for edge in v if edge.source_stage not in nodes
        }
        unknown_destinations = {
            edge.destination_stage for edge in v if edge.destination_stage not in nodes
        }
        unknown = unknown_sources | unknown_destinations
        
        if unknown:
            raise ValueError(
                f"Data flow edges reference unknown stage(s): {', '.join(sorted(unknown))}"
            )
        return v

    def visualize(
        self,
        figsize: tuple[int, int] = (16, 12),
        node_size: int = 2000,
        font_size: int = 10,
        show_legend: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 300,
    ) -> None:
        """
        Visualize the DAG with topologically sorted layers for clear execution flow.

        Parameters:
        -----------
        figsize : tuple[int, int]
            Figure size (width, height) in inches
        node_size : int
            Size of nodes in the graph
        font_size : int
            Font size for node labels
        show_legend : bool
            Whether to show a legend
        save_path : Optional[str]
            Path to save the plot (if None, displays the plot)
        dpi : int
            DPI for saved images
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for node_name in self.nodes.keys():
            G.add_node(node_name)

        # Add data flow edges
        for edge in self.data_flow_edges:
            G.add_edge(edge.source_stage, edge.destination_stage, edge_type="data_flow")

        # Add execution order dependency edges
        if self.exec_order_deps:
            for stage_name, deps in self.exec_order_deps.items():
                for dep in deps:
                    G.add_edge(
                        dep.stage_name,
                        stage_name,
                        edge_type="exec_order",
                        condition=dep.condition,
                    )

        # Create figure
        plt.figure(figsize=figsize)

        # Use topological sorting for layered layout
        try:
            # Get topological sort
            topo_order = list(nx.topological_sort(G))

            # Create layered layout
            layers = self._create_layered_layout(G, topo_order)
            pos = self._position_nodes_in_layers(layers, figsize)

        except nx.NetworkXError:
            # Fallback to spring layout if there are cycles
            print(
                "Warning: Graph contains cycles, using spring layout instead of layered layout"
            )
            pos = nx.spring_layout(G, k=3, iterations=50)

        # Define colors for different node types
        colors = []
        node_labels = {}

        entry_points_set: set[str] = set()
        exec_order_stages = set()
        if self.exec_order_deps:
            for deps in self.exec_order_deps.values():
                for dep in deps:
                    exec_order_stages.add(dep.stage_name)

        for node in G.nodes():
            node_labels[node] = node

            # Color coding:
            # - Entry points: Green
            # - Execution order dependency targets: Orange
            # - Regular nodes: Light blue
            if node in entry_points_set:
                colors.append("#2E8B57")  # Sea green for entry points
            elif node in exec_order_stages:
                colors.append("#FF8C00")  # Dark orange for exec order deps
            else:
                colors.append("#87CEEB")  # Sky blue for regular nodes

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color=colors, node_size=node_size, alpha=0.8
        )

        # Draw node labels
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, font_size=font_size, font_weight="bold"
        )

        # Draw edges with different styles based on type and condition
        data_flow_edges = [
            (u, v)
            for (u, v, d) in G.edges(data=True)
            if d.get("edge_type") == "data_flow"
        ]
        exec_order_edges = [
            (u, v)
            for (u, v, d) in G.edges(data=True)
            if d.get("edge_type") == "exec_order"
        ]

        # Draw data flow edges with curved routing
        if data_flow_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=data_flow_edges,
                edge_color="gray",
                width=2,
                arrows=True,
                arrowsize=20,
                arrowstyle="->",
                alpha=0.7,
                connectionstyle="arc3,rad=0.1",  # Slight curve to avoid overlaps
            )

        # Draw execution order edges with different colors based on condition
        if exec_order_edges:
            # Group edges by condition
            success_edges = []
            failure_edges = []
            always_edges = []

            for u, v in exec_order_edges:
                edge_data = G.get_edge_data(u, v)
                condition = edge_data.get("condition", "always")
                if condition == "success":
                    success_edges.append((u, v))
                elif condition == "failure":
                    failure_edges.append((u, v))
                else:  # always
                    always_edges.append((u, v))

            # Draw success edges (green) with different curve directions
            if success_edges:
                for i, (u, v) in enumerate(success_edges):
                    curve_rad = 0.1 + (i * 0.05) % 0.2  # Vary curve direction
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        edge_color="#228B22",  # Forest green
                        width=3,
                        arrows=True,
                        arrowsize=20,
                        arrowstyle="->",
                        alpha=0.8,
                        style="dashed",
                        connectionstyle=f"arc3,rad={curve_rad}",
                    )

            # Draw failure edges (red) with different curve directions
            if failure_edges:
                for i, (u, v) in enumerate(failure_edges):
                    curve_rad = -0.1 - (i * 0.05) % 0.2  # Vary curve direction
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        edge_color="#DC143C",  # Crimson red
                        width=3,
                        arrows=True,
                        arrowsize=20,
                        arrowstyle="->",
                        alpha=0.8,
                        style="dotted",
                        connectionstyle=f"arc3,rad={curve_rad}",
                    )

            # Draw always edges (purple) with different curve directions
            if always_edges:
                for i, (u, v) in enumerate(always_edges):
                    curve_rad = 0.15 + (i * 0.03) % 0.3  # Vary curve direction
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        edge_color="#8A2BE2",  # Blue violet
                        width=3,
                        arrows=True,
                        arrowsize=20,
                        arrowstyle="->",
                        alpha=0.8,
                        style="solid",
                        connectionstyle=f"arc3,rad={curve_rad}",
                    )

        # Add legend if requested
        if show_legend:
            legend_elements = [
                # Node types
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#2E8B57",
                    markersize=10,
                    label="Entry Points",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#FF8C00",
                    markersize=10,
                    label="Execution Order Dependencies",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#87CEEB",
                    markersize=10,
                    label="Regular Nodes",
                ),
                # Edge types
                plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=2,
                    label="Regular Dependencies",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="#228B22",
                    linewidth=3,
                    linestyle="--",
                    label="On Success Dependencies",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="#DC143C",
                    linewidth=3,
                    linestyle=":",
                    label="On Failure Dependencies",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="#8A2BE2",
                    linewidth=3,
                    linestyle="-",
                    label="Always After Dependencies",
                ),
            ]
            plt.legend(
                handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1)
            )

        plt.title(
            "DAG Specification Visualization (Topologically Sorted)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Graph saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def _create_layered_layout(
        self, G: nx.DiGraph, topo_order: List[str]
    ) -> List[List[str]]:
        """
        Create layered layout based on topological sort with better layer assignment.

        Parameters:
        -----------
        G : nx.DiGraph
            The directed graph
        topo_order : List[str]
            Topologically sorted node order

        Returns:
        --------
        List[List[str]] : List of layers, each containing nodes at the same level
        """
        # Calculate layer for each node based on longest path from entry points
        layers = {}
        entry_points: set[str] = set()

        # Without entry points, initialize layers from predecessors only

        # Calculate layer for each node
        for node in topo_order:
            if node not in layers:
                # Find the maximum layer of predecessors
                pred_layers = [
                    layers.get(pred, 0) for pred in G.predecessors(node)
                ]
                layers[node] = max(pred_layers) + 1 if pred_layers else 0

        # Group nodes by layer
        layer_groups = {}
        for node, layer in layers.items():
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append(node)

        # Return layers in order
        return [layer_groups[i] for i in sorted(layer_groups.keys())]

    def _position_nodes_in_layers(
        self, layers: List[List[str]], figsize: tuple[int, int]
    ) -> Dict[str, tuple[float, float]]:
        """
        Position nodes in layers with proper spacing to avoid edge overlaps.

        Parameters:
        -----------
        layers : List[List[str]]
            List of layers, each containing nodes at the same level
        figsize : tuple[int, int]
            Figure size for scaling

        Returns:
        --------
        Dict[str, tuple[float, float]] : Node positions
        """
        pos = {}
        width, height = figsize

        # Calculate layer spacing with more space between layers
        layer_height = height / (len(layers) + 2)  # Add extra space

        for layer_idx, layer in enumerate(layers):
            # Calculate y position for this layer
            y = height - (layer_idx + 1.5) * layer_height

            # Calculate node spacing within layer with more spread
            if len(layer) == 1:
                x_positions = [width / 2]
            else:
                # Use more of the width to spread nodes out
                margin = width * 0.1  # 10% margin on each side
                usable_width = width - 2 * margin
                spacing = (
                    usable_width / (len(layer) - 1) if len(layer) > 1 else 0
                )
                x_positions = [margin + spacing * i for i in range(len(layer))]

            # Assign positions to nodes in this layer
            for node_idx, node in enumerate(layer):
                pos[node] = (x_positions[node_idx], y)

        return pos

    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the DAG structure.

        Returns:
        --------
        Dict containing DAG statistics and structure information
        """
        G = nx.DiGraph()

        # Add nodes and edges
        for node_name in self.nodes.keys():
            G.add_node(node_name)

        for edge in self.data_flow_edges:
            G.add_edge(edge.source_stage, edge.destination_stage)

        entry_points_set: set[str] = set()
        exec_order_stages = set()
        exec_order_details = {}

        if self.exec_order_deps:
            for stage_name, deps in self.exec_order_deps.items():
                exec_order_details[stage_name] = []
                for dep in deps:
                    exec_order_stages.add(dep.stage_name)
                    exec_order_details[stage_name].append(
                        {
                            "dependency": dep.stage_name,
                            "condition": dep.condition,
                        }
                    )

        return {
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
            "entry_points": [],
            "execution_order_dependencies": list(exec_order_stages),
            "execution_order_details": exec_order_details,
            "is_dag": nx.is_directed_acyclic_graph(G),
            "has_cycles": not nx.is_directed_acyclic_graph(G),
            "max_parallel_stages": self.max_parallel_stages,
            "node_degrees": dict(G.degree()),
            "in_degrees": dict(G.in_degree()),
            "out_degrees": dict(G.out_degree()),
        }

from typing import Dict

from graphviz import Digraph

from evolutionary.history import SolutionHistoryKey, SolutionHistoryItem
from evolutionary_imaging.processing import get_images_for_candidate


def visualize_family_tree(history: Dict[SolutionHistoryKey, SolutionHistoryItem],
                          root_key: SolutionHistoryKey,
                          depth: int = 3,
                          engine: str = "dot",
                          format: str = "pdf") -> Digraph:
    """
    Recursively builds a Graphviz Digraph representing the family tree of a solution,
    using distinct nodes for the candidate's image and its short text description.

    Args:
        history: Dictionary mapping SolutionHistoryKey to SolutionHistoryItem.
        root_key: The starting candidate's key.
        depth: Maximum number of generations to traverse.
        engine: The Graphviz layout engine (e.g., "dot", "neato", "fdp").
        format: Output format (e.g., "pdf" to preserve images).
    Returns:
        A Graphviz Digraph object.
    """
    dot = Digraph(comment="Solution Family Tree", format=format, engine=engine)
    # Set bottom-to-top layout and reduce spacing.
    dot.attr(rankdir="BT")
    dot.attr(overlap="false", nodesep="0.1", ranksep="0.1")
    # Use plain nodes; we won't rely on HTML labels.
    dot.attr('node', shape="none", margin="0", fontsize="10", fontname="Helvetica")
    visited = set()

    def add_candidate(key: SolutionHistoryKey, current_depth: int):
        if current_depth < 0 or key not in history:
            return
        if key in visited:
            return
        visited.add(key)
        item = history[key]
        # Create distinct node IDs for the image and label nodes.
        candidate_img_id = key.short_str() + "_img"
        candidate_lbl_id = key.short_str() + "_lbl"

        # Group the two nodes in a subgraph to force same rank.
        with dot.subgraph() as s:
            s.attr(rank="same")
            image_paths = get_images_for_candidate(index=item.index,
                                                   generation=item.generation,
                                                   ident=item.ident)
            if image_paths:
                s.node(candidate_img_id,
                       label="",
                       image=image_paths[0],
                       shape="none",
                       fixedsize="true",
                       width="0.8",   # Adjust size as needed.
                       height="0.8")
            else:
                s.node(candidate_img_id, label="No Image", shape="box")
            # Use a plaintext label node; add a background if desired via the "style" attribute.
            s.node(candidate_lbl_id, label=item.short_str(), shape="plaintext", style="filled", fillcolor="white")
            # Connect them with an invisible edge with minlen=0 to force them to stick together.
            s.edge(candidate_img_id, candidate_lbl_id, dir="none", style="invis", weight="100", minlen="0")

        # Draw edges from parent's image node to this candidate's image node.
        if item.parent_1 is not None:
            parent1_img_id = item.parent_1.short_str() + "_img"
            dot.edge(parent1_img_id, candidate_img_id, label="P1", minlen="1")
            add_candidate(item.parent_1, current_depth - 1)
        if item.parent_2 is not None:
            parent2_img_id = item.parent_2.short_str() + "_img"
            dot.edge(parent2_img_id, candidate_img_id, label="P2", minlen="1")
            add_candidate(item.parent_2, current_depth - 1)

    add_candidate(root_key, depth)
    return dot
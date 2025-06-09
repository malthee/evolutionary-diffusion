import math
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.figure import Figure

from evolutionary.evolution_base import Fitness


def plot_fitness_statistics(
    num_generations: int,
    best_fitness: Optional[List[Fitness]] = None,
    worst_fitness: Optional[List[Fitness]] = None,
    avg_fitness:   Optional[List[Fitness]] = None,
    title: str = "Fitness Statistics over Generations",
    labels: Optional[List[str]] = None,
    multi_objective_plot_index: Optional[int] = None,
) -> Figure:
    """
    Plots fitness statistics over generations.
    *Back-compatible signature*: behaves exactly like the original.
    Returns the `matplotlib.figure.Figure`, so callers can `.savefig(...)`.
    """

    def _extract(fitness: List[Fitness]) -> List[float]:
        """Return the relevant (possibly single-objective) fitness list."""
        if multi_objective_plot_index is None:
            return fitness
        return [f[multi_objective_plot_index] for f in fitness]

    # --- Sanity check -------------------------------------------------------
    expected = num_generations
    for name, series in (
        ("best_fitness",  best_fitness),
        ("worst_fitness", worst_fitness),
        ("avg_fitness",   avg_fitness),
    ):
        if series is not None and len(series) != expected:
            raise ValueError(
                f"{name} has length {len(series)} but num_generations={expected}"
            )

    generations = list(range(num_generations))

    # --- Start drawing ------------------------------------------------------
    fig, ax = plt.subplots()

    if best_fitness is not None:
        ax.plot(
            generations,
            _extract(best_fitness),
            label=(
                [f"{lbl} Best Fitness" for lbl in labels]
                if labels else
                "Best Fitness"
            ),
        )

    if worst_fitness is not None:
        ax.plot(
            generations,
            _extract(worst_fitness),
            label=(
                [f"{lbl} Worst Fitness" for lbl in labels]
                if labels else
                "Worst Fitness"
            ),
        )

    if avg_fitness is not None:
        ax.plot(
            generations,
            _extract(avg_fitness),
            label=(
                [f"{lbl} Average Fitness" for lbl in labels]
                if labels else
                "Average Fitness"
            ),
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()  # make sure everything fits

    return fig


def plot_time_statistics(
    evaluation_time: List[float],
    creation_time:  List[float],
    post_evaluation_time: Optional[List[float]] = None,
    title: str = "Time Statistics over Generations in Seconds",
    *,
    overlap_threshold_pct: float = 8.0,
) -> Figure:
    """
    Pie-chart of time distribution with manual label placement:
      • slices ≥ overlap_threshold_pct get their %/value centered inside
      • smaller slices get their labels placed outside on staggered radii
        with leader lines, to avoid any overlap.
    """
    # 1) Prepare data
    labels = ["Evaluation", "Creation"] + (["Post Evaluation"] if post_evaluation_time else [])
    sizes  = [sum(evaluation_time), sum(creation_time)] + ([sum(post_evaluation_time)] if post_evaluation_time else [])
    total  = sum(sizes)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, _ = ax.pie(
        sizes,
        labels=None,
        startangle=90,
        radius=1.0,
        wedgeprops={"linewidth":1, "edgecolor":"white"},
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(
        wedges,
        labels,
        title="Phase",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.4, 1),
    )

    # 2) Decide which are “small”
    pctdistance = 0.80
    small_idxs = [i for i,s in enumerate(sizes) if 100*s/total < overlap_threshold_pct]

    # 3) Sort small slices by their mid‐angle (so stacking is predictable)
    angles = [ (wedges[i].theta1 + wedges[i].theta2)/2 for i in small_idxs ]
    small_order = [i for _,i in sorted(zip(angles, small_idxs))]

    # 4) Stagger settings
    base_out = 0.3    # base offset beyond pctdistance
    delta    = 0.15    # extra per‐slice radial step

    # 5) Draw every label
    for idx in range(len(wedges)):
        wedge = wedges[idx]
        size  = sizes[idx]
        pct   = 100.0 * size / total
        txt   = f"{pct:.1f}%\n({int(size)}s)"

        ang = (wedge.theta1 + wedge.theta2) / 2
        theta = math.radians(ang)
        cx, cy = wedge.center
        r = wedge.r

        if idx in small_order:
            # compute the rank (0,1,2…) in our sorted small list
            rank = small_order.index(idx)
            # start of leader: the slice edge
            x0, y0 = cx + r*math.cos(theta), cy + r*math.sin(theta)
            # radial position: pctdistance + base + rank*delta
            r_text = r*(pctdistance + base_out + rank*delta)
            x1, y1 = cx + r_text*math.cos(theta), cy + r_text*math.sin(theta)
            # leader line + label
            ax.plot([x0, x1], [y0, y1], color=wedge.get_facecolor(), lw=0.8)
            ax.text(x1, y1, txt, ha="center", va="center")
        else:
            # large slices: label inside at 60% of radius
            r_text = 0.6 * r
            x, y = cx + r_text*math.cos(theta), cy + r_text*math.sin(theta)
            ax.text(x, y, txt, ha="center", va="center")

    fig.tight_layout()
    return fig

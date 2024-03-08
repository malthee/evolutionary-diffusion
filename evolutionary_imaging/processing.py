from math import pi
from typing import List, Any, Optional, Tuple

import numpy as np
from PIL import Image
from evolutionary.evolution_base import SolutionCandidate, Fitness
from evolutionary_imaging.image_base import ImageSolutionData
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import glob
import imageio

"""
These functions are used to save and visualize images from the solution candidates of an evolutionary algorithm.
Images are saved in the RESULTS_FOLDER, have their fitness and index in the filename.
"""

RESULTS_FOLDER = "results"


def parse_fitness_from_filename(filename: str) -> Fitness:
    """
    Match "fitness_" followed by one or more groups of one or more digits
    (with optional decimal point and negative sign), separated by underscores
    """
    matches = re.findall(r"fitness_((-?\d+(\.\d+)?_?)+)\.png", filename)
    if matches:
        fitness_values = matches[0][0].split('_')
        if len(fitness_values) == 1:  # Single objective
            return float(fitness_values[0])
        else:
            return [float(value) for value in fitness_values]
    return 0.0


def parse_id_from_filename(filename: str) -> Optional[int]:
    """
    Match "id_" followed by one or more groups of one or more digits
    Returns None if no match
    """
    match = re.search(r"id_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def fitness_filename_sorting_key(filename: str) -> float:
    """
    Sorting key for filenames based on fitness. Used to sort images by fitness.
    For multi-objective optimization, the sum of the fitness values is used.
    """
    f = parse_fitness_from_filename(filename)
    if isinstance(f, list):  # If multi-objective, compute the sum, simple way to compare
        return sum(f)
    return f


def save_images_from_generation(population: List[SolutionCandidate[Any, ImageSolutionData, Fitness]],
                                generation: int, ident: Optional[int] = None) -> None:
    """
    Saves images from a given generation of solution candidates to the RESULTS folder with a sub-folder for
    the generation.

    Args:
    - population (List[SolutionCandidate[Any, ImageSolutionData]]): The population of solution candidates.
    - generation (int): The current generation number.
    - ident (Optional[int]): An id to add to the filename id_xxx. Useful for distinguishing between
    algorithms. Must be positive.
    """
    generation_dir = os.path.join(RESULTS_FOLDER, str(generation))
    os.makedirs(generation_dir, exist_ok=True)

    for index, candidate in enumerate(population):
        image_solution_data = candidate.result

        # Check if the fitness is single or multi-objective
        if isinstance(candidate.fitness, list):
            fitness_str = "_".join(f"{fit:.3f}" for fit in candidate.fitness)
        else:
            fitness_str = f"{candidate.fitness:.3f}"

        for i, image in enumerate(image_solution_data.images):
            image_name = f"{index}_{i}_fitness_{fitness_str}.png"
            if ident is not None:
                image_name = f"id_{ident}_{image_name}"
            image_path = os.path.join(generation_dir, image_name)
            image.save(image_path)


def create_generation_image_grid(generation: int, images_per_row: int = 5, max_images: Optional[int] = None,
                                 label_fontsize: int = 16,
                                 save_to_folder: bool = True, ident_mapper: Optional[list] = None) -> plt.Figure:
    """
    Creates a grid of images from a specific generation.
    Recommend for single-objective optimization or multi-objective optimization with 2 objectives.
    Otherwise, use create_generation_radar_chart_grid().

    Args:
    - generation (int): The generation number for which to create the image grid. Used as the title.
    - images_per_row (int): Number of images per row in the grid.
    - max_images (int): Maximum number of images to include in the grid. If None, all images in the generation folder
    will be included.
    - safe_to_folder (bool): Whether to save the figure to the generation folder. If False, the figure will only
    be returned
    - ident_mapper (Optional[dict]): A list mapping candidate indices to identifiers. If provided, the identifiers
    will be used as labels for the images in the grid.

    Returns:
    - plt.Figure: The matplotlib figure object containing the image grid.
    """
    generation_dir = os.path.join(RESULTS_FOLDER, f"{generation}")

    # Delete plot if already exists
    output_filepath = os.path.join(generation_dir, f"grid_{generation}.png")
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    image_files = glob.glob(os.path.join(generation_dir, "*.png"))
    image_files.sort(key=fitness_filename_sorting_key, reverse=True)

    if max_images is not None:
        image_files = image_files[:max_images]

    nrows = (len(image_files) + images_per_row - 1) // images_per_row
    fig = plt.figure(figsize=(15, 3 * (nrows + 0.5)))  # +1 for the title row
    gs = gridspec.GridSpec(nrows + 1, images_per_row, figure=fig, height_ratios=[0.5] + [3] * nrows)

    # Title row
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.5, f"Generation {generation}", fontsize=20, va='center', ha='center')
    ax_title.axis('off')

    # Image rows
    for i, img_file in enumerate(image_files):
        row = 1 + i // images_per_row  # Start from 1 to leave space for the title
        col = i % images_per_row
        ax = fig.add_subplot(gs[row, col])
        img = mpimg.imread(img_file)
        ax.imshow(img)
        ax.axis('off')
        fitness = parse_fitness_from_filename(img_file)
        ident = parse_id_from_filename(img_file)
        # Fitness may be single or multi-objective
        fitness_str = f"{fitness:.3f}" if not isinstance(fitness, list) else ", ".join(f"{fit:.1f}" for fit in fitness)
        # Optionally add identifier to the fitness string
        if ident is not None and ident_mapper is not None:
            fitness_str = ident_mapper[ident] + "\n" + fitness_str
        ax.set_title(fitness_str, fontsize=label_fontsize)

    plt.tight_layout(pad=1.5)
    if save_to_folder:
        plt.savefig(output_filepath)
    plt.close(fig)
    return fig


def _plot_radar_chart(ax: plt.Axes, scores: List[float], labels: List[str],
                      label_fontsize: int, label_padding: int) -> None:
    num_vars = len(scores)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]  # Complete the loop by duplicating the first score at the end
    angles += angles[:1]  # Complete the loop for angles

    ax.fill(angles, scores, color='red', alpha=0.25)
    ax.plot(angles, scores, color='red', linewidth=2)
    ax.set_xticks(angles[:-1])

    ax.set_xticklabels(labels, fontsize=label_fontsize)
    ax.tick_params(axis='x', which='major', pad=label_padding)
    ax.set_yticklabels([])


def create_generation_radar_chart_grid(generation: int, descriptors: Tuple[str, ...], images_per_row: int = 2,
                                       label_fontsize: int = 8, label_padding: int = 10,
                                       max_images: Optional[int] = None, save_to_folder: bool = True) -> plt.Figure:
    """
    Creates a grid of images from a specific generation, each accompanied by a radar chart
    visualizing multi-objective fitness values.

    Args:
    - generation (int): The generation number for which to create the grid.
    - descriptors (Tuple[str]): Descriptors for the radar chart, corresponding to fitness objectives.
    - images_per_row (int): Number of image-radar chart pairs per row in the grid.
    - max_images (Optional[int]): Maximum number of image-radar chart pairs to include in the grid.
    - save_to_folder (bool): Whether to save the figure to the generation folder.

    Returns:
    - plt.Figure: The matplotlib figure object containing the grid.
    """
    generation_dir = os.path.join(RESULTS_FOLDER, f"{generation}")
    output_filepath = os.path.join(generation_dir, f"grid_{generation}.png")

    # Cleanup previous file if exists
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    # Gather and sort image files by fitness
    image_files = glob.glob(os.path.join(generation_dir, "*.png"))
    image_files.sort(key=fitness_filename_sorting_key, reverse=True)

    if max_images is not None:
        image_files = image_files[:max_images]

    total_items = len(image_files)
    total_rows = (total_items + images_per_row - 1) // images_per_row
    fig = plt.figure(figsize=(images_per_row * 6, total_rows * 3.5))  # Adjust figsize as needed

    for idx in range(images_per_row * 2 * total_rows):
        col_idx = idx % (images_per_row * 2)  # Determine current column in the grid

        # Check if the current index corresponds to an available image file
        if idx // 2 < len(image_files):
            if col_idx % 2 == 0:  # Even index, plot image
                img_file = image_files[idx // 2]
                img_ax = fig.add_subplot(total_rows, images_per_row * 2, idx + 1)
                img = mpimg.imread(img_file)
                img_ax.imshow(img)
                img_ax.axis('off')
            else:  # Odd index, plot radar chart if there's a corresponding image
                fitness = parse_fitness_from_filename(image_files[idx // 2])
                scores = fitness if isinstance(fitness, list) else [fitness for _ in range(len(descriptors))]
                radar_ax = fig.add_subplot(total_rows, images_per_row * 2, idx + 1, polar=True)
                _plot_radar_chart(radar_ax, scores, list(descriptors), label_fontsize, label_padding)
        else:
            fig.add_subplot(total_rows, images_per_row * 2, idx + 1).axis('off')  # Hide unused plots

    fig.suptitle(f"Generation {generation}", fontsize=20)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    if save_to_folder:
        plt.savefig(output_filepath)
    plt.close(fig)
    return fig


def create_animation_from_generations(num_generations: int, output_file: str = "evolution.mp4",
                                      time_per_frame: int = 1000, time_last_frame: int = 5000, fps: int = 5) -> str:
    """
    Creates an animated MP4 video from the image grids of multiple generations using imageio with ffmpeg.

    Args:
    - num_generations (int): The number of generations to include in the animation.
    - output_file (str): The filename for the saved animation (preferably .mp4 for efficiency). Will be put in
    the RESULTS folder.
    - time_per_frame (int): Duration for each frame in the animation in milliseconds.
    - time_last_frame (int): Additional time for the last frame in milliseconds.
    - fps (int): Frames per second for the animation. Adjust this when using a different time_per_frame.
    """
    fps = fps

    output_location = os.path.join(RESULTS_FOLDER, output_file)
    writer = imageio.get_writer(output_location, fps=fps)

    # Calculate the number of times to append each frame
    frame_repetitions = time_per_frame // (1000 // fps)
    last_frame_repetitions = time_last_frame // (1000 // fps)

    last_frame = None
    for generation in range(num_generations):
        generation_path = os.path.join(RESULTS_FOLDER, f"{generation}", f"grid_{generation}.png")
        frame = imageio.v2.imread(generation_path)
        last_frame = frame

        # Append each frame according to its desired duration
        for _ in range(frame_repetitions):
            writer.append_data(frame)

    # Handle the last frame separately if there are any frames
    if last_frame is not None:
        for _ in range(last_frame_repetitions):
            writer.append_data(last_frame)

    writer.close()
    return output_location


def create_animation_from_generations_pil(num_generations: int, output_file: str = "evolution.gif",
                                          time_per_frame: int = 1000, time_last_frame: int = 5000) -> str:
    """
    First call create_generation_image_grid() for each generation.
    Creates an animated file from the image grids of multiple generations.
    Fallback when ffmpeg is not available.

    Args:
    - num_generations (int): The number of generations to include in the animation. Each one must have a grid image in
      its folder.
    - output_file (str): The filename for the saved animation.
    - time_per_frame (int): Duration for each frame in the animation (milliseconds).
    - time_last_frame (int): Duration for the last frame in the animation (milliseconds).
    """
    output_location = os.path.join(RESULTS_FOLDER, output_file)

    frames = []
    for generation in range(num_generations):
        generation_path = os.path.join(RESULTS_FOLDER, f"{generation}", f"grid_{generation}.png")
        frames.append(Image.open(generation_path))

    durations = [time_per_frame] * (len(frames) - 1) + [time_last_frame]
    frames[0].save(output_location, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    return output_location

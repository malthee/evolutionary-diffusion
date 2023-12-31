import warnings
from typing import List, Any
from PIL import Image
from evolutionary.evolution_base import SolutionCandidate
from evolutionary.image_base import ImageSolutionData
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import glob
import imageio

RESULTS_FOLDER = "results"


def _parse_fitness_from_filename(filename: str) -> float:
    match = re.search(r"fitness_([0-9.]+)\.png", filename)
    return float(match.group(1)) if match else 0


def save_images_from_generation(population: List[SolutionCandidate[Any, ImageSolutionData]], generation: int) -> None:
    """
    Saves images from a given generation of solution candidates.

    Args:
    - population (List[SolutionCandidate[Any, ImageSolutionData]]): The population of solution candidates.
    - generation (int): The current generation number.
    """
    generation_dir = os.path.join(RESULTS_FOLDER, f"{generation}")
    os.makedirs(generation_dir, exist_ok=True)

    for index, candidate in enumerate(population):
        image_solution_data = candidate.result
        fitness_str = f"{candidate.fitness:.3f}"

        for i, image in enumerate(image_solution_data.images):
            image_name = f"{index}_{i}_fitness_{fitness_str}.png"
            image_path = os.path.join(generation_dir, image_name)
            image.save(image_path)


def create_generation_image_grid(generation: int, images_per_row: int = 5, safe_to_folder: bool = True) -> plt.Figure:
    """
    Creates a grid of images from a specific generation.

    Args:
    - generation (int): The generation number for which to create the image grid. Used as the title.
    - images_per_row (int): Number of images per row in the grid.

    Returns:
    - plt.Figure: The matplotlib figure object containing the image grid.
    """
    generation_dir = os.path.join(RESULTS_FOLDER, f"{generation}")

    # Delete plot if already exists
    output_filepath = os.path.join(generation_dir, f"grid_{generation}.png")
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    image_files = glob.glob(os.path.join(generation_dir, "*.png"))
    image_files.sort(key=_parse_fitness_from_filename, reverse=True)

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
        fitness = _parse_fitness_from_filename(img_file)
        ax.set_title(f"{fitness:.3f}", fontsize=16)

    plt.tight_layout(pad=1.5)
    if safe_to_folder:
        plt.savefig(output_filepath)
    plt.close(fig)
    return fig


def create_animation_from_generations(num_generations: int, output_file: str = "ga_evolution.mp4",
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
        frame = imageio.imread(generation_path)
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


def create_animation_from_generations_pil(num_generations: int, output_file: str = "ga_evolution.gif",
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

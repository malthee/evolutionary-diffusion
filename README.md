# Evolutionary-Diffusion
*Combining Evolutionary Computing with Diffusion Models*

* 🎨 Aesthetics Maximization/Minimization using [LAION Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor)
* 📊 Multi-Objective Optimization with CLIP-IQA metrics
* 🛡️ Evading AI-Image Detection by optimizing against a [fine-tuned SDXL AI-Image-Detector](https://huggingface.co/Organika/sdxl-detector)
* 🧭 Navigating the CLIP-Score Landscape for Prompt-Matching

Goals: Augment the process of art generation, improve A-to-I ratio, explore possibilities of combination, optimize and automize.

## Try it out in Google Colab

| Notebook                 | Link                                                                                                                                                                                                          |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Genetic Algorithm        | [![Genetic Algorithm](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malthee/evolutionary-diffusion/blob/main/notebooks/ga_notebook.ipynb)               |
| Island Genetic Algorithm | [![Island Genetic Algorithm](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malthee/evolutionary-diffusion/blob/main/notebooks/island_ga_notebook.ipynb) |
| NSGA                     | [![Genetic Algorithm](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malthee/evolutionary-diffusion/blob/main/notebooks/nsga_notebook.ipynb)             |

Image results will be saved in your Google Drive in the folder `evolutionary`. Each generation
creates a new folder where the images will be saved in. You can change the folders in the notebook.

## Example - Creating the most Aesthetic Image

### Optimizing for Aesthetics using the Aesthetics Predictor V2 from LAION with a GA and SDXL-Turbo
Optimizing the aesthetics predictor as a maximization problem, the algorithm came to a max Aesthetics score of **8.67**.
This score is higher than [the examples from the real LAION English Subset dataset have](http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html), with the red line showing the limit.
A wide variety of prompts (inspired by parti prompts) was used for the initial population.

![Ga200Gen100PopFitnessChartAesthetics](https://github.com/malthee/evolutionary-diffusion/assets/18032233/9afe41f2-6ee8-4af0-bed1-2b0a77df6f3e)

Parameters: 
```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1

creator = SDXLPromptEmbeddingImageCreator(pipeline_factory=setup_pipeline, batch_size=batch_size, inference_steps=3)
evaluator = AestheticsImageEvaluator()  
crossover = PooledArithmeticCrossover(0.5, 0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=2, clamp_range=(-900, 900)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.3, clamp_range=(-8, 8))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)
```

## Example - Island GA with Artists on each Island
Performing an Island GA by creating random embeddings and mixing them with artist embeddings to get mixtures of styles and new ideas.

|    |    |    |    |
| --- | --- | --- | --- |
| ![Mark Rothko chairs](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/161643_6_0_fitness_5.871.png) | ![Sketching Person Picasso](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/230414_3_0_fitness_6.594.png) | ![Dali Angles Crazy](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/154957_4_0_fitness_5.895.png) | ![Landscape Van Gogh](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/163139_6_0_fitness_6.788.png) |  
| ![Character Walls Unique](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/193932_8_0_fitness_4.660.png) | ![Pattern Colorful](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/194326_6_0_fitness_4.825.png) | ![Woman Butterfly Landscape](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/055009_5_0_fitness_7.375.png) | ![Green Car City](https://github.com/malthee/evolutionary-diffusion-results/blob/main/interesting_images/island_aesthetics/054934_8_0_fitness_6.767.png)

## Detailed Results and Notebooks
More detailed results can be found in a separate repository dedicated to the results of the experiments:
https://github.com/malthee/evolutionary-diffusion-results

## Evaluators
* AestheticsImageEvaluator: Uses the [LAION Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor). Blog: https://laion.ai/blog/laion-aesthetics/
* CLIPScoreEvaluator: Using the [torchmetrics implementation for CLIP-Score](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html)
* (Single/Multi)CLIPIQAEvaluator: Using the [torchmetrics implementation for CLIP Image Quality Assessment](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html).
* AIDetectionImageEvaluator: Using the [original Version from HuggingFace](https://huggingface.co/umm-maybe/AI-image-detector), or the [fine-tuned one for SDXL generated images](https://huggingface.co/Organika/sdxl-detector)

## Image Creators
Current supported creators working in the prompt embedding space:
* SDXLPromptEmbeddingImageCreator: Supports the SDXL pipeline, creates both prompt- and pooled-prompt-embeddings.
* SDPromptEmbeddingImageCreator: Only has prompt-embeddings, is faster but produces less quality results than SDXL.

## Package Structure and Base Classes
![Package Diagram](https://github.com/malthee/evolutionary-diffusion/assets/18032233/4943b577-faa9-45ce-8f8a-b781e65734be)

![Solution Candidate Class Diagram](https://github.com/malthee/evolutionary-diffusion/assets/18032233/3f334c9c-b5b2-4ecc-914f-485e89fada32)

## (Pre-Testing) Evaluating Models for Evolutionary use
There are multiple notebooks exploring the speed and quality of models for generation and fitness-evaluation. 
These notebooks also allow for simple inference so that any model can be tried out easily.

* diffusion_model_comparison: tries out different diffusion models with varying arguments (inference steps, batch size) to find out the optimal model for image generation in an evolutionary context (generation speed & quality)
* clip_evaluators: uses torch metrics with CLIPScore and CLIP IQA. CLIPScore could define the fitness for "prompt fulfillment" or "image alignment" while CLIP IQA has many possible metrics like "quality, brightness, happiness..."
* ai_detection_evaluator: uses a pre-trained model for AI image detection. This could be a fitness criteria to minimize "AI-likeness" in images.
* aesthetics_evaluator: uses a pre-trained model from the maintainers of the LAION image dataset, which scores an image 0-10 depending on how "aesthetic" it is. Could be used as a maximization criteria for the fitness of images.
* clamp_range: testing the usual prompt-embedding min and max values for different models, so that a CLAMP range can be set in the mutator for example. [Using the parti prompts.](https://github.com/rromb/parti-prompts)https://github.com/rromb/parti-prompts
* crossover_mutation_experiments: testing different crossover and mutation strategies to see how they work in the prompt embedding space


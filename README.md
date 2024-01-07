# Evolutionary-Diffusion
*Combining Evolutionary Computing with Diffusion Models*

Goals: Augment the process of art generation, improve A-to-I ratio, explore possibilities of combination, optimize and automize.

## Results 
### Simple Genetic Algorithm with SDXL-Turbo 
This was one of the first experiments. Using a small population size and number of generations, the algorithm was able to optimize the Aesthetic fitness from
start ~7.5 to ~7.8. The video shows the evolution of prompt embeddings.

Prompt: "amazing breathtaking painting weltwunder city landscape famous"  
Video:   

https://github.com/malthee/evolutionary-diffusion/assets/18032233/a6b58313-79ba-4965-b7e3-15e3e80b479a


Arguments: 
```python
population_size = 10
num_generations = 30 
creator = SDXLPromptEmbeddingImageCreator(pipeline=pipe, batch_size=1, inference_steps=3)
evaluator = AestheticsImageEvaluator()  
crossover = PooledArithmeticCrossover(crossover_rate=0.5, crossover_rate_pooled=0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=3, clamp_range=(-900, 900))
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.5, clamp_range=(-8, 8))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)
```

## Installation & Environment
SOON Collab-Compatible!  
Python, Torch...

## Evaluators
* AestheticsImageEvaluator: Uses the LAION Aesthetics Predictor V2. Source: https://laion.ai/blog/laion-aesthetics/ and GitHub https://github.com/christophschuhmann/improved-aesthetic-predictor. 

## Image Creators
The first experiments were made with PromptEmbeddingImageCreators, which extract the 
logic of text tokenization and encoding from the DiffusionPipeline. This allows
the evolutionary algorithm to move in the prompt embedding space. Other possible 
image creators could work with prompts themselves, as a form of Genetic Programming or also
with image embeddings. Current supported creators:
* SDXLPromptEmbeddingImageCreator: Supports the SDXL pipeline, creates both prompt- and pooled-prompt-embeddings.
* SDPromptEmbeddingImageCreator: In the works, only has prompt-embeddings.

## Class Diagrams


## (Pre-Testing) Evaluating Models for Evolutionary use
There are multiple notebooks exploring the speed and quality of models for generation and fitness-evaluation. 
These notebooks also allow for simple inference so that any model can be tried out easily.

* diffusion_model_comparison: tries out different diffusion models with varying arguments (inference steps, batch size) to find out the optimal model for image generation in an evolutionary context (generation speed & quality)
* clip_evaluators: uses torch metrics with CLIPScore and CLIP IQA. CLIPScore could define the fitness for "prompt fulfillment" or "image alignment" while CLIP IQA has many possible metrics like "quality, brightness, happiness..."
* ai_detection_evaluator: uses a pre-trained model for AI image detection. This could be a fitness criteria to minimize "AI-likeness" in images.
* aesthetics_evaluator: uses a pre-trained model from the maintainers of the LAION image dataset, which scores an image 0-10 depending on how "aesthetic" it is. Could be used as a maximization criteria for the fitness of images.



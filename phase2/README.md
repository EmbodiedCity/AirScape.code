# Phase-2

Here we provide the detailed instruction on how Airscape's phase-2 training is carried out and how it should be used.

## Installation

Enter each subfolders to see the exact dependencies needed.


## Self-play loop

Our idea is to iteratively use this loop to let the model we get in phase-1 automatically improved based on the guidance from a MoE teacher.

### 1. Prompts generation

Use mature VLM to generate 8 prompts based on the given frame. Then these data will be used in model inference.

Open [prompts_generate](https://github.com/EmbodiedCity/AirScape.code/edit/main/phase2/prompts_generate) to see more details.

### 2. Inference(based on Phase 1 model)

Use phase-1 airscape to get outcomes based on different prompts and seeds, which boosts diversity that allows for foreseeable capbability for evolution.

### 3. Discriminator

This discriminator acts as a MoE teacher that leads the model to get stronger.

## Usage



## Contributing



## License




## Acknowledgements

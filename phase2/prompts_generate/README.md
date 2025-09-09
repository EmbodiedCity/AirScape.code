# Prompts generation guide

We use VLM to help generate prompts based on the given single frames extracted from arbitrary videos

The usage of this part is quite easy, just make sure you have your target model's api key. The detailed api uploding method varies based on different chosen models.

## Install dependencies

Change to fit your needs.
```
conda create -n prompt
conda activate prompt
cd prompts_generate
pip install -r requirements_gemini.txt
# or pip install -r requirements_gpt.txt
```

## How to get frames

Specify the target video directory and output directory that stores the samples framse, and use:
```
python get_random_frames.py
```

## How to generate prompts

We use gemini-2.5-flash/gpt-4o to generate prompts
> Remember the api key is needed and the model can be chosen as you like

Just use the following script

```
python prompts_generate.py
```


# Discriminator used in self-play

The core idea is to get throught 2 levels of quality checking to let the discriminator select the best sample among all the outputs videos.

For one, we check if the trajectories obey our instruction, that is, Instruction Following ability. According to empirical observation, in high probability, the most of the generated videos basically follow the input prompt. So we firstly discarded those obviously bad-traj outliers by isolating anomaly based on VGGT data.

For another, we use video quality benchmark to see how well the Temporal-Spacial Continuity and basic Physical Consistency is.

## 1. Install dependencies

Since the heterogeneity of the development environments, you can see the dependencies in each subfolder to satisfy the specific requirements respectively.

## 1. Prepare the metrics

### Trajectory similarity metric

To get the trajectory similarity, we resorted to VGGT, the 3D-reconstruction model. To see detailed infomation, go through [README.md](traj_similarity_metric/README.md) in traj_similarity_metric directory.

### Video quality metrics

### Put all together

Run the following script

```
python discriminator_model/prepare_csv_data.py
```

to generate the source data with all 5 metrics before training and inference.


## 2. Training

We adopted isolation forest algorithm to get the outliers. In discriminator_model directory, you will see how the dataset was built up and how the simple training was carried out.

Human-annotated labels should be added into the source CSV data when training as you can see in the `dataset.py`. 

Run the following script

```
python discriminator_model\train\dataset.py
``` 

and then the whole dataset will be splited into training set, validation set and test set according to the proportion of 6:1:1.


At last, run the following script

```
python discriminator_model\train\train.py
```

to simply get the trained model.

## 3. Best selection


Finally, since we already get the CSV data that contains each video's 5 metrics as well as the trained model, we can use the model to select the best quality video in a tournament-like mode and feed the winner into the next self-play loop. 

Just run the following script

```
python discriminator_model\run_model.py
```







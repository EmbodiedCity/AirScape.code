# Instruction on discriminator model

## Dependencies

Here we use basically the same python environment dependencies as traj_similarity_metric, you can access to it via [README.md](best_selection\traj_similarity_metric\README.md).

Of course, we don't need COLMAP and VGGT here.

## Cautions

1. Please be careful about the exact CSV content format, specifically the column names.
2. We tried and compared several simple models, and selected random forest as the most robust and outperforming chioce. This, however, can vary according to your needs.
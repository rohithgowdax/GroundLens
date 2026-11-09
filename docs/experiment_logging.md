# Experiment Logging

This project uses [Weights & Biases (WandB)](https://wandb.ai/) for experiment tracking and visualization. Training scripts include an argument for experiment name, so that each of experiment can be structured into projects. We use it to log performance metrics and structure our experiments. 


![Projects](../assets/wandb_illustrations/projects.png)

![Runs Logged](../assets/wandb_illustrations/runs_logged.png)

## Setup

1. Create a WandB account at https://wandb.ai/
2. Add your API key to the `.env` file:

```bash
WANDB_API_KEY=your_wandb_api_key_here
```

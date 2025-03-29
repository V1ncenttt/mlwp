import yaml
import wandb

# Load YAML config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Init wandb
wandb.init(
    project=config["project"],
    entity=config["entity"],
    name=config["name"],
    config=config
)

# Access values
lr = config["train"]["learning_rate"]
model_type = config["model"]["type"]

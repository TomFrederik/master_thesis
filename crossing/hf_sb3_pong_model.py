from huggingface_sb3 import load_from_hub
checkpoint = load_from_hub(
	repo_id="sb3/ppo-PongNoFrameskip-v4",
	filename="{MODEL FILENAME}.zip",
)
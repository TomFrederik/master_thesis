import stable_baselines3 as sb3




model = sb3.PPO(policy, env)
model.learn(total_timesteps=10000)


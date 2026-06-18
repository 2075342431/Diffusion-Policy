import torch
import gymnasium as gym
import mani_skill
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion import DiffusionPolicy
import cv2

CKPT_DIR = "outputs/train/pickcube_diffusion/checkpoints/050000/pretrained_model"
policy = DiffusionPolicy.from_pretrained(CKPT_DIR).to("cuda")
policy.eval()

env = gym.make("PickCube-v1", obs_mode="rgb+depth", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
obs, info = env.reset()
policy.reset()

rgb = obs["sensor_data"]["base_camera"]["rgb"]
state = obs["agent"]["qpos"]
print(f"Init rgb shape: {rgb.shape}, type: {type(rgb)}")
print(f"Init state shape: {state.shape}, type: {type(state)}")

if rgb.ndim == 4: rgb = rgb[0]
if state.ndim == 2: state = state[0]
if isinstance(rgb, torch.Tensor): rgb = rgb.cpu().numpy()
if isinstance(state, torch.Tensor): state = state.cpu().numpy()

rgb_resized = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA)
img_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
state_tensor = torch.from_numpy(state).float()

batch = {
    "observation.images.base_camera": img_tensor.unsqueeze(0).to("cuda"),
    "observation.state": state_tensor.unsqueeze(0).to("cuda")
}
print(f"Batch img shape: {batch['observation.images.base_camera'].shape}")
print(f"Batch state shape: {batch['observation.state'].shape}")

action = policy.select_action(batch)
print(f"Policy output action: {action.shape}, type: {type(action)}")
print(f"Action values: {action}")

# Now step the environment
action_np = action.cpu().numpy()
if action_np.ndim == 1: action_np = action_np[np.newaxis, :]
# Wait, should we pass numpy or tensor? Let's try passing the tensor directly!
action_tensor = action.clone()
if action_tensor.ndim == 1: action_tensor = action_tensor.unsqueeze(0)
obs, reward, term, trunc, info = env.step(action_tensor)
print(f"Reward after step: {reward}")


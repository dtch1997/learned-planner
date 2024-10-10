# %%
from cleanba.environments import BoxobanConfig
from cleanba import cleanba_impala
from learned_planner.interp.utils import jax_to_th, load_jax_model_to_torch
import pathlib
import torch
from learned_planner.interp.utils import load_jax_model_to_torch
from huggingface_hub import snapshot_download

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000/" # DRC(3, 3) 2B checkpoint
MODEL_BASE_PATH = pathlib.Path(
    snapshot_download("AlignmentResearch/learned-planner", allow_patterns=[MODEL_PATH_IN_REPO + "*"]),
) # only download the specific model
MODEL_PATH = MODEL_BASE_PATH / MODEL_PATH_IN_REPO

BOXOBAN_CACHE = 'boxoban-cache'

env = BoxobanConfig(
    cache_path= BOXOBAN_CACHE,
    num_envs=1,
    max_episode_steps=120,
    asynchronous=False,
    tinyworld_obs=True,
).make()

#jax_policy, carry_t, jax_args, train_state, _ = cleanba_impala.load_train_state(MODEL_PATH, env)
cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, env)

# %%
from torchinfo import summary

summary(policy_th)

# %%
# observation is of shape (1, 3, 10, 10)?
import matplotlib.pyplot as plt
obs, info = env.reset()
import einops
plt.imshow(einops.rearrange(obs[0], "c h w -> h w c"))

# %%
obs, info = env.reset()
obs = torch.Tensor(obs)
state = policy_th.recurrent_initial_state(1)
episode_starts = torch.Tensor([1]).bool()
action, value, logprobs, state = policy_th(obs, state, episode_starts)


policy_th.mlp_extractor
policy_th.mlp_extractor.policy_net
action, value, logprobs, state = policy_th(obs, state, episode_starts)


# %%
#init state is a list of three tensors of shape torch.Size([1?, 1?, 32, 10, 10]) full of zeros?
# initial_state = policy_th.recurrent_initial_state()
# episode_starts = torch.Tensor([1]).bool()
# policy_th(torch.Tensor(obs), initial_state, episode_starts)

# %%
obs, info = env.reset()
obs = torch.Tensor(obs)
state = policy_th.recurrent_initial_state(1)
episode_starts = torch.Tensor([1]).bool()
action, value, logprobs, state = policy_th(obs, state, episode_starts)

# %%
def show_obs(obs):
    obs = einops.rearrange(obs.squeeze(), 'c h w -> h w c')
    plt.imshow(obs)
    plt.show()


obs, info = env.reset()
obs = torch.Tensor(obs)
state = policy_th.recurrent_initial_state(1)
episode_starts = torch.Tensor([1]).bool()

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def play_sokoban(env, policy_th, num_steps=100):
    obs, info = env.reset(seed=65)
    obs = torch.Tensor(obs)
    state = policy_th.recurrent_initial_state(1)
    episode_starts = torch.Tensor([1]).bool()

    frames = []
    values = []
    actions = []

    for step in range(num_steps):
        with torch.no_grad():
            action, value, logprobs, next_state = policy_th(obs, state, episode_starts, deterministic=True)
        
        action_int = action.item()
        obs, reward, done, truncated, info = env.step([action_int])
        
        obs = torch.Tensor(obs)
        state = next_state
        episode_starts = torch.Tensor([0]).bool()
        
        frames.append(obs.squeeze().permute(1, 2, 0).numpy())
        values.append(value.item())
        actions.append(action_int)
        
        if done or truncated:
            break

    return frames, values, actions

def plot_sokoban_game(frames, values, actions):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], 
                        specs=[[{"type": "image"}, {"type": "scatter"}]])

    fig.add_trace(go.Image(z=frames[0]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[values[0]], mode='lines+markers'), row=1, col=2)

    # Calculate initial y-axis range
    y_min, y_max = min(values), max(values)
    y_range = y_max - y_min
    y_min -= 0.1 * y_range  # Add 10% padding
    y_max += 0.1 * y_range

    fig.update_layout(
        title="Sokoban Agent Play",
        xaxis2=dict(title="Step", range=[0, len(frames)]),
        yaxis2=dict(title="Value", range=[y_min, y_max]),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="⏮", method="animate", args=[["previous"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
                    dict(label="⏭", method="animate", args=[["next"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
                    dict(label="▶", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [i],
                        {"frame": {"duration": 0, "redraw": True},
                         "mode": "immediate",
                         "transition": {"duration": 0}}
                    ],
                    "label": str(i),
                    "method": "animate"
                } for i in range(len(frames))
            ]
        }]
    )

    fig_frames = []
    for i in range(len(frames)):
        # Update y-axis range if necessary
        current_values = values[:i+1]
        y_min = min(min(current_values), y_min)
        y_max = max(max(current_values), y_max)
        y_range = y_max - y_min
        # y_min -= 0.1 * y_range  # Add 10% padding
        # y_max += 0.1 * y_range

        frame = go.Frame(
            data=[
                go.Image(z=frames[i]),
                go.Scatter(x=list(range(i+1)), y=current_values, mode='lines+markers')
            ],
            layout=go.Layout(
                title_text=f"Step {i+1}, Action: {actions[i]}",
                yaxis2=dict(range=[y_min, y_max])
            ),
            name=str(i)
        )
        fig_frames.append(frame)

    fig.frames = fig_frames
    return fig

obs, info = env.reset()
frames, values, actions = play_sokoban(env, policy_th)
fig = plot_sokoban_game(frames, values, actions)
fig.show()

# %%




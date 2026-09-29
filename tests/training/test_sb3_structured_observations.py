import numpy as np
import pytest
import torch

sb3_contrib = pytest.importorskip("sb3_contrib")

from skat_rl.training.train_sb3_ppo import _make_vec_env


def test_maskable_ppo_trains_and_predicts_with_dict_observations(tmp_path):
    torch.set_num_threads(1)
    env = _make_vec_env({"learning_player": 2, "fixed_declarer": 2, "seed": 17}, 2)
    try:
        model = sb3_contrib.MaskablePPO(
            "MultiInputPolicy", env, n_steps=12, batch_size=12, n_epochs=1,
            policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}}, device="cpu",
        )
        model.learn(total_timesteps=48)
        path = tmp_path / "model.zip"
        model.save(path)
        loaded = sb3_contrib.MaskablePPO.load(path, env=env, device="cpu")
        observation = env.reset()
        masks = np.asarray(env.env_method("action_masks"))
        actions, _ = loaded.predict(observation, action_masks=masks, deterministic=True)
        assert masks[np.arange(2), actions].all()
        assert len(model.ep_info_buffer) >= 2
        assert observation["card_status"].shape == (2, 32)
    finally:
        env.close()

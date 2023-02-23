import numpy as np
import torch


def load_test_anim(filename, device):
    anim = np.load(filename)
    anim = torch.tensor(anim, device=device, dtype=torch.float)
    poses = anim[:, :-3]
    assert poses.shape[1] == 72
    loc = anim[:, -3:]
    loc[..., 1] += 1.1174
    loc = loc.unsqueeze(1)

    return poses, loc


def load_cmu_mocap_anim(filename, device):
    # 72 of 156 poses are picked from
    # https://github.com/NetEase-GameAI/MoCap-Solver/blob/main/SyntheticDataGeneration/generate_test_data.py#L139
    cdata = torch.from_numpy(np.load(filename)["poses"][:, :])
    N, D = cdata.shape
    assert D == 156

    poses = torch.zeros((N, 72), device=device, dtype=torch.float)
    poses[:, :66] = cdata[:, :66]
    poses[:, 66:69] = cdata[:, 75:78]
    poses[:, 69:72] = cdata[:, 120:123]

    return poses

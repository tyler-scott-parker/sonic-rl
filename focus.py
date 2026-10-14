"""
focus.py — what the base-game agent is paying attention to, for the viewer
(watch_main.py --focus, V cycles it).

The network has no gaze to read off: it is three convolutions over the
84x84 grayscale picture (the last 4 agent frames). So this asks it instead
(perturbation saliency, Greydanus et al. 2018, "Visualizing and Understanding
Atari Agents"): blur one small patch of what it sees, in all 4 stacked
frames, and measure how much that moves
  - its decision (orange): its scores for the 9 button combinations (the
    log-probabilities, their common shift removed), squared distance; and
  - its expectation (cyan): the critic's estimate of what's coming.
Done for a patch centred every ~5 pixels (a 17 x 17 grid of patches),
one batch through the network, then smoothed back up to 84 x 84. A bright
spot = "this moves what I'd press" (decision) or "this tells me how well
things will go" (expectation).

Why the scores and not the change in its action probabilities (KL): mg2 at
2.58M steps on AIZ1 from the start, 300 steps: the KL map was ~blank on 75%
of steps (a confident decision doesn't flip when one patch blurs) and its peak
varied 366x from step to step; the scores' peak varied 4x and was never blank.

Caveat for whoever reads the picture: a blurred patch is something the
network never saw in training, so a bright spot means "the network reacts to
this changing", which is not proof it uses it. The HUD (always crisp in
training) lights up a lot: the ring count (a hit kills at 0) and the clock
(its pace against Tyler's) would be real reasons, but an in-distribution test
(change the RAM, not the pixels) is what would show it.

Scale: each map is divided by a running average of the recent passes' peaks,
not by its own peak: a typical moment shows at full strength, a quiet one
looks quiet instead of having its noise stretched to full brightness, a
striking one saturates. (A decaying running maximum was tried first: peaks run
2-4x a typical pass, so typical moments fell under the outline and vanished.)

Device: the GPU when there is one (a pass is a few ms there, so the map is
ready within the agent step it explains); on one CPU thread a pass is ~90 ms
(~135 ms beside training), so the viewer's overlay trails the picture by a
couple of steps, ~100 px at full speed. The viewer runs it on a thread of its
own either way. GPU: never while Tyler is gaming (it's a viewer of a training
run, which doesn't run then either).
"""
import copy

import cv2
import numpy as np
import torch

OBS = 84
N = 17                              # a 17 x 17 grid of patches, one every ~5 px
SIGMA_MASK = 5.0                    # the patch: a Gaussian of this radius (px of the 84 x 84)
SIGMA_BLUR = 3.0                    # what replaces it: the picture blurred this much
TRACK = 0.05                        # the scale: an average of the passes' peaks (~20 passes)
CENTRES = (np.arange(N) + 0.5) * OBS / N - 0.5      # where cv2.resize puts the grid's cells
_yy, _xx = np.mgrid[:OBS, :OBS]
MASKS = np.stack([np.exp(-((_yy - cy) ** 2 + (_xx - cx) ** 2) / (2 * SIGMA_MASK ** 2))
                  for cy in CENTRES for cx in CENTRES]).astype(np.float32)[:, None]     # (289, 1, 84, 84)

COLOURS = {'decision': (255, 140, 0), 'value': (0, 220, 255)}
MODES = ['off', 'decision', 'value', 'both']        # V cycles these


class Focus:
    """f.use(model); f(obs) -> {'decision': map, 'value': map}, each 84 x 84 in [0, 1]."""

    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.masks = torch.as_tensor(MASKS, device=self.device)
        self.scale = {'decision': None, 'value': None}
        self.src = self.policy = None

    def use(self, model):
        """The model to explain (its own copy on the GPU, made again when the model changes)."""
        if model is not self.src:
            self.src = model
            self.policy = (model.policy if self.device.type == 'cpu'
                           else copy.deepcopy(model.policy).to(self.device))
            self.policy.set_training_mode(False)

    @torch.no_grad()
    def raw(self, obs):
        """The unscaled 17 x 17 grids for one observation (84, 84, 4) uint8, as the env gives it."""
        img = obs.astype(np.float32)
        blur = cv2.GaussianBlur(img, (0, 0), SIGMA_BLUR)               # each stacked frame
        img, blur = (torch.as_tensor(x, device=self.device).permute(2, 0, 1)[None] for x in (img, blur))
        t = torch.cat([img, img * (1 - self.masks) + blur * self.masks])   # (290, 4, 84, 84)
        p = self.policy
        feats = p.extract_features(t)                                  # / 255 inside
        if p.share_features_extractor:
            latent_pi, latent_vf = p.mlp_extractor(feats)
        else:
            latent_pi = p.mlp_extractor.forward_actor(feats[0])
            latent_vf = p.mlp_extractor.forward_critic(feats[1])
        logp = torch.log_softmax(p.action_net(latent_pi), -1)          # (290, 9)
        scores = logp - logp.mean(-1, keepdim=True)                    # the common shift removed
        value = p.value_net(latent_vf)[:, 0]
        return {'decision': (0.5 * ((scores[1:] - scores[0]) ** 2).sum(-1)).cpu().numpy().reshape(N, N),
                'value': (value[1:] - value[0]).abs().cpu().numpy().reshape(N, N)}

    def __call__(self, obs):
        out = {}
        for name, grid in self.raw(obs).items():
            peak, sc = float(grid.max()), self.scale[name]
            self.scale[name] = sc = peak if sc is None else sc + TRACK * (peak - sc)
            up = cv2.resize(grid, (OBS, OBS), interpolation=cv2.INTER_CUBIC)
            out[name] = np.clip(up / max(sc, 1e-9), 0, 1)
        return out


def overlay(maps, mode, size=(320, 224), alpha=0.55, level=0.5, line=1):
    """An RGBA picture (h, w, 4) to lay over the game picture of size (w, h)
    (the viewer makes it at the game's 320 x 224 and scales it with the
    picture): each shown map as a tint that deepens with the map, outlined
    where it passes `level`. The 84 x 84 covers the whole screen (the
    observation is the whole screen squeezed), so stretching it back lines
    it up."""
    w, h = size
    out = np.zeros((h, w, 4), np.uint8)
    names = ['decision', 'value'] if mode == 'both' else [mode]
    for name in names:
        big = cv2.resize(maps[name], (w, h), interpolation=cv2.INTER_LINEAR)
        a = np.clip((big - 0.15) / 0.85, 0, 1) ** 1.5 * alpha * 255
        tint = a > out[..., 3]
        out[tint, :3] = COLOURS[name]
        out[..., 3] = np.maximum(out[..., 3], a.astype(np.uint8))
        lines, _ = cv2.findContours((big >= level).astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, lines, -1, (*COLOURS[name], 255), line, cv2.LINE_AA)
    return out


def eye(obs, maps, mode, px):
    """What the agent sees (the newest of its 4 frames, 84 x 84 gray, with
    the air bar and ring compass drawn in), px x px, with the same overlay,
    as RGB (px, px, 3)."""
    g = cv2.resize(obs[..., -1], (px, px), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if mode != 'off':
        o = overlay(maps, mode, (px, px), line=2).astype(np.float32)
        a = o[..., 3:] / 255
        rgb = rgb * (1 - a) + o[..., :3] * a
    return rgb.astype(np.uint8)

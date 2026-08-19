#!/usr/bin/env python3
"""
make_states.py — Records special stage save states at natural speed increase points.

Usage:
    python3 make_states.py --stage 1 --output ~/sonic-rl/states/

This lets the game run naturally until each speed tier is reached,
then captures the state. No RAM patching — everything is authentic.

Speed increases every 1800 frames (30 seconds at 60fps).
Speed tiers: 0 (start), 1 (~1800f), 2 (~3600f), 3 (~5400f)
"""

import os
import argparse
import stable_retro

# Speed tier frame targets (with small buffer to ensure timer has reset)
SPEED_TIER_FRAMES = {
    0: 0,      # start of stage
    1: 1850,   # just past first speed increase
    2: 3700,   # just past second speed increase
    3: 5550,   # just past third speed increase
}

STAGE_STATES = {
    1: 'curriculum_stage_1.state',
    2: 'curriculum_stage_2.state',
    3: 'curriculum_stage_3.state',
    4: 'curriculum_stage_4.state',
    5: 'curriculum_stage_5.state',
    6: 'curriculum_stage_6.state',
    7: 'curriculum_stage_7.state',
}


def make_state(base_state_path, output_dir, stage, tier, frames):
    """Run the stage for `frames` steps with no-op, then save state."""
    env = stable_retro.make(
        game='SonicAndKnuckles3-Genesis-v0',
        use_restricted_actions=stable_retro.Actions.FILTERED
    )
    env.reset()

    with open(base_state_path, 'rb') as f:
        env.em.set_state(f.read())

    print(f"  Running {frames} frames...")
    prev_speed_timer = None
    speed_level = 0

    for i in range(frames):
        _, _, done, _, info = env.step([0] * 12)

        # Track speed increases for verification
        speed_timer = info.get('ss_speed_timer', None)
        if speed_timer is not None and prev_speed_timer is not None:
            if speed_timer > prev_speed_timer + 100:
                speed_level += 1
                print(f"  Speed increased to level {speed_level} at frame {i}")
        prev_speed_timer = speed_timer

        if done:
            print(f"  WARNING: Stage ended unexpectedly at frame {i}")
            break

    # Verify speed level matches expected tier
    if speed_level != tier:
        print(f"  WARNING: Expected speed tier {tier}, got {speed_level}")
    else:
        print(f"  Speed tier {tier} confirmed.")

    # Save state
    state = env.em.get_state()
    filename = f'stage_{stage}_speed{tier}.state'
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as f:
        f.write(state)

    print(f"  Saved: {output_path}")
    env.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Record SS save states at natural speed tiers')
    parser.add_argument('--stage', type=int, default=1, choices=range(1, 8),
                        help='Special stage number (1-7)')
    parser.add_argument('--output', type=str,
                        default=os.path.expanduser('~/sonic-rl/states/'),
                        help='Output directory for save states')
    parser.add_argument('--tiers', type=int, nargs='+', default=[0, 1, 2, 3],
                        choices=[0, 1, 2, 3],
                        help='Speed tiers to record (default: all)')
    parser.add_argument('--states-dir', type=str,
                        default=os.path.expanduser('~/sonic-rl/states/'),
                        help='Directory containing base stage save states')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    state_filename = STAGE_STATES.get(args.stage)
    if not state_filename:
        print(f"Unknown stage: {args.stage}")
        return

    base_state_path = os.path.join(args.states_dir, state_filename)
    if not os.path.exists(base_state_path):
        print(f"Base state not found: {base_state_path}")
        print(f"Record {state_filename} first by playing to the stage start.")
        return

    print(f"Recording Stage {args.stage} states in {args.output}")
    print(f"Base state: {base_state_path}")
    print()

    saved = []
    for tier in args.tiers:
        frames = SPEED_TIER_FRAMES[tier]
        print(f"Recording speed tier {tier} ({frames} frames)...")
        path = make_state(base_state_path, args.output, args.stage, tier, frames)
        saved.append(path)
        print()

    print("Done. Recorded states:")
    for path in saved:
        print(f"  {path}")

    print()
    print("To use in training, update SpecialStageEnv to randomly sample")
    print("from these states at episode reset for speed curriculum training.")


if __name__ == '__main__':
    main()

"""
run_free_energy.py -- Free Energy Arena: DishBrain-inspired reward via input structure.

Instead of explicit reward signals (deliver_reward), the brain learns from the
STRUCTURE of its sensory input:
  - Near food: clean, structured sensory patterns (low entropy, predictable)
  - Far from food: noisy, random patterns (high entropy, unpredictable)

The brain's own STDP naturally drives it toward states that produce predictable
input, because predictable patterns create consistent pre-post timing relationships
that strengthen synapses, while random noise creates inconsistent timing that
averages to no change or weakening.

This is the free energy principle: the brain minimizes surprise by learning to
produce behavior that leads to predictable sensory input.

No deliver_reward() is called. The reward IS the structure of the input.

Usage:
    py run_free_energy.py --brain brains/metabolic_high_v2_s200.db
    py run_free_energy.py --brain brains/cortex_heavy_v2_s100.db --ticks 100000
    py run_free_energy.py --brain brains/metabolic_high_v2_s200.db --sessions 5
"""
import numpy as np
import time
import os
import sys
import argparse
import json
import functools

print = functools.partial(print, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(os.path.dirname(BASE), 'inner-models-v8')
sys.path.insert(0, V8)

from engine.loader import load
from engine.runner import Brain
from arena import Arena
from worm_body import WormBody


def build_electrodes(brain_data, n, rng):
    """Build population-encoded electrode assignments (same as viewer)."""
    # Sensory: 16 audio-like channels x 8 pop = 128 input neurons
    # Motor: 4 output channels x 16 pop = 64 motor neurons
    # (Smaller than viewer since we only need basic sensory-motor)
    audio_channels = 16
    audio_pop = 8
    motor_channels = 4
    motor_pop = 16

    audio_total = audio_channels * audio_pop  # 128
    motor_total = motor_channels * motor_pop   # 64

    in_deg = np.zeros(n, dtype=int)
    out_deg = np.zeros(n, dtype=int)
    for s in brain_data['synapses']:
        out_deg[s['source']] += 1
        in_deg[s['target']] += 1

    # Input: top out-degree
    out_sorted = np.argsort(out_deg)[::-1]
    audio_list = out_sorted[:audio_total].tolist()

    # Motor: top in-degree, excluding input
    input_set = set(audio_list)
    in_sorted = np.argsort(in_deg)[::-1]
    motor_list = []
    for idx in in_sorted:
        if int(idx) not in input_set:
            motor_list.append(int(idx))
        if len(motor_list) >= motor_total:
            break

    # Population encoding: tuning curves for sensory channels
    intercepts = np.zeros((audio_channels, audio_pop))
    gains = np.zeros((audio_channels, audio_pop))
    for ch in range(audio_channels):
        intercepts[ch] = np.linspace(0.05, 0.95, audio_pop) + rng.randn(audio_pop) * 0.05
        intercepts[ch] = np.clip(intercepts[ch], 0.0, 1.0)
        gains[ch] = rng.uniform(2.0, 8.0, audio_pop)

    # Motor decoding weights
    motor_decoders = rng.uniform(0.1, 1.0, (motor_channels, motor_pop))
    motor_decoders /= motor_decoders.sum(axis=1, keepdims=True)

    audio_out = np.mean(out_deg[audio_list]) if audio_list else 0
    motor_in = np.mean(in_deg[motor_list]) if motor_list else 0
    print(f"  Electrodes: {audio_total} sensory (avg out={audio_out:.0f}), "
          f"{motor_total} motor (avg in={motor_in:.0f})")

    return {
        'audio_idx': np.array(audio_list, dtype=np.intp),
        'motor_idx': np.array(motor_list, dtype=np.intp),
        'audio_channels': audio_channels,
        'audio_pop': audio_pop,
        'motor_channels': motor_channels,
        'motor_pop': motor_pop,
        'intercepts': intercepts,
        'gains': gains,
        'motor_decoders': motor_decoders,
    }


def encode_population(signal, intercepts, gains, gain_scale):
    """Encode channel signals into per-neuron current injection."""
    sig = np.clip(signal, 0, 1)[:, None]
    activation = np.maximum(0, gains * (sig - intercepts))
    return (activation * gain_scale).ravel()


def decode_motor(fire_counts, decoders):
    """Decode motor population firing into output channels."""
    n_channels, pop_size = decoders.shape
    counts = fire_counts.reshape(n_channels, pop_size)
    return np.sum(counts * decoders, axis=1)


def generate_structured_pattern(channels, tick, pattern_type='sweep',
                                food_angle=0.0):
    """Generate a structured, predictable sensory pattern.

    Near food: the brain receives clean, repeating patterns.
    These create consistent spike timing -> STDP strengthens connections.

    food_angle: angle to food relative to heading (-pi to pi).
    Used in 'directional' pattern type to encode food direction.
    """
    signal = np.zeros(channels)
    if pattern_type == 'sweep':
        # Frequency sweep: one channel active at a time, cycling
        cycle_len = channels * 5  # 80 ticks per full sweep
        phase = (tick % cycle_len) / cycle_len
        active_ch = int(phase * channels) % channels
        signal[active_ch] = 0.9
        # Neighbors get partial activation (smooth sweep)
        if active_ch > 0:
            signal[active_ch - 1] = 0.3
        if active_ch < channels - 1:
            signal[active_ch + 1] = 0.3
    elif pattern_type == 'pulse':
        # Regular pulses: all channels activate together periodically
        period = 20
        if tick % period < 5:
            signal[:] = 0.8
    elif pattern_type == 'alternating':
        # Alternating halves: even/odd channels alternate
        period = 30
        if (tick // period) % 2 == 0:
            signal[::2] = 0.7
        else:
            signal[1::2] = 0.7
    elif pattern_type == 'directional':
        # Directional sweep: sweep direction encodes food direction
        # food_angle < 0 = food is to the left -> sweep channels 0 -> N
        # food_angle > 0 = food is to the right -> sweep channels N -> 0
        # food_angle ~ 0 = food is ahead -> pulse (all channels together)
        abs_angle = abs(food_angle)
        if abs_angle < 0.3:
            # Food ahead: strong pulse
            period = 15
            if tick % period < 5:
                signal[:] = 0.9
        else:
            # Directional sweep
            cycle_len = channels * 4
            phase = (tick % cycle_len) / cycle_len
            if food_angle < 0:
                active_ch = int(phase * channels) % channels
            else:
                active_ch = (channels - 1 - int(phase * channels)) % channels
            # Intensity proportional to angle magnitude
            intensity = min(0.9, 0.4 + abs_angle * 0.3)
            signal[active_ch] = intensity
            if active_ch > 0:
                signal[active_ch - 1] = intensity * 0.35
            if active_ch < channels - 1:
                signal[active_ch + 1] = intensity * 0.35
    return signal


def generate_noise_pattern(channels, rng):
    """Generate random, unpredictable sensory pattern.

    Far from food: the brain receives noise.
    Random timing -> STDP averages to zero or weakening.
    """
    return rng.uniform(0, 0.8, channels)


def run_free_energy(db_path, ticks=60000, seed=42, tonic=2.8,
                    sensory_gain=6.0, sessions=1, report_interval=5000,
                    structure_mode='steep', explicit_reward=False,
                    curriculum=False):
    """Run free-energy arena simulation."""

    print(f"{'='*70}")
    print(f"  FREE ENERGY ARENA")
    print(f"  Brain: {os.path.basename(db_path)}")
    print(f"  Ticks: {ticks:,d} x {sessions} sessions, tonic: {tonic}, seed: {seed}")
    if explicit_reward:
        print(f"  Reward: FREE ENERGY + EXPLICIT (hybrid)")
    else:
        print(f"  Reward: NONE (free energy principle)")
    print(f"  Near food -> structured patterns, Far -> noise")
    print(f"  Structure mode: {structure_mode}")
    print(f"{'='*70}")

    data = load(db_path)
    brain = Brain(data, learn=True)
    n = brain.n
    rng = np.random.RandomState(seed)

    print(f"  Brain: {n}N, {len(data['synapses'])} synapses")

    # Electrode assignment
    electrodes = build_electrodes(data, n, rng)
    audio_idx = electrodes['audio_idx']
    motor_idx = electrodes['motor_idx']
    motor_map = {int(idx): i for i, idx in enumerate(motor_idx)}

    # Arena
    arena = Arena(radius=30.0)
    arena.add_food(15.0, 0.0, peak=1.0, sigma=12.0)
    food_x, food_y = 15.0, 0.0

    all_results = []

    for session in range(sessions):
        sess_rng = np.random.RandomState(seed + session * 100)

        # Start position
        angle = sess_rng.uniform(0, 2 * np.pi)
        if curriculum:
            # Start close, gradually increase: 5 -> 10 -> 15 -> 20 -> 25...
            start_dist = min(5.0 + session * 5.0, 25.0)
        else:
            start_dist = 20.0
        start_x = food_x + start_dist * np.cos(angle)
        start_y = food_y + start_dist * np.sin(angle)

        body = WormBody()
        body.pos = np.array([start_x, start_y], dtype=np.float64)
        body.heading = sess_rng.uniform(0, 2 * np.pi)
        body.pos = arena.clamp_to_boundary(body.pos)

        d0 = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)
        c0 = arena.concentration_at(body.pos[0], body.pos[1])

        print(f"\n  Session {session+1}/{sessions}: start=({body.pos[0]:.1f}, {body.pos[1]:.1f}), "
              f"dist={d0:.1f}, conc={c0:.4f}")

        # Tracking
        trajectory = []
        conc_log = []
        dist_log = []
        total_spikes = 0
        motor_spikes = 0
        structured_ticks = 0
        noise_ticks = 0
        reward_total = 0.0
        prev_conc = c0

        # Pattern state
        pattern_types = ['sweep', 'pulse', 'alternating']
        current_pattern = 0

        start_time = time.perf_counter()

        print(f"  {'Tick':>8s} | {'Conc':>6s} | {'Dist':>6s} | {'Struct%':>7s} | "
              f"{'Motor':>6s} | {'Total':>6s} | {'Pattern':>10s}")
        print(f"  {'-'*70}")

        for tick in range(ticks):
            # 1. Get environment state
            head = body.pos
            conc = arena.concentration_at(head[0], head[1])
            dist = np.sqrt((head[0] - food_x)**2 + (head[1] - food_y)**2)

            # 2. Free energy encoding
            # structure_fraction: 0 = pure noise, 1 = pure structure
            if structure_mode == 'flat':
                # Original: too generous, 75% structured even at dist=20
                structure_fraction = min(1.0, conc * 3.0)
            elif structure_mode == 'steep':
                # Steep: conc^2 * 4. Strong gradient.
                # d=0: 100%, d=15: 84%, d=20: 25%, d=25: 5%, d=30: 1%
                structure_fraction = min(1.0, conc ** 2 * 4.0)
            elif structure_mode == 'binary':
                # Sharp threshold: structured only when close
                structure_fraction = 1.0 if conc > 0.4 else 0.0
            elif structure_mode == 'directional':
                # Steep gradient + directional encoding in the pattern itself
                structure_fraction = min(1.0, conc ** 2 * 4.0)
            else:
                structure_fraction = min(1.0, conc * 3.0)

            # Compute angle to food relative to heading
            food_angle = np.arctan2(food_y - head[1], food_x - head[0]) - body.heading
            # Normalize to [-pi, pi]
            food_angle = (food_angle + np.pi) % (2 * np.pi) - np.pi

            if rng.random() < structure_fraction:
                # Structured pattern
                if structure_mode == 'directional':
                    pattern_name = 'directional'
                else:
                    pattern_name = pattern_types[current_pattern % len(pattern_types)]
                signal = generate_structured_pattern(
                    electrodes['audio_channels'], tick, pattern_name,
                    food_angle=food_angle)
                structured_ticks += 1
            else:
                # Noise
                signal = generate_noise_pattern(electrodes['audio_channels'], rng)
                noise_ticks += 1

            # Rotate pattern type slowly (variety within structure)
            if tick % 500 == 0 and tick > 0:
                current_pattern += 1

            # 3. Population encode and inject
            currents = encode_population(
                signal, electrodes['intercepts'], electrodes['gains'], sensory_gain)
            I_ext = np.full(n, tonic)
            n_inject = min(len(currents), len(audio_idx))
            I_ext[audio_idx[:n_inject]] += currents[:n_inject]

            # 4. Tick brain (STDP active, NO explicit reward)
            fired = brain.tick(external_I=I_ext)
            n_fired = len(fired)
            total_spikes += n_fired

            # 5. Read motor output
            motor_fire = np.zeros(len(motor_idx), dtype=int)
            for f in fired:
                fi = int(f)
                if fi in motor_map:
                    motor_fire[motor_map[fi]] += 1
                    motor_spikes += 1

            # 6. Decode motor into movement
            motor_decoded = decode_motor(motor_fire, electrodes['motor_decoders'])
            # Map 4 channels to: forward, backward, turn_left, turn_right
            fwd = max(0, motor_decoded[0] - motor_decoded[1])
            turn = motor_decoded[2] - motor_decoded[3]

            # Apply movement (simple steering model)
            speed = fwd * 0.003  # Scale to arena units
            body.heading += turn * 0.02
            body.pos[0] += np.cos(body.heading) * speed
            body.pos[1] += np.sin(body.heading) * speed

            # Also step the worm body physics for natural motion
            body.pos = arena.clamp_to_boundary(body.pos)

            # Optional explicit reward (hybrid mode)
            if explicit_reward and tick % 200 == 199:
                new_conc = arena.concentration_at(body.pos[0], body.pos[1])
                dc = new_conc - prev_conc
                if abs(dc) > 0.0005:
                    reward = float(np.clip(dc * 5.0, -1.0, 1.0))
                    brain.deliver_reward(reward)
                    reward_total += reward
                prev_conc = new_conc

            # 7. Log
            if tick % 100 == 0:
                trajectory.append((float(head[0]), float(head[1])))
                conc_log.append(float(conc))
                dist_log.append(float(dist))

            if tick % report_interval == 0 and tick > 0:
                struct_pct = structured_ticks / max(structured_ticks + noise_ticks, 1) * 100
                elapsed = time.perf_counter() - start_time
                rate = tick / elapsed
                print(f"  {tick:>8d} | {conc:.4f} | {dist:>5.1f} | {struct_pct:>5.1f}% | "
                      f"{motor_spikes:>6d} | {total_spikes:>6d} | {pattern_types[current_pattern % len(pattern_types)]}")

        # Session summary
        elapsed = time.perf_counter() - start_time
        final_dist = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)
        min_dist = min(dist_log) if dist_log else final_dist
        struct_pct = structured_ticks / max(structured_ticks + noise_ticks, 1) * 100

        # Chemotaxis index: (d_start - d_end) / d_start
        ci = (d0 - final_dist) / max(d0, 0.01)

        result = {
            'session': session + 1,
            'seed': seed + session * 100,
            'ticks': ticks,
            'd_start': float(d0),
            'd_end': float(final_dist),
            'd_min': float(min_dist),
            'ci': float(ci),
            'total_spikes': total_spikes,
            'motor_spikes': motor_spikes,
            'struct_pct': float(struct_pct),
            'reward_total': float(reward_total),
            'elapsed': float(elapsed),
            'ticks_per_sec': float(ticks / elapsed),
        }
        all_results.append(result)

        print(f"\n  Session {session+1} results:")
        print(f"    CI: {ci:+.3f}")
        print(f"    Distance: {d0:.1f} -> {final_dist:.1f} (min: {min_dist:.1f})")
        print(f"    Structured input: {struct_pct:.0f}% of ticks")
        print(f"    Spikes: {total_spikes:,d} total, {motor_spikes:,d} motor")
        print(f"    Speed: {ticks/elapsed:.0f} ticks/sec")

        # Sleep between sessions (homeostasis)
        if session < sessions - 1 and hasattr(brain, 'sleep'):
            print(f"\n  Sleeping between sessions...")
            brain.sleep(5000, compression=0.8)

    # Overall summary
    print(f"\n{'='*70}")
    print(f"  OVERALL RESULTS ({sessions} sessions)")
    print(f"{'='*70}")
    cis = [r['ci'] for r in all_results]
    print(f"  CI: mean={np.mean(cis):+.3f}, std={np.std(cis):.3f}")
    print(f"  CI range: [{min(cis):+.3f}, {max(cis):+.3f}]")
    for r in all_results:
        print(f"    S{r['session']}: CI={r['ci']:+.3f}, d={r['d_start']:.1f}->{r['d_end']:.1f}, "
              f"struct={r['struct_pct']:.0f}%")

    # Save results
    results_dir = os.path.join(BASE, 'results')
    os.makedirs(results_dir, exist_ok=True)
    brain_name = os.path.splitext(os.path.basename(db_path))[0]
    results_path = os.path.join(results_dir, f"free_energy_{brain_name}_{structure_mode}_s{seed}.json")
    summary = {
        'brain': brain_name,
        'structure_mode': structure_mode,
        'explicit_reward': explicit_reward,
        'tonic': tonic,
        'seed': seed,
        'sessions': all_results,
        'mean_ci': float(np.mean(cis)),
        'std_ci': float(np.std(cis)),
    }
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Results: {results_path}")

    return all_results


def main():
    p = argparse.ArgumentParser(description='Free Energy Arena')
    p.add_argument('--brain', required=True, help='Path to brain DB')
    p.add_argument('--ticks', type=int, default=60000, help='Ticks per session')
    p.add_argument('--sessions', type=int, default=3, help='Number of sessions')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--tonic', type=float, default=2.8, help='Tonic drive')
    p.add_argument('--sensory-gain', type=float, default=6.0, help='Sensory gain')
    p.add_argument('--structure-mode', default='steep',
                   choices=['flat', 'steep', 'binary', 'directional'],
                   help='Structure fraction formula (directional = steep + sweep direction encodes food angle)')
    p.add_argument('--reward', action='store_true',
                   help='Add explicit reward on top of free energy (hybrid mode)')
    p.add_argument('--curriculum', action='store_true',
                   help='Start close to food, gradually increase distance across sessions')
    args = p.parse_args()

    run_free_energy(
        args.brain, ticks=args.ticks, seed=args.seed,
        tonic=args.tonic, sensory_gain=args.sensory_gain,
        sessions=args.sessions, structure_mode=args.structure_mode,
        explicit_reward=args.reward, curriculum=args.curriculum)


if __name__ == '__main__':
    main()

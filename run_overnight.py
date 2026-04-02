"""
run_overnight.py -- Run arena chemotaxis tests on all V9 brains.

Uses the V8 arena engine (run_arena.py) to test V9-grown brains.
Each brain gets multiple sessions with random starting positions.

Usage:
  py run_overnight.py                        # All brains, full runs
  py run_overnight.py --brain brains/regional_balanced_s42.db
  py run_overnight.py --quick                # 3 sessions, 10K ticks
"""
import os
import sys
import sqlite3
import shutil
import numpy as np
import argparse
import functools
import time
import json

print = functools.partial(print, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(os.path.dirname(BASE), 'inner-models-v8')
sys.path.insert(0, V8)


def run_session(db_path, ticks, tonic, seed):
    """Run one arena session using the V8 engine directly."""
    from engine.loader import load
    from engine.runner import Brain
    from arena import Arena
    from worm_body import WormBody

    data = load(db_path)
    n = data['n']
    brain = Brain(data, learn=True)
    body_map = data.get('body_map', {})
    sensor_map = data.get('sensor_map', {})

    # Classify sensors
    chemical_sensors = []
    mechanical_sensors = []
    left_sensors = []
    right_sensors = []

    for idx, entry in sensor_map.items():
        mod = entry.get('modality', '')
        if mod == 'chemical':
            chemical_sensors.append(idx)
            side = entry.get('side', '')
            if side in ('left', 'L'):
                left_sensors.append(idx)
            elif side in ('right', 'R'):
                right_sensors.append(idx)
        elif mod == 'mechanical':
            mechanical_sensors.append(idx)

    # Arena: food at (15, 0)
    arena = Arena(radius=30.0)
    arena.add_food(15.0, 0.0, peak=1.0, sigma=12.0)
    food_x, food_y = 15.0, 0.0

    # Body: random start
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 2 * np.pi)
    start_distance = 20.0
    start_x = food_x + start_distance * np.cos(angle)
    start_y = food_y + start_distance * np.sin(angle)

    body = WormBody()
    body.pos = np.array([start_x, start_y], dtype=np.float64)
    body.heading = rng.uniform(0, 2 * np.pi)
    body.pos = arena.clamp_to_boundary(body.pos)

    d0 = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)
    c0 = arena.concentration_at(body.pos[0], body.pos[1])

    # Adaptation state
    adapted_conc = c0
    tau_adapt = 500.0
    prev_conc = c0
    sensory_gain = 5.0

    total_spikes = 0
    motor_spikes = 0
    sensory_spikes = 0
    reward_total = 0.0

    start_time = time.perf_counter()

    for tick in range(ticks):
        head = body.pos
        heading = body.heading
        conc = arena.concentration_at(head[0], head[1])

        # Derivative sensing
        alpha = 1.0 / tau_adapt
        delta_conc = conc - adapted_conc
        adapted_conc += alpha * (conc - adapted_conc)

        sensory_I = np.zeros(n)

        # Chemical sensors
        for idx in chemical_sensors:
            sensory_I[idx] = conc * sensory_gain + max(0, delta_conc) * sensory_gain * 3.0

        # Spatial sensing
        head_half_width = 0.04
        normal_x = -np.sin(heading)
        normal_y = np.cos(heading)
        conc_left = arena.concentration_at(
            head[0] + normal_x * head_half_width,
            head[1] + normal_y * head_half_width)
        conc_right = arena.concentration_at(
            head[0] - normal_x * head_half_width,
            head[1] - normal_y * head_half_width)

        for idx in left_sensors:
            sensory_I[idx] += conc_left * sensory_gain * 0.5
        for idx in right_sensors:
            sensory_I[idx] += conc_right * sensory_gain * 0.5

        # Wall detection
        d_wall = arena.radius - np.linalg.norm(head)
        if d_wall < 2.0:
            wall_strength = (2.0 - d_wall) / 2.0 * 15.0
            for idx in mechanical_sensors:
                sensory_I[idx] += wall_strength

        # Combine currents
        I_ext = np.full(n, tonic)
        I_ext += sensory_I

        # Brain tick
        fired = brain.tick(external_I=I_ext)
        total_spikes += len(fired)

        # Motor output
        for fi in fired:
            if fi in body_map:
                entry = body_map[fi]
                body.apply_motor_spike(entry['segment'], entry['side'], entry['effect'])
                motor_spikes += 1
            if fi in sensor_map:
                sensory_spikes += 1

        # Body physics
        body.step(dt=0.001)
        body.pos = arena.clamp_to_boundary(body.pos)

        # Reward every 200 ticks
        if tick % 200 == 199:
            new_conc = arena.concentration_at(body.pos[0], body.pos[1])
            dc = new_conc - prev_conc
            if abs(dc) > 0.0005:
                reward = float(np.clip(dc * 5.0, -1.0, 1.0))
                brain.deliver_reward(reward)
                reward_total += reward
            prev_conc = new_conc

    elapsed = time.perf_counter() - start_time
    d_final = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)
    ci = (d0 - d_final) / max(d0, 1.0)

    # Weight stats
    w_mean = 0
    w_std = 0
    if hasattr(brain, 'reward_idx') and len(brain.reward_idx) > 0:
        rw = np.array([brain.synapses[si]['weight'] for si in brain.reward_idx])
        w_mean = float(np.mean(np.abs(rw)))
        w_std = float(np.std(np.abs(rw)))

    return {
        'ci': float(ci),
        'start_dist': float(d0),
        'end_dist': float(d_final),
        'total_spikes': int(total_spikes),
        'motor_spikes': int(motor_spikes),
        'sensory_spikes': int(sensory_spikes),
        'spikes_per_tick': total_spikes / max(ticks, 1),
        'reward_total': float(reward_total),
        'w_mean': w_mean,
        'w_std': w_std,
        'time_s': elapsed,
        'ticks_per_sec': ticks / elapsed,
    }


def run_brain(db_path, n_sessions=10, ticks_per_session=30000, tonic=2.8, quick=False):
    """Run multiple sessions on one brain."""
    if quick:
        n_sessions = 3
        ticks_per_session = 10000

    name = os.path.basename(db_path).replace('.db', '')

    # Work on a copy
    work_db = db_path + '.running'
    shutil.copy2(db_path, work_db)

    print(f"\n  --- {name} ---")

    # DB info
    conn = sqlite3.connect(work_db)
    n_neurons = conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    n_synapses = conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
    n_reward = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='reward_plastic'").fetchone()[0]
    conn.close()

    print(f"  {n_neurons}N, {n_synapses} syn, {n_reward} reward_plastic")
    print(f"  Running {n_sessions} sessions x {ticks_per_session} ticks at tonic {tonic}")

    session_results = []
    for s in range(n_sessions):
        seed = 42 + s * 17  # Different starting position each session
        try:
            result = run_session(work_db, ticks_per_session, tonic, seed)
            session_results.append(result)

            print(f"    S{s}: CI={result['ci']:+.3f}, "
                  f"spk/t={result['spikes_per_tick']:.1f}, "
                  f"motor={result['motor_spikes']}, "
                  f"reward={result['reward_total']:+.2f}, "
                  f"{result['ticks_per_sec']:.0f} t/s")
        except Exception as e:
            print(f"    S{s}: ERROR - {e}")

    # Cleanup
    if os.path.exists(work_db):
        os.remove(work_db)

    if not session_results:
        print(f"  FAIL: No successful sessions")
        return None

    # Summary
    cis = [r['ci'] for r in session_results]
    mean_ci = np.mean(cis)
    w_means = [r['w_mean'] for r in session_results]

    summary = {
        'name': name,
        'neurons': n_neurons,
        'synapses': n_synapses,
        'reward_plastic': n_reward,
        'sessions': n_sessions,
        'ticks_per_session': ticks_per_session,
        'mean_ci': float(mean_ci),
        'ci_std': float(np.std(cis)),
        'best_ci': float(max(cis)),
        'worst_ci': float(min(cis)),
        'mean_spk_rate': float(np.mean([r['spikes_per_tick'] for r in session_results])),
        'mean_motor': float(np.mean([r['motor_spikes'] for r in session_results])),
        'final_w_mean': float(w_means[-1]) if w_means else 0,
        'session_details': session_results,
    }

    print(f"  RESULT: mean CI={mean_ci:+.3f} +/- {np.std(cis):.3f}")
    print(f"          best={max(cis):+.3f}, worst={min(cis):+.3f}")

    return summary


def main():
    p = argparse.ArgumentParser(description='Overnight arena tests on V9 brains')
    p.add_argument('--brain', default=None)
    p.add_argument('--sessions', type=int, default=10)
    p.add_argument('--ticks', type=int, default=30000)
    p.add_argument('--tonic', type=float, default=2.8)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()

    if args.brain:
        db_files = [args.brain]
    else:
        brain_dir = os.path.join(BASE, 'brains')
        if not os.path.exists(brain_dir):
            print("No brains/ directory. Run build_five.py first.")
            return
        db_files = sorted([os.path.join(brain_dir, f) for f in os.listdir(brain_dir)
                          if f.endswith('.db') and not f.endswith('.running')])

    if not db_files:
        print("No brain databases found.")
        return

    print(f"{'='*60}")
    print(f"  V9 OVERNIGHT ARENA TESTS")
    print(f"  Brains: {len(db_files)}")
    n_sess = 3 if args.quick else args.sessions
    n_ticks = 10000 if args.quick else args.ticks
    print(f"  Sessions: {n_sess}, Ticks/session: {n_ticks}")
    print(f"  Tonic: {args.tonic}")
    print(f"{'='*60}")

    results_dir = os.path.join(BASE, 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    start_time = time.perf_counter()

    for db_path in db_files:
        result = run_brain(db_path, args.sessions, args.ticks, args.tonic, args.quick)
        if result:
            all_results.append(result)
            # Save individual result
            rname = result['name']
            with open(os.path.join(results_dir, f"{rname}_result.json"), 'w') as f:
                json.dump(result, f, indent=2, default=str)

    total_time = time.perf_counter() - start_time

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Brain':<35s} {'N':>6s} {'CI':>8s} {'Best':>8s} {'Spk/t':>7s} {'Motor':>8s}")
    for r in sorted(all_results, key=lambda x: -x['mean_ci']):
        name = r['name'][:33]
        print(f"  {name:<35s} {r['neurons']:>6d} {r['mean_ci']:>+8.3f} "
              f"{r['best_ci']:>+8.3f} {r['mean_spk_rate']:>7.1f} {r['mean_motor']:>8.0f}")

    print(f"\n  Total time: {total_time/60:.1f} minutes")

    # Save combined
    with open(os.path.join(results_dir, 'overnight_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {results_dir}/")


if __name__ == '__main__':
    main()

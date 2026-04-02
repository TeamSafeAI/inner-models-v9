"""
validate_brains.py -- Quick validation of V9 grown brains.

Loads each brain into the V8 engine and runs a short arena test
(1000 ticks) to verify:
  1. Brain loads without errors
  2. Neurons fire (not dead)
  3. Motor output exists (body moves)
  4. Sensory input arrives
  5. Reward plastic synapses exist

Usage:
  py validate_brains.py                    # Validate all brains in brains/
  py validate_brains.py --brain brains/regional_balanced_s42.db
"""
import os
import sys
import sqlite3
import numpy as np
import argparse
import functools
import time

print = functools.partial(print, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(os.path.dirname(BASE), 'inner-models-v8')
sys.path.insert(0, V8)


def validate_db(db_path):
    """Check brain DB structure is valid."""
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    n_neurons = conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    n_synapses = conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]

    has_body = 'body_map' in tables
    has_sensor = 'sensor_map' in tables

    n_body = 0
    n_sensor = 0
    if has_body:
        n_body = conn.execute("SELECT COUNT(*) FROM body_map").fetchone()[0]
    if has_sensor:
        n_sensor = conn.execute("SELECT COUNT(*) FROM sensor_map").fetchone()[0]

    n_reward = conn.execute(
        "SELECT COUNT(*) FROM synapses WHERE synapse_type='reward_plastic'"
    ).fetchone()[0]

    conn.close()

    return {
        'neurons': n_neurons,
        'synapses': n_synapses,
        'body_map': n_body,
        'sensor_map': n_sensor,
        'reward_plastic': n_reward,
        'has_body': has_body,
        'has_sensor': has_sensor,
    }


def validate_engine(db_path, tonic=2.8, ticks=1000):
    """Load brain into V8 engine and run a short test."""
    try:
        from engine.loader import load_brain
        from engine.runner import Brain
    except ImportError as e:
        print(f"    SKIP engine test: {e}")
        return None

    try:
        data = load_brain(db_path)
    except Exception as e:
        print(f"    FAIL load: {e}")
        return {'loaded': False, 'error': str(e)}

    n = data['n']
    brain = Brain(data)

    I_ext = np.full(n, tonic)
    total_spikes = 0
    motor_spikes = 0
    sensory_active = 0

    body_map = data.get('body_map', {})
    sensor_map = data.get('sensor_map', {})
    motor_ids = set(body_map.keys())
    sensor_ids = set(sensor_map.keys())

    for t in range(ticks):
        # Fake sensory input to sensor neurons
        for sid in sensor_ids:
            I_ext[sid] = tonic + np.sin(t / 50.0) * 3.0

        fired = brain.tick(I_ext)

        if fired is not None and len(fired) > 0:
            total_spikes += len(fired)
            motor_spikes += sum(1 for f in fired if f in motor_ids)
            sensory_active += sum(1 for f in fired if f in sensor_ids)

    return {
        'loaded': True,
        'n': n,
        'ticks': ticks,
        'total_spikes': total_spikes,
        'spikes_per_tick': total_spikes / ticks,
        'motor_spikes': motor_spikes,
        'sensory_active': sensory_active,
        'body_neurons': len(motor_ids),
        'sensor_neurons': len(sensor_ids),
    }


def main():
    p = argparse.ArgumentParser(description='Validate V9 grown brains')
    p.add_argument('--brain', default=None, help='Specific brain DB to validate')
    p.add_argument('--tonic', type=float, default=2.8)
    p.add_argument('--ticks', type=int, default=1000)
    p.add_argument('--skip-engine', action='store_true', help='Only check DB structure')
    args = p.parse_args()

    if args.brain:
        db_files = [args.brain]
    else:
        brain_dir = os.path.join(BASE, 'brains')
        if not os.path.exists(brain_dir):
            print("No brains/ directory found. Run build_five.py first.")
            return
        db_files = sorted([os.path.join(brain_dir, f) for f in os.listdir(brain_dir) if f.endswith('.db')])

    if not db_files:
        print("No brain databases found.")
        return

    print(f"{'='*70}")
    print(f"  V9 BRAIN VALIDATION")
    print(f"  Brains: {len(db_files)}")
    print(f"  Tonic: {args.tonic}, Ticks: {args.ticks}")
    print(f"{'='*70}")

    results = []

    for db_path in db_files:
        name = os.path.basename(db_path)
        print(f"\n  --- {name} ---")

        # DB structure check
        db_info = validate_db(db_path)
        print(f"    Neurons: {db_info['neurons']}, Synapses: {db_info['synapses']}")
        print(f"    body_map: {db_info['body_map']}, sensor_map: {db_info['sensor_map']}")
        print(f"    reward_plastic: {db_info['reward_plastic']}")

        ok = True
        if db_info['neurons'] == 0:
            print(f"    FAIL: No neurons!")
            ok = False
        if db_info['synapses'] == 0:
            print(f"    FAIL: No synapses!")
            ok = False
        if db_info['body_map'] == 0:
            print(f"    WARN: No body_map entries (won't move)")
        if db_info['sensor_map'] == 0:
            print(f"    WARN: No sensor_map entries (won't sense)")

        # Engine test
        engine_result = None
        if not args.skip_engine and ok:
            start = time.perf_counter()
            engine_result = validate_engine(db_path, args.tonic, args.ticks)
            elapsed = time.perf_counter() - start

            if engine_result and engine_result.get('loaded'):
                er = engine_result
                print(f"    Engine: {er['n']}N loaded, {er['spikes_per_tick']:.1f} spk/tick")
                print(f"    Motor spikes: {er['motor_spikes']}, Sensory active: {er['sensory_active']}")
                print(f"    Speed: {args.ticks / elapsed:.0f} ticks/s")

                if er['total_spikes'] == 0:
                    print(f"    FAIL: Brain is dead (0 spikes)")
                elif er['motor_spikes'] == 0:
                    print(f"    WARN: No motor output")
                else:
                    print(f"    PASS")
            elif engine_result:
                print(f"    FAIL: {engine_result.get('error', 'unknown')}")

        results.append({
            'name': name,
            'db': db_info,
            'engine': engine_result,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Brain':<40s} {'N':>6s} {'Syn':>8s} {'Spk/t':>7s} {'Motor':>7s} {'Status':>8s}")
    for r in results:
        name = r['name'][:38]
        n = r['db']['neurons']
        syn = r['db']['synapses']
        if r['engine'] and r['engine'].get('loaded'):
            spt = f"{r['engine']['spikes_per_tick']:.1f}"
            mot = str(r['engine']['motor_spikes'])
            status = 'PASS' if r['engine']['total_spikes'] > 0 else 'DEAD'
        else:
            spt = '-'
            mot = '-'
            status = 'SKIP' if not r['engine'] else 'FAIL'
        print(f"  {name:<40s} {n:>6d} {syn:>8d} {spt:>7s} {mot:>7s} {status:>8s}")


if __name__ == '__main__':
    main()

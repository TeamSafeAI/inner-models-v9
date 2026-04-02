"""
build_five.py -- Build 5 regional brain variants for testing.

Variants:
  1. balanced      -- Standard 7-region, balanced proportions
  2. cortex_heavy  -- 50% cortex, strong association
  3. memory_dense  -- 25% hippocampus, dense recurrence (DENSE)
  4. thalamic_hub  -- Thalamus as super-hub, strong relay (DENSE)
  5. amygdala_driven -- Emotional fast-path, strong amygdala

Each brain gets body_map (brainstem -> motor) and sensor_map
(sensory -> chemical/mechanical) populated so it can run in the arena.

Usage:
  py build_five.py --small          # 1000N each, quick validation
  py build_five.py --full           # 8000-12000N, overnight runs
  py build_five.py --config balanced --neurons 1000  # Single variant
"""
import subprocess
import sys
import os
import sqlite3
import numpy as np
import argparse
import functools
import time

print = functools.partial(print, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    'balanced':       {'neurons': 8000,  'label': 'NORMAL'},
    'cortex_heavy':   {'neurons': 8000,  'label': 'NORMAL'},
    'memory_dense':   {'neurons': 12000, 'label': 'DENSE'},
    'thalamic_hub':   {'neurons': 10000, 'label': 'DENSE'},
    'amygdala_driven':{'neurons': 8000,  'label': 'WILD'},
}

SMALL_N = 1000


def populate_body_sensor_maps(db_path, rng):
    """Add body_map and sensor_map to a V9-grown brain.

    Assigns:
      - brainstem neurons -> body_map (motor output to worm segments)
      - sensory neurons -> sensor_map (chemical + mechanical input)

    Uses pos_z to identify regions (brainstem = low z, sensory = high z).
    """
    conn = sqlite3.connect(db_path)

    # Check if tables exist
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    if 'body_map' not in tables:
        conn.execute("""CREATE TABLE IF NOT EXISTS body_map (
            neuron_id INTEGER NOT NULL,
            segment INTEGER NOT NULL,
            side TEXT NOT NULL,
            effect TEXT NOT NULL
        )""")

    if 'sensor_map' not in tables:
        conn.execute("""CREATE TABLE IF NOT EXISTS sensor_map (
            neuron_id INTEGER NOT NULL,
            modality TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT 'head',
            response_type TEXT NOT NULL DEFAULT 'tonic',
            side TEXT NOT NULL DEFAULT 'bilateral'
        )""")

    # Get all neurons with positions
    neurons = conn.execute(
        "SELECT id, pos_x, pos_y, pos_z, neuron_type FROM neurons ORDER BY id"
    ).fetchall()

    n = len(neurons)

    # Identify motor neurons (low z = brainstem region, z < 200)
    motor_candidates = [(nid, x, y, z) for nid, x, y, z, nt in neurons if z < 200]
    # Identify sensory neurons (high z = sensory region, z > 300 and y < 200)
    sensory_candidates = [(nid, x, y, z) for nid, x, y, z, nt in neurons if z > 300 and y < 200]

    # If not enough, use position heuristic more broadly
    if len(motor_candidates) < 20:
        motor_candidates = sorted([(nid, x, y, z) for nid, x, y, z, nt in neurons],
                                  key=lambda r: r[3])[:max(20, n // 50)]
    if len(sensory_candidates) < 20:
        sensory_candidates = sorted([(nid, x, y, z) for nid, x, y, z, nt in neurons],
                                    key=lambda r: -r[3])[:max(20, n // 50)]

    # 24-segment worm body
    n_segments = 24

    # Assign motor neurons to segments
    # Each segment gets dorsal+ventral on L+R sides, with excitatory and cross-inhibitory
    motor_per_seg = max(1, len(motor_candidates) // n_segments)
    rng.shuffle(motor_candidates)
    body_count = 0
    for i, (nid, x, y, z, *_) in enumerate(motor_candidates[:n_segments * motor_per_seg]):
        seg = i // motor_per_seg
        if seg >= n_segments:
            break
        side = 'dorsal' if y > 250 else 'ventral'
        # Alternate excitatory/inhibitory
        effect = 'excitatory' if i % 2 == 0 else 'inhibitory'
        conn.execute("INSERT OR IGNORE INTO body_map (neuron_id, segment, side, effect) VALUES (?, ?, ?, ?)",
                     (nid, seg, side, effect))
        body_count += 1

    # Assign sensory neurons: 60% chemical, 40% mechanical
    # V8 schema: (neuron_id, modality, location, response_type, side)
    rng.shuffle(sensory_candidates)
    n_chem = int(len(sensory_candidates) * 0.6)
    sensor_count = 0
    for i, (nid, x, y, z, *_) in enumerate(sensory_candidates):
        modality = 'chemical' if i < n_chem else 'mechanical'
        location = 'head' if i % 3 != 2 else 'tail'
        response_type = 'tonic' if modality == 'chemical' else 'phasic'
        side = 'left' if x < 250 else 'right'
        conn.execute(
            "INSERT OR IGNORE INTO sensor_map (neuron_id, modality, location, response_type, side) VALUES (?, ?, ?, ?, ?)",
            (nid, modality, location, response_type, side))
        sensor_count += 1

    conn.commit()
    conn.close()

    print(f"    body_map: {body_count} motor neurons across {n_segments} segments")
    print(f"    sensor_map: {sensor_count} sensory neurons ({n_chem} chem, {sensor_count - n_chem} mech)")


def grow_one(config_name, n_neurons, seed=42):
    """Grow a single brain variant."""
    name = f"regional_{config_name}_s{seed}"
    db_path = os.path.join(BASE, 'brains', f"{name}.db")

    if os.path.exists(db_path):
        print(f"  {name} already exists, skipping growth")
        return db_path

    cmd = [
        sys.executable, os.path.join(BASE, 'grow_regional.py'),
        '--neurons', str(n_neurons),
        '--seed', str(seed),
        '--config', config_name,
        '--name', name,
    ]

    print(f"\n{'='*60}")
    print(f"  GROWING: {name} ({n_neurons}N, config={config_name})")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Growth failed for {name}")
        return None

    # Add body_map and sensor_map
    if os.path.exists(db_path):
        rng = np.random.RandomState(seed + 1000)
        print(f"  Populating body_map and sensor_map...")
        populate_body_sensor_maps(db_path, rng)

    return db_path


def main():
    p = argparse.ArgumentParser(description='Build 5 regional brain variants')
    p.add_argument('--small', action='store_true', help=f'Small validation ({SMALL_N}N each)')
    p.add_argument('--full', action='store_true', help='Full size (8K-12K each)')
    p.add_argument('--config', default=None, help='Build single config')
    p.add_argument('--neurons', type=int, default=None, help='Override neuron count')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if not args.small and not args.full and not args.config:
        print("Usage: py build_five.py --small  OR  --full  OR  --config <name>")
        print(f"Configs: {', '.join(CONFIGS.keys())}")
        return

    os.makedirs(os.path.join(BASE, 'brains'), exist_ok=True)

    start = time.perf_counter()

    if args.config:
        # Single config
        n = args.neurons or CONFIGS.get(args.config, {}).get('neurons', 8000)
        grow_one(args.config, n, args.seed)
    else:
        # Build all 5
        for cfg_name, cfg in CONFIGS.items():
            n = SMALL_N if args.small else (args.neurons or cfg['neurons'])
            grow_one(cfg_name, n, args.seed)

    elapsed = time.perf_counter() - start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()

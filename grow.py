"""
grow.py -- Grow a neural network from spatial dynamics.

Instead of specifying connections, we simulate axon growth.
Each neuron has a 3D position. Axons extend outward following
chemical gradients (chemoattraction). When a growth cone comes
within proximity of another neuron's soma/dendrite, a synapse forms.

Growth follows multiplicative dynamics:
  - Axons extend in small steps
  - At each step: steer toward nearby chemoattractant sources
  - Probabilistic branching (more branches = more connection opportunities)
  - Connection on proximity (growth cone within contact_radius of target soma)
  - Same pair can form MULTIPLE synapses (multi-contact, like real cortex)

This naturally produces:
  - Power-law degree distribution (preferential attachment via spatial density)
  - Log-normal connection distances (multiplicative growth steps)
  - Hub neurons (neurons in dense regions attract more growth cones)
  - Convergent funneling (many axons converge on spatially central neurons)
  - Rich club (hubs are spatially close, so they find each other)

Parameters encode what DNA encodes: growth rules, not wiring diagrams.

Usage:
  py grow.py                    # Grow 8000 neurons, default params
  py grow.py --neurons 5000     # Smaller brain
  py grow.py --seed 42          # Reproducible
  py grow.py --validate         # Compare stats to H01 human cortex
"""
import numpy as np
import time
import os
import sys
import argparse
import sqlite3
import json

# Force unbuffered output on Windows
import functools
print = functools.partial(print, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Growth Parameters ──────────────────────────────────────────────
# These are the "genome" -- the growth rules that produce the network.

DEFAULT_PARAMS = {
    # Neuron placement
    'n_neurons': 8000,
    'volume_size': 500.0,       # um, cubic volume side length
    'n_layers': 6,              # cortical layers
    'layer_fractions': [0.05, 0.25, 0.25, 0.21, 0.17, 0.07],  # L1-L6

    # Cell types
    'exc_fraction': 0.65,       # pyramidal / excitatory
    'inh_fraction': 0.35,       # interneuron / inhibitory

    # Axon growth
    'axon_steps': 200,          # growth steps per axon
    'step_size': 5.0,           # um per step
    'branch_prob': 0.03,        # probability of branching per step
    'max_branches': 8,          # max active growth cones per neuron
    'turn_rate': 0.3,           # how much growth cones steer toward attractant (0=straight, 1=direct)

    # Chemoattraction
    'attract_radius': 100.0,    # um, range of chemoattractant signal
    'attract_decay': 2.0,       # power of distance decay (2.0 = inverse square)

    # Synapse formation
    'contact_radius': 8.0,      # um, proximity for synapse formation
    'multi_contact': True,       # allow multiple synapses between same pair
    'max_synapses_per_pair': 12, # cap on multi-contact

    # Inhibitory targeting (interneurons prefer pyramidals)
    'inh_exc_preference': 3.0,  # multiplier on contact_radius for inh->exc

    # Layer connectivity bias
    'feedforward_bias': 2.0,    # multiplier for downward (L2->L3->L4->L5) growth
    'feedback_fraction': 0.3,   # fraction of axon branches that grow upward

    # Neuron parameters (Izhikevich)
    'param_jitter': 0.15,       # jitter on a,b,c,d parameters

    # D1/D2 neuromodulator assignment
    'd1_fraction': 0.4,
    'd2_fraction': 0.4,
    # remaining 0.2 = mixed/neutral
}


def place_neurons(params, rng):
    """Place neurons in 3D space with layer structure."""
    n = params['n_neurons']
    size = params['volume_size']
    fracs = params['layer_fractions']

    # X, Y uniform in volume; Z stratified by layer
    x = rng.uniform(0, size, n)
    y = rng.uniform(0, size, n)

    # Assign layers based on fractions
    layer_boundaries = np.cumsum([0] + fracs)
    layers = np.zeros(n, dtype=int)
    z = np.zeros(n)

    idx = 0
    for layer_id in range(len(fracs)):
        n_in_layer = int(fracs[layer_id] * n)
        if layer_id == len(fracs) - 1:
            n_in_layer = n - idx  # remainder
        end = min(idx + n_in_layer, n)

        layers[idx:end] = layer_id + 1  # 1-indexed layers
        z_lo = layer_boundaries[layer_id] * size
        z_hi = layer_boundaries[layer_id + 1] * size
        z[idx:end] = rng.uniform(z_lo, z_hi, end - idx)
        idx = end

    # Assign types: excitatory or inhibitory
    exc_n = int(params['exc_fraction'] * n)
    types = np.array(['EXC'] * exc_n + ['INH'] * (n - exc_n))
    rng.shuffle(types)

    # Assign neuron model types based on cell type
    # EXC: mostly RS, some IB in L5
    # INH: FS (basket), LTS (Martinotti)
    neuron_types = []
    for i in range(n):
        if types[i] == 'EXC':
            if layers[i] == 5 and rng.random() < 0.4:
                neuron_types.append('ib')  # intrinsic bursting in L5
            else:
                neuron_types.append('rs')  # regular spiking
        else:
            if rng.random() < 0.6:
                neuron_types.append('fs')  # fast spiking (basket)
            else:
                neuron_types.append('lts')  # low-threshold (Martinotti)

    return {
        'x': x, 'y': y, 'z': z,
        'layers': layers,
        'cell_types': types,
        'neuron_types': neuron_types,
        'n': n,
    }


def grow_axons(neurons, params, rng):
    """Simulate axon growth and synapse formation.

    Each neuron sprouts an axon that grows step by step.
    Growth cones steer toward nearby neurons (chemoattraction).
    When a cone gets close enough to a target soma, a synapse forms.
    Cones can branch, creating multiple growth tips per neuron.

    Returns list of (source, target) synapse pairs (with duplicates for multi-contact).
    """
    n = neurons['n']
    x, y, z = neurons['x'], neurons['y'], neurons['z']
    cell_types = neurons['cell_types']
    layers = neurons['layers']

    steps = params['axon_steps']
    step_size = params['step_size']
    branch_prob = params['branch_prob']
    max_branches = params['max_branches']
    turn_rate = params['turn_rate']
    attract_radius = params['attract_radius']
    attract_decay = params['attract_decay']
    contact_radius = params['contact_radius']
    multi_contact = params['multi_contact']
    max_syn_pair = params['max_synapses_per_pair']
    inh_pref = params['inh_exc_preference']
    ff_bias = params['feedforward_bias']
    fb_frac = params['feedback_fraction']

    # Positions array for fast distance computation
    positions = np.column_stack([x, y, z])

    # Build spatial index (simple grid)
    grid_size = attract_radius
    grid = {}
    for i in range(n):
        gx = int(x[i] / grid_size)
        gy = int(y[i] / grid_size)
        gz = int(z[i] / grid_size)
        key = (gx, gy, gz)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    def get_nearby(px, py, pz, radius):
        """Get neuron indices within radius using grid."""
        gx = int(px / grid_size)
        gy = int(py / grid_size)
        gz = int(pz / grid_size)
        nearby = []
        r = max(1, int(np.ceil(radius / grid_size)))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    key = (gx + dx, gy + dy, gz + dz)
                    if key in grid:
                        for idx in grid[key]:
                            d = np.sqrt((x[idx] - px)**2 + (y[idx] - py)**2 + (z[idx] - pz)**2)
                            if d < radius:
                                nearby.append((idx, d))
        return nearby

    # Track synapses
    synapses = []
    pair_counts = {}  # (src, tgt) -> count

    # Grow each neuron's axon
    start_time = time.perf_counter()

    for src in range(n):
        if src % 1000 == 0 and src > 0:
            elapsed = time.perf_counter() - start_time
            rate = src / elapsed
            print(f"  Growing neuron {src}/{n} ({rate:.0f} neurons/s, {len(synapses)} synapses)...")

        src_layer = layers[src]
        is_inh = cell_types[src] == 'INH'

        # Initial growth direction: biased by layer
        # Feedforward: grow toward higher layers (deeper)
        # Some branches grow upward (feedback)
        if rng.random() < fb_frac:
            z_direction = -1.0  # feedback (upward)
        else:
            z_direction = 1.0 * ff_bias  # feedforward (downward)

        # Start growth cones (list of [px, py, pz, dx, dy, dz])
        direction = rng.randn(3)
        direction[2] = abs(direction[2]) * z_direction  # bias Z
        direction /= np.linalg.norm(direction) + 1e-10

        cones = [[
            x[src] + direction[0] * step_size,
            y[src] + direction[1] * step_size,
            z[src] + direction[2] * step_size,
            direction[0], direction[1], direction[2]
        ]]

        for step in range(steps):
            if not cones:
                break

            new_cones = []
            for cone in cones:
                px, py, pz, dx, dy, dz = cone

                # Check for contacts (synapse formation)
                c_radius = contact_radius
                if is_inh:
                    # Inhibitory neurons have wider contact for excitatory targets
                    c_radius_exc = contact_radius * inh_pref
                else:
                    c_radius_exc = contact_radius

                nearby = get_nearby(px, py, pz, max(c_radius, c_radius_exc if is_inh else c_radius))
                for tgt, dist in nearby:
                    if tgt == src:
                        continue

                    # Determine effective contact radius
                    if is_inh and cell_types[tgt] == 'EXC':
                        eff_radius = c_radius_exc
                    else:
                        eff_radius = c_radius

                    if dist < eff_radius:
                        pair = (src, tgt)
                        count = pair_counts.get(pair, 0)
                        if not multi_contact and count > 0:
                            continue
                        if count >= max_syn_pair:
                            continue
                        synapses.append(pair)
                        pair_counts[pair] = count + 1

                # Steer toward nearby neurons (chemoattraction)
                attract = get_nearby(px, py, pz, attract_radius)
                if attract:
                    # Weighted attraction: closer neurons pull harder
                    ax, ay, az = 0.0, 0.0, 0.0
                    for tgt, dist in attract:
                        if tgt == src:
                            continue
                        if dist < 1.0:
                            dist = 1.0
                        weight = 1.0 / (dist ** attract_decay)
                        ax += (x[tgt] - px) * weight
                        ay += (y[tgt] - py) * weight
                        az += (z[tgt] - pz) * weight

                    norm = np.sqrt(ax*ax + ay*ay + az*az)
                    if norm > 1e-10:
                        ax /= norm
                        ay /= norm
                        az /= norm

                    # Blend current direction with attraction
                    dx = dx * (1 - turn_rate) + ax * turn_rate
                    dy = dy * (1 - turn_rate) + ay * turn_rate
                    dz = dz * (1 - turn_rate) + az * turn_rate

                # Add noise to direction
                dx += rng.randn() * 0.2
                dy += rng.randn() * 0.2
                dz += rng.randn() * 0.2

                # Normalize
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
                    dx /= norm
                    dy /= norm
                    dz /= norm

                # Advance
                new_px = px + dx * step_size
                new_py = py + dy * step_size
                new_pz = pz + dz * step_size

                # Boundary: wrap or reflect
                size = params['volume_size']
                new_px = max(0, min(size, new_px))
                new_py = max(0, min(size, new_py))
                new_pz = max(0, min(size, new_pz))

                new_cones.append([new_px, new_py, new_pz, dx, dy, dz])

                # Branching
                if len(new_cones) < max_branches and rng.random() < branch_prob:
                    # New branch in random direction (biased by parent)
                    bdx = dx + rng.randn() * 0.5
                    bdy = dy + rng.randn() * 0.5
                    bdz = dz + rng.randn() * 0.5
                    bnorm = np.sqrt(bdx*bdx + bdy*bdy + bdz*bdz)
                    if bnorm > 1e-10:
                        bdx /= bnorm
                        bdy /= bnorm
                        bdz /= bnorm
                    new_cones.append([new_px, new_py, new_pz, bdx, bdy, bdz])

            cones = new_cones

    elapsed = time.perf_counter() - start_time
    print(f"  Growth complete: {len(synapses)} synapses in {elapsed:.1f}s")

    return synapses, pair_counts


def compute_stats(neurons, synapses, pair_counts):
    """Compute network statistics for comparison with H01."""
    from collections import Counter

    n = neurons['n']
    cell_types = neurons['cell_types']

    # Degree distributions
    out_degree = Counter()
    in_degree = Counter()
    for src, tgt in synapses:
        out_degree[src] += 1
        in_degree[tgt] += 1

    out_vals = [out_degree.get(i, 0) for i in range(n)]
    in_vals = [in_degree.get(i, 0) for i in range(n)]

    active_out = [v for v in out_vals if v > 0]
    active_in = [v for v in in_vals if v > 0]

    print(f"\n{'='*60}")
    print(f"  NETWORK STATISTICS")
    print(f"{'='*60}")
    print(f"  Neurons: {n} ({sum(1 for t in cell_types if t == 'EXC')} exc, "
          f"{sum(1 for t in cell_types if t == 'INH')} inh)")
    print(f"  Total synapses: {len(synapses)}")
    print(f"  Unique pairs: {len(pair_counts)}")
    multi = sum(1 for c in pair_counts.values() if c > 1)
    print(f"  Multi-contact pairs: {multi} ({multi/max(len(pair_counts),1)*100:.1f}%)")

    # Degree stats
    print(f"\n  Out-degree: mean={np.mean(active_out):.1f}, median={np.median(active_out):.0f}, "
          f"max={max(active_out) if active_out else 0}, std={np.std(active_out):.1f}")
    print(f"  In-degree:  mean={np.mean(active_in):.1f}, median={np.median(active_in):.0f}, "
          f"max={max(active_in) if active_in else 0}, std={np.std(active_in):.1f}")

    # Power law
    if active_out:
        deg_counts = Counter(active_out)
        degs = sorted(deg_counts.keys())
        if len(degs) > 3:
            counts = [deg_counts[d] for d in degs]
            log_d = np.log(np.array(degs, dtype=float))
            log_c = np.log(np.array(counts, dtype=float))
            coeffs = np.polyfit(log_d, log_c, 1)
            r2 = 1 - np.sum((log_c - np.polyval(coeffs, log_d))**2) / np.sum((log_c - np.mean(log_c))**2)
            print(f"  Power law exponent: {-coeffs[0]:.2f} (H01: 1.89), R2={r2:.3f}")

    # E/I connectivity
    exc_exc = sum(1 for s, t in synapses if cell_types[s] == 'EXC' and cell_types[t] == 'EXC')
    exc_inh = sum(1 for s, t in synapses if cell_types[s] == 'EXC' and cell_types[t] == 'INH')
    inh_exc = sum(1 for s, t in synapses if cell_types[s] == 'INH' and cell_types[t] == 'EXC')
    inh_inh = sum(1 for s, t in synapses if cell_types[s] == 'INH' and cell_types[t] == 'INH')
    total_typed = exc_exc + exc_inh + inh_exc + inh_inh
    if total_typed > 0:
        print(f"\n  E->E: {exc_exc/total_typed*100:.1f}% (H01: 37.8%)")
        print(f"  E->I: {exc_inh/total_typed*100:.1f}% (H01: 40.5%)")
        print(f"  I->E: {inh_exc/total_typed*100:.1f}% (H01: 16.9%)")
        print(f"  I->I: {inh_inh/total_typed*100:.1f}% (H01: 4.8%)")
        # Inhibitory targeting
        total_inh_out = inh_exc + inh_inh
        if total_inh_out > 0:
            print(f"  INH targets EXC: {inh_exc/total_inh_out*100:.1f}% (H01: 78.0%)")

    # Convergence vs divergence
    # Convergence: neurons with many inputs, divergence: neurons with many outputs
    high_in = sum(1 for v in in_vals if v > np.mean(active_in) + np.std(active_in))
    high_out = sum(1 for v in out_vals if v > np.mean(active_out) + np.std(active_out))
    print(f"\n  High in-degree (convergent): {high_in}")
    print(f"  High out-degree (divergent): {high_out}")
    print(f"  Convergence ratio: {np.mean(active_in)/np.mean(active_out):.2f} (H01 funnel: >1)")

    # Rich club
    threshold = np.percentile(active_out, 95) if active_out else 0
    hubs = set(i for i in range(n) if out_degree.get(i, 0) >= threshold)
    if hubs:
        hub_to_hub = sum(1 for s, t in synapses if s in hubs and t in hubs)
        hub_total = sum(1 for s, t in synapses if s in hubs)
        hub_frac = len(hubs) / n
        actual = hub_to_hub / max(hub_total, 1)
        print(f"\n  Rich club (top 5%, >={threshold:.0f} out):")
        print(f"    Hubs: {len(hubs)}")
        print(f"    Hub->Hub: {actual*100:.1f}% (expected random: {hub_frac*100:.1f}%)")
        print(f"    Rich club coefficient: {actual/max(hub_frac,0.001):.2f}x (H01: 2.72x)")

    # Distance distribution
    distances = []
    for src, tgt in synapses[:10000]:  # sample
        d = np.sqrt((neurons['x'][src] - neurons['x'][tgt])**2 +
                     (neurons['y'][src] - neurons['y'][tgt])**2 +
                     (neurons['z'][src] - neurons['z'][tgt])**2)
        distances.append(d)
    if distances:
        d_arr = np.array(distances)
        log_d = np.log(d_arr[d_arr > 0])
        skew = float(np.mean(((log_d - np.mean(log_d))/np.std(log_d))**3)) if len(log_d) > 10 else 0
        print(f"\n  Distances: mean={np.mean(d_arr):.1f}um, median={np.median(d_arr):.1f}um")
        print(f"  Log-distance skew: {skew:.2f} (0 = log-normal)")
        local = sum(1 for d in distances if d < 50)
        print(f"  Local (<50um): {local/len(distances)*100:.1f}% (H01: 93.6%)")

    # Multi-contact distribution
    mc_counts = Counter(pair_counts.values())
    print(f"\n  Multi-contact distribution:")
    for n_syn in sorted(mc_counts.keys())[:8]:
        print(f"    {n_syn} synapses: {mc_counts[n_syn]} pairs")

    return {
        'out_degree': out_vals,
        'in_degree': in_vals,
    }


def save_to_db(neurons, synapses, pair_counts, params, db_path, rng):
    """Save grown network to V8-compatible SQLite database."""
    # Import V8 schema
    v8_path = os.path.join(os.path.dirname(BASE), 'inner-models-v8')
    sys.path.insert(0, v8_path)
    from schema import create_brain_db

    n = neurons['n']
    jitter = params['param_jitter']

    # Izhikevich parameters by type
    TYPE_PARAMS = {
        'rs':  {'a': 0.02,  'b': 0.2,  'c': -65.0, 'd': 8.0},
        'fs':  {'a': 0.1,   'b': 0.2,  'c': -65.0, 'd': 2.0},
        'ib':  {'a': 0.02,  'b': 0.2,  'c': -55.0, 'd': 4.0},
        'lts': {'a': 0.02,  'b': 0.25, 'c': -65.0, 'd': 2.0},
        'ch':  {'a': 0.02,  'b': 0.2,  'c': -50.0, 'd': 2.0},
    }

    conn = create_brain_db(db_path)

    # Insert neurons
    for i in range(n):
        nt = neurons['neuron_types'][i]
        base = TYPE_PARAMS[nt]
        nt_upper = nt.upper()  # V8 engine expects uppercase
        a = base['a'] * (1 + rng.uniform(-jitter, jitter))
        b = base['b'] * (1 + rng.uniform(-jitter, jitter))
        c = base['c'] * (1 + rng.uniform(-jitter, jitter))
        d = base['d'] * (1 + rng.uniform(-jitter, jitter))

        # D1/D2 assignment
        r = rng.random()
        if r < params['d1_fraction']:
            dopa_sens = rng.uniform(0.3, 1.0)
        elif r < params['d1_fraction'] + params['d2_fraction']:
            dopa_sens = rng.uniform(-1.0, -0.3)
        else:
            dopa_sens = rng.uniform(-0.2, 0.2)

        conn.execute(
            """INSERT INTO neurons
               (neuron_type, a, b, c, d, v, u, last_spike,
                pos_x, pos_y, pos_z, dopamine_sens, excitability, activity_trace)
               VALUES (?, ?, ?, ?, ?, -65, ?, -1000, ?, ?, ?, ?, 0, 0)""",
            (nt_upper, a, b, c, d, b * -65.0,
             neurons['x'][i], neurons['y'][i], neurons['z'][i],
             dopa_sens)
        )

    # Build neuron ID mapping (1-indexed from DB)
    db_ids = [row[0] for row in conn.execute("SELECT id FROM neurons ORDER BY id")]
    idx_to_id = {i: db_ids[i] for i in range(n)}

    # Determine synapse types and weights
    cell_types = neurons['cell_types']

    # Group synapses by unique pair and count contacts
    for (src, tgt), n_contacts in pair_counts.items():
        src_exc = cell_types[src] == 'EXC'
        src_layer = neurons['layers'][src]
        tgt_layer = neurons['layers'][tgt]

        # Weight: base weight scaled by number of contacts (multi-synapse = stronger)
        if src_exc:
            base_w = 2.0
        else:
            base_w = -2.5

        # Multi-contact scaling: each additional contact adds reliability
        weight = base_w * (1.0 + 0.3 * (n_contacts - 1))

        # Determine synapse type
        # Cross-layer excitatory: reward_plastic (learnable)
        # Same-layer: fixed or plastic
        if src_exc and src_layer != tgt_layer:
            syn_type = 'reward_plastic'
        elif src_exc:
            syn_type = 'plastic'
        else:
            syn_type = 'fixed'

        # Distance-based delay
        d = np.sqrt((neurons['x'][src] - neurons['x'][tgt])**2 +
                     (neurons['y'][src] - neurons['y'][tgt])**2 +
                     (neurons['z'][src] - neurons['z'][tgt])**2)
        delay = max(1, int(d / 50.0))  # ~1ms per 50um

        params_json = json.dumps({'n_contacts': n_contacts})

        conn.execute(
            """INSERT INTO synapses
               (source, target, synapse_type, weight, delay, params, state)
               VALUES (?, ?, ?, ?, ?, ?, '{}')""",
            (idx_to_id[src], idx_to_id[tgt], syn_type, abs(weight), delay, params_json)
        )

    conn.commit()

    # Stats
    n_syn = conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
    n_reward = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='reward_plastic'").fetchone()[0]
    n_plastic = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='plastic'").fetchone()[0]
    n_fixed = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='fixed'").fetchone()[0]

    conn.close()

    print(f"\n  Saved to: {db_path}")
    print(f"  {n} neurons, {n_syn} synapses")
    print(f"  Types: {n_fixed} fixed, {n_plastic} plastic, {n_reward} reward_plastic")

    return db_path


def main():
    p = argparse.ArgumentParser(description='Grow a neural network from spatial dynamics')
    p.add_argument('--neurons', type=int, default=8000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validate', action='store_true', help='Compare stats to H01')
    p.add_argument('--save', action='store_true', default=True, help='Save to V8-compatible DB')
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--name', default=None, help='Brain name (default: grown_s{seed})')
    # Growth parameter overrides
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--branch-prob', type=float, default=0.03)
    p.add_argument('--contact-radius', type=float, default=8.0)
    p.add_argument('--attract-radius', type=float, default=100.0)
    p.add_argument('--volume', type=float, default=500.0)
    args = p.parse_args()

    params = dict(DEFAULT_PARAMS)
    params['n_neurons'] = args.neurons
    params['axon_steps'] = args.steps
    params['branch_prob'] = args.branch_prob
    params['contact_radius'] = args.contact_radius
    params['attract_radius'] = args.attract_radius
    params['volume_size'] = args.volume

    rng = np.random.RandomState(args.seed)

    print(f"{'='*60}")
    print(f"  V9 NETWORK GROWTH")
    print(f"  Neurons: {params['n_neurons']}, Steps: {params['axon_steps']}")
    print(f"  Volume: {params['volume_size']}um, Contact: {params['contact_radius']}um")
    print(f"  Branch prob: {params['branch_prob']}, Attract: {params['attract_radius']}um")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}")

    # 1. Place neurons
    print("\n  Placing neurons...")
    neurons = place_neurons(params, rng)
    print(f"  Placed {neurons['n']} neurons in 6 layers")

    # 2. Grow axons
    print("\n  Growing axons...")
    synapses, pair_counts = grow_axons(neurons, params, rng)

    # 3. Statistics
    stats = compute_stats(neurons, synapses, pair_counts)

    # 4. Save
    if args.save and not args.no_save:
        name = args.name or f"grown_s{args.seed}"
        # Save to V8 brains directory for compatibility
        v8_zoo = os.path.join(os.path.dirname(BASE), 'inner-models-v8', 'brains', 'zoo')
        os.makedirs(v8_zoo, exist_ok=True)
        db_path = os.path.join(v8_zoo, f"{name}.db")
        save_to_db(neurons, synapses, pair_counts, params, db_path, rng)

    print(f"\n{'='*60}")
    print(f"  GROWTH COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

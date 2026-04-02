"""
grow_regional.py -- Grow a brain with distinct anatomical regions.

Places brain regions (cortex, hippocampus, amygdala, basal ganglia,
thalamus, brainstem) at specific spatial locations. Each region has
its own cell type composition, density, and chemoattractant profile.

Connections are NOT specified -- they emerge from axon growth.
Each region emits chemical signals that attract growth cones from
other regions. The GROWTH discovers the wiring.

This is the V9 approach: define WHERE the regions ARE, define what
chemical signals they emit, then let spatial dynamics determine
every single connection.

Usage:
  py grow_regional.py                     # 8000 neurons, default regions
  py grow_regional.py --neurons 12000     # Bigger brain
  py grow_regional.py --seed 42           # Reproducible
"""
import numpy as np
import time
import os
import sys
import argparse
import sqlite3
import json
import functools

print = functools.partial(print, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Brain Regions ──────────────────────────────────────────────────
# Each region has:
#   center: (x, y, z) in um
#   radius: spatial extent
#   n_fraction: fraction of total neurons
#   exc_ratio: excitatory fraction (varies by region)
#   neuron_types: distribution of Izhikevich types
#   attractant: how strongly this region attracts growth cones from OTHER regions
#   local_density: how tightly packed (affects local connectivity)
#   description: what it does

REGIONS = {
    'cortex': {
        'center': (250, 250, 400),  # top of brain (dorsal)
        'radius': 150,
        'n_fraction': 0.35,
        'exc_ratio': 0.65,
        'neuron_types': {'rs': 0.50, 'fs': 0.25, 'lts': 0.15, 'ib': 0.10},
        'attractant': 1.0,         # moderate -- attracts from thalamus, hippocampus
        'local_density': 1.0,
        'description': 'Association cortex. Layered. Decision-making.',
    },
    'hippocampus': {
        'center': (150, 400, 250),  # medial temporal
        'radius': 80,
        'n_fraction': 0.15,
        'exc_ratio': 0.80,         # hippocampus is heavily excitatory
        'neuron_types': {'rs': 0.70, 'fs': 0.15, 'lts': 0.10, 'ch': 0.05},
        'attractant': 1.5,         # strong -- memory hub, attracts from cortex + amygdala
        'local_density': 2.0,      # dense local recurrence (CA3)
        'description': 'Memory formation. Dense recurrent. Spatial navigation.',
    },
    'amygdala': {
        'center': (350, 400, 200),  # medial temporal, ventral
        'radius': 60,
        'n_fraction': 0.08,
        'exc_ratio': 0.60,
        'neuron_types': {'rs': 0.40, 'fs': 0.30, 'lts': 0.20, 'ib': 0.10},
        'attractant': 2.0,         # very strong -- fast emotional processing
        'local_density': 1.5,
        'description': 'Emotional valence. Fast threat/reward. Outputs to basal ganglia.',
    },
    'basal_ganglia': {
        'center': (250, 300, 250),  # deep central
        'radius': 70,
        'n_fraction': 0.12,
        'exc_ratio': 0.40,         # heavily inhibitory (striatal MSNs)
        'neuron_types': {'rs': 0.30, 'fs': 0.40, 'lts': 0.20, 'ib': 0.10},
        'attractant': 0.8,
        'local_density': 1.2,
        'description': 'Action selection. D1 go / D2 no-go. Inhibitory gating.',
    },
    'thalamus': {
        'center': (250, 250, 250),  # central relay
        'radius': 60,
        'n_fraction': 0.10,
        'exc_ratio': 0.75,         # relay neurons are excitatory
        'neuron_types': {'rs': 0.60, 'fs': 0.20, 'lts': 0.15, 'ch': 0.05},
        'attractant': 1.8,         # strong -- everything passes through thalamus
        'local_density': 0.8,      # less local, more relay
        'description': 'Sensory relay. Gates information to cortex. Central hub.',
    },
    'brainstem': {
        'center': (250, 150, 100),  # ventral, posterior
        'radius': 80,
        'n_fraction': 0.12,
        'exc_ratio': 0.55,
        'neuron_types': {'rs': 0.35, 'fs': 0.20, 'ib': 0.30, 'lts': 0.15},
        'attractant': 0.5,         # low -- receives from above, outputs to body
        'local_density': 1.0,
        'description': 'Motor output. CPG. Connects to body_map.',
    },
    'sensory': {
        'center': (250, 100, 350),  # anterior, dorsal
        'radius': 70,
        'n_fraction': 0.08,
        'exc_ratio': 0.70,
        'neuron_types': {'rs': 0.60, 'fs': 0.25, 'lts': 0.10, 'ib': 0.05},
        'attractant': 0.6,         # moderate -- receives external input
        'local_density': 1.0,
        'description': 'Sensory input. Chemical + mechanical. Connects to sensor_map.',
    },
}


def place_regional_neurons(n_total, regions, rng):
    """Place neurons within their assigned brain regions."""
    x_all, y_all, z_all = [], [], []
    region_labels = []
    cell_types = []
    neuron_types = []

    for rname, rinfo in regions.items():
        n_region = int(rinfo['n_fraction'] * n_total)
        cx, cy, cz = rinfo['center']
        radius = rinfo['radius']
        exc_ratio = rinfo['exc_ratio']
        type_dist = rinfo['neuron_types']

        # Place neurons in sphere (Gaussian distribution for density)
        for i in range(n_region):
            # Gaussian placement within region (denser at center)
            dx = rng.randn() * radius * 0.4
            dy = rng.randn() * radius * 0.4
            dz = rng.randn() * radius * 0.4
            # Clamp to region radius
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            if d > radius:
                scale = radius / d
                dx *= scale
                dy *= scale
                dz *= scale

            x_all.append(cx + dx)
            y_all.append(cy + dy)
            z_all.append(cz + dz)
            region_labels.append(rname)

            # Cell type
            is_exc = rng.random() < exc_ratio
            cell_types.append('EXC' if is_exc else 'INH')

            # Neuron type from region distribution
            r = rng.random()
            cumulative = 0
            chosen = 'rs'
            for ntype, prob in type_dist.items():
                cumulative += prob
                if r < cumulative:
                    chosen = ntype
                    break
            neuron_types.append(chosen)

    n = len(x_all)
    print(f"  Placed {n} neurons across {len(regions)} regions:")
    from collections import Counter
    region_counts = Counter(region_labels)
    for rname in regions:
        print(f"    {rname:15s}: {region_counts[rname]:5d} neurons")

    return {
        'x': np.array(x_all),
        'y': np.array(y_all),
        'z': np.array(z_all),
        'regions': region_labels,
        'cell_types': cell_types,
        'neuron_types': neuron_types,
        'n': n,
    }


def grow_regional_axons(neurons, regions, params, rng):
    """Grow axons with region-aware chemoattraction.

    Growth cones are attracted to regions based on their attractant strength.
    This means thalamus (high attractant) naturally becomes a hub,
    hippocampus attracts from cortex, amygdala attracts from sensory, etc.
    """
    n = neurons['n']
    x, y, z = neurons['x'], neurons['y'], neurons['z']
    cell_types = neurons['cell_types']
    region_labels = neurons['regions']

    steps = params.get('axon_steps', 200)
    step_size = params.get('step_size', 5.0)
    branch_prob = params.get('branch_prob', 0.04)
    max_branches = params.get('max_branches', 8)
    turn_rate = params.get('turn_rate', 0.3)
    contact_radius = params.get('contact_radius', 10.0)
    max_syn_pair = params.get('max_synapses_per_pair', 12)
    inh_pref = params.get('inh_exc_preference', 2.5)

    # Build spatial grid
    grid_size = 50.0  # um
    grid = {}
    for i in range(n):
        key = (int(x[i] / grid_size), int(y[i] / grid_size), int(z[i] / grid_size))
        grid.setdefault(key, []).append(i)

    def get_nearby(px, py, pz, radius):
        gx, gy, gz = int(px / grid_size), int(py / grid_size), int(pz / grid_size)
        r = max(1, int(np.ceil(radius / grid_size)))
        result = []
        for ddx in range(-r, r + 1):
            for ddy in range(-r, r + 1):
                for ddz in range(-r, r + 1):
                    key = (gx + ddx, gy + ddy, gz + ddz)
                    if key in grid:
                        for idx in grid[key]:
                            d = np.sqrt((x[idx] - px)**2 + (y[idx] - py)**2 + (z[idx] - pz)**2)
                            if d < radius:
                                result.append((idx, d))
        return result

    # Region centers for long-range attraction
    region_centers = {}
    region_attract = {}
    for rname, rinfo in regions.items():
        region_centers[rname] = np.array(rinfo['center'], dtype=float)
        region_attract[rname] = rinfo['attractant']

    synapses = []
    pair_counts = {}

    start_time = time.perf_counter()

    for src in range(n):
        if src % 500 == 0 and src > 0:
            elapsed = time.perf_counter() - start_time
            rate = src / elapsed
            eta = (n - src) / rate / 60
            print(f"  Growing {src}/{n} ({rate:.0f}/s, {len(synapses)} syn, ETA {eta:.1f}m)")

        src_region = region_labels[src]
        is_inh = cell_types[src] == 'INH'

        # Growth direction: influenced by nearby region attractants
        direction = rng.randn(3)

        # Bias growth toward high-attractant regions (NOT own region)
        attract_dir = np.zeros(3)
        for rname, center in region_centers.items():
            if rname == src_region:
                continue  # don't attract to own region (those connections are local)
            to_region = center - np.array([x[src], y[src], z[src]])
            dist = np.linalg.norm(to_region)
            if dist > 10:
                strength = region_attract[rname] / (dist ** 0.5)
                attract_dir += to_region / dist * strength

        if np.linalg.norm(attract_dir) > 0.01:
            attract_dir /= np.linalg.norm(attract_dir)
            direction = direction * 0.5 + attract_dir * 0.5

        direction /= np.linalg.norm(direction) + 1e-10

        # Start growth
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

                # Contact check
                eff_radius = contact_radius * inh_pref if is_inh else contact_radius
                nearby = get_nearby(px, py, pz, eff_radius)
                for tgt, dist in nearby:
                    if tgt == src:
                        continue
                    # Inhibitory preference for excitatory targets
                    actual_radius = contact_radius
                    if is_inh and cell_types[tgt] == 'EXC':
                        actual_radius = contact_radius * inh_pref
                    if dist < actual_radius:
                        pair = (src, tgt)
                        count = pair_counts.get(pair, 0)
                        if count >= max_syn_pair:
                            continue
                        synapses.append(pair)
                        pair_counts[pair] = count + 1

                # Steering: local chemoattraction + regional signals
                local = get_nearby(px, py, pz, 80.0)
                ax, ay, az = 0.0, 0.0, 0.0
                for tgt, dist in local:
                    if tgt == src or dist < 1.0:
                        continue
                    weight = 1.0 / (dist * dist)
                    ax += (x[tgt] - px) * weight
                    ay += (y[tgt] - py) * weight
                    az += (z[tgt] - pz) * weight

                # Also steer toward distant region centers
                for rname, center in region_centers.items():
                    to_r = center - np.array([px, py, pz])
                    d_r = np.linalg.norm(to_r)
                    if d_r > 10:
                        w = region_attract[rname] / (d_r ** 1.5) * 50.0
                        ax += to_r[0] / d_r * w
                        ay += to_r[1] / d_r * w
                        az += to_r[2] / d_r * w

                norm = np.sqrt(ax*ax + ay*ay + az*az)
                if norm > 1e-10:
                    ax /= norm; ay /= norm; az /= norm
                    dx = dx * (1 - turn_rate) + ax * turn_rate
                    dy = dy * (1 - turn_rate) + ay * turn_rate
                    dz = dz * (1 - turn_rate) + az * turn_rate

                # Noise
                dx += rng.randn() * 0.15
                dy += rng.randn() * 0.15
                dz += rng.randn() * 0.15

                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
                    dx /= norm; dy /= norm; dz /= norm

                new_px = px + dx * step_size
                new_py = py + dy * step_size
                new_pz = pz + dz * step_size

                new_cones.append([new_px, new_py, new_pz, dx, dy, dz])

                # Branch
                if len(new_cones) < max_branches and rng.random() < branch_prob:
                    bdx = dx + rng.randn() * 0.5
                    bdy = dy + rng.randn() * 0.5
                    bdz = dz + rng.randn() * 0.5
                    bnorm = np.sqrt(bdx*bdx + bdy*bdy + bdz*bdz)
                    if bnorm > 1e-10:
                        bdx /= bnorm; bdy /= bnorm; bdz /= bnorm
                    new_cones.append([new_px, new_py, new_pz, bdx, bdy, bdz])

            cones = new_cones

    elapsed = time.perf_counter() - start_time
    print(f"  Growth complete: {len(synapses)} synapses in {elapsed:.1f}s")

    return synapses, pair_counts


def analyze_regional(neurons, synapses, regions):
    """Analyze inter-region connectivity."""
    from collections import Counter

    region_labels = neurons['regions']
    n = neurons['n']

    # Region-to-region connection matrix
    region_names = list(regions.keys())
    conn_matrix = {}
    for rn in region_names:
        for rn2 in region_names:
            conn_matrix[(rn, rn2)] = 0

    for src, tgt in synapses:
        r1 = region_labels[src]
        r2 = region_labels[tgt]
        conn_matrix[(r1, r2)] += 1

    print(f"\n  INTER-REGION CONNECTIVITY MATRIX")
    print(f"  {'':15s}", end='')
    for rn in region_names:
        print(f" {rn[:6]:>6s}", end='')
    print()

    for r1 in region_names:
        print(f"  {r1:15s}", end='')
        for r2 in region_names:
            c = conn_matrix[(r1, r2)]
            if c > 0:
                print(f" {c:6d}", end='')
            else:
                print(f"     .", end='')
        total = sum(conn_matrix[(r1, r2)] for r2 in region_names)
        local = conn_matrix[(r1, r1)]
        print(f"  | {total:6d} total, {local/max(total,1)*100:.0f}% local")

    # Which regions are hubs?
    in_by_region = Counter()
    out_by_region = Counter()
    for src, tgt in synapses:
        out_by_region[region_labels[src]] += 1
        in_by_region[region_labels[tgt]] += 1

    print(f"\n  REGION HUB ANALYSIS")
    region_sizes = Counter(region_labels)
    for rn in region_names:
        sz = region_sizes[rn]
        out_per = out_by_region[rn] / max(sz, 1)
        in_per = in_by_region[rn] / max(sz, 1)
        print(f"  {rn:15s}: {sz:4d}N, out/N={out_per:.1f}, in/N={in_per:.1f}, "
              f"ratio={in_per/max(out_per,0.01):.2f}")


def main():
    p = argparse.ArgumentParser(description='Grow a regional brain')
    p.add_argument('--neurons', type=int, default=8000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--contact-radius', type=float, default=10.0)
    p.add_argument('--branch-prob', type=float, default=0.04)
    p.add_argument('--name', default=None)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)

    params = {
        'axon_steps': args.steps,
        'step_size': 5.0,
        'branch_prob': args.branch_prob,
        'max_branches': 8,
        'turn_rate': 0.3,
        'contact_radius': args.contact_radius,
        'max_synapses_per_pair': 12,
        'inh_exc_preference': 2.5,
    }

    print(f"{'='*60}")
    print(f"  V9 REGIONAL BRAIN GROWTH")
    print(f"  Neurons: {args.neurons}, Steps: {args.steps}")
    print(f"  Contact: {args.contact_radius}um, Branch: {args.branch_prob}")
    print(f"  Seed: {args.seed}")
    print(f"  Regions: {', '.join(REGIONS.keys())}")
    print(f"{'='*60}")

    # 1. Place neurons
    print(f"\n  Placing neurons...")
    neurons = place_regional_neurons(args.neurons, REGIONS, rng)

    # 2. Grow axons
    print(f"\n  Growing axons (this takes a while)...")
    synapses, pair_counts = grow_regional_axons(neurons, REGIONS, params, rng)

    # 3. Analyze
    analyze_regional(neurons, synapses, REGIONS)

    # 4. General stats (import from grow.py)
    from grow import compute_stats
    compute_stats(neurons, synapses, pair_counts)

    print(f"\n{'='*60}")
    print(f"  REGIONAL GROWTH COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

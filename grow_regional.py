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
        'birth_order': 5,          # late -- last to develop (like real cortex)
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
        'birth_order': 3,          # mid -- archicortex develops before neocortex
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
        'birth_order': 2,          # early -- emotional circuits develop early
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
        'birth_order': 3,          # mid
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
        'birth_order': 2,          # early -- relay develops before cortex
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
        'birth_order': 0,          # FIRST -- brainstem develops earliest
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
        'birth_order': 1,          # early -- sensory input develops early
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

            # Cell type derived from neuron type (FS/LTS = inhibitory)
            is_exc = chosen not in ('fs', 'lts')
            cell_types.append('EXC' if is_exc else 'INH')

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
    """Grow axons with region-aware chemoattraction + metabolic cost.

    Growth cones are attracted to regions based on their attractant strength.
    Metabolic cost: longer axons have increasing probability of death per step.
    Birth-time ordering: neurons grow in developmental order (brainstem first, cortex last).
    Early-born neurons' existing axons become part of the landscape for later neurons.

    Uses scipy cKDTree for O(log n) contact detection instead of O(n) grid scan.
    """
    from scipy.spatial import cKDTree

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
    inh_target_exc_prob = params.get('inh_target_exc_prob', 0.78)

    # Metabolic cost: constant per-step death probability for growth cones.
    # Uses half-life model: P(survive N steps) = (1 - cost)^N
    # cost=0.005 -> half-life ~138 steps, 61% survive 100 steps (gentle local bias)
    # cost=0.01  -> half-life ~69 steps,  37% survive 100 steps (moderate)
    # cost=0.02  -> half-life ~34 steps,  13% survive 100 steps (strong local bias)
    metabolic_cost = params.get('metabolic_cost', 0.0)  # 0 = off (legacy behavior)

    # Build KD-tree for fast spatial queries (O(log n) vs O(n) grid)
    positions = np.column_stack([x, y, z])
    tree = cKDTree(positions)

    # Precompute region center arrays and attractant strengths
    region_names = list(regions.keys())
    n_regions = len(region_names)
    rc_array = np.array([regions[rn]['center'] for rn in region_names], dtype=float)  # (R, 3)
    ra_array = np.array([regions[rn]['attractant'] for rn in region_names], dtype=float)  # (R,)
    region_name_to_idx = {rn: i for i, rn in enumerate(region_names)}

    # EXC lookup as boolean array for fast INH targeting
    is_exc = np.array([ct == 'EXC' for ct in cell_types])

    # Birth-time ordering: sort neurons by region birth_order
    # Within each birth wave, neurons grow in random order
    birth_orders = np.array([regions[region_labels[i]].get('birth_order', 3)
                             for i in range(n)])
    # Stable sort by birth_order preserves original within-group ordering
    # Then shuffle within each birth wave for randomness
    growth_order = np.argsort(birth_orders, kind='stable')
    # Shuffle within birth waves
    unique_waves = np.unique(birth_orders)
    for wave in unique_waves:
        wave_mask = birth_orders[growth_order] == wave
        wave_indices = np.where(wave_mask)[0]
        rng.shuffle(growth_order[wave_indices[0]:wave_indices[-1]+1])

    synapses = []
    pair_counts = {}
    synapse_distances = {}  # track formation distance for weight scaling

    start_time = time.perf_counter()

    # Local steering parameters (scale-aware)
    local_freq = max(20, n // 200)
    local_radius = min(40.0, 20.0 + 20000.0 / max(n, 1))

    report_interval = max(100, n // 20)
    cones_killed = 0
    cones_total = 0

    for grow_idx, src in enumerate(growth_order):
        if grow_idx % report_interval == 0 and grow_idx > 0:
            elapsed = time.perf_counter() - start_time
            rate = grow_idx / elapsed
            eta = (n - grow_idx) / rate / 60
            killed_pct = cones_killed / max(cones_total, 1) * 100
            print(f"  Growing {grow_idx}/{n} ({rate:.0f}/s, {len(synapses)} syn, "
                  f"ETA {eta:.1f}m, {killed_pct:.0f}% cones died)")

        src_pos = positions[src]
        src_region = region_labels[src]
        src_region_idx = region_name_to_idx[src_region]
        is_inh = cell_types[src] == 'INH'

        # Growth direction: biased toward high-attractant regions (NOT own region)
        direction = rng.randn(3)

        to_regions = rc_array - src_pos  # (R, 3)
        dists = np.linalg.norm(to_regions, axis=1)  # (R,)
        mask = (dists > 10)
        mask[src_region_idx] = False  # exclude own region
        if np.any(mask):
            strengths = ra_array[mask] / np.sqrt(dists[mask])  # (K,)
            unit_dirs = to_regions[mask] / dists[mask, None]  # (K, 3)
            attract_dir = np.sum(unit_dirs * strengths[:, None], axis=0)
            anorm = np.linalg.norm(attract_dir)
            if anorm > 0.01:
                attract_dir /= anorm
                direction = direction * 0.5 + attract_dir * 0.5

        dnorm = np.linalg.norm(direction)
        if dnorm > 1e-10:
            direction /= dnorm

        # Start growth from soma
        # Cone format: [px, py, pz, dx, dy, dz, distance_traveled]
        cones = [[
            src_pos[0] + direction[0] * step_size,
            src_pos[1] + direction[1] * step_size,
            src_pos[2] + direction[2] * step_size,
            direction[0], direction[1], direction[2],
            step_size  # initial distance
        ]]

        for step in range(steps):
            if not cones:
                break

            new_cones = []
            for cone in cones:
                cpx, cpy, cpz, dx, dy, dz, dist_traveled = cone
                cones_total += 1

                # Metabolic cost: constant per-step death (half-life model)
                if metabolic_cost > 0:
                    if rng.random() < metabolic_cost:
                        cones_killed += 1
                        continue  # cone dies, no more growth from this tip

                # Contact check via KD-tree (O(log n))
                nearby_idx = tree.query_ball_point([cpx, cpy, cpz], contact_radius)
                for tgt in nearby_idx:
                    if tgt == src:
                        continue
                    # INH preferential targeting
                    if is_inh:
                        if is_exc[tgt]:
                            if rng.random() > inh_target_exc_prob:
                                continue
                        else:
                            if rng.random() > (1.0 - inh_target_exc_prob):
                                continue
                    pair = (src, tgt)
                    count = pair_counts.get(pair, 0)
                    if count >= max_syn_pair:
                        continue
                    synapses.append(pair)
                    pair_counts[pair] = count + 1
                    # Record formation distance (for weight scaling)
                    if pair not in synapse_distances:
                        synapse_distances[pair] = dist_traveled

                # Steering: region centers (vectorized) + occasional local
                cone_pos = np.array([cpx, cpy, cpz])
                to_r = rc_array - cone_pos  # (R, 3)
                d_r = np.linalg.norm(to_r, axis=1)  # (R,)
                rmask = d_r > 10
                if np.any(rmask):
                    weights = ra_array[rmask] / (d_r[rmask] ** 1.5) * 50.0
                    unit_r = to_r[rmask] / d_r[rmask, None]
                    accel = np.sum(unit_r * weights[:, None], axis=0)
                    ax, ay, az = accel[0], accel[1], accel[2]
                else:
                    ax, ay, az = 0.0, 0.0, 0.0

                # Local cell attraction (KD-tree, scale-aware frequency)
                if step % local_freq == 0:
                    local_idx = tree.query_ball_point([cpx, cpy, cpz], local_radius)
                    if local_idx:
                        local_pos = positions[local_idx]  # (K, 3)
                        diffs = local_pos - cone_pos  # (K, 3)
                        local_dists = np.linalg.norm(diffs, axis=1)  # (K,)
                        valid = (local_dists > 1.0)
                        # Exclude self
                        for vi, li in enumerate(local_idx):
                            if li == src:
                                valid[vi] = False
                        if np.any(valid):
                            w = 1.0 / (local_dists[valid] ** 2)
                            pull = np.sum(diffs[valid] * w[:, None], axis=0)
                            ax += pull[0]; ay += pull[1]; az += pull[2]

                norm = ax*ax + ay*ay + az*az
                if norm > 1e-20:
                    norm = np.sqrt(norm)
                    ax /= norm; ay /= norm; az /= norm
                    dx = dx * (1 - turn_rate) + ax * turn_rate
                    dy = dy * (1 - turn_rate) + ay * turn_rate
                    dz = dz * (1 - turn_rate) + az * turn_rate

                # Noise
                dx += rng.randn() * 0.15
                dy += rng.randn() * 0.15
                dz += rng.randn() * 0.15

                norm = dx*dx + dy*dy + dz*dz
                if norm > 1e-20:
                    norm = np.sqrt(norm)
                    dx /= norm; dy /= norm; dz /= norm

                new_px = cpx + dx * step_size
                new_py = cpy + dy * step_size
                new_pz = cpz + dz * step_size
                new_dist = dist_traveled + step_size

                new_cones.append([new_px, new_py, new_pz, dx, dy, dz, new_dist])

                # Branch (metabolic cost reduces branching for distant cones)
                effective_branch_prob = branch_prob
                if metabolic_cost > 0 and dist_traveled > step_size * 30:
                    # Only reduce branching after 30 steps (150um) -- local branching stays full
                    effective_branch_prob *= 0.5
                if len(new_cones) < max_branches and rng.random() < effective_branch_prob:
                    bdx = dx + rng.randn() * 0.5
                    bdy = dy + rng.randn() * 0.5
                    bdz = dz + rng.randn() * 0.5
                    bnorm = bdx*bdx + bdy*bdy + bdz*bdz
                    if bnorm > 1e-20:
                        bnorm = np.sqrt(bnorm)
                        bdx /= bnorm; bdy /= bnorm; bdz /= bnorm
                    new_cones.append([new_px, new_py, new_pz, bdx, bdy, bdz, new_dist])

            cones = new_cones

    elapsed = time.perf_counter() - start_time
    if metabolic_cost > 0:
        killed_pct = cones_killed / max(cones_total, 1) * 100
        print(f"  Growth complete: {len(synapses)} synapses in {elapsed:.1f}s "
              f"(metabolic cost={metabolic_cost}, {killed_pct:.0f}% cones died)")
    else:
        print(f"  Growth complete: {len(synapses)} synapses in {elapsed:.1f}s")

    return synapses, pair_counts, synapse_distances


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


def prune_weak_synapses(synapses, pair_counts, synapse_distances, prune_fraction, rng):
    """Drop the weakest synapses post-growth. Sharpens regional structure.

    Weakness = long formation distance + few contacts. Multi-contact pairs
    survive even at long range (they represent strong biological connections).
    """
    if prune_fraction <= 0:
        return synapses, pair_counts, synapse_distances

    # Score each unique pair: lower = weaker = more likely to be pruned
    pair_scores = {}
    for pair, count in pair_counts.items():
        dist = synapse_distances.get(pair, 100.0)
        # Score: more contacts = stronger, shorter distance = stronger
        pair_scores[pair] = count / (1.0 + dist * 0.01)

    # Sort by score, prune the bottom fraction
    sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1])
    n_prune = int(len(sorted_pairs) * prune_fraction)
    pruned_set = set(pair for pair, _ in sorted_pairs[:n_prune])

    new_synapses = [s for s in synapses if (s[0], s[1]) not in pruned_set]
    new_pair_counts = {p: c for p, c in pair_counts.items() if p not in pruned_set}
    new_distances = {p: d for p, d in synapse_distances.items() if p not in pruned_set}

    print(f"  Pruned {n_prune} weak pairs ({prune_fraction*100:.0f}%), "
          f"{len(new_pair_counts)} pairs remain ({len(new_synapses)} synapses)")

    return new_synapses, new_pair_counts, new_distances


def save_regional_db(neurons, synapses, pair_counts, params, regions_used, db_path, rng,
                     synapse_distances=None, synapse_mode='plastic'):
    """Save grown regional network to V8-compatible SQLite database."""
    v8_path = os.path.join(os.path.dirname(BASE), 'inner-models-v8')
    sys.path.insert(0, v8_path)
    from schema import create_brain_db

    n = neurons['n']
    jitter = params.get('param_jitter', 0.15)

    TYPE_PARAMS = {
        'rs':  {'a': 0.02,  'b': 0.2,  'c': -65.0, 'd': 8.0},
        'fs':  {'a': 0.1,   'b': 0.2,  'c': -65.0, 'd': 2.0},
        'ib':  {'a': 0.02,  'b': 0.2,  'c': -55.0, 'd': 4.0},
        'lts': {'a': 0.02,  'b': 0.25, 'c': -65.0, 'd': 2.0},
        'ch':  {'a': 0.02,  'b': 0.2,  'c': -50.0, 'd': 2.0},
    }

    conn = create_brain_db(db_path)

    # Insert neurons with region metadata
    for i in range(n):
        nt = neurons['neuron_types'][i]
        base = TYPE_PARAMS[nt]
        nt_upper = nt.upper()  # V8 engine expects uppercase
        a = base['a'] * (1 + rng.uniform(-jitter, jitter))
        b = base['b'] * (1 + rng.uniform(-jitter, jitter))
        c = base['c'] * (1 + rng.uniform(-jitter, jitter))
        d = base['d'] * (1 + rng.uniform(-jitter, jitter))

        # D1/D2 by region -- basal_ganglia gets strong D1/D2, others mixed
        region = neurons['regions'][i]
        if region == 'basal_ganglia':
            r = rng.random()
            dopa_sens = rng.uniform(0.5, 1.0) if r < 0.5 else rng.uniform(-1.0, -0.5)
        elif region in ('amygdala', 'cortex'):
            dopa_sens = rng.uniform(-0.5, 0.5)
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

    db_ids = [row[0] for row in conn.execute("SELECT id FROM neurons ORDER BY id")]
    idx_to_id = {i: db_ids[i] for i in range(n)}

    cell_types = neurons['cell_types']
    region_labels = neurons['regions']

    for (src, tgt), n_contacts in pair_counts.items():
        src_exc = cell_types[src] == 'EXC'
        src_region = region_labels[src]
        tgt_region = region_labels[tgt]

        # Weight: base scaled by contacts, sign from neuron type
        # Distance penalty: longer formation distance = weaker initial weight
        if src_exc:
            base_w = 2.0
        else:
            base_w = -2.5  # inhibitory -> negative weight
        weight = base_w * (1.0 + 0.3 * (n_contacts - 1))
        # Apply distance-based attenuation if we have formation distances
        if synapse_distances and (src, tgt) in synapse_distances:
            form_dist = synapse_distances[(src, tgt)]
            # Exponential decay: local synapses full strength, distant ones weaker
            # Half-life ~200um (40 steps * 5um/step)
            dist_scale = np.exp(-form_dist / 200.0)
            weight *= max(0.3, dist_scale)  # floor at 30% to not kill long-range entirely

        # Synapse type assignment based on mode:
        # INH -> always fixed (inhibition provides structure, doesn't learn)
        # EXC -> depends on synapse_mode:
        #   'plastic':       all EXC = plastic (full STDP with LTP+LTD)
        #   'developmental': 1-2 contact = developmental (STDP+pruning), 3+ = plastic
        #   'reward':        all EXC = reward_plastic (3-factor, needs deliver_reward)
        #   'mixed':         same-region = plastic (Hebbian), cross-region = reward_plastic (dopamine)
        if not src_exc:
            syn_type = 'fixed'
        elif synapse_mode == 'plastic':
            syn_type = 'plastic'
        elif synapse_mode == 'reward':
            syn_type = 'reward_plastic'
        elif synapse_mode == 'developmental':
            if n_contacts >= 3:
                syn_type = 'plastic'
            else:
                syn_type = 'developmental'
        elif synapse_mode == 'mixed':
            # Local circuits: Hebbian STDP (learns from activity patterns)
            # Cross-region projections: reward-gated (learns from dopamine)
            if src_region == tgt_region:
                syn_type = 'plastic'
            else:
                syn_type = 'reward_plastic'

        # Distance-based delay
        d = np.sqrt((neurons['x'][src] - neurons['x'][tgt])**2 +
                     (neurons['y'][src] - neurons['y'][tgt])**2 +
                     (neurons['z'][src] - neurons['z'][tgt])**2)
        delay = max(1, int(d / 50.0))

        params_dict = {'n_contacts': n_contacts, 'src_region': src_region, 'tgt_region': tgt_region}
        # Set w_min/w_max for plastic types based on sign
        if syn_type in ('plastic', 'reward_plastic', 'developmental'):
            if weight < 0:
                params_dict['w_min'] = -10.0
                params_dict['w_max'] = 0.0
            else:
                params_dict['w_min'] = 0.0
                params_dict['w_max'] = 10.0
        # Developmental: longer critical period for complex brains
        if syn_type == 'developmental':
            params_dict['critical_period'] = 30000   # 30K ticks to prove worth
            params_dict['pruning_threshold'] = 0.01  # lenient -- even weak correlation keeps you alive
            params_dict['eval_interval'] = 5000      # evaluate every 5K ticks
        params_json = json.dumps(params_dict)

        conn.execute(
            """INSERT INTO synapses
               (source, target, synapse_type, weight, delay, params, state)
               VALUES (?, ?, ?, ?, ?, ?, '{}')""",
            (idx_to_id[src], idx_to_id[tgt], syn_type, weight, delay, params_json)
        )

    conn.commit()

    n_syn = conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
    n_reward = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='reward_plastic'").fetchone()[0]
    n_plastic = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='plastic'").fetchone()[0]
    n_fixed = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='fixed'").fetchone()[0]
    n_dev = conn.execute("SELECT COUNT(*) FROM synapses WHERE synapse_type='developmental'").fetchone()[0]

    conn.close()

    print(f"\n  Saved to: {db_path}")
    print(f"  {n} neurons, {n_syn} unique-pair synapses")
    type_parts = [f"{n_fixed} fixed"]
    if n_plastic > 0:
        type_parts.append(f"{n_plastic} plastic")
    if n_reward > 0:
        type_parts.append(f"{n_reward} reward_plastic")
    if n_dev > 0:
        type_parts.append(f"{n_dev} developmental")
    print(f"  Types: {', '.join(type_parts)}")

    return db_path


def auto_contact_radius(n_neurons, regions):
    """Estimate contact_radius to yield ~10-15 synapses per neuron.

    Synapse count scales roughly as contact_radius^3 * density * steps * branches.
    We use a conservative formula since too-many is worse than too-few
    (too-few still produces interesting topology, too-many is noise).
    """
    # Effective volume from all region spheres
    total_vol = 0
    for rinfo in regions.values():
        r = rinfo['radius']
        total_vol += (4/3) * np.pi * r**3

    density = n_neurons / total_vol

    # Target: ~12 synapses per neuron
    # Empirical: at density ~8e-5 N/um^3, radius 2.0um works for ~10-15 syn/N
    # Scale: r ~ (target_density / actual_density)^(1/3) * base_r
    ref_density = 8e-5
    ref_radius = 2.0

    auto_r = ref_radius * (ref_density / density) ** (1/3)

    # Clamp to reasonable range
    auto_r = max(1.5, min(auto_r, 6.0))
    return auto_r


def main():
    p = argparse.ArgumentParser(description='Grow a regional brain')
    p.add_argument('--neurons', type=int, default=8000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--contact-radius', type=float, default=None,
                   help='Contact radius in um (auto-scaled if not set)')
    p.add_argument('--branch-prob', type=float, default=0.04)
    p.add_argument('--name', default=None)
    p.add_argument('--save', action='store_true', default=True)
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--config', default=None,
                   help='Region config preset: balanced, cortex_heavy, memory_dense, thalamic_hub, spread')
    p.add_argument('--metabolic-cost', type=float, default=0.0,
                   help='Metabolic cost per growth step (0=off, 0.003=low, 0.005=med, 0.01=high)')
    p.add_argument('--prune', type=float, default=0.0,
                   help='Post-growth prune fraction (0=off, 0.1=10%%, 0.15=15%%)')
    p.add_argument('--synapse-mode', default='plastic',
                   choices=['plastic', 'developmental', 'reward', 'mixed'],
                   help='EXC synapse type: plastic (STDP), developmental (STDP+pruning), reward (3-factor), mixed (local=plastic, cross=reward)')
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)

    # Select region configuration
    regions = dict(REGIONS)  # deep copy
    regions = {k: dict(v) for k, v in regions.items()}

    if args.config == 'cortex_heavy':
        regions['cortex']['n_fraction'] = 0.50
        regions['cortex']['attractant'] = 1.5
        regions['cortex']['radius'] = 180
        regions['hippocampus']['n_fraction'] = 0.10
        regions['sensory']['n_fraction'] = 0.05
        regions['brainstem']['n_fraction'] = 0.08
        regions['basal_ganglia']['n_fraction'] = 0.09
        regions['thalamus']['n_fraction'] = 0.10
        regions['amygdala']['n_fraction'] = 0.08
    elif args.config == 'memory_dense':
        regions['hippocampus']['n_fraction'] = 0.25
        regions['hippocampus']['local_density'] = 3.0
        regions['hippocampus']['attractant'] = 2.0
        regions['amygdala']['n_fraction'] = 0.15
        regions['amygdala']['attractant'] = 2.5
        regions['cortex']['n_fraction'] = 0.25
        regions['basal_ganglia']['n_fraction'] = 0.10
        regions['thalamus']['n_fraction'] = 0.10
        regions['brainstem']['n_fraction'] = 0.08
        regions['sensory']['n_fraction'] = 0.07
    elif args.config == 'thalamic_hub':
        regions['thalamus']['attractant'] = 3.5
        regions['thalamus']['n_fraction'] = 0.15
        regions['thalamus']['radius'] = 80
        regions['brainstem']['attractant'] = 1.5
        regions['cortex']['attractant'] = 0.7
        regions['cortex']['n_fraction'] = 0.30
        regions['hippocampus']['n_fraction'] = 0.12
        regions['amygdala']['n_fraction'] = 0.08
        regions['basal_ganglia']['n_fraction'] = 0.10
        regions['sensory']['n_fraction'] = 0.07
        regions['brainstem']['n_fraction'] = 0.18
    elif args.config == 'spread':
        for rn in regions:
            regions[rn]['radius'] = int(regions[rn]['radius'] * 1.5)
            regions[rn]['local_density'] = max(0.5, regions[rn]['local_density'] * 0.7)
            regions[rn]['attractant'] = regions[rn]['attractant'] * 0.8
    elif args.config == 'amygdala_driven':
        regions['amygdala']['n_fraction'] = 0.18
        regions['amygdala']['attractant'] = 3.0
        regions['amygdala']['radius'] = 80
        regions['basal_ganglia']['n_fraction'] = 0.15
        regions['basal_ganglia']['attractant'] = 1.2
        regions['cortex']['n_fraction'] = 0.25
        regions['hippocampus']['n_fraction'] = 0.12
        regions['thalamus']['n_fraction'] = 0.10
        regions['brainstem']['n_fraction'] = 0.12
        regions['sensory']['n_fraction'] = 0.08

    # Auto-calculate contact radius if not specified
    if args.contact_radius is None:
        cr = auto_contact_radius(args.neurons, regions)
    else:
        cr = args.contact_radius

    params = {
        'axon_steps': args.steps,
        'step_size': 5.0,
        'branch_prob': args.branch_prob,
        'max_branches': 8,
        'turn_rate': 0.3,
        'contact_radius': cr,
        'max_synapses_per_pair': 12,
        'inh_target_exc_prob': 0.78,
        'param_jitter': 0.15,
        'metabolic_cost': args.metabolic_cost,
    }

    config_label = args.config or 'balanced'
    print(f"{'='*60}")
    print(f"  V9 REGIONAL BRAIN GROWTH")
    print(f"  Config: {config_label}")
    print(f"  Neurons: {args.neurons}, Steps: {args.steps}")
    print(f"  Contact: {cr:.2f}um {'(auto)' if args.contact_radius is None else ''}, Branch: {args.branch_prob}")
    print(f"  Metabolic: {args.metabolic_cost} {'(off)' if args.metabolic_cost == 0 else ''}, Prune: {args.prune*100:.0f}%")
    print(f"  Synapse mode: {args.synapse_mode}")
    print(f"  Seed: {args.seed}")
    print(f"  Regions: {', '.join(regions.keys())}")
    print(f"  Birth order: {' -> '.join(rn for rn, _ in sorted(regions.items(), key=lambda x: x[1].get('birth_order', 3)))}")
    print(f"{'='*60}")

    # 1. Place neurons
    print(f"\n  Placing neurons...")
    neurons = place_regional_neurons(args.neurons, regions, rng)

    # 2. Grow axons
    print(f"\n  Growing axons (this takes a while)...")
    synapses, pair_counts, synapse_distances = grow_regional_axons(neurons, regions, params, rng)

    # 2.5. Post-growth pruning (drop weak long-range, low-contact pairs)
    if args.prune > 0:
        print(f"\n  Pruning weakest {args.prune*100:.0f}% of connections...")
        synapses, pair_counts, synapse_distances = prune_weak_synapses(
            synapses, pair_counts, synapse_distances, args.prune, rng)

    # 3. Analyze
    analyze_regional(neurons, synapses, regions)

    # 4. General stats (import from grow.py)
    from grow import compute_stats
    compute_stats(neurons, synapses, pair_counts)

    # 5. Save to DB
    if args.save and not args.no_save:
        name = args.name or f"regional_{config_label}_s{args.seed}"
        db_dir = os.path.join(BASE, 'brains')
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"{name}.db")
        save_regional_db(neurons, synapses, pair_counts, params, regions, db_path, rng,
                         synapse_distances=synapse_distances, synapse_mode=args.synapse_mode)

    print(f"\n{'='*60}")
    print(f"  REGIONAL GROWTH COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

"""
analyze_brain.py -- Deep analysis of V9 grown brains.

Measures:
1. Spontaneous oscillation detection (do healthy brains oscillate without input?)
2. Population activity patterns (what representations emerge?)
3. Weight distribution and evolution
4. Information flow analysis (which paths carry signal?)
5. Criticality metrics (branching ratio, avalanche sizes)

Usage:
    py analyze_brain.py --brain brains/metabolic_med_v2_s200.db
    py analyze_brain.py --brain brains/cortex_heavy_v2_s100.db --ticks 20000
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


def analyze_oscillations(brain, n, tonic, ticks=10000, rng=None):
    """Detect spontaneous oscillations with tonic drive only (no input).

    Real brains show spontaneous rhythms: delta (1-4Hz), theta (4-8Hz),
    alpha (8-13Hz), beta (13-30Hz), gamma (30-80Hz).

    At 1ms tick resolution:
    - 1Hz = 1000 ticks/cycle
    - 10Hz = 100 ticks/cycle
    - 40Hz = 25 ticks/cycle
    """
    print(f"\n  OSCILLATION ANALYSIS (tonic={tonic}, {ticks} ticks, no input)")

    # Record population firing rate in 10ms bins
    bin_size = 10  # 10ms bins
    n_bins = ticks // bin_size
    rates = np.zeros(n_bins)

    I_ext = np.full(n, tonic)
    for tick in range(ticks):
        fired = brain.tick(external_I=I_ext)
        rates[tick // bin_size] += len(fired)

    # Normalize to firing rate (spikes per neuron per second)
    rates = rates / n / (bin_size / 1000.0)

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    cv = std_rate / max(mean_rate, 1e-6)

    print(f"  Mean firing rate: {mean_rate:.2f} Hz")
    print(f"  Rate CV: {cv:.3f}")

    if mean_rate < 0.01:
        print(f"  Brain is silent at tonic {tonic} -- no oscillations possible")
        return {'mean_rate': mean_rate, 'cv': cv, 'dominant_freq': 0, 'power': {}}

    # FFT to find dominant frequencies
    fft = np.fft.rfft(rates - mean_rate)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n_bins, d=bin_size / 1000.0)  # Hz

    # Find peaks in biologically relevant bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80),
    }

    band_power = {}
    total_power = np.sum(power[1:])  # exclude DC
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp = np.sum(power[mask])
        band_power[band_name] = float(bp / max(total_power, 1e-10))

    # Dominant frequency
    peak_idx = np.argmax(power[1:]) + 1
    dominant_freq = freqs[peak_idx]

    print(f"  Dominant frequency: {dominant_freq:.1f} Hz")
    print(f"  Band power:")
    for band_name, bp in band_power.items():
        bar = '#' * int(bp * 50)
        print(f"    {band_name:>8s}: {bp*100:5.1f}% {bar}")

    return {
        'mean_rate': float(mean_rate),
        'cv': float(cv),
        'dominant_freq': float(dominant_freq),
        'power': band_power,
    }


def analyze_criticality(brain, n, tonic, ticks=10000):
    """Measure criticality via branching ratio and avalanche sizes.

    Branching ratio ~1.0 = critical (edge of chaos, maximum information processing)
    < 1.0 = subcritical (activity dies out)
    > 1.0 = supercritical (activity explodes)
    """
    print(f"\n  CRITICALITY ANALYSIS ({ticks} ticks)")

    I_ext = np.full(n, tonic)
    prev_fired = 0
    ratios = []
    avalanche_sizes = []
    current_avalanche = 0
    silent_count = 0

    for tick in range(ticks):
        fired = brain.tick(external_I=I_ext)
        n_fired = len(fired)

        if prev_fired > 0:
            ratios.append(n_fired / prev_fired)

        if n_fired > 0:
            current_avalanche += n_fired
            silent_count = 0
        else:
            if current_avalanche > 0:
                avalanche_sizes.append(current_avalanche)
                current_avalanche = 0
            silent_count += 1

        prev_fired = n_fired

    if current_avalanche > 0:
        avalanche_sizes.append(current_avalanche)

    branching = np.median(ratios) if ratios else 0
    mean_avalanche = np.mean(avalanche_sizes) if avalanche_sizes else 0
    max_avalanche = max(avalanche_sizes) if avalanche_sizes else 0

    print(f"  Branching ratio: {branching:.4f} (target: 1.0)")
    print(f"  Avalanche sizes: mean={mean_avalanche:.1f}, max={max_avalanche}")
    print(f"  N avalanches: {len(avalanche_sizes)}")

    # Power-law test on avalanche sizes
    if len(avalanche_sizes) > 10:
        sizes = np.array(avalanche_sizes)
        sizes = sizes[sizes > 0]
        if len(sizes) > 3:
            from collections import Counter
            size_counts = Counter(sizes.astype(int))
            s_vals = sorted(size_counts.keys())
            if len(s_vals) > 3 and max(s_vals) > 1:
                counts = [size_counts[s] for s in s_vals]
                log_s = np.log(np.array(s_vals, dtype=float))
                log_c = np.log(np.array(counts, dtype=float))
                coeffs = np.polyfit(log_s, log_c, 1)
                r2 = 1 - np.sum((log_c - np.polyval(coeffs, log_s))**2) / np.sum((log_c - np.mean(log_c))**2)
                print(f"  Avalanche power law: exponent={-coeffs[0]:.2f}, R2={r2:.3f}")
                print(f"    (Critical systems show exponent ~1.5)")

    return {
        'branching': float(branching),
        'mean_avalanche': float(mean_avalanche),
        'max_avalanche': int(max_avalanche),
        'n_avalanches': len(avalanche_sizes),
    }


def analyze_weights(brain_data):
    """Analyze weight distribution of the brain."""
    print(f"\n  WEIGHT ANALYSIS")

    synapses = brain_data['synapses']
    weights = np.array([s['weight'] for s in synapses])

    n_pos = np.sum(weights > 0)
    n_neg = np.sum(weights < 0)
    n_zero = np.sum(weights == 0)

    print(f"  Total synapses: {len(weights)}")
    print(f"  Positive (exc): {n_pos} ({n_pos/len(weights)*100:.1f}%)")
    print(f"  Negative (inh): {n_neg} ({n_neg/len(weights)*100:.1f}%)")
    print(f"  Zero: {n_zero}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  Mean |w|: {np.mean(np.abs(weights)):.3f}")
    print(f"  Std |w|: {np.std(np.abs(weights)):.3f}")

    # By synapse type
    type_weights = {}
    for s in synapses:
        st = s['type']
        if st not in type_weights:
            type_weights[st] = []
        type_weights[st].append(s['weight'])

    print(f"\n  By synapse type:")
    for st, ws in sorted(type_weights.items()):
        ws = np.array(ws)
        print(f"    {st:15s}: n={len(ws):5d}, mean={np.mean(ws):+.3f}, "
              f"std={np.std(ws):.3f}, range=[{ws.min():.3f}, {ws.max():.3f}]")

    return {
        'n_synapses': len(weights),
        'n_exc': int(n_pos),
        'n_inh': int(n_neg),
        'mean_abs': float(np.mean(np.abs(weights))),
        'weight_range': [float(weights.min()), float(weights.max())],
    }


def analyze_information_flow(brain_data, brain, n, tonic, ticks=5000):
    """Measure information flow from input to output electrodes.

    Inject structured input into top out-degree neurons and measure
    whether activity propagates to top in-degree neurons.
    """
    print(f"\n  INFORMATION FLOW ANALYSIS ({ticks} ticks)")

    # Find input/output neurons (same logic as viewer)
    in_deg = np.zeros(n, dtype=int)
    out_deg = np.zeros(n, dtype=int)
    for s in brain_data['synapses']:
        out_deg[s['source']] += 1
        in_deg[s['target']] += 1

    n_input = min(64, n // 10)
    n_output = min(64, n // 10)

    out_sorted = np.argsort(out_deg)[::-1]
    input_idx = out_sorted[:n_input]

    input_set = set(input_idx.tolist())
    in_sorted = np.argsort(in_deg)[::-1]
    output_idx = []
    for idx in in_sorted:
        if int(idx) not in input_set:
            output_idx.append(int(idx))
        if len(output_idx) >= n_output:
            break
    output_set = set(output_idx)

    # Phase 1: baseline (tonic only)
    print(f"  Phase 1: Baseline ({ticks//2} ticks, tonic only)...")
    baseline_output_rate = 0
    I_ext = np.full(n, tonic)
    for tick in range(ticks // 2):
        fired = brain.tick(external_I=I_ext)
        for f in fired:
            if int(f) in output_set:
                baseline_output_rate += 1
    baseline_output_rate /= (ticks // 2)

    # Phase 2: structured input
    print(f"  Phase 2: Structured input ({ticks//2} ticks, pulse on input neurons)...")
    stim_output_rate = 0
    for tick in range(ticks // 2):
        I_ext = np.full(n, tonic)
        # Pulse every 20 ticks
        if tick % 20 < 5:
            I_ext[input_idx] += 8.0
        fired = brain.tick(external_I=I_ext)
        for f in fired:
            if int(f) in output_set:
                stim_output_rate += 1
    stim_output_rate /= (ticks // 2)

    transfer = stim_output_rate / max(baseline_output_rate, 0.001)
    print(f"  Baseline output rate: {baseline_output_rate:.4f} spikes/tick")
    print(f"  Stimulated output rate: {stim_output_rate:.4f} spikes/tick")
    print(f"  Transfer ratio: {transfer:.2f}x")
    print(f"    (>1.0 = input reaches output, <1.0 = input doesn't propagate)")

    return {
        'baseline_rate': float(baseline_output_rate),
        'stim_rate': float(stim_output_rate),
        'transfer_ratio': float(transfer),
    }


def main():
    p = argparse.ArgumentParser(description='Analyze V9 brain')
    p.add_argument('--brain', required=True, help='Path to brain DB')
    p.add_argument('--tonic', type=float, default=2.8, help='Tonic drive')
    p.add_argument('--ticks', type=int, default=10000, help='Analysis ticks')
    args = p.parse_args()

    print(f"{'='*70}")
    print(f"  BRAIN ANALYSIS: {os.path.basename(args.brain)}")
    print(f"{'='*70}")

    data = load(args.brain)
    brain = Brain(data, learn=False)  # no learning during analysis
    n = brain.n

    print(f"  Brain: {n}N, {len(data['synapses'])} synapses")

    results = {}

    # Weight analysis (no simulation needed)
    results['weights'] = analyze_weights(data)

    # Oscillations
    results['oscillations'] = analyze_oscillations(brain, n, args.tonic, args.ticks)

    # Criticality
    results['criticality'] = analyze_criticality(brain, n, args.tonic, args.ticks)

    # Information flow
    results['info_flow'] = analyze_information_flow(data, brain, n, args.tonic, args.ticks)

    # Save
    results_dir = os.path.join(BASE, 'results')
    os.makedirs(results_dir, exist_ok=True)
    brain_name = os.path.splitext(os.path.basename(args.brain))[0]
    out_path = os.path.join(results_dir, f"analysis_{brain_name}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()

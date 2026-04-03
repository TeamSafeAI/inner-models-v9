"""
brain_viewer_server.py -- WebSocket server streaming brain activity to a 3D viewer.

Loads a V9-grown brain, ticks it with tonic drive, and sends spike data
to a browser client for real-time 3D visualization. Supports sensory I/O:
audio in (mic FFT), visual in (webcam pixels), touch in (electrode probe),
and motor out (firing readout from designated neurons).

Usage:
    py brain_viewer_server.py
    py brain_viewer_server.py --brain brains/regional_cortex_heavy_s42.db
    py brain_viewer_server.py --tonic 3.0 --speed 2.0
"""
import os, sys, json, time, asyncio, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(os.path.dirname(BASE), 'inner-models-v8')
sys.path.insert(0, V8)

from engine.loader import load
from engine.runner import Brain


NEURON_TYPE_COLORS = {
    'RS':  [0.3, 0.6, 1.0],   # blue -- regular spiking (excitatory)
    'FS':  [1.0, 0.25, 0.25], # red -- fast spiking (inhibitory)
    'IB':  [0.3, 0.9, 0.4],   # green -- intrinsically bursting
    'CH':  [1.0, 0.8, 0.2],   # yellow -- chattering
    'LTS': [0.9, 0.4, 0.8],   # magenta -- low-threshold spiking (inhibitory)
}

# I/O electrode configuration -- population encoding (Nengo-inspired)
# Each sensory channel maps to an ENSEMBLE of neurons with different tuning curves.
# Motor output is a weighted population decode, not single-neuron readout.
IO = {
    # Audio: 16 frequency bands, each encoded by a population of 8 neurons
    'audio_channels': 16,     # FFT bins (client downsamples to this)
    'audio_pop_size': 8,      # neurons per frequency band
    'audio_gain': 6.0,        # mA per unit FFT magnitude

    # Visual: 64 pixels (8x8 grid), each encoded by 4 neurons
    'visual_channels': 64,    # pixels (8x8 downsampled webcam)
    'visual_pop_size': 4,     # neurons per pixel
    'visual_gain': 5.0,       # mA per unit pixel intensity

    # Motor: 8 output channels, each decoded from a population of 16 neurons
    'motor_channels': 8,      # output dimensions (decoded values)
    'motor_pop_size': 16,     # neurons per output channel

    # Touch
    'touch_strength': 15.0,   # mA default touch probe
    'touch_radius': 40.0,     # spatial units for touch spread
}


def build_population_encoding(n_channels, pop_size, rng):
    """Build Nengo-style population encoding parameters.

    Each neuron in a channel's ensemble has:
    - intercept: where in [0, 1] it starts responding (tuning curve shift)
    - gain: how steeply it responds (tuning curve width)

    Returns: intercepts (channels, pop_size), gains (channels, pop_size)
    """
    # Spread intercepts evenly across [0, 1] with jitter
    intercepts = np.zeros((n_channels, pop_size))
    gains = np.zeros((n_channels, pop_size))
    for ch in range(n_channels):
        # Uniform spread of preferred stimuli + noise
        intercepts[ch] = np.linspace(0.05, 0.95, pop_size) + rng.randn(pop_size) * 0.05
        intercepts[ch] = np.clip(intercepts[ch], 0.0, 1.0)
        # Gains vary so some neurons are sharp responders, some broad
        gains[ch] = rng.uniform(2.0, 8.0, pop_size)
    return intercepts, gains


def encode_population(signal, intercepts, gains, gain_scale):
    """Encode signal values into per-neuron current injection.

    signal: (n_channels,) -- values in [0, 1]
    Returns: (n_channels * pop_size,) -- current injection per neuron
    """
    n_channels, pop_size = intercepts.shape
    # Tuning curve: ReLU(gain * (signal - intercept))
    # Each neuron fires more when signal exceeds its intercept
    sig = np.clip(signal, 0, 1)[:, None]  # (channels, 1)
    activation = np.maximum(0, gains * (sig - intercepts))  # (channels, pop_size)
    return (activation * gain_scale).ravel()


def build_motor_decoders(motor_pop_size, motor_channels, rng):
    """Build linear decoding weights for motor population readout.

    Returns: (motor_channels, motor_pop_size) weight matrix
    """
    # Random initial decoders -- will be refined by observing activity
    # Positive weights, normalized per channel
    W = rng.uniform(0.1, 1.0, (motor_channels, motor_pop_size))
    W /= W.sum(axis=1, keepdims=True)
    return W


def decode_motor(motor_fire_counts, decoders):
    """Decode motor population firing into output channel values.

    motor_fire_counts: (motor_channels * motor_pop_size,) -- spike counts per neuron
    decoders: (motor_channels, motor_pop_size) -- decoding weights
    Returns: (motor_channels,) -- decoded output values
    """
    n_channels, pop_size = decoders.shape
    counts = motor_fire_counts.reshape(n_channels, pop_size)
    return np.sum(counts * decoders, axis=1)


def build_config(brain_data, electrodes):
    """Build initial config message with neuron positions, types, and electrode map."""
    neurons = brain_data['neurons']
    synapses = brain_data['synapses']
    n = len(neurons)

    neuron_list = []
    type_counts = {}
    for i, neuron in enumerate(neurons):
        nt = neuron['type']
        color = NEURON_TYPE_COLORS.get(nt, [0.5, 0.5, 0.5])
        is_inh = nt in ('FS', 'LTS')
        type_counts[nt] = type_counts.get(nt, 0) + 1
        neuron_list.append({
            'x': float(neuron['pos_x']),
            'y': float(neuron['pos_y']),
            'z': float(neuron['pos_z']),
            'type': nt,
            'color': color,
            'inh': is_inh,
        })

    syn_types = {}
    for s in synapses:
        st = s['type']
        syn_types[st] = syn_types.get(st, 0) + 1

    return {
        'type': 'config',
        'n_neurons': n,
        'n_synapses': len(synapses),
        'neurons': neuron_list,
        'syn_types': syn_types,
        'type_counts': type_counts,
        'type_colors': NEURON_TYPE_COLORS,
        'electrodes': electrodes,
        'io': IO,
    }


def get_frame(brain, neuron_types, tick, fired, motor_fire):
    """Build per-frame state: spikes, type activity, motor output."""
    fired_list = fired.tolist() if hasattr(fired, 'tolist') else list(fired)

    type_fire = {t: 0 for t in NEURON_TYPE_COLORS}
    for fi in fired_list:
        nt = neuron_types[fi]
        type_fire[nt] = type_fire.get(nt, 0) + 1

    return {
        'type': 'frame',
        'tick': tick,
        'fired': fired_list,
        'n_fired': len(fired_list),
        'type_fire': type_fire,
        'motor': motor_fire.tolist(),
    }


async def simulation_loop(websocket, brain, brain_data, neuron_types,
                          positions, electrodes, pop_params, args):
    """Run brain with population-encoded I/O and stream activity to client."""
    n = brain.n
    speed = args.speed
    ticks_per_frame = max(1, int(33 * speed))

    # Electrode index arrays
    audio_idx = np.array(electrodes['audio'], dtype=np.intp)
    visual_idx = np.array(electrodes['visual'], dtype=np.intp)
    motor_idx = np.array(electrodes['motor'], dtype=np.intp)
    motor_map = {int(idx): i for i, idx in enumerate(motor_idx)}

    # Population encoding parameters
    audio_intercepts = pop_params['audio_intercepts']
    audio_gains = pop_params['audio_gains']
    visual_intercepts = pop_params['visual_intercepts']
    visual_gains = pop_params['visual_gains']
    motor_decoders = pop_params['motor_decoders']

    # I/O state -- raw channel signals (not per-neuron)
    audio_signal = np.zeros(IO['audio_channels'])   # 16 frequency bands
    visual_signal = np.zeros(IO['visual_channels'])  # 64 pixels
    touch_inject = np.zeros(n)

    # Send config
    config = build_config(brain_data, electrodes)
    config['io'] = IO  # send updated IO config to client
    await websocket.send(json.dumps(config))

    paused = False
    running = True

    while running:
        frame_start = time.time()

        # Drain all pending client messages
        while True:
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                cmd = json.loads(msg)
                action = cmd.get('action')

                if action == 'pause':
                    paused = not paused
                elif action == 'speed':
                    speed = cmd.get('value', 1.0)
                    ticks_per_frame = max(1, int(33 * speed))
                elif action == 'tonic':
                    args.tonic = cmd.get('value', 2.8)

                # --- Sensory I/O (channel-level, not per-neuron) ---
                elif action == 'audio':
                    fft = cmd.get('fft', [])
                    nc = min(len(fft), len(audio_signal))
                    if nc > 0:
                        audio_signal[:nc] = fft[:nc]
                elif action == 'visual':
                    px = cmd.get('pixels', [])
                    nc = min(len(px), len(visual_signal))
                    if nc > 0:
                        visual_signal[:nc] = px[:nc]
                elif action == 'touch':
                    idx = cmd.get('idx', -1)
                    strength = cmd.get('strength', IO['touch_strength'])
                    if 0 <= idx < n:
                        pos = positions[idx]
                        dists = np.linalg.norm(positions - pos, axis=1)
                        nearby = np.where(dists < IO['touch_radius'])[0]
                        touch_inject[nearby] = strength
                elif action == 'stop':
                    running = False
                    break
            except (asyncio.TimeoutError, asyncio.CancelledError):
                break

        if not paused:
            # Build external current: tonic base + population-encoded sensory input
            I_ext = np.full(n, args.tonic)

            # Population encoding: channel signals -> per-neuron currents via tuning curves
            audio_currents = encode_population(
                audio_signal, audio_intercepts, audio_gains, IO['audio_gain'])
            visual_currents = encode_population(
                visual_signal, visual_intercepts, visual_gains, IO['visual_gain'])

            # Inject into electrode neurons
            n_audio = min(len(audio_currents), len(audio_idx))
            n_visual = min(len(visual_currents), len(visual_idx))
            I_ext[audio_idx[:n_audio]] += audio_currents[:n_audio]
            I_ext[visual_idx[:n_visual]] += visual_currents[:n_visual]
            I_ext += touch_inject
            touch_inject *= 0.7  # touch decays over frames

            # Tick brain, accumulate spikes
            all_fired = set()
            motor_fire = np.zeros(len(motor_idx), dtype=int)
            for _ in range(ticks_per_frame):
                fired = brain.tick(external_I=I_ext)
                for f in fired:
                    fi = int(f)
                    all_fired.add(fi)
                    if fi in motor_map:
                        motor_fire[motor_map[fi]] += 1

            # Population decode: motor neuron firing -> channel values
            motor_decoded = decode_motor(motor_fire, motor_decoders)

            frame = get_frame(brain, neuron_types, brain.tick_count,
                              np.array(list(all_fired)), motor_fire)
            frame['speed'] = speed
            frame['paused'] = False
            frame['tonic'] = float(args.tonic)
            frame['motor_decoded'] = motor_decoded.tolist()
            await websocket.send(json.dumps(frame))
        else:
            await websocket.send(json.dumps({
                'type': 'frame', 'tick': brain.tick_count,
                'fired': [], 'n_fired': 0, 'paused': True,
                'type_fire': {t: 0 for t in NEURON_TYPE_COLORS},
                'motor': [0] * len(motor_idx),
                'motor_decoded': [0.0] * IO['motor_channels'],
                'speed': speed, 'tonic': float(args.tonic),
            }))

        # Maintain ~30 FPS
        elapsed = time.time() - frame_start
        sleep_time = max(0, (1.0 / 30.0) - elapsed)
        await asyncio.sleep(sleep_time)


async def handler(websocket, brain, brain_data, neuron_types,
                  positions, electrodes, pop_params, args):
    """Handle a single WebSocket connection."""
    print(f"  Client connected")
    try:
        await simulation_loop(websocket, brain, brain_data, neuron_types,
                              positions, electrodes, pop_params, args)
    except Exception as e:
        if 'closed' not in str(e).lower():
            print(f"  Error: {e}")
    finally:
        print(f"  Client disconnected")


async def main():
    p = argparse.ArgumentParser(description='Brain activity viewer server')
    p.add_argument('--brain', default=None, help='Path to brain DB')
    p.add_argument('--tonic', type=float, default=2.8, help='Tonic drive level')
    p.add_argument('--speed', type=float, default=1.0, help='Speed multiplier')
    p.add_argument('--port', type=int, default=8891, help='WebSocket port')
    p.add_argument('--no-learn', action='store_true', help='Disable learning')
    args = p.parse_args()

    # Find brain
    if args.brain:
        db_path = args.brain
    else:
        brains_dir = os.path.join(BASE, 'brains')
        dbs = sorted([f for f in os.listdir(brains_dir) if f.endswith('.db')])
        if not dbs:
            print("No brains found in brains/ directory")
            sys.exit(1)
        db_path = os.path.join(brains_dir, dbs[0])

    print(f"  Loading brain: {db_path}")
    brain_data = load(db_path)
    brain = Brain(brain_data, learn=not args.no_learn)
    neuron_types = [nn['type'] for nn in brain_data['neurons']]
    n = brain.n

    # Neuron 3D positions (for touch proximity)
    positions = np.array([[nn['pos_x'], nn['pos_y'], nn['pos_z']]
                          for nn in brain_data['neurons']])

    # Population-based electrode assignment (Nengo-inspired)
    # Input ensembles: highest out-degree neurons (they broadcast widest)
    # Output ensembles: highest in-degree neurons (they collect from most sources)
    rng = np.random.RandomState(42)  # reproducible electrode assignment

    audio_total = IO['audio_channels'] * IO['audio_pop_size']   # 128
    visual_total = IO['visual_channels'] * IO['visual_pop_size'] # 256
    motor_total = IO['motor_channels'] * IO['motor_pop_size']    # 128
    total_electrodes = audio_total + visual_total + motor_total   # 512

    # Clamp to available neurons
    if total_electrodes > n // 2:
        scale = (n // 2) / total_electrodes
        audio_total = int(audio_total * scale)
        visual_total = int(visual_total * scale)
        motor_total = int(motor_total * scale)

    in_deg = np.zeros(n, dtype=int)
    out_deg = np.zeros(n, dtype=int)
    for s in brain_data['synapses']:
        out_deg[s['source']] += 1
        in_deg[s['target']] += 1

    # Input electrodes: top out-degree neurons
    out_sorted = np.argsort(out_deg)[::-1]
    input_neurons = out_sorted[:audio_total + visual_total]
    audio_list = input_neurons[:audio_total].tolist()
    visual_list = input_neurons[audio_total:audio_total + visual_total].tolist()

    # Motor electrodes: top in-degree neurons, excluding input
    input_set = set(audio_list + visual_list)
    in_sorted = np.argsort(in_deg)[::-1]
    motor_list = []
    for idx in in_sorted:
        if int(idx) not in input_set:
            motor_list.append(int(idx))
        if len(motor_list) >= motor_total:
            break

    electrodes = {
        'audio': audio_list,
        'visual': visual_list,
        'motor': motor_list,
    }

    # Build population encoding/decoding parameters
    audio_intercepts, audio_gains = build_population_encoding(
        IO['audio_channels'], IO['audio_pop_size'], rng)
    visual_intercepts, visual_gains = build_population_encoding(
        IO['visual_channels'], IO['visual_pop_size'], rng)
    motor_decoders = build_motor_decoders(IO['motor_pop_size'], IO['motor_channels'], rng)

    pop_params = {
        'audio_intercepts': audio_intercepts,
        'audio_gains': audio_gains,
        'visual_intercepts': visual_intercepts,
        'visual_gains': visual_gains,
        'motor_decoders': motor_decoders,
    }

    # Report electrode stats
    audio_out = np.mean(out_deg[audio_list]) if audio_list else 0
    visual_out = np.mean(out_deg[visual_list]) if visual_list else 0
    motor_in = np.mean(in_deg[motor_list]) if motor_list else 0

    # Type distribution
    tc = {}
    for nt in neuron_types:
        tc[nt] = tc.get(nt, 0) + 1
    type_str = ', '.join(f"{k}:{v}" for k, v in sorted(tc.items()))

    print(f"  Brain: {n}N, {len(brain_data['synapses'])} synapses")
    print(f"  Types: {type_str}")
    print(f"  Electrodes (population encoded):")
    print(f"    Audio: {IO['audio_channels']} channels x {IO['audio_pop_size']} neurons = {len(audio_list)} (avg out={audio_out:.0f})")
    print(f"    Visual: {IO['visual_channels']} channels x {IO['visual_pop_size']} neurons = {len(visual_list)} (avg out={visual_out:.0f})")
    print(f"    Motor: {IO['motor_channels']} channels x {IO['motor_pop_size']} neurons = {len(motor_list)} (avg in={motor_in:.0f})")
    print(f"  Tonic: {args.tonic}, Speed: {args.speed}x")
    print(f"  Viewer: http://localhost:{args.port - 1}/brain_viewer.html")

    # HTTP server for static files
    import http.server, threading
    http_port = args.port - 1

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=BASE, **kw)
        def log_message(self, format, *a):
            pass

    httpd = http.server.HTTPServer(('localhost', http_port), QuietHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"  HTTP server on port {http_port}")

    try:
        import websockets
    except ImportError:
        print("  Installing websockets...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'websockets', '-q'])
        import websockets

    async with websockets.serve(
        lambda ws: handler(ws, brain, brain_data, neuron_types,
                           positions, electrodes, pop_params, args),
        'localhost', args.port
    ):
        print(f"\n  Ready. Open browser to http://localhost:{http_port}/brain_viewer.html\n")
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())

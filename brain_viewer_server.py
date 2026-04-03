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

# I/O electrode configuration
# Arbitrary neuron assignments -- we're just poking with electrodes
IO = {
    'audio_n': 64,        # frequency bands (mic FFT bins)
    'visual_n': 256,      # pixels (16x16 downsampled webcam)
    'motor_n': 32,        # output readout channels
    'audio_gain': 5.0,    # mA per unit FFT magnitude
    'visual_gain': 4.0,   # mA per unit pixel intensity
    'touch_strength': 15.0,  # mA default touch probe
    'touch_radius': 40.0,    # spatial units for touch spread
}


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
                          positions, electrodes, args):
    """Run brain with I/O and stream activity to client."""
    n = brain.n
    speed = args.speed
    ticks_per_frame = max(1, int(33 * speed))

    # Electrode index arrays
    audio_idx = np.array(electrodes['audio'], dtype=np.intp)
    visual_idx = np.array(electrodes['visual'], dtype=np.intp)
    motor_idx = np.array(electrodes['motor'], dtype=np.intp)
    motor_map = {int(idx): i for i, idx in enumerate(motor_idx)}

    # I/O state -- current injection from sensory input
    audio_data = np.zeros(len(audio_idx))
    visual_data = np.zeros(len(visual_idx))
    touch_inject = np.zeros(n)

    # Send config
    config = build_config(brain_data, electrodes)
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

                # --- Sensory I/O ---
                elif action == 'audio':
                    fft = cmd.get('fft', [])
                    nc = min(len(fft), len(audio_data))
                    if nc > 0:
                        audio_data[:nc] = fft[:nc]
                elif action == 'visual':
                    px = cmd.get('pixels', [])
                    nc = min(len(px), len(visual_data))
                    if nc > 0:
                        visual_data[:nc] = px[:nc]
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
            # Build external current: tonic base + sensory input
            I_ext = np.full(n, args.tonic)
            I_ext[audio_idx] += audio_data * IO['audio_gain']
            I_ext[visual_idx] += visual_data * IO['visual_gain']
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

            frame = get_frame(brain, neuron_types, brain.tick_count,
                              np.array(list(all_fired)), motor_fire)
            frame['speed'] = speed
            frame['paused'] = False
            frame['tonic'] = float(args.tonic)
            await websocket.send(json.dumps(frame))
        else:
            await websocket.send(json.dumps({
                'type': 'frame', 'tick': brain.tick_count,
                'fired': [], 'n_fired': 0, 'paused': True,
                'type_fire': {t: 0 for t in NEURON_TYPE_COLORS},
                'motor': [0] * len(motor_idx),
                'speed': speed, 'tonic': float(args.tonic),
            }))

        # Maintain ~30 FPS
        elapsed = time.time() - frame_start
        sleep_time = max(0, (1.0 / 30.0) - elapsed)
        await asyncio.sleep(sleep_time)


async def handler(websocket, brain, brain_data, neuron_types,
                  positions, electrodes, args):
    """Handle a single WebSocket connection."""
    print(f"  Client connected")
    try:
        await simulation_loop(websocket, brain, brain_data, neuron_types,
                              positions, electrodes, args)
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

    # Electrode assignments -- arbitrary index ranges, pure electrode probing
    audio_n = min(IO['audio_n'], n // 10)
    visual_n = min(IO['visual_n'], n // 4)
    motor_n = min(IO['motor_n'], n // 10)
    electrodes = {
        'audio': list(range(0, audio_n)),
        'visual': list(range(audio_n, audio_n + visual_n)),
        'motor': list(range(n - motor_n, n)),
    }

    # Type distribution
    tc = {}
    for nt in neuron_types:
        tc[nt] = tc.get(nt, 0) + 1
    type_str = ', '.join(f"{k}:{v}" for k, v in sorted(tc.items()))

    print(f"  Brain: {n}N, {len(brain_data['synapses'])} synapses")
    print(f"  Types: {type_str}")
    print(f"  Electrodes: {audio_n} audio IN, {visual_n} visual IN, {motor_n} motor OUT")
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
                           positions, electrodes, args),
        'localhost', args.port
    ):
        print(f"\n  Ready. Open browser to http://localhost:{http_port}/brain_viewer.html\n")
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())

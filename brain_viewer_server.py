"""
brain_viewer_server.py -- WebSocket server streaming brain activity to a 3D viewer.

Loads a V9-grown brain, ticks it with tonic drive, and sends spike data
to a browser client for real-time 3D visualization.

Usage:
    py brain_viewer_server.py                                    # default cortex_heavy
    py brain_viewer_server.py --brain brains/regional_balanced_s42.db
    py brain_viewer_server.py --tonic 3.0 --speed 2.0
    py brain_viewer_server.py --port 8891
"""
import os, sys, json, time, asyncio, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(os.path.dirname(BASE), 'inner-models-v8')
sys.path.insert(0, V8)

from engine.loader import load
from engine.runner import Brain


NEURON_TYPE_COLORS = {
    'RS':  [0.3, 0.6, 1.0],   # blue -- regular spiking (most common excitatory)
    'FS':  [1.0, 0.25, 0.25], # red -- fast spiking (inhibitory)
    'IB':  [0.3, 0.9, 0.4],   # green -- intrinsically bursting
    'CH':  [1.0, 0.8, 0.2],   # yellow -- chattering
    'LTS': [0.9, 0.4, 0.8],   # magenta -- low-threshold spiking (inhibitory)
}



def build_config(brain_data):
    """Build initial config message with neuron positions and types."""
    neurons = brain_data['neurons']
    synapses = brain_data['synapses']
    n = len(neurons)

    # Neuron data: position, type, color by neuron type
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

    # Synapse type counts
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
    }


def get_frame(brain, neuron_types, tick, fired):
    """Build per-frame state: which neurons fired, activity by type."""
    fired_list = fired.tolist() if hasattr(fired, 'tolist') else list(fired)

    # Firing counts by neuron type
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
    }


async def simulation_loop(websocket, brain, brain_data, neuron_types, args):
    """Run brain and stream activity to client."""
    n = brain.n
    tonic = np.full(n, args.tonic)
    speed = args.speed
    ticks_per_frame = max(1, int(33 * speed))

    # Send config (includes all neuron positions)
    config = build_config(brain_data)
    await websocket.send(json.dumps(config))

    paused = False
    running = True

    while running:
        frame_start = time.time()

        # Check for client messages
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
                tonic[:] = cmd.get('value', 2.8)
            elif action == 'stop':
                running = False
                break
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        if not paused:
            # Accumulate fired neurons over all ticks in this frame
            all_fired = set()
            for _ in range(ticks_per_frame):
                fired = brain.tick(external_I=tonic)
                for f in fired:
                    all_fired.add(int(f))

            frame = get_frame(brain, neuron_types, brain.tick_count,
                              np.array(list(all_fired)))
            frame['speed'] = speed
            frame['paused'] = False
            frame['tonic'] = float(tonic[0])
            await websocket.send(json.dumps(frame))
        else:
            await websocket.send(json.dumps({
                'type': 'frame', 'tick': brain.tick_count,
                'fired': [], 'n_fired': 0, 'paused': True,
                'type_fire': {t: 0 for t in NEURON_TYPE_COLORS},
                'speed': speed, 'tonic': float(tonic[0]),
            }))

        # Maintain ~30 FPS
        elapsed = time.time() - frame_start
        sleep_time = max(0, (1.0 / 30.0) - elapsed)
        await asyncio.sleep(sleep_time)


async def handler(websocket, brain, brain_data, neuron_types, args):
    """Handle a single WebSocket connection."""
    print(f"  Client connected")
    try:
        await simulation_loop(websocket, brain, brain_data, neuron_types, args)
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
        # Default: first brain in brains/
        brains_dir = os.path.join(BASE, 'brains')
        dbs = sorted([f for f in os.listdir(brains_dir) if f.endswith('.db')])
        if not dbs:
            print("No brains found in brains/ directory")
            sys.exit(1)
        db_path = os.path.join(brains_dir, dbs[0])

    print(f"  Loading brain: {db_path}")
    brain_data = load(db_path)
    brain = Brain(brain_data, learn=not args.no_learn)
    neuron_types = [n['type'] for n in brain_data['neurons']]
    n = brain.n

    # Type distribution
    type_counts = {}
    for nt in neuron_types:
        type_counts[nt] = type_counts.get(nt, 0) + 1
    type_str = ', '.join(f"{k}:{v}" for k, v in sorted(type_counts.items()))

    print(f"  Brain: {n}N, {len(brain_data['synapses'])} synapses")
    print(f"  Types: {type_str}")
    print(f"  Tonic: {args.tonic}, Speed: {args.speed}x")
    print(f"  Viewer: http://localhost:{args.port - 1}/brain_viewer.html")
    print(f"  WebSocket: ws://localhost:{args.port}")

    # Serve the HTML file via simple HTTP
    import http.server
    import threading

    html_dir = BASE
    http_port = args.port - 1

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=html_dir, **kw)
        def log_message(self, format, *a):
            pass  # suppress logs

    httpd = http.server.HTTPServer(('localhost', http_port), QuietHandler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    print(f"  HTTP server on port {http_port}")

    try:
        import websockets
    except ImportError:
        print("  Installing websockets...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'websockets', '-q'])
        import websockets

    async with websockets.serve(
        lambda ws: handler(ws, brain, brain_data, neuron_types, args),
        'localhost', args.port
    ):
        print(f"\n  Ready. Open browser to http://localhost:{http_port}/brain_viewer.html\n")
        await asyncio.Future()  # run forever


if __name__ == '__main__':
    asyncio.run(main())

"""stream_out.py — the viewer's picture and the game's sound as a live video
stream for a phone (watch_main.py --stream PORT).

Tyler 2026-10-12: "are you able to make it so that this watcher is viewable
from my phone? it's so entertaining to watch". ffmpeg turns raw frames and
samples into HLS (H.264 + AAC, 2-second segments, a few seconds behind live),
which iPhone Safari plays natively (other browsers via hls.js). A small web
server shows it, with buttons that flip the viewer's switches, on the
tailnet address only (tailscale ip -4), so only Tyler's own devices see it.

Same interface as game_audio.GameAudio (push / pace / clear / close), so the
viewer's sound code drives it unchanged: a ticker thread hands ffmpeg one
picture and exactly 1/30 s of sound 30 times a second (silence when the game
isn't playing, the last picture when nothing new was drawn), and pace() holds
the game to that clock, so picture and sound stay together.
"""
import http.server
import json
import os
import queue
import subprocess
import threading
import time

FPS = 30
PACE_S = 0.1                 # pace(): keep about this much sound queued
CAP_S = 0.3                  # ... and never more than this
HLS_JS = 'https://cdn.jsdelivr.net/npm/hls.js@1.5.13/dist/hls.min.js'

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonic agent live</title>
<style>
  body { margin: 0; background: #000; color: #ddd; font: 15px -apple-system, system-ui, sans-serif; }
  video { width: 100%; display: block; background: #000; }
  .bar { display: flex; gap: 8px; padding: 10px 12px; flex-wrap: wrap; }
  button { font: inherit; padding: 8px 12px; border-radius: 8px; border: 1px solid #555;
           background: #222; color: #ddd; }
  button.on { background: #2a5; border-color: #2a5; color: #fff; }
  p { margin: 4px 12px; color: #999; font-size: 13px; }
</style></head>
<body>
<video id="v" controls playsinline autoplay muted></video>
<div class="bar">
  <button id="skip_ss" onclick="flip('skip_ss')">Skip special stages</button>
  <button id="ring" onclick="flip('ring')">Giant-ring tries</button>
  <button id="fresh" onclick="flip('fresh')">From the act start</button>
  <button id="act" onclick="flip('act')">Next act</button>
</div>
<p>Tap the video for sound. It runs a few seconds behind live; switches apply from the next run.</p>
<script src="HLS_JS"></script>
<script>
const v = document.getElementById('v'), src = 'live.m3u8';
if (v.canPlayType('application/vnd.apple.mpegurl')) v.src = src;
else if (window.Hls && Hls.isSupported()) { const h = new Hls({liveSyncDurationCount: 2}); h.loadSource(src); h.attachMedia(v); }
async function state() {
  try {
    const s = await (await fetch('state', {cache: 'no-store'})).json();
    for (const k of ['skip_ss', 'ring', 'fresh']) document.getElementById(k).classList.toggle('on', !!s[k]);
  } catch (e) {}
}
async function flip(k) { await fetch('flip?' + k, {method: 'POST'}); state(); }
state(); setInterval(state, 3000);
</script>
</body></html>
""".replace('HLS_JS', HLS_JS)


def tailnet_ip():
    try:
        ip = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True,
                            timeout=5).stdout.split()
        return ip[0] if ip else None
    except (OSError, subprocess.SubprocessError):
        return None


class StreamOut:
    def __init__(self, out_dir, size, port, switches, rate=44100, host=None):
        """size: (width, height) of the frames given to frame(), RGB. switches:
        {name: (get, set)} the page's buttons flip."""
        self.rate = int(round(rate))
        self.chunk = self.rate // FPS * 4                    # bytes of 16-bit stereo per tick
        self.w, self.h = size
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):                        # last run's segments
            if f.endswith(('.ts', '.m3u8', '.fifo')):
                os.remove(os.path.join(out_dir, f))
        with open(os.path.join(out_dir, 'index.html'), 'w') as f:
            f.write(PAGE)
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.picture = bytes(self.w * self.h * 3)
        self.running = True
        fifo = os.path.join(out_dir, 'audio.fifo')
        os.mkfifo(fifo)
        cmd = ['ffmpeg', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.w}x{self.h}', '-framerate', str(FPS),
               '-thread_queue_size', '64', '-i', 'pipe:0',
               '-f', 's16le', '-ar', str(self.rate), '-ac', '2', '-thread_queue_size', '64', '-i', fifo,
               '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency', '-pix_fmt', 'yuv420p',
               '-g', str(FPS * 2), '-b:v', '1200k', '-maxrate', '1600k', '-bufsize', '3200k',
               '-c:a', 'aac', '-b:a', '96k',
               '-f', 'hls', '-hls_time', '2', '-hls_list_size', '8',
               '-hls_flags', 'delete_segments+independent_segments+omit_endlist',
               os.path.join(out_dir, 'live.m3u8')]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.vq, self.aq = queue.Queue(maxsize=FPS), queue.Queue(maxsize=FPS)
        threading.Thread(target=self._write, args=(self.vq, self.proc.stdin), daemon=True).start()
        threading.Thread(target=self._write_fifo, args=(fifo,), daemon=True).start()
        threading.Thread(target=self._tick, daemon=True).start()
        self.host = host or tailnet_ip() or '127.0.0.1'
        self.url = f'http://{self.host}:{port}/'
        self.server = http.server.ThreadingHTTPServer((self.host, port), self._handler(switches))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    # ── ffmpeg's two inputs ──────────────────────────────────────────────

    def _write(self, q, pipe):
        while self.running:
            data = q.get()
            try:
                pipe.write(data)
            except (BrokenPipeError, ValueError):
                return

    def _write_fifo(self, path):
        with open(path, 'wb') as f:                          # blocks until ffmpeg opens it
            self._write(self.aq, f)

    def _tick(self):
        """30 times a second: one picture and exactly 1/30 s of sound."""
        nxt = time.monotonic()
        while self.running:
            with self.lock:
                snd = bytes(self.buf[:self.chunk])
                del self.buf[:len(snd)]
            snd += bytes(self.chunk - len(snd))              # silence when the game isn't playing
            for q, data in ((self.vq, self.picture), (self.aq, snd)):
                try:
                    q.put_nowait(data)
                except queue.Full:                           # ffmpeg stalled: drop, stay live
                    pass
            nxt += 1 / FPS
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 1:                   # fell far behind: start the clock again
                nxt = time.monotonic()

    # ── the viewer's side ───────────────────────────────────────────────

    def frame(self, rgb_bytes):
        self.picture = rgb_bytes

    def push(self, samples):
        data = samples.tobytes()
        cap = int(self.rate * CAP_S) * 4
        with self.lock:
            self.buf += data
            if len(self.buf) > cap:
                del self.buf[:len(self.buf) - cap]

    def pace(self, timeout=0.2):
        limit = int(self.rate * PACE_S) * 4
        end = time.monotonic() + timeout
        while len(self.buf) > limit and time.monotonic() < end:
            time.sleep(0.002)

    def clear(self):
        with self.lock:
            self.buf.clear()

    def close(self):
        self.running = False
        self.server.shutdown()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.terminate()

    # ── the page ────────────────────────────────────────────────────────

    def _handler(self, switches):
        directory = self.dir

        class Handler(http.server.SimpleHTTPRequestHandler):
            extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                              '.m3u8': 'application/vnd.apple.mpegurl', '.ts': 'video/mp2t'}

            def __init__(self, *a, **kw):
                super().__init__(*a, directory=directory, **kw)

            def end_headers(self):
                if self.path.split('?')[0].endswith(('.m3u8', 'state', '/')):
                    self.send_header('Cache-Control', 'no-store')
                super().end_headers()

            def do_GET(self):
                if self.path.split('?')[0] == '/state':
                    return self._json({k: bool(g()) for k, (g, _) in switches.items()})
                return super().do_GET()

            def do_POST(self):
                path, _, name = self.path.partition('?')
                if path == '/flip' and name in switches:
                    get, put = switches[name]
                    put(not get())
                    return self._json({name: bool(get())})
                self.send_error(404)

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        return Handler

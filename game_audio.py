"""game_audio.py — the emulator's sound, streamed to the speakers while a
recorder runs (record_play.py, record_demo.py).

stable-retro hands back each frame's samples (`em.get_audio()`: int16 stereo,
~735 per frame at `em.get_audio_rate()` = 44,100 Hz). SDL pulls them from a
short buffer on its own thread. The buffer waits for ~2 frames before playing
(so frame-to-frame jitter doesn't click) and is capped at ~0.1 s (so sound
never trails the picture). Feed it only at 100% speed: slowed or paused play
produces sound slower than SDL plays it, which would only stutter, so the
recorder goes quiet then (`clear()`).

Output only: reading the samples doesn't touch the emulator, so recordings and
replays are unchanged.
"""
import subprocess
import threading
import time

PRIME = 2 * 735 * 4          # bytes before playback starts: ~2 frames of 16-bit stereo
CAP_S = 0.1                  # seconds of sound buffered at most
PACE_S = 0.05                # pace(): keep about this much queued


def default_output(names):
    """The SDL output device matching the system's default sink (PipeWire /
    PulseAudio), so the sound goes where every other program's does; the
    first device if that can't be found. pygame's AudioDevice won't take
    None for "default" (2.6.1 raises "devicename must be a string")."""
    try:
        want = subprocess.run(['pactl', 'get-default-sink'], capture_output=True,
                              text=True, timeout=3).stdout.strip()
        sinks = subprocess.run(['pactl', 'list', 'sinks'], capture_output=True,
                               text=True, timeout=3).stdout
        name = None
        for line in sinks.splitlines():
            line = line.strip()
            if line.startswith('Name:'):
                name = line.split(':', 1)[1].strip()
            elif line.startswith('Description:') and name == want:
                desc = line.split(':', 1)[1].strip()
                if desc in names:
                    return desc
    except (OSError, subprocess.SubprocessError):
        pass
    return names[0]


class GameAudio:
    def __init__(self, rate, volume=1.0):
        from pygame._sdl2 import audio as sdl_audio
        self.rate = int(round(rate))
        self.cap = int(self.rate * CAP_S) * 4
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.primed = False
        self.pulled = 0          # bytes SDL has taken (a check that it's playing)
        self.dropped = 0         # bytes dropped because the buffer was full
        self.volume = volume
        names = sdl_audio.get_audio_device_names(False)
        if not names:
            raise RuntimeError('no sound output device')
        self.device = default_output(names)
        self.dev = sdl_audio.AudioDevice(
            devicename=self.device, iscapture=False, frequency=self.rate,
            audioformat=sdl_audio.AUDIO_S16, numchannels=2, chunksize=512,
            allowed_changes=0, callback=self._fill)
        self.dev.pause(0)

    def _fill(self, _dev, mem):
        n = len(mem)
        with self.lock:
            if not self.primed and len(self.buf) >= PRIME:
                self.primed = True
            take = bytes(self.buf[:n]) if self.primed else b''
            del self.buf[:len(take)]
            if self.primed and not self.buf:
                self.primed = False              # ran dry: re-prime before resuming
        mem[:len(take)] = take
        if len(take) < n:
            mem[len(take):] = bytes(n - len(take))
        self.pulled += len(take)

    def push(self, samples):
        """One frame's samples (int16, shape (n, 2)) from em.get_audio()."""
        if self.volume != 1.0:
            samples = (samples * self.volume).astype(samples.dtype)
        data = samples.tobytes()
        with self.lock:
            self.buf += data
            if len(self.buf) > self.cap:         # running behind: drop the oldest
                self.dropped += len(self.buf) - self.cap
                del self.buf[:len(self.buf) - self.cap]

    def pace(self, timeout=0.1):
        """Use the sound card as the game clock at 100% speed: wait while more
        than PACE_S of sound is queued. pygame's Clock.tick(60) ran ~61 fps
        here, which overfills the buffer and drops a sliver of sound about once
        a second; paced by the card the game runs at exactly 44100/735 = 60 fps
        and nothing is dropped. Gives up after `timeout` (a stalled device)."""
        limit = int(self.rate * PACE_S) * 4
        end = time.monotonic() + timeout
        while len(self.buf) > limit and time.monotonic() < end:
            time.sleep(0.001)

    def clear(self):
        with self.lock:
            self.buf.clear()
            self.primed = False

    def close(self):
        self.dev.pause(1)
        self.dev.close()

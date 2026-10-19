"""um_glitch: say "um" and your webcam glitches like a bad internet connection.

    pip install -r requirements.txt
    python um_glitch.py              # default camera
    python um_glitch.py --camera 1   # another camera

Keys: space = glitch now · d = show the detector's numbers · q / esc = quit

Everything happens on your computer. Nothing is sent anywhere, and nothing is saved to disk.
"""
import argparse
import queue

import cv2
import numpy as np
import sounddevice as sd

# =====================================================================================
# 1. SETTINGS
# =====================================================================================
SAMPLE_RATE = 48000
HOP = 512           # the detector runs every 512 samples (~94 times a second)...
FRAME = 2048        # ...on the last 2048 samples (~43 ms) of microphone audio

# What counts as an "um". Press `d` while you talk to see these numbers and tune them.
WINDOW = 0.3        # look at the last 0.3 seconds of sound
VOICED = 0.85       # at least 85% of it must be "voiced" (a clear pitch, like a hum or vowel)
FLUX = 5.3          # and the sound must barely change (median spectral change below 5.3 dB).
                    # Normal talking sits around 6-9 because it keeps switching vowels and consonants.
CLARITY = 0.6       # how clearly pitched (0..1) a moment must be to count as voiced
MIN_LEVEL = 0.006   # ignore anything quieter than this
COOLDOWN = 0.6      # seconds before another um can count

# How the bad connection plays out
DROP = 3.5          # seconds of glitching per um (another um restarts the clock)
ATTACK = 0.25       # seconds to go from fine to fully broken
RELEASE = 1.0       # the last second eases back to normal

rng = np.random.default_rng()


# =====================================================================================
# 2. HEARING: is this sound an "um"?
# =====================================================================================
# Pitch search range: 75-350 Hz covers most speaking voices.
LAG_MIN, LAG_MAX = SAMPLE_RATE // 350, SAMPLE_RATE // 75
SPAN = FRAME - LAG_MAX - 2
WINDOW_FN = np.blackman(FRAME)
_HZ_PER_BIN = SAMPLE_RATE / FRAME
# 16 frequency bands, spaced evenly on a log scale from 120 Hz to 4 kHz
BANDS = []
for i in range(16):
    lo = round(120 * (4000 / 120) ** (i / 16) / _HZ_PER_BIN)
    hi = round(120 * (4000 / 120) ** ((i + 1) / 16) / _HZ_PER_BIN)
    BANDS.append((lo, max(lo + 1, hi)))


def pitch(frame):
    """Find the voice's pitch by autocorrelation: slide the sound against itself and see where it
    lines up best. Returns (pitch in Hz, clarity 0..1). A held hum lines up almost perfectly
    (clarity near 1); consonants and noise don't."""
    x = frame[:SPAN]
    shifted = np.lib.stride_tricks.sliding_window_view(frame, SPAN)[LAG_MIN:LAG_MAX + 2]
    energy = np.concatenate([[0.0], np.cumsum(frame.astype(np.float64) ** 2)])
    shifted_energy = energy[LAG_MIN + SPAN:LAG_MAX + 2 + SPAN] - energy[LAG_MIN:LAG_MAX + 2]
    corr = (shifted @ x) / np.sqrt((x @ x) * shifted_energy + 1e-12)
    k = int(np.argmax(corr[:-1]))
    best = corr[k]
    for j in range(1, k):            # prefer the shortest lag that's nearly as good (avoids octave errors)
        if corr[j] > 0.92 * best and corr[j] >= corr[j - 1] and corr[j] >= corr[j + 1]:
            k = j
            break
    return SAMPLE_RATE / (LAG_MIN + k), float(best)


def sound_shape(frame):
    """The loudness of 16 frequency bands, relative to their average: the *shape* of the sound,
    not how loud it is. Each vowel and consonant has its own shape."""
    power = np.abs(np.fft.rfft(frame * WINDOW_FN)) ** 2
    db = np.array([10 * np.log10(power[a:b].mean() + 1e-12) for a, b in BANDS])
    return db - db.mean()


class UmDetector:
    """Feed it microphone audio; it calls on_um() when it hears a held "ummm" / "uhhh".

    The idea: an "um" is ONE steady voiced sound. Over 0.3 s it's almost all pitched, and its
    shape hardly changes. Talking is also pitched, but it jumps between sounds several times a
    second, so its shape keeps changing."""

    def __init__(self, on_um):
        self.on_um = on_um
        self.noise_floor = 0.01
        self.armed = True            # one trigger per held sound
        self.recent = []             # (time, voiced, flux) for the last WINDOW seconds
        self.shapes = []             # recent sound shapes, to compare with ~60 ms ago
        self.readout = ""

    def hear(self, frame, now):
        level = np.sqrt(np.mean(frame ** 2))
        _, clarity = pitch(frame)
        loud = level > max(MIN_LEVEL, 2.5 * self.noise_floor)
        voiced = loud and clarity > CLARITY

        # how much has the sound's shape changed in the last 60 ms?
        shape = sound_shape(frame)
        self.shapes.append((now, shape))
        while len(self.shapes) > 2 and now - self.shapes[1][0] >= 0.06:
            self.shapes.pop(0)
        flux = np.abs(shape - self.shapes[0][1]).mean()

        # look back over the last WINDOW seconds
        self.recent.append((now, voiced, flux))
        while now - self.recent[0][0] > WINDOW:
            self.recent.pop(0)
        voiced_share = np.mean([v for _, v, _ in self.recent])
        change = np.median([f for _, v, f in self.recent if v] or [99])

        if voiced_share < 0.4:
            self.armed = True
        if self.armed and voiced_share >= VOICED and change < FLUX:
            self.armed = False
            self.on_um(now)

        if not loud:                 # slowly learn how loud the room is when you're silent
            rate = 0.2 if level < self.noise_floor else 0.02
            self.noise_floor += (min(level, 0.2) - self.noise_floor) * rate
        self.readout = (f"voiced {voiced_share * 100:3.0f}% (need {VOICED * 100:.0f}%)   "
                        f"change {min(change, 99):4.1f} dB (need < {FLUX})")


# =====================================================================================
# 3. GLITCHING: make the pixels look like a bad connection
# =====================================================================================
class BadConnection:
    """How broken the connection is right now: 0 = perfect, 1 = worst."""

    def __init__(self):
        self.start = self.until = self.last = -9.0

    def trigger(self, now, forced=False):
        if not forced and now - self.last < COOLDOWN:
            return
        self.last = now
        if now > self.until:
            self.start = now
        self.until = now + DROP

    def badness(self, now):
        if now >= self.until:
            return 0.0
        ramp_in = min(1, (now - self.start) / ATTACK)
        ramp_out = min(1, (self.until - now) / RELEASE)
        return ramp_in * ramp_out


class Glitch:
    """Turns clean webcam frames into bad-connection frames: freezes, a choppy frame rate,
    blocky compression, and frames where only some blocks arrive."""

    def __init__(self, fps=30):
        self.fps = fps
        self.shown = None            # the last frame we showed (stale pixels live here)
        self.frozen_until = 0.0
        self.next_update = 0.0

    def apply(self, frame, badness, now):
        h, w = frame.shape[:2]
        if self.shown is None or self.shown.shape != frame.shape or badness <= 0:
            self.shown = frame.copy()
            return self.shown

        if now < self.frozen_until:                       # FREEZE: keep showing the old frame
            return self.shown
        if rng.random() < badness * 1.6 / self.fps:       # ~1-2 freezes a second at worst
            self.frozen_until = now + rng.uniform(0.4, 0.6 + badness)
            return self.shown

        if now < self.next_update:                        # CHOPPY: only a few new frames a second
            return self.shown
        self.next_update = now + rng.uniform(0.05, 0.12 + 0.45 * badness)

        # BLOCKY: shrink the frame and blow it back up with hard edges
        block = 1 + round(badness * rng.uniform(0.3, 1) * 22)
        small = cv2.resize(frame, (max(8, w // block), max(5, h // block)), interpolation=cv2.INTER_AREA)
        blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST if block > 2 else cv2.INTER_LINEAR)

        if rng.random() < badness * 0.5:
            # HALF-ARRIVED: only some of a 16x9 grid of tiles update; the rest stay stale
            arrived = rng.random((9, 16)) < rng.uniform(0.35, 0.8)
            mask = cv2.resize(arrived.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            self.shown[mask] = blocky[mask]
        else:
            self.shown = blocky
        return self.shown


# =====================================================================================
# 4. MAIN: microphone -> detector, webcam -> glitch -> window
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description='Say "um" and your webcam glitches like a bad connection.')
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    connection = BadConnection()
    detector = UmDetector(on_um=connection.trigger)
    glitch = Glitch()

    # The mic callback only hands audio to the main loop; all the thinking happens there.
    audio_blocks = queue.Queue()
    mic = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=HOP, channels=1, dtype="float32",
                         callback=lambda data, frames, t, status: audio_blocks.put(data[:, 0].copy()))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera. On macOS, allow camera access for your terminal.")

    buffer = np.zeros(FRAME, np.float32)
    samples_heard = 0
    show_numbers = False
    print('Say "ummm"... (space = glitch now, d = numbers, q = quit)')

    with mic:
        while True:
            # 1) run the detector on every new block of audio
            while not audio_blocks.empty():
                block = audio_blocks.get()
                buffer = np.concatenate([buffer[len(block):], block])
                samples_heard += len(block)
                detector.hear(buffer, samples_heard / SAMPLE_RATE)
            now = samples_heard / SAMPLE_RATE

            # 2) glitch the webcam frame by how bad the connection is right now
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                    # mirror, like a video call
            bad = connection.badness(now)
            out = glitch.apply(frame, bad, now).copy()

            if bad > 0:
                cv2.putText(out, "Your connection is unstable", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if show_numbers:
                cv2.putText(out, detector.readout, (20, out.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 160), 1, cv2.LINE_AA)
            cv2.imshow("um glitch", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                connection.trigger(now, forced=True)
            if key == ord("d"):
                show_numbers = not show_numbers

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

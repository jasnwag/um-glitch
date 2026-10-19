# um-glitch

Say **"ummm"** on camera and your video glitches like your internet just dropped: it freezes, gets choppy, goes blocky, then recovers on its own. Perfect for dodging a hard question on a call.

One Python file, about 250 lines with comments, no AI models.

## Run it

```bash
pip install -r requirements.txt
python um_glitch.py
```

A window opens with your webcam. Say "ummm" (hold it for half a second) and watch it break.

| Key | Does |
|---|---|
| `space` | glitch now |
| `d` | show the detector's numbers |
| `q` / `esc` | quit |

macOS will ask your terminal for camera and microphone access the first time. Use `--camera 1` for a different camera.

## How it works

**Hearing an "um"** (`UmDetector` in `um_glitch.py`). About 94 times a second it looks at the last ~43 ms of microphone audio and asks two things:

1. **Is it voiced?** A hum or vowel has a clear pitch. It finds the pitch by *autocorrelation*: it slides the sound against itself and checks how well it lines up. Consonants and background noise don't line up.
2. **Is the sound changing?** It splits the sound into 16 frequency bands, which gives the sound's "shape", and measures how much that shape changed over the last 60 ms.

Talking is mostly voiced too, but it jumps between vowels and consonants several times a second, so its shape keeps changing. An "ummm" is one held sound. So the rule is: **if the last 0.3 s was at least 85% voiced and its shape barely changed, it's an um.**

**Faking the bad connection** (`Glitch`). For 3.5 seconds after an um, frames get randomly:
- **frozen**: the old frame stays on screen,
- **choppy**: only a few new frames a second,
- **blocky**: shrunk and blown back up with hard pixel edges,
- **half-arrived**: only some tiles of a 16×9 grid update, while the rest stay stale.

It ramps in fast and eases back out over the last second.

## Tuning

All the knobs are in the `SETTINGS` block at the top of the file. Press `d` and watch the numbers while you talk and while you say "ummm":
- normal talking triggers it → lower `FLUX`
- your ums are missed → raise `FLUX` a little, or lower `VOICED`
- longer or shorter glitches → `DROP`

## Privacy

Everything runs on your computer. The program makes no network requests and saves nothing to disk: no recordings, no logs. It only reads your camera and microphone while the window is open.

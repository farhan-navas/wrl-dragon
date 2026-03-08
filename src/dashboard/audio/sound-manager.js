/**
 * SoundManager — background music + procedural SFX via Web Audio API.
 *
 * Background music: loads from "audio/bg-music.ogg" if present,
 *                   otherwise generates a soft procedural ambient loop.
 * SFX: short procedural tones triggered by game events.
 */
const SoundManager = {
  _ctx: null,
  _masterGain: null,
  _musicGain: null,
  _sfxGain: null,
  _muted: false,
  _musicPlaying: false,
  _bgSource: null,
  _bgBuffer: null,
  _ambientInterval: null,
  _unlocked: false,

  // Volume levels (0–1)
  _musicVolume: 0.5,
  _sfxVolume: 0.6,

  init() {
    this._createMuteButton();
    this._bindUnlock();
    console.log("[SoundManager] initialized — click anywhere to start audio");
  },

  // ── Bootstrap ──────────────────────────────────────────

  _bindUnlock() {
    const self = this;
    const unlock = async () => {
      if (self._unlocked) return;
      self._unlocked = true;
      console.log("[SoundManager] unlocking audio context");

      try {
        if (!self._ctx) self._createContext();
        if (!self._ctx) return; // creation failed

        if (self._ctx.state === "suspended") {
          await self._ctx.resume();
        }
        if (!self._musicPlaying) self._startMusic();
      } catch (e) {
        console.warn("[SoundManager] unlock failed:", e);
        self._unlocked = false; // allow retry on next click
        return;
      }

      document.removeEventListener("click", unlock, true);
      document.removeEventListener("keydown", unlock, true);
    };
    // Use capture phase so stopPropagation on child elements won't block us
    document.addEventListener("click", unlock, true);
    document.addEventListener("keydown", unlock, true);
  },

  _createContext() {
    try {
      this._ctx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 44100,
        latencyHint: "playback",
      });

      this._ctx.onstatechange = () => {
        console.log("[SoundManager] AudioContext state:", this._ctx.state);
      };

      this._masterGain = this._ctx.createGain();
      this._masterGain.connect(this._ctx.destination);

      this._musicGain = this._ctx.createGain();
      this._musicGain.gain.value = this._musicVolume;
      this._musicGain.connect(this._masterGain);

      this._sfxGain = this._ctx.createGain();
      this._sfxGain.gain.value = this._sfxVolume;
      this._sfxGain.connect(this._masterGain);

      console.log("[SoundManager] AudioContext created, state:", this._ctx.state);
    } catch (e) {
      console.warn("[SoundManager] failed to create AudioContext:", e);
      this._ctx = null;
    }
  },

  _createMuteButton() {
    const btn = document.createElement("button");
    btn.id = "sound-toggle";
    btn.textContent = "\u266A";
    btn.title = "Toggle sound";
    btn.addEventListener("click", () => {
      this.toggleMute();
    });
    const controls = document.getElementById("header-controls");
    if (controls) {
      controls.insertBefore(btn, controls.firstChild);
    }
  },

  toggleMute() {
    this._muted = !this._muted;
    if (this._masterGain) {
      this._masterGain.gain.value = this._muted ? 0 : 1;
    }
    const btn = document.getElementById("sound-toggle");
    if (btn) {
      btn.textContent = this._muted ? "\u266A\u0338" : "\u266A";
      btn.classList.toggle("muted", this._muted);
    }
  },

  // ── Background Music ──────────────────────────────────

  async _startMusic() {
    this._musicPlaying = true;

    // Try loading an mp3 file first
    try {
      const resp = await fetch("audio/bg-music.ogg");
      // Check content-type to avoid decoding HTML from SPA fallback
      const ct = resp.headers.get("content-type") || "";
      if (resp.ok && (ct.includes("audio") || ct.includes("ogg"))) {
        const buf = await resp.arrayBuffer();
        this._bgBuffer = await this._ctx.decodeAudioData(buf);
        this._playMusicBuffer();
        console.log("[SoundManager] playing bg-music.ogg");
        return;
      }
    } catch {
      // No file — fall through to procedural
    }

    // Procedural ambient: soft pad chords
    console.log("[SoundManager] no mp3 found, starting procedural ambient");
    this._startProceduralAmbient();
  },

  _playMusicBuffer() {
    if (!this._bgBuffer) return;
    const src = this._ctx.createBufferSource();
    src.buffer = this._bgBuffer;
    src.loop = true;
    src.connect(this._musicGain);
    src.start(0);
    this._bgSource = src;
  },

  _startProceduralAmbient() {
    // Ambient pad: cycle through gentle chords using detuned oscillators
    const chords = [
      [220, 277, 330],     // Am
      [196, 247, 294],     // G
      [175, 220, 262],     // F
      [196, 247, 330],     // Em/G
    ];
    let chordIdx = 0;

    const playChord = () => {
      if (!this._ctx || this._ctx.state === "closed") return;
      const t = this._ctx.currentTime;
      const notes = chords[chordIdx % chords.length];
      chordIdx++;

      for (const freq of notes) {
        // Main tone
        const osc = this._ctx.createOscillator();
        osc.type = "sine";
        osc.frequency.value = freq;

        // Slight detune for warmth
        const osc2 = this._ctx.createOscillator();
        osc2.type = "sine";
        osc2.frequency.value = freq * 1.003;

        // Sub octave for body
        const osc3 = this._ctx.createOscillator();
        osc3.type = "sine";
        osc3.frequency.value = freq / 2;

        const env = this._ctx.createGain();
        env.gain.setValueAtTime(0, t);
        env.gain.linearRampToValueAtTime(0.18, t + 1.2);
        env.gain.linearRampToValueAtTime(0.12, t + 2.5);
        env.gain.linearRampToValueAtTime(0, t + 4.2);

        osc.connect(env);
        osc2.connect(env);
        osc3.connect(env);
        env.connect(this._musicGain);

        osc.start(t);
        osc.stop(t + 4.5);
        osc2.start(t);
        osc2.stop(t + 4.5);
        osc3.start(t);
        osc3.stop(t + 4.5);
      }
    };

    playChord();
    this._ambientInterval = setInterval(playChord, 4000);
  },

  // ── Sound Effects ─────────────────────────────────────

  playSFX(eventType, ctx) {
    if (!this._ctx || this._muted) return;

    switch (eventType) {
      case "agent_spawned":
        this._sfxSpawn();
        break;
      case "task_assigned":
        this._sfxNotify();
        break;
      case "rollout_started":
        this._sfxGo();
        break;
      case "rollout_completed":
        this._sfxResult(ctx);
        break;
      case "phase_change":
        this._sfxPhase();
        break;
      case "insight":
        this._sfxPing();
        break;
    }
  },

  // Ascending arpeggio — new agent joins
  _sfxSpawn() {
    const t = this._ctx.currentTime;
    const notes = [523, 659, 784]; // C5 E5 G5
    notes.forEach((freq, i) => {
      this._beep(freq, t + i * 0.1, 0.15, "square", 0.2);
    });
  },

  // Short blip — task delegation
  _sfxNotify() {
    const t = this._ctx.currentTime;
    this._beep(880, t, 0.08, "square", 0.18);
    this._beep(1100, t + 0.08, 0.08, "square", 0.18);
  },

  // Rising sweep — rollout begins
  _sfxGo() {
    const t = this._ctx.currentTime;
    const osc = this._ctx.createOscillator();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(200, t);
    osc.frequency.exponentialRampToValueAtTime(600, t + 0.2);

    const env = this._ctx.createGain();
    env.gain.setValueAtTime(0.18, t);
    env.gain.linearRampToValueAtTime(0, t + 0.25);

    osc.connect(env);
    env.connect(this._sfxGain);
    osc.start(t);
    osc.stop(t + 0.25);
  },

  // Success/fail chime based on reward
  _sfxResult(ctx) {
    const t = this._ctx.currentTime;
    const reward = ctx?.total_reward ?? 0;

    if (reward >= 100) {
      // Triumphant — major arpeggio up
      [523, 659, 784, 1047].forEach((f, i) => {
        this._beep(f, t + i * 0.08, 0.18, "square", 0.2);
      });
    } else if (reward >= 30) {
      // Decent — two-note chime
      this._beep(659, t, 0.12, "triangle", 0.18);
      this._beep(784, t + 0.1, 0.18, "triangle", 0.18);
    } else {
      // Low — descending two-note
      this._beep(440, t, 0.1, "triangle", 0.15);
      this._beep(330, t + 0.12, 0.18, "triangle", 0.15);
    }
  },

  // Phase transition sweep
  _sfxPhase() {
    const t = this._ctx.currentTime;
    const osc = this._ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(300, t);
    osc.frequency.exponentialRampToValueAtTime(800, t + 0.15);
    osc.frequency.exponentialRampToValueAtTime(600, t + 0.3);

    const env = this._ctx.createGain();
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(0.2, t + 0.05);
    env.gain.linearRampToValueAtTime(0, t + 0.35);

    osc.connect(env);
    env.connect(this._sfxGain);
    osc.start(t);
    osc.stop(t + 0.35);
  },

  // Subtle ping — insight notification
  _sfxPing() {
    const t = this._ctx.currentTime;
    this._beep(1200, t, 0.08, "sine", 0.12);
    this._beep(1500, t + 0.07, 0.12, "sine", 0.1);
  },

  // ── Utility ───────────────────────────────────────────

  _beep(freq, time, duration, type, volume) {
    const osc = this._ctx.createOscillator();
    osc.type = type || "square";
    osc.frequency.value = freq;

    const env = this._ctx.createGain();
    env.gain.setValueAtTime(0, time);
    env.gain.linearRampToValueAtTime(volume || 0.15, time + 0.01);
    env.gain.linearRampToValueAtTime(0, time + duration);

    osc.connect(env);
    env.connect(this._sfxGain);
    osc.start(time);
    osc.stop(time + duration + 0.01);
  },
};

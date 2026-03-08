// Simple localStorage state store for persistence across page refreshes
const Store = {
  _prefix: "wrl_",
  _maxEvents: 200,

  // ── Raw read/write ──────────────────────────────────────

  _get(key) {
    try {
      const raw = localStorage.getItem(this._prefix + key);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  },

  _set(key, value) {
    try {
      localStorage.setItem(this._prefix + key, JSON.stringify(value));
    } catch {
      // Storage full — clear old data and retry once
      this.clear();
      try {
        localStorage.setItem(this._prefix + key, JSON.stringify(value));
      } catch { /* give up */ }
    }
  },

  // ── Events ──────────────────────────────────────────────

  pushEvent(event) {
    const events = this.getEvents();
    events.push(event);
    // Keep only the last N events
    if (events.length > this._maxEvents) {
      events.splice(0, events.length - this._maxEvents);
    }
    this._set("events", events);
  },

  getEvents() {
    return this._get("events") || [];
  },

  // ── Agents ──────────────────────────────────────────────

  setAgents(agents) {
    // Store minimal agent data needed to restore canvas
    const minimal = agents.map(a => ({
      id: a.id,
      tier: a.tier,
      env: a.env,
      name: a.displayName || a.name || a.id,
    }));
    this._set("agents", minimal);
  },

  getAgents() {
    return this._get("agents") || [];
  },

  // ── Feed state (stats + entries) ────────────────────────

  setFeedState(stats, entries) {
    this._set("feed", {
      stats: {
        round: stats.round,
        totalRounds: stats.totalRounds,
        phase: stats.phase,
        agentsOnline: stats.agentsOnline,
        rolloutsRun: stats.rolloutsRun,
        bestReward: stats.bestReward === -Infinity ? null : stats.bestReward,
        bestEnv: stats.bestEnv,
      },
      entries: entries.slice(-100),
    });
  },

  getFeedState() {
    return this._get("feed");
  },

  // ── Clear ───────────────────────────────────────────────

  clear() {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(this._prefix)) keys.push(k);
    }
    for (const k of keys) localStorage.removeItem(k);
  },
};

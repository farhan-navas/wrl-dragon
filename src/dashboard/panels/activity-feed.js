// Activity Feed — gamified narrative panel showing agent thought process
const ActivityFeed = {
  _entries: [],
  _maxEntries: 100,
  _container: null,
  _phaseEl: null,
  _roundEl: null,
  _statsEl: null,

  // Track stats for the scoreboard
  _stats: {
    round: 0,
    totalRounds: 0,
    phase: "idle",
    agentsOnline: 0,
    rolloutsRun: 0,
    bestReward: -Infinity,
    bestEnv: "",
  },

  init() {
    this._container = document.getElementById("feed-entries");
    this._phaseEl = document.getElementById("feed-phase");
    this._roundEl = document.getElementById("feed-round");
    this._statsEl = document.getElementById("feed-stats");
    this._updatePhaseDisplay();
  },

  reset() {
    this._entries = [];
    this._stats = {
      round: 0, totalRounds: 0, phase: "idle",
      agentsOnline: 0, rolloutsRun: 0,
      bestReward: -Infinity, bestEnv: "",
    };
    if (this._container) this._container.innerHTML = "";
    this._updatePhaseDisplay();
    this._updateStatsDisplay();
  },

  // Called by App.handleEvent for every incoming event
  handleEvent(event) {
    switch (event.type) {
      case "agent_spawned":
        this._onAgentSpawned(event);
        break;
      case "task_assigned":
        this._onTaskAssigned(event);
        break;
      case "rollout_started":
        this._onRolloutStarted(event);
        break;
      case "rollout_completed":
        this._onRolloutCompleted(event);
        break;
      case "phase_change":
        this._onPhaseChange(event);
        break;
      case "insight":
        this._onInsight(event);
        break;
      // reward_update is too noisy, skip it
    }
  },

  _onAgentSpawned(e) {
    this._stats.agentsOnline++;
    const name = e.name || e.agent_id;
    const tierTag = this._tierTag(e.tier);
    this._addEntry(
      "spawn",
      `${tierTag} <span class="feed-name">${name}</span> has joined the team`,
    );
    this._updateStatsDisplay();
  },

  _onTaskAssigned(e) {
    const fromName = this._agentName(e.from);
    const toName = this._agentName(e.to);
    const task = e.task || "new task";
    this._addEntry(
      "task",
      `<span class="feed-name">${fromName}</span> delegates to <span class="feed-name">${toName}</span>: <span class="feed-task-text">"${this._truncate(task, 50)}"</span>`,
    );
  },

  _onRolloutStarted(e) {
    const name = this._agentName(e.agent_id);
    this._addEntry(
      "rollout",
      `<span class="feed-name">${name}</span> begins trial run...`,
    );
  },

  _onRolloutCompleted(e) {
    this._stats.rolloutsRun++;
    const name = this._agentName(e.agent_id);
    const reward = e.total_reward ?? 0;
    const steps = e.steps ?? "?";
    const grade = this._rewardGrade(reward, e.env);

    if (reward > this._stats.bestReward) {
      this._stats.bestReward = reward;
      this._stats.bestEnv = e.env || "";
    }

    this._addEntry(
      "result",
      `<span class="feed-name">${name}</span> reports: <span class="feed-reward">${reward.toFixed(1)}</span> reward in ${steps} steps <span class="feed-grade grade-${grade}">${grade}</span>`,
    );
    this._updateStatsDisplay();
  },

  _onPhaseChange(e) {
    this._stats.phase = e.phase;
    this._stats.round = e.round_num || this._stats.round;
    this._stats.totalRounds = e.total_rounds || this._stats.totalRounds;

    const phaseLabels = {
      generate: "GENERATE",
      execute: "EXECUTE",
      learn: "LEARN",
    };
    const label = phaseLabels[e.phase] || e.phase.toUpperCase();

    this._addEntry(
      "phase",
      `<span class="feed-phase-label">${label}</span> ${e.message || ""}`,
    );
    this._updatePhaseDisplay();
  },

  _onInsight(e) {
    const source = e.source || "system";
    const sourceLabels = {
      ceo: "CEO",
      analyst: "Analyst",
      coder: "Coder",
      system: "System",
    };
    const label = sourceLabels[source] || source;

    // Clean up JSON fragments from insight messages
    let msg = (e.message || "")
      .replace(/[{}\[\]"]/g, "")
      .replace(/\s+/g, " ")
      .trim();

    this._addEntry(
      "insight",
      `<span class="feed-insight-source">[${label}]</span> ${this._truncate(msg, 100)}`,
    );
  },

  // ── Rendering ───────────────────────────────────────────

  _addEntry(category, html) {
    const entry = { category, html, ts: Date.now() };
    this._entries.push(entry);
    if (this._entries.length > this._maxEntries) {
      this._entries.shift();
    }
    this._renderEntry(entry);
  },

  _renderEntry(entry) {
    if (!this._container) return;

    const div = document.createElement("div");
    div.className = `feed-entry feed-${entry.category}`;

    const timeStr = new Date(entry.ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    div.innerHTML = `<span class="feed-time">${timeStr}</span> ${entry.html}`;
    this._container.insertBefore(div, this._container.firstChild);

    // Animate in
    div.style.opacity = "0";
    div.style.transform = "translateX(-8px)";
    requestAnimationFrame(() => {
      div.style.transition = "opacity 0.3s, transform 0.3s";
      div.style.opacity = "1";
      div.style.transform = "translateX(0)";
    });

    // Trim old entries from DOM
    while (this._container.children.length > this._maxEntries) {
      this._container.removeChild(this._container.lastChild);
    }
  },

  _updatePhaseDisplay() {
    if (!this._phaseEl || !this._roundEl) return;

    const s = this._stats;
    const phases = ["generate", "execute", "learn"];
    const dots = phases
      .map((p) => {
        const active = p === s.phase;
        const cls = active ? "phase-dot active" : "phase-dot";
        const label = p[0].toUpperCase();
        return `<span class="${cls}">${label}</span>`;
      })
      .join('<span class="phase-arrow">></span>');

    this._phaseEl.innerHTML = dots;
    this._roundEl.textContent =
      s.round > 0
        ? `ROUND ${s.round}${s.totalRounds ? " / " + s.totalRounds : ""}`
        : "STANDBY";
  },

  _updateStatsDisplay() {
    if (!this._statsEl) return;
    const s = this._stats;
    const bestStr =
      s.bestReward > -Infinity ? `${s.bestReward.toFixed(1)}` : "--";
    this._statsEl.innerHTML =
      `<span>Agents: ${s.agentsOnline}</span>` +
      `<span> Rollouts: ${s.rolloutsRun}</span>` +
      `<span> Best: ${bestStr}</span>`;
  },

  // ── Helpers ─────────────────────────────────────────────

  _agentName(agentId) {
    if (!agentId) return "???";
    const agent = Renderer.getAgent(agentId);
    return agent?.displayName || agentId;
  },

  _tierTag(tier) {
    const tags = {
      ceo: '<span class="feed-tier tier-ceo">CEO</span>',
      coder: '<span class="feed-tier tier-coder">DEV</span>',
      qa: '<span class="feed-tier tier-qa">QA</span>',
    };
    return (
      tags[tier] ||
      `<span class="feed-tier">${(tier || "?").toUpperCase()}</span>`
    );
  },

  _rewardGrade(reward) {
    // Simple grading thresholds
    if (reward >= 400) return "S";
    if (reward >= 200) return "A";
    if (reward >= 100) return "B";
    if (reward >= 50) return "C";
    return "D";
  },

  _truncate(str, max) {
    if (str.length <= max) return str;
    return str.slice(0, max) + "...";
  },
};

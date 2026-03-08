// Main application entry point
const App = {
  state: {
    agents: [],
    events: [],
  },
  _ready: false,       // true once agents are loaded and ready for events
  _pendingEvents: [],  // buffer events that arrive before _ready
  _demoTimer: null,
  _serverConnected: false,

  init() {
    // Initialize canvas renderer
    Renderer.init("agent-canvas");
    Renderer.onAgentClick = (agent) => this.onAgentClick(agent);

    // Initialize panels
    RolloutViewer.init();
    RewardChart.init();
    ActivityFeed.init();

    // Connect WebSocket
    WSClient.onEvent = (event) => this.handleEvent(event);
    WSClient.connect();

    // Load agents, then replay history, then mark ready
    this.loadAgents().then(() => this.replayHistory()).then(() => {
      this._ready = true;
      // Flush any events that arrived while loading
      for (const evt of this._pendingEvents) {
        this._processEvent(evt);
      }
      this._pendingEvents = [];

      // Start demo loop if no server agents
      if (!this._serverConnected) {
        this._startDemoLoop();
      }
    });

    console.log("WRL-Dragon Dashboard initialized");
  },

  // Default demo agents so the dashboard always shows characters
  DEMO_AGENTS: [
    { id: "ceo-1", tier: "ceo", name: "CEO" },
    { id: "analyst-1", tier: "ceo", name: "Analyst" },
    { id: "coder-1", tier: "coder", name: "Coder-1" },
    { id: "coder-2", tier: "coder", name: "Coder-2" },
    { id: "qa-1", tier: "qa", name: "QA-1" },
    { id: "qa-2", tier: "qa", name: "QA-2" },
  ],

  async loadAgents() {
    let agents = [];
    try {
      const resp = await fetch("/api/agents");
      agents = await resp.json();
    } catch {
      // Server not running yet
    }

    if (agents.length > 0) {
      this._serverConnected = true;
      // Real agents from server
      Renderer.agents = [];
      for (const agent of agents) {
        Renderer.addAgent(agent);
      }
      Layout.invalidateBackground();
    } else {
      // No server — use demo agents as placeholders
      for (const agent of this.DEMO_AGENTS) {
        if (!Renderer.getAgent(agent.id)) {
          Renderer.addAgent(agent);
        }
      }
    }
  },

  async replayHistory() {
    let events = [];
    try {
      const resp = await fetch("/api/events/history");
      events = await resp.json();
    } catch {
      return;
    }

    if (!events.length) return;

    this._serverConnected = true;

    // Only replay events after the last run_started (discard stale state)
    const lastResetIdx = events.findLastIndex(e => e.type === "run_started");
    if (lastResetIdx >= 0) {
      events = events.slice(lastResetIdx + 1);
    }

    if (!events.length) return;
    console.log(`[App] Replaying ${events.length} historical events`);

    for (const event of events) {
      // Replay into activity feed and event log only (skip canvas animations for old events)
      this.addEventToLog(event);
      ActivityFeed.handleEvent(event);

      // For agent_spawned, ensure the agent exists in Renderer
      if (event.type === "agent_spawned" && event.agent_id) {
        if (!Renderer.getAgent(event.agent_id)) {
          Renderer.addAgent({
            id: event.agent_id,
            tier: event.tier,
            env: event.env,
            name: event.name || event.agent_id,
          });
        }
      }
    }
  },

  handleEvent(event) {
    if (!this._ready) {
      this._pendingEvents.push(event);
      return;
    }
    this._processEvent(event);
  },

  _processEvent(event) {
    console.log("[App] event received:", event.type, event);

    // Stop demo loop on first real event
    if (this._demoTimer) {
      this._stopDemoLoop();
    }
    this._serverConnected = true;

    // New run — reset all state
    if (event.type === "run_started") {
      this._resetState();
      return;
    }

    // Add to event log
    this.addEventToLog(event);

    // Feed narrative panel
    ActivityFeed.handleEvent(event);

    // Run animation via factory
    AnimationFactory.run(event.type, event);
  },

  _resetState() {
    console.log("[App] run_started — resetting all state");
    Store.clear();

    // Clear event log UI
    const container = document.getElementById("event-entries");
    if (container) container.innerHTML = "";

    // Reset activity feed
    ActivityFeed.reset();

    // Clear agents — they'll be re-added by agent_spawned events
    Renderer.agents = [];
    Layout.invalidateBackground();
  },

  onAgentClick(agent) {
    if (agent.tier === "qa") {
      RolloutViewer.open(agent);
    } else {
      // Show basic info for non-QA agents
      Renderer.showSpeechBubble(agent.id, agent.displayName || agent.id, 2000);
    }
  },

  addEventToLog(event) {
    const container = document.getElementById("event-entries");
    const div = document.createElement("div");
    div.className = `event-entry ${event.type}`;

    const time = new Date((event.ts || 0) * 1000).toLocaleTimeString();
    let text = `[${time}] ${event.type}`;

    if (event.agent_id) text += ` | ${event.agent_id}`;
    if (event.task) text += ` | ${event.task}`;
    if (event.total_reward !== undefined)
      text += ` | R:${event.total_reward.toFixed(1)}`;
    if (event.run_id) text += ` | ${event.run_id}`;

    div.textContent = text;
    container.insertBefore(div, container.firstChild);

    // Keep max 50 entries
    while (container.children.length > 50) {
      container.removeChild(container.lastChild);
    }
  },

  // ── Demo loop ────────────────────────────────────────────

  _demoSequence: [
    { key: "demo_ceo_thinking", delay: 3000 },
    { key: "demo_ceo_assigns_coder", delay: 4000 },
    { key: "demo_coders_coding", delay: 4000 },
    { key: "demo_coder_assigns_qa", delay: 3000, ctx: { from: "coder-1", to: "qa-1" } },
    { key: "demo_coder_assigns_qa", delay: 3000, ctx: { from: "coder-2", to: "qa-2" } },
    { key: "demo_qa_running", delay: 4000 },
    { key: "demo_qa_running_mid", delay: 3000 },
    { key: "demo_qa_reports", delay: 5000 },
    { key: "demo_reset", delay: 3000 },
  ],
  _demoStep: 0,

  _startDemoLoop() {
    this._demoStep = 0;
    this._runNextDemo();
  },

  _runNextDemo() {
    if (this._serverConnected) return;

    const step = this._demoSequence[this._demoStep];
    AnimationFactory.run(step.key, step.ctx || {});

    this._demoStep = (this._demoStep + 1) % this._demoSequence.length;
    this._demoTimer = setTimeout(() => this._runNextDemo(), step.delay);
  },

  _stopDemoLoop() {
    if (this._demoTimer) {
      clearTimeout(this._demoTimer);
      this._demoTimer = null;
    }
    // Reset all demo agents to idle
    AnimationFactory.run("demo_reset");
  },
};

// Boot
document.addEventListener("DOMContentLoaded", () => App.init());

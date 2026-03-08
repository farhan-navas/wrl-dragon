// Main application entry point
const App = {
    state: {
        agents: [],
        events: [],
    },
    _demoTimer: null,
    _demoStep: 0,
    _realEventsReceived: false,

    init() {
        // Initialize canvas renderer
        Renderer.init("agent-canvas");
        Renderer.onAgentClick = (agent) => this.onAgentClick(agent);

        // Initialize panels
        RolloutViewer.init();
        RewardChart.init();

        // Connect WebSocket
        WSClient.onEvent = (event) => this.handleEvent(event);
        WSClient.connect();

        // Load initial agents
        this.loadAgents();

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

    DEMO_SEQUENCE: [
        { key: "demo_ceo_thinking" },
        { key: "demo_ceo_assigns_coder" },
        { key: "demo_coders_coding" },
        { key: "demo_coder_assigns_qa", context: { from: "coder-1", to: "qa-1" } },
        { key: "demo_coder_assigns_qa", context: { from: "coder-2", to: "qa-2" } },
        { key: "demo_qa_running" },
        { key: "demo_qa_running_mid" },
        { key: "demo_qa_reports" },
        { key: "demo_reset" },
    ],

    async loadAgents() {
        let agents = [];
        try {
            const resp = await fetch("/api/agents");
            agents = await resp.json();
        } catch {
            // Server not running yet
        }

        // Use server agents if available, otherwise spawn demo agents
        const toSpawn = agents.length > 0 ? agents : this.DEMO_AGENTS;
        for (const agent of toSpawn) {
            if (!Renderer.getAgent(agent.id)) {
                Renderer.addAgent(agent);
            }
        }

        // Start demo simulation if using demo agents
        if (agents.length === 0) {
            this._startDemoSimulation();
        }
    },

    _startDemoSimulation() {
        if (this._demoTimer) return;

        this._demoStep = 0;
        this._demoTimer = setInterval(() => {
            if (this._realEventsReceived) {
                this._stopDemoSimulation();
                return;
            }
            const step = this.DEMO_SEQUENCE[this._demoStep];
            AnimationFactory.run(step.key, step.context || {});
            this._demoStep = (this._demoStep + 1) % this.DEMO_SEQUENCE.length;
        }, 3000);
    },

    _stopDemoSimulation() {
        if (this._demoTimer) {
            clearInterval(this._demoTimer);
            this._demoTimer = null;
        }
    },

    handleEvent(event) {
        // Stop demo simulation when real events arrive
        if (!this._realEventsReceived) {
            this._realEventsReceived = true;
            this._stopDemoSimulation();
        }

        // Add to event log
        this.addEventToLog(event);

        // Run animation via factory
        AnimationFactory.run(event.type, event);
    },

    onAgentClick(agent) {
        if (agent.tier === "qa") {
            RolloutViewer.open(agent);
        } else {
            // Show basic info for non-QA agents
            Renderer.showSpeechBubble(agent.id, `${agent.tier.toUpperCase()} agent`, 2000);
        }
    },

    addEventToLog(event) {
        const container = document.getElementById("event-entries");
        const div = document.createElement("div");
        div.className = `event-entry ${event.type}`;

        const time = new Date(event.ts * 1000).toLocaleTimeString();
        let text = `[${time}] ${event.type}`;

        if (event.agent_id) text += ` | ${event.agent_id}`;
        if (event.task) text += ` | ${event.task}`;
        if (event.total_reward !== undefined) text += ` | R:${event.total_reward.toFixed(1)}`;
        if (event.run_id) text += ` | ${event.run_id}`;

        div.textContent = text;
        container.insertBefore(div, container.firstChild);

        // Keep max 50 entries
        while (container.children.length > 50) {
            container.removeChild(container.lastChild);
        }
    }
};

// Boot
document.addEventListener("DOMContentLoaded", () => App.init());

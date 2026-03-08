// Main application entry point
const App = {
    state: {
        agents: [],
        events: [],
    },

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

    async loadAgents() {
        let agents = [];
        try {
            const resp = await fetch("/api/agents");
            agents = await resp.json();
        } catch {
            // Server not running yet
        }

        if (agents.length > 0) {
            // Real agents from server — clear any existing demo agents first
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

    handleEvent(event) {
        console.log("[App] event received:", event.type, event);

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
            Renderer.showSpeechBubble(agent.id, agent.displayName || agent.id, 2000);
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

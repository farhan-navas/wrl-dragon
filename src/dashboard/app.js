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

        // Use server agents if available, otherwise spawn demo agents
        const toSpawn = agents.length > 0 ? agents : this.DEMO_AGENTS;
        for (const agent of toSpawn) {
            if (!Renderer.getAgent(agent.id)) {
                Renderer.addAgent(agent);
            }
        }
    },

    handleEvent(event) {
        // Add to event log
        this.addEventToLog(event);

        switch (event.type) {
            case "agent_spawned":
                this.onAgentSpawned(event);
                break;
            case "task_assigned":
                this.onTaskAssigned(event);
                break;
            case "rollout_started":
                this.onRolloutStarted(event);
                break;
            case "rollout_completed":
                this.onRolloutCompleted(event);
                break;
            case "reward_update":
                this.onRewardUpdate(event);
                break;
        }
    },

    onAgentSpawned(event) {
        if (!Renderer.getAgent(event.agent_id)) {
            Renderer.addAgent({
                id: event.agent_id,
                tier: event.tier,
                env: event.env,
                name: event.agent_id,
            });
        }
        Renderer.showSpeechBubble(event.agent_id, "Ready!", 2000);
    },

    onTaskAssigned(event) {
        const fromId = event.from;
        const toId = event.to;

        Renderer.setAgentState(fromId, "assigning");
        Renderer.showSpeechBubble(fromId, event.task || "New task", 3000);
        Renderer.addTaskLine(fromId, toId);

        // Target agent starts working
        const toAgent = Renderer.getAgent(toId);
        if (toAgent) {
            if (toAgent.tier === "coder") {
                Renderer.setAgentState(toId, "coding");
            } else {
                Renderer.setAgentState(toId, "thinking");
            }
            Renderer.showSpeechBubble(toId, "On it!", 2000);
        }

        // Reset assigner after delay
        setTimeout(() => Renderer.setAgentState(fromId, "thinking"), 3000);
    },

    onRolloutStarted(event) {
        Renderer.setAgentState(event.agent_id, "running");
        Renderer.showSpeechBubble(event.agent_id, `Run: ${event.run_id}`, 2000);
        RewardChart.resetLiveData();
    },

    onRolloutCompleted(event) {
        Renderer.setAgentState(event.agent_id, "reporting");
        Renderer.showSpeechBubble(
            event.agent_id,
            `R:${event.total_reward?.toFixed(0) || "?"} / ${event.steps || "?"}s`,
            4000
        );

        // Flash effect
        setTimeout(() => Renderer.setAgentState(event.agent_id, "idle"), 4000);
    },

    onRewardUpdate(event) {
        RewardChart.addPoint(event.step, event.reward, event.cumulative);
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

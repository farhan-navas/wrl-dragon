// Main canvas renderer with game loop
const Renderer = {
    canvas: null,
    ctx: null,
    agents: [],
    taskLines: [],
    running: false,

    init(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext("2d");
        this.canvas.width = Layout.canvas.width;
        this.canvas.height = Layout.canvas.height;

        // Click handler
        this.canvas.addEventListener("click", (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.canvas.width / rect.width;
            const scaleY = this.canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            const agent = Layout.getAgentAtPosition(x, y, this.agents);
            if (agent) {
                this.onAgentClick(agent);
            }
        });

        // Hover cursor
        this.canvas.addEventListener("mousemove", (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.canvas.width / rect.width;
            const scaleY = this.canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            const agent = Layout.getAgentAtPosition(x, y, this.agents);
            this.canvas.style.cursor = agent ? "pointer" : "default";
        });

        this.running = true;
        this.loop(performance.now());
    },

    onAgentClick(agent) {
        // Overridden by app.js
        console.log("Agent clicked:", agent.id);
    },

    addAgent(agentData) {
        const tierCounts = { ceo: 0, coder: 0, qa: 0 };
        for (const a of this.agents) {
            tierCounts[a.tier] = (tierCounts[a.tier] || 0) + 1;
        }

        const index = tierCounts[agentData.tier] || 0;
        const pos = Layout.getNextDeskPosition(agentData.tier, index);

        const agent = {
            ...agentData,
            position: { ...pos },
            animState: "idle",
            displayName: agentData.name || agentData.id,
            speechBubble: null,
            speechBubbleExpiry: 0,
        };
        this.agents.push(agent);
        return agent;
    },

    getAgent(agentId) {
        return this.agents.find(a => a.id === agentId);
    },

    setAgentState(agentId, state) {
        const agent = this.getAgent(agentId);
        if (agent) agent.animState = state;
    },

    showSpeechBubble(agentId, text, duration) {
        const agent = this.getAgent(agentId);
        if (agent) {
            agent.speechBubble = text;
            agent.speechBubbleExpiry = performance.now() + (duration || 3000);
        }
    },

    addTaskLine(fromId, toId) {
        this.taskLines.push({
            from: fromId,
            to: toId,
            startTime: performance.now(),
            duration: 2000,
        });
    },

    loop(timestamp) {
        if (!this.running) return;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw floors
        Layout.drawFloors(ctx);

        // Draw desks
        for (const [tier, desks] of Object.entries(Layout.deskPositions)) {
            for (const desk of desks) {
                Layout.drawDesk(ctx, desk.x, desk.y, desk.label);
            }
        }

        // Draw task lines
        this.drawTaskLines(ctx, timestamp);

        // Draw agents (z-sorted by y)
        const sorted = [...this.agents].sort((a, b) =>
            (a.position?.y || 0) - (b.position?.y || 0)
        );
        for (const agent of sorted) {
            AgentSprites.drawAgent(ctx, agent, timestamp);
        }

        requestAnimationFrame((t) => this.loop(t));
    },

    drawTaskLines(ctx, timestamp) {
        this.taskLines = this.taskLines.filter(line => {
            const elapsed = timestamp - line.startTime;
            if (elapsed > line.duration) return false;

            const fromAgent = this.getAgent(line.from);
            const toAgent = this.getAgent(line.to);
            if (!fromAgent || !toAgent) return false;

            const progress = elapsed / line.duration;
            const alpha = 1 - progress;

            ctx.strokeStyle = `rgba(255, 255, 107, ${alpha})`;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(fromAgent.position.x, fromAgent.position.y + 12);

            // Animated endpoint
            const endX = fromAgent.position.x + (toAgent.position.x - fromAgent.position.x) * progress;
            const endY = fromAgent.position.y + 12 + (toAgent.position.y + 12 - fromAgent.position.y - 12) * progress;
            ctx.lineTo(endX, endY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Dot at end
            ctx.fillStyle = `rgba(255, 255, 107, ${alpha})`;
            ctx.beginPath();
            ctx.arc(endX, endY, 3, 0, Math.PI * 2);
            ctx.fill();

            return true;
        });
    },

    destroy() {
        this.running = false;
    }
};

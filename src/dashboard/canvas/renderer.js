// Main canvas renderer with game loop
const Renderer = {
    canvas: null,
    ctx: null,
    agents: [],
    taskLines: [],
    running: false,
    _ready: false,

    init(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext("2d");
        this.canvas.width = Layout.canvas.width;
        this.canvas.height = Layout.canvas.height;
        this.ctx.imageSmoothingEnabled = false;

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

        // Wait for tilesets before starting game loop
        TilesetLoader.ready().then(() => {
            this._ready = true;
            Layout.invalidateBackground();
            this.running = true;
            this.loop(performance.now());
        });
    },

    onAgentClick(agent) {
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
            _tierIndex: index,
        };
        this.agents.push(agent);
        return agent;
    },

    getAgent(agentId) {
        return this.agents.find(a => a.id === agentId);
    },

    setAgentState(agentId, state) {
        const agent = this.getAgent(agentId);
        if (agent) {
            console.log(`[Renderer] setAgentState "${agentId}" → "${state}"`);
            agent.animState = state;
        } else {
            console.warn(`[Renderer] setAgentState: unknown agent "${agentId}" (known: ${this.agents.map(a => a.id).join(", ")})`);
        }
    },

    showSpeechBubble(agentId, text, duration) {
        const agent = this.getAgent(agentId);
        if (agent) {
            agent.speechBubble = text;
            agent.speechBubbleExpiry = performance.now() + (duration || 3000);
        } else {
            console.warn(`[Renderer] showSpeechBubble: unknown agent "${agentId}"`);
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
        ctx.imageSmoothingEnabled = false;

        // Layer 1: Pre-rendered static background (floors, walls, furniture)
        Layout.drawBackground(ctx);

        // Layer 2: Task lines
        this.drawTaskLines(ctx, timestamp);

        // Layer 3: Agents (z-sorted by y position)
        const sorted = [...this.agents].sort((a, b) =>
            (a.position?.y || 0) - (b.position?.y || 0)
        );
        for (const agent of sorted) {
            AgentSprites.drawAgent(ctx, agent, timestamp);
        }

        requestAnimationFrame((t) => this.loop(t));
    },

    drawTaskLines(ctx, timestamp) {
        const spriteH = CharacterPngSprites.TILE_H * AgentSprites.scale;

        this.taskLines = this.taskLines.filter(line => {
            const elapsed = timestamp - line.startTime;
            if (elapsed > line.duration) return false;

            const fromAgent = this.getAgent(line.from);
            const toAgent = this.getAgent(line.to);
            if (!fromAgent || !toAgent) return false;

            const progress = elapsed / line.duration;
            const alpha = 1 - progress;

            // Connect from agent center (sprite anchored at bottom-center)
            const fromY = fromAgent.position.y - spriteH / 2;
            const toY = toAgent.position.y - spriteH / 2;

            ctx.strokeStyle = `rgba(255, 255, 107, ${alpha})`;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(fromAgent.position.x, fromY);

            const endX = fromAgent.position.x + (toAgent.position.x - fromAgent.position.x) * progress;
            const endY = fromY + (toY - fromY) * progress;
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

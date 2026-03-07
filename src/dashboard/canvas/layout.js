// 3-tier office layout for agent visualization
const Layout = {
    canvas: { width: 800, height: 600 },

    // Floor definitions
    floors: {
        ceo: { y: 40, height: 160, label: "TIER 1: CEO + ANALYST", color: "#1a0a2e" },
        coder: { y: 210, height: 160, label: "TIER 2: CODERS", color: "#0a1a2e" },
        qa: { y: 380, height: 160, label: "TIER 3: QA WORKERS", color: "#0a2e1a" },
    },

    // Desk positions within each floor
    deskPositions: {
        ceo: [
            { x: 250, y: 100, label: "CEO" },
            { x: 500, y: 100, label: "Analyst" },
        ],
        coder: [
            { x: 150, y: 280 },
            { x: 350, y: 280 },
            { x: 550, y: 280 },
        ],
        qa: [
            { x: 150, y: 450 },
            { x: 350, y: 450 },
            { x: 550, y: 450 },
        ],
    },

    getFloorForTier(tier) {
        return this.floors[tier] || this.floors.qa;
    },

    getNextDeskPosition(tier, index) {
        const desks = this.deskPositions[tier] || this.deskPositions.qa;
        return desks[index % desks.length];
    },

    drawFloors(ctx) {
        for (const [key, floor] of Object.entries(this.floors)) {
            // Floor background
            ctx.fillStyle = floor.color;
            ctx.fillRect(0, floor.y, this.canvas.width, floor.height);

            // Floor border
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.strokeRect(0, floor.y, this.canvas.width, floor.height);

            // Floor label
            ctx.fillStyle = "#555";
            ctx.font = "8px 'Press Start 2P', monospace";
            ctx.fillText(floor.label, 10, floor.y + 15);
        }

        // Connection lines between floors
        ctx.strokeStyle = "#222";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(400, 200);
        ctx.lineTo(400, 210);
        ctx.moveTo(400, 370);
        ctx.lineTo(400, 380);
        ctx.stroke();
        ctx.setLineDash([]);
    },

    drawDesk(ctx, x, y, label) {
        // Desk (table)
        ctx.fillStyle = "#2a2a3a";
        ctx.fillRect(x - 25, y + 15, 50, 12);
        ctx.strokeStyle = "#444";
        ctx.strokeRect(x - 25, y + 15, 50, 12);

        // Monitor
        ctx.fillStyle = "#1a3a1a";
        ctx.fillRect(x - 10, y + 5, 20, 12);
        ctx.strokeStyle = "#3a5a3a";
        ctx.strokeRect(x - 10, y + 5, 20, 12);

        if (label) {
            ctx.fillStyle = "#666";
            ctx.font = "6px 'Press Start 2P', monospace";
            ctx.textAlign = "center";
            ctx.fillText(label, x, y + 38);
            ctx.textAlign = "left";
        }
    },

    getAgentAtPosition(x, y, agents) {
        for (const agent of agents) {
            const pos = agent.position;
            if (!pos) continue;
            const dx = x - pos.x;
            const dy = y - pos.y;
            if (Math.abs(dx) < 20 && Math.abs(dy) < 20) {
                return agent;
            }
        }
        return null;
    }
};

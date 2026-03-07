// Pixel-art agent characters with animation states
const AgentSprites = {
    colors: {
        ceo: { body: "#ff6b6b", head: "#ffaaaa" },
        coder: { body: "#6bb5ff", head: "#aad4ff" },
        qa: { body: "#6bff6b", head: "#aaffaa" },
    },

    animations: {
        idle: { frames: 2, speed: 800 },
        thinking: { frames: 4, speed: 400 },
        coding: { frames: 3, speed: 200 },
        running: { frames: 4, speed: 150 },
        assigning: { frames: 3, speed: 300 },
        reporting: { frames: 2, speed: 500 },
    },

    drawAgent(ctx, agent, timestamp) {
        const pos = agent.position;
        if (!pos) return;

        const tier = agent.tier || "qa";
        const colors = this.colors[tier] || this.colors.qa;
        const anim = this.animations[agent.animState || "idle"];
        const frame = Math.floor(timestamp / anim.speed) % anim.frames;

        const x = pos.x;
        const y = pos.y;

        // Shadow
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(x - 6, y + 12, 12, 3);

        // Body (bounces slightly on active animations)
        const bounce = (agent.animState === "idle") ? 0 : Math.sin(frame * Math.PI / anim.frames) * 2;

        // Legs
        ctx.fillStyle = "#333";
        ctx.fillRect(x - 4, y + 6, 3, 6);
        ctx.fillRect(x + 1, y + 6, 3, 6);

        // Body
        ctx.fillStyle = colors.body;
        ctx.fillRect(x - 5, y - 2 - bounce, 10, 10);

        // Head
        ctx.fillStyle = colors.head;
        ctx.fillRect(x - 4, y - 9 - bounce, 8, 8);

        // Eyes
        ctx.fillStyle = "#111";
        const blink = (frame === 0 && Math.random() > 0.7);
        if (!blink) {
            ctx.fillRect(x - 2, y - 6 - bounce, 2, 2);
            ctx.fillRect(x + 1, y - 6 - bounce, 2, 2);
        } else {
            ctx.fillRect(x - 2, y - 5 - bounce, 2, 1);
            ctx.fillRect(x + 1, y - 5 - bounce, 2, 1);
        }

        // Animation-specific details
        if (agent.animState === "coding") {
            // Typing sparks
            if (frame % 2 === 0) {
                ctx.fillStyle = "#ffff6b";
                ctx.fillRect(x + 8, y - bounce, 2, 2);
            }
        } else if (agent.animState === "thinking") {
            // Thought dots
            ctx.fillStyle = "#fff";
            for (let i = 0; i < (frame % 3) + 1; i++) {
                ctx.fillRect(x + 10 + i * 4, y - 12 - bounce - i * 3, 2, 2);
            }
        } else if (agent.animState === "running") {
            // Speed lines
            ctx.strokeStyle = "#6bff6b44";
            ctx.beginPath();
            ctx.moveTo(x - 10, y - bounce);
            ctx.lineTo(x - 18, y - bounce + (frame % 2 ? 2 : -2));
            ctx.stroke();
        }

        // Agent label
        ctx.fillStyle = colors.body;
        ctx.font = "6px 'Press Start 2P', monospace";
        ctx.textAlign = "center";
        ctx.fillText(agent.displayName || agent.id, x, y + 22);
        ctx.textAlign = "left";

        // Speech bubble
        if (agent.speechBubble && agent.speechBubbleExpiry > timestamp) {
            this.drawSpeechBubble(ctx, x, y - 20 - bounce, agent.speechBubble);
        }
    },

    drawSpeechBubble(ctx, x, y, text) {
        ctx.font = "6px 'Press Start 2P', monospace";
        const width = Math.min(ctx.measureText(text).width + 10, 120);

        ctx.fillStyle = "#222";
        ctx.strokeStyle = "#555";
        ctx.fillRect(x - width / 2, y - 18, width, 14);
        ctx.strokeRect(x - width / 2, y - 18, width, 14);

        // Arrow
        ctx.fillStyle = "#222";
        ctx.beginPath();
        ctx.moveTo(x - 3, y - 4);
        ctx.lineTo(x + 3, y - 4);
        ctx.lineTo(x, y);
        ctx.fill();

        ctx.fillStyle = "#fff";
        ctx.textAlign = "center";
        ctx.fillText(text, x, y - 9);
        ctx.textAlign = "left";
    }
};

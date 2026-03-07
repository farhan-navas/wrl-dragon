// Pixel Agents-style character sprites with animation states
// Based on pablodelucca/pixel-agents (Metro City character pack)
const AgentSprites = {
    // Tier to palette index: CEO/Analyst -> 0/1, Coder -> 2/3, QA -> 4/5
    tierPalette: {
        ceo: 0,
        coder: 2,
        qa: 4,
    },

    scale: 2,

    // animState -> { mode: 'walk'|'typing'|'reading', frames: N, speed: ms }
    animConfig: {
        idle: { mode: "walk", frameIdx: 1, frames: 2, speed: 800 },
        thinking: { mode: "reading", frames: 2, speed: 400 },
        coding: { mode: "typing", frames: 2, speed: 200 },
        running: { mode: "walk", frames: 4, speed: 150 },
        assigning: { mode: "typing", frames: 2, speed: 300 },
        reporting: { mode: "typing", frames: 2, speed: 500 },
    },

    getPaletteIndex(agent) {
        const tier = agent.tier || "qa";
        const base = this.tierPalette[tier] ?? this.tierPalette.qa;
        // Analyst uses palette 1 for variety
        if (tier === "ceo" && (agent.displayName || agent.name || "").toLowerCase().includes("analyst")) {
            return 1;
        }
        return base;
    },

    getSprite(agent, timestamp) {
        const paletteIdx = this.getPaletteIndex(agent);
        const animState = agent.animState || "idle";
        const config = this.animConfig[animState] || this.animConfig.idle;

        let frame;
        if (config.frameIdx !== undefined) {
            frame = config.frameIdx;
        } else {
            const frameCount = config.frames || 2;
            const speed = config.speed || 400;
            frame = Math.floor(timestamp / speed) % frameCount;
        }

        if (typeof CharacterPngSprites === "undefined") return null;
        return CharacterPngSprites.getFrame(paletteIdx, config.mode, frame);
    },

    drawAgent(ctx, agent, timestamp) {
        const pos = agent.position;
        if (!pos) return;

        const sprite = this.getSprite(agent, timestamp);
        if (!sprite) return;

        const s = this.scale;
        const w = sprite.sw * s;
        const h = sprite.sh * s;
        const x = pos.x - w / 2;
        const y = pos.y - h;

        // Shadow
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(pos.x - 8, pos.y + 4, 16, 4);

        // PNG sprite (anchor bottom-center)
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(
            sprite.image,
            sprite.sx,
            sprite.sy,
            sprite.sw,
            sprite.sh,
            Math.round(x),
            Math.round(y),
            Math.round(w),
            Math.round(h)
        );

        // Agent label color by tier
        const tier = agent.tier || "qa";
        const labelColors = {
            ceo: "#ff6b6b",
            coder: "#6bb5ff",
            qa: "#6bff6b",
        };
        ctx.fillStyle = labelColors[tier] || "#ffffff";
        ctx.font = "6px 'Press Start 2P', monospace";
        ctx.textAlign = "center";
        ctx.fillText(agent.displayName || agent.id, pos.x, pos.y + 22);
        ctx.textAlign = "left";

        // Speech bubble
        if (agent.speechBubble && agent.speechBubbleExpiry > timestamp) {
            this.drawSpeechBubble(ctx, pos.x, y - 4, agent.speechBubble);
        }
    },

    drawSpeechBubble(ctx, x, y, text) {
        ctx.font = "6px 'Press Start 2P', monospace";
        const width = Math.min(ctx.measureText(text).width + 10, 120);

        ctx.fillStyle = "#222";
        ctx.strokeStyle = "#555";
        ctx.fillRect(x - width / 2, y - 18, width, 14);
        ctx.strokeRect(x - width / 2, y - 18, width, 14);

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
    },
};

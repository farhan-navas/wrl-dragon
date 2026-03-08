// Agent sprite drawing using Modern tiles_Free character sprites
const AgentSprites = {
    scale: 2,

    // Map agent tier+index to character name
    CHARACTER_MAP: {
        ceo: ["Alex", "Amelia"],     // CEO = Alex, Analyst = Amelia
        coder: ["Bob", "Adam"],       // Cycle by index
        qa: ["Adam", "Bob"],          // Cycle by index (inverted from coder)
    },

    // Animation state -> sprite strip + frame config
    // idle_anim: 6 frames of down-facing breathing (not idle which has 1 frame per direction)
    animConfig: {
        idle:      { strip: "idle_anim", frames: 6, speed: 300 },
        thinking:  { strip: "phone",     frames: 9, speed: 250 },
        coding:    { strip: "sit",       frames: 3, speed: 200, sitting: true },
        running:   { strip: "idle_anim", frames: 6, speed: 400 },
        assigning: { strip: "sit3",      frames: 3, speed: 300, sitting: true },
        reporting: { strip: "sit2",      frames: 3, speed: 300, sitting: true },
    },

    getCharacterName(agent) {
        const tier = agent.tier || "qa";
        const chars = this.CHARACTER_MAP[tier] || this.CHARACTER_MAP.qa;
        const index = agent._tierIndex || 0;
        return chars[index % chars.length];
    },

    getSprite(agent, timestamp) {
        const charName = this.getCharacterName(agent);
        const animState = agent.animState || "idle";
        const config = this.animConfig[animState] || this.animConfig.idle;

        const frameCount = config.frames || 4;
        const speed = config.speed || 300;
        const frameIndex = Math.floor(timestamp / speed) % frameCount;

        return CharacterPngSprites.getFrame(charName, config.strip, frameIndex);
    },

    drawAgent(ctx, agent, timestamp) {
        const pos = agent.position;
        if (!pos) return;

        const sprite = this.getSprite(agent, timestamp);
        if (!sprite) return;

        const s = this.scale;
        const w = sprite.sw * s;  // 16 * 3 = 48
        const h = sprite.sh * s;  // 32 * 3 = 96
        const x = pos.x - w / 2;

        // Sitting offset: shift character up slightly when seated
        const animState = agent.animState || "idle";
        const config = this.animConfig[animState] || this.animConfig.idle;
        const sitOffset = config.sitting ? 6 : 0;
        const y = pos.y - h + sitOffset;

        // Shadow
        ctx.fillStyle = "rgba(0,0,0,0.25)";
        const shadowW = w * 0.6;
        ctx.fillRect(pos.x - shadowW / 2, pos.y - 2, shadowW, 6);

        // Draw sprite
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(
            sprite.image,
            sprite.sx, sprite.sy, sprite.sw, sprite.sh,
            Math.round(x), Math.round(y),
            Math.round(w), Math.round(h)
        );

        // Agent label
        const tier = agent.tier || "qa";
        const labelColors = {
            ceo: "#ff6b6b",
            coder: "#6bb5ff",
            qa: "#6bff6b",
        };
        ctx.fillStyle = labelColors[tier] || "#ffffff";
        ctx.font = "6px 'Press Start 2P', monospace";
        ctx.textAlign = "center";
        ctx.fillText(agent.displayName || agent.id, pos.x, pos.y + 14);
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

        // Bubble tail
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

// Tile-based 3-tier office layout using Room Builder floors + procedural furniture
const Layout = {
    canvas: { width: 800, height: 600 },
    TILE_SIZE: 16,
    SCALE: 2,

    get SCALED_TILE() { return this.TILE_SIZE * this.SCALE; },

    // Floor styles per tier (procedural rendering — reliable and clean)
    FLOOR_STYLES: {
        ceo: {
            base: "#1a0a2e",
            pattern: "wood",
            accent: "rgba(139,105,20,0.25)",
        },
        coder: {
            base: "#0a1a2e",
            pattern: "carpet",
            accent: "rgba(68,119,170,0.15)",
        },
        qa: {
            base: "#0a2e1a",
            pattern: "tile",
            accent: "rgba(68,170,102,0.2)",
        },
    },

    floors: {
        ceo: {
            y: 40,
            height: 160,
            label: "TIER 1: CEO + ANALYST",
        },
        coder: {
            y: 210,
            height: 160,
            label: "TIER 2: CODERS",
        },
        qa: {
            y: 380,
            height: 160,
            label: "TIER 3: QA WORKERS",
        },
    },

    deskPositions: {
        ceo: [
            { x: 250, y: 150, label: "CEO" },
            { x: 550, y: 150, label: "Analyst" },
        ],
        coder: [
            { x: 150, y: 320 },
            { x: 400, y: 320 },
            { x: 650, y: 320 },
        ],
        qa: [
            { x: 150, y: 490 },
            { x: 400, y: 490 },
            { x: 650, y: 490 },
        ],
    },

    _bgCanvas: null,
    _bgDirty: true,

    getFloorForTier(tier) {
        return this.floors[tier] || this.floors.qa;
    },

    getNextDeskPosition(tier, index) {
        const desks = this.deskPositions[tier] || this.deskPositions.qa;
        return desks[index % desks.length];
    },

    // Pre-render static background
    renderBackground() {
        if (!this._bgCanvas) {
            this._bgCanvas = document.createElement("canvas");
            this._bgCanvas.width = this.canvas.width;
            this._bgCanvas.height = this.canvas.height;
        }
        const ctx = this._bgCanvas.getContext("2d");
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Dark base
        ctx.fillStyle = "#0d0d20";
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Try tile-based rendering first
        const usedTiles = OfficeTiles.renderAll(ctx, this.floors);

        if (!usedTiles) {
            // Fallback: procedural rendering if tilesets not loaded
            for (const [tierKey, floor] of Object.entries(this.floors)) {
                this._drawFloorTiles(ctx, tierKey, floor);
                this._drawWallStrip(ctx, tierKey, floor);
            }
            // Procedural furniture
            for (const [tier, desks] of Object.entries(this.deskPositions)) {
                for (const desk of desks) {
                    this._drawDeskSetup(ctx, desk.x, desk.y, tier, desk.label);
                }
            }
            this._drawDecorations(ctx);
        }

        // Tier labels (always draw on top)
        for (const floor of Object.values(this.floors)) {
            this._drawTierLabel(ctx, floor);
        }

        // Tier borders
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        for (const floor of Object.values(this.floors)) {
            ctx.strokeRect(0, floor.y, this.canvas.width, floor.height);
        }

        this._bgDirty = false;
    },

    _drawFloorTiles(ctx, tierKey, floor) {
        const style = this.FLOOR_STYLES[tierKey];
        const { width } = this.canvas;
        const y0 = floor.y;
        const h = floor.height;

        // Base color
        ctx.fillStyle = style.base;
        ctx.fillRect(0, y0, width, h);

        // Pattern overlay
        ctx.strokeStyle = style.accent;
        ctx.lineWidth = 1;

        if (style.pattern === "wood") {
            // Horizontal wood planks
            for (let r = 0; r < h; r += 20) {
                ctx.beginPath();
                ctx.moveTo(0, y0 + r);
                ctx.lineTo(width, y0 + r);
                ctx.stroke();
            }
            // Staggered vertical joints
            for (let r = 0; r < h; r += 20) {
                const offset = (r / 20) % 2 === 0 ? 0 : 60;
                for (let c = offset; c < width; c += 120) {
                    ctx.beginPath();
                    ctx.moveTo(c, y0 + r);
                    ctx.lineTo(c, y0 + r + 20);
                    ctx.stroke();
                }
            }
        } else if (style.pattern === "carpet") {
            // Subtle diamond/crosshatch
            ctx.setLineDash([6, 6]);
            for (let c = 0; c < width; c += 60) {
                ctx.beginPath();
                ctx.moveTo(c, y0);
                ctx.lineTo(c, y0 + h);
                ctx.stroke();
            }
            for (let r = 0; r < h; r += 60) {
                ctx.beginPath();
                ctx.moveTo(0, y0 + r);
                ctx.lineTo(width, y0 + r);
                ctx.stroke();
            }
            ctx.setLineDash([]);
        } else {
            // Tile grid (QA lab)
            for (let c = 0; c < width; c += 40) {
                for (let r = 0; r < h; r += 40) {
                    ctx.strokeRect(c, y0 + r, 40, 40);
                }
            }
        }
    },

    _drawWallStrip(ctx, tierKey, floor) {
        // Simple procedural wall strip - reliable and clean
        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.fillRect(0, floor.y, this.canvas.width, 24);

        // Accent line at bottom of wall
        const accents = { ceo: "#8B6914", coder: "#4477AA", qa: "#44AA66" };
        ctx.fillStyle = accents[tierKey] || "#555";
        ctx.fillRect(0, floor.y + 22, this.canvas.width, 2);
    },

    _drawTierLabel(ctx, floor) {
        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(4, floor.y + 3, 200, 16);
        ctx.fillStyle = "#888";
        ctx.font = "7px 'Press Start 2P', monospace";
        ctx.fillText(floor.label, 8, floor.y + 14);
    },

    _drawDeskSetup(ctx, x, y, tier, label) {
        // Desk surface
        FurnitureSprites.drawDesk(ctx, x - 30, y + 10, 48, 16, 2);

        // Chair
        FurnitureSprites.drawChair(ctx, x - 8, y - 6, 2);

        // Monitor on desk
        FurnitureSprites.drawMonitor(ctx, x + 12, y - 14, 2);

        // CEO extras
        if (tier === "ceo" && label === "CEO") {
            FurnitureSprites.drawLamp(ctx, x + 36, y - 4, 2);
        }
        if (tier === "ceo" && label === "Analyst") {
            FurnitureSprites.drawPlant(ctx, x + 40, y - 2, 2);
        }

        // Coder: lamp per desk
        if (tier === "coder") {
            FurnitureSprites.drawLamp(ctx, x + 32, y - 2, 2);
        }

        // Desk label
        if (label) {
            ctx.fillStyle = "#666";
            ctx.font = "6px 'Press Start 2P', monospace";
            ctx.textAlign = "center";
            ctx.fillText(label, x, y + 42);
            ctx.textAlign = "left";
        }
    },

    _drawDecorations(ctx) {
        const ceo = this.floors.ceo;
        const coder = this.floors.coder;
        const qa = this.floors.qa;

        // CEO: windows along top wall
        for (let i = 0; i < 3; i++) {
            FurnitureSprites.drawWindow(ctx, 100 + i * 250, ceo.y + 28, 2);
        }

        // CEO: rug in center
        FurnitureSprites.drawRug(ctx, 350, ceo.y + 100, 100, 50);

        // Coder: plants at edges
        FurnitureSprites.drawPlant(ctx, 40, coder.y + 90, 2);
        FurnitureSprites.drawPlant(ctx, 745, coder.y + 90, 2);

        // QA: bookshelves on walls
        FurnitureSprites.drawBookshelf(ctx, 24, qa.y + 28, 2);
        FurnitureSprites.drawBookshelf(ctx, 750, qa.y + 28, 2);

        // Divider lines between tiers
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

    drawBackground(ctx) {
        if (this._bgDirty || !this._bgCanvas) {
            this.renderBackground();
        }
        if (this._bgCanvas) {
            ctx.drawImage(this._bgCanvas, 0, 0);
        }
    },

    invalidateBackground() {
        this._bgDirty = true;
    },

    getAgentAtPosition(x, y, agents) {
        const w = CharacterPngSprites.TILE_W * AgentSprites.scale;
        const h = CharacterPngSprites.TILE_H * AgentSprites.scale;
        for (const agent of agents) {
            const pos = agent.position;
            if (!pos) continue;
            if (Math.abs(x - pos.x) < w && Math.abs(y - pos.y) < h + 12) {
                return agent;
            }
        }
        return null;
    },
};

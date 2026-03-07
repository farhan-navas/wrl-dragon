// 3-tier office layout with Pixel Agents-style themed backgrounds
const Layout = {
    canvas: { width: 800, height: 600 },

    furnitureScale: 2,

    floors: {
        ceo: {
            y: 40,
            height: 160,
            label: "TIER 1: CEO + ANALYST",
            color: "#1a0a2e",
            floorPattern: "executive", // wood / rug
            accent: "#8B6914",
        },
        coder: {
            y: 210,
            height: 160,
            label: "TIER 2: CODERS",
            color: "#0a1a2e",
            floorPattern: "carpet", // gray carpet
            accent: "#4477AA",
        },
        qa: {
            y: 380,
            height: 160,
            label: "TIER 3: QA WORKERS",
            color: "#0a2e1a",
            floorPattern: "tiled", // lab tiling
            accent: "#44AA66",
        },
    },

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

    drawFloorPattern(ctx, floor, floorY) {
        const { width } = this.canvas;
        const height = floor.height;

        if (floor.floorPattern === "executive") {
            // Wood floor - horizontal planks
            ctx.fillStyle = floor.color;
            ctx.fillRect(0, floorY, width, height);
            ctx.strokeStyle = "rgba(139,105,20,0.3)";
            ctx.lineWidth = 1;
            for (let r = 0; r < 8; r++) {
                const y = floorY + r * 20;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
        } else if (floor.floorPattern === "carpet") {
            // Carpet - subtle grid
            ctx.fillStyle = floor.color;
            ctx.fillRect(0, floorY, width, height);
            ctx.strokeStyle = "rgba(68,119,170,0.15)";
            ctx.setLineDash([8, 8]);
            for (let c = 0; c < width; c += 80) {
                ctx.beginPath();
                ctx.moveTo(c, floorY);
                ctx.lineTo(c, floorY + height);
                ctx.stroke();
            }
            ctx.setLineDash([]);
        } else {
            // Tiled (QA lab)
            ctx.fillStyle = floor.color;
            ctx.fillRect(0, floorY, width, height);
            ctx.strokeStyle = "rgba(68,170,102,0.2)";
            ctx.lineWidth = 1;
            for (let c = 0; c < width; c += 40) {
                for (let r = 0; r < height; r += 40) {
                    ctx.strokeRect(c, floorY + r, 40, 40);
                }
            }
        }
    },

    drawPixelRect(ctx, x, y, w, h, color, scale) {
        const s = scale || 2;
        ctx.fillStyle = color;
        ctx.fillRect(Math.round(x), Math.round(y), Math.round(w * s), Math.round(h * s));
    },

    // Pixel-ish oval table using rect strips (keeps crisp edges at integer scale)
    drawOvalTable(ctx, centerX, centerY, radiusX, radiusY, scale) {
        const s = scale || 2;
        const woodDark = "#6B4E0A";
        const woodMid = "#8B6914";
        const woodLight = "#B8922E";

        // Shadow under table
        ctx.fillStyle = "rgba(0,0,0,0.25)";
        ctx.fillRect(
            Math.round(centerX - radiusX * s),
            Math.round(centerY + (radiusY * 0.55) * s),
            Math.round(radiusX * 2 * s),
            Math.round(3 * s)
        );

        // Build oval from horizontal strips (each strip is 1 sprite-pixel tall)
        for (let ry = -radiusY; ry <= radiusY; ry++) {
            const t = 1 - (ry * ry) / (radiusY * radiusY);
            const rx = Math.floor(radiusX * Math.sqrt(Math.max(0, t)));

            // Border on top/bottom strips for definition
            const isEdge = Math.abs(ry) >= radiusY - 1;
            const baseColor = isEdge ? woodDark : woodMid;
            const highlight = ry < -radiusY * 0.25 ? woodLight : baseColor;

            ctx.fillStyle = highlight;
            ctx.fillRect(
                Math.round((centerX - rx) * s),
                Math.round((centerY + ry) * s),
                Math.round((rx * 2 + 1) * s),
                Math.round(1 * s)
            );
        }

        // Inner inlay stripe
        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = Math.max(1, Math.round(1 * s));
        ctx.beginPath();
        ctx.ellipse(centerX * s, centerY * s, (radiusX - 2) * s, (radiusY - 2) * s, 0, 0, Math.PI * 2);
        ctx.stroke();
    },

    drawTierBackground(ctx, tierKey, floor) {
        const s = this.furnitureScale;
        const y0 = floor.y;
        const y1 = floor.y + floor.height;

        // Simple wall strip at top of each tier
        ctx.fillStyle = "rgba(0,0,0,0.18)";
        ctx.fillRect(0, y0, this.canvas.width, 18);

        if (tierKey === "ceo") {
            // Executive wall accents + window panels
            ctx.fillStyle = "rgba(255,255,255,0.06)";
            ctx.fillRect(40, y0 + 26, 720, 2);
            ctx.fillRect(40, y0 + 60, 720, 2);

            // Windows
            ctx.fillStyle = "#101a30";
            for (let i = 0; i < 4; i++) {
                const wx = 90 + i * 160;
                ctx.fillRect(wx, y0 + 30, 120, 40);
                ctx.strokeStyle = "rgba(255,255,255,0.10)";
                ctx.strokeRect(wx + 0.5, y0 + 30.5, 120, 40);
                ctx.fillStyle = "rgba(120,170,255,0.10)";
                ctx.fillRect(wx + 4, y0 + 34, 112, 32);
                ctx.fillStyle = "#101a30";
            }

            // Big oval C-suite table in the middle of Tier 1
            this.drawOvalTable(ctx, 400, y0 + 115, 70, 26, s / 2);

            // Chairs around table (pixel props)
            if (typeof FurnitureSprites !== "undefined") {
                const chair = FurnitureSprites.CHAIR;
                const drawC = (cx, cy) => FurnitureSprites.drawSprite(ctx, chair, cx, cy, s - 0.5);
                drawC(400 - 110, y0 + 88);
                drawC(400 + 92, y0 + 88);
                drawC(400 - 70, y0 + 148);
                drawC(400 + 52, y0 + 148);
            }
        }

        if (tierKey === "coder") {
            // Coder office: cubicle partitions + extra monitors vibe
            const wall = "rgba(255,255,255,0.06)";
            ctx.fillStyle = wall;
            // Horizontal partition line
            ctx.fillRect(30, y0 + 48, 740, 2);
            // Vertical partitions creating 3 booths aligned with desks
            const boothXs = [90, 290, 490];
            for (const bx of boothXs) {
                // Left wall
                ctx.fillRect(bx, y0 + 48, 2, 98);
                // Right wall
                ctx.fillRect(bx + 190, y0 + 48, 2, 98);
                // Little header strip
                ctx.fillRect(bx, y0 + 48, 192, 2);

                // Pinboard / poster in each booth
                ctx.fillStyle = "rgba(255,255,255,0.05)";
                ctx.fillRect(bx + 16, y0 + 62, 52, 30);
                ctx.strokeStyle = "rgba(68,119,170,0.25)";
                ctx.strokeRect(bx + 16 + 0.5, y0 + 62 + 0.5, 52, 30);
                ctx.fillStyle = wall;
            }

            // A couple of plants along the corridor
            if (typeof FurnitureSprites !== "undefined") {
                FurnitureSprites.drawSprite(ctx, FurnitureSprites.PLANT, 50, y0 + 104, s - 0.5);
                FurnitureSprites.drawSprite(ctx, FurnitureSprites.PLANT, 740, y0 + 104, s - 0.5);
            }

            // Subtle cable/LED strip near bottom
            ctx.strokeStyle = "rgba(68,119,170,0.20)";
            ctx.setLineDash([6, 6]);
            ctx.beginPath();
            ctx.moveTo(40, y1 - 18);
            ctx.lineTo(760, y1 - 18);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    },

    drawFloors(ctx) {
        for (const [key, floor] of Object.entries(this.floors)) {
            this.drawFloorPattern(ctx, floor, floor.y);
            this.drawTierBackground(ctx, key, floor);

            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.strokeRect(0, floor.y, this.canvas.width, floor.height);

            ctx.fillStyle = "#555";
            ctx.font = "8px 'Press Start 2P', monospace";
            ctx.fillText(floor.label, 10, floor.y + 15);
        }

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

    drawDeskSetup(ctx, x, y, tier, label) {
        const s = this.furnitureScale;

        // Desk surface (back layer)
        if (typeof FurnitureSprites !== "undefined") {
            FurnitureSprites.drawDeskSurface(ctx, x - 48, y + 8, s / 2);
        } else {
            ctx.fillStyle = "#2a2a3a";
            ctx.fillRect(x - 25, y + 15, 50, 12);
            ctx.strokeStyle = "#444";
            ctx.strokeRect(x - 25, y + 15, 50, 12);
        }

        // Chair (agent sits here)
        if (typeof FurnitureSprites !== "undefined") {
            FurnitureSprites.drawSprite(ctx, FurnitureSprites.CHAIR, x - 16, y - 20, s);
            // Monitor beside the desk (to the right)
            FurnitureSprites.drawSprite(ctx, FurnitureSprites.PC, x + 8, y - 32, s);
        }

        // CEO zone: add lamp next to first desk
        if (tier === "ceo" && label === "CEO" && typeof FurnitureSprites !== "undefined") {
            FurnitureSprites.drawSprite(ctx, FurnitureSprites.LAMP, x + 25, y - 20, s);
        }

        // Coder zone: add lamp per desk
        if (tier === "coder" && typeof FurnitureSprites !== "undefined") {
            FurnitureSprites.drawSprite(ctx, FurnitureSprites.LAMP, x + 20, y - 18, s - 0.5);
        }

        // Plant in CEO corner
        if (tier === "ceo" && label === "Analyst" && typeof FurnitureSprites !== "undefined") {
            FurnitureSprites.drawSprite(ctx, FurnitureSprites.PLANT, x + 30, y - 10, s - 0.5);
        }

        if (label) {
            ctx.fillStyle = "#666";
            ctx.font = "6px 'Press Start 2P', monospace";
            ctx.textAlign = "center";
            ctx.fillText(label, x, y + 38);
            ctx.textAlign = "left";
        }
    },

    drawDesk(ctx, x, y, label) {
        const tier = this.getTierForPosition(y);
        this.drawDeskSetup(ctx, x, y, tier, label);
    },

    getTierForPosition(y) {
        if (y < 200) return "ceo";
        if (y < 380) return "coder";
        return "qa";
    },

    getAgentAtPosition(x, y, agents) {
        const w = 16 * 2;
        const h = 24 * 2;
        for (const agent of agents) {
            const pos = agent.position;
            if (!pos) continue;
            const dx = x - pos.x;
            const dy = y - pos.y;
            if (Math.abs(dx) < w && Math.abs(dy) < h + 8) {
                return agent;
            }
        }
        return null;
    },
};

// Office furniture drawing - procedural shapes with optional tileset upgrade
// Procedural approach guarantees correct visuals; tileset coords can be added later
const FurnitureSprites = {

    // Simple brown desk surface
    drawDesk(ctx, x, y, w, h, scale) {
        const s = scale || 2;
        const dw = (w || 48) * s / 2;
        const dh = (h || 16) * s / 2;
        // Shadow
        ctx.fillStyle = "rgba(0,0,0,0.2)";
        ctx.fillRect(x + 2, y + dh, dw, 4);
        // Desk surface
        ctx.fillStyle = "#6B4E0A";
        ctx.fillRect(x, y, dw, dh);
        ctx.fillStyle = "#8B6914";
        ctx.fillRect(x + 1, y + 1, dw - 2, dh - 3);
        // Highlight
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        ctx.fillRect(x + 2, y + 2, dw - 4, 2);
    },

    // Small office chair
    drawChair(ctx, x, y, scale) {
        const s = scale || 2;
        const sz = 10 * s / 2;
        ctx.fillStyle = "#5C3D0A";
        ctx.fillRect(x, y, sz, sz + 4);
        ctx.fillStyle = "#A07828";
        ctx.fillRect(x + 1, y + 1, sz - 2, sz - 2);
    },

    // Monitor/screen
    drawMonitor(ctx, x, y, scale) {
        const s = scale || 2;
        const mw = 12 * s / 2;
        const mh = 10 * s / 2;
        // Screen frame
        ctx.fillStyle = "#444";
        ctx.fillRect(x, y, mw, mh);
        // Screen
        ctx.fillStyle = "#3A5C8B";
        ctx.fillRect(x + 1, y + 1, mw - 2, mh - 3);
        // Screen glow
        ctx.fillStyle = "rgba(100,180,255,0.15)";
        ctx.fillRect(x + 2, y + 2, mw - 4, mh - 5);
        // Stand
        ctx.fillStyle = "#444";
        ctx.fillRect(x + mw / 2 - 2, y + mh, 4, 3);
        ctx.fillRect(x + mw / 2 - 4, y + mh + 3, 8, 2);
    },

    // Potted plant
    drawPlant(ctx, x, y, scale) {
        const s = scale || 2;
        // Pot
        ctx.fillStyle = "#8B4422";
        ctx.fillRect(x + 2, y + 10, 10, 8);
        ctx.fillStyle = "#B85C3A";
        ctx.fillRect(x + 3, y + 11, 8, 6);
        // Leaves
        ctx.fillStyle = "#3D8B37";
        ctx.fillRect(x + 1, y + 2, 12, 8);
        ctx.fillStyle = "#2D6B27";
        ctx.fillRect(x + 3, y, 8, 4);
        ctx.fillRect(x + 5, y + 4, 4, 6);
    },

    // Desk lamp
    drawLamp(ctx, x, y, scale) {
        const s = scale || 2;
        // Shade
        ctx.fillStyle = "#FFDD55";
        ctx.fillRect(x + 1, y, 8, 5);
        ctx.fillStyle = "#FFEE88";
        ctx.fillRect(x + 2, y + 1, 6, 3);
        // Pole
        ctx.fillStyle = "#888";
        ctx.fillRect(x + 4, y + 5, 2, 8);
        // Base
        ctx.fillStyle = "#666";
        ctx.fillRect(x + 2, y + 13, 6, 2);
    },

    // Bookshelf
    drawBookshelf(ctx, x, y, scale) {
        const s = scale || 2;
        const bw = 24;
        const bh = 36;
        // Frame
        ctx.fillStyle = "#6B4E0A";
        ctx.fillRect(x, y, bw, bh);
        ctx.fillStyle = "#8B6914";
        ctx.fillRect(x + 1, y + 1, bw - 2, bh - 2);
        // Shelves
        ctx.fillStyle = "#6B4E0A";
        for (let i = 1; i < 4; i++) {
            ctx.fillRect(x + 1, y + i * 9, bw - 2, 1);
        }
        // Books (colored rectangles)
        const bookColors = ["#CC4444", "#4488CC", "#44AA66", "#AA55CC", "#CCAA33"];
        for (let shelf = 0; shelf < 3; shelf++) {
            let bx = x + 2;
            for (let b = 0; b < 4; b++) {
                const c = bookColors[(shelf * 4 + b) % bookColors.length];
                ctx.fillStyle = c;
                ctx.fillRect(bx, y + shelf * 9 + 2, 4, 7);
                bx += 5;
            }
        }
    },

    // Window (wall decoration)
    drawWindow(ctx, x, y, scale) {
        const s = scale || 2;
        const ww = 28;
        const wh = 24;
        // Frame
        ctx.fillStyle = "#555";
        ctx.fillRect(x, y, ww, wh);
        // Glass
        ctx.fillStyle = "#1a2a40";
        ctx.fillRect(x + 2, y + 2, ww - 4, wh - 4);
        // Sky glow
        ctx.fillStyle = "rgba(100,140,200,0.15)";
        ctx.fillRect(x + 3, y + 3, ww - 6, wh / 2 - 2);
        // Cross bar
        ctx.fillStyle = "#555";
        ctx.fillRect(x + ww / 2 - 1, y + 2, 2, wh - 4);
        ctx.fillRect(x + 2, y + wh / 2 - 1, ww - 4, 2);
    },

    // Rug / floor mat
    drawRug(ctx, x, y, w, h) {
        ctx.fillStyle = "rgba(120,40,40,0.25)";
        ctx.fillRect(x, y, w || 80, h || 40);
        ctx.strokeStyle = "rgba(180,80,40,0.3)";
        ctx.strokeRect(x + 2, y + 2, (w || 80) - 4, (h || 40) - 4);
    },
};

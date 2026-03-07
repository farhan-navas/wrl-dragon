/**
 * Pixel Agents-style office furniture sprites (from pablodelucca/pixel-agents)
 * Procedural 16x16 / 32x32 top-down office props.
 */
(function (global) {
    "use strict";

    const _ = "";

    function drawSprite(ctx, sprite, x, y, scale) {
        const s = scale || 2;
        ctx.imageSmoothingEnabled = false;
        for (let r = 0; r < sprite.length; r++) {
            for (let c = 0; c < (sprite[r] || []).length; c++) {
                const color = sprite[r][c];
                if (!color) continue;
                ctx.fillStyle = color;
                ctx.fillRect(Math.round(x + c * s), Math.round(y + r * s), s, s);
            }
        }
    }

    // Chair 16x16 (from Pixel Agents CHAIR_SPRITE)
    const CHAIR = (() => {
        const W = "#8B6914",
            D = "#6B4E0A",
            B = "#5C3D0A",
            S = "#A07828";
        return [
            [_, _, _, _, _, D, D, D, D, D, D, _, _, _, _, _],
            [_, _, _, _, D, B, B, B, B, B, B, D, _, _, _, _],
            [_, _, _, _, D, B, S, S, S, S, B, D, _, _, _, _],
            [_, _, _, _, D, B, S, S, S, S, B, D, _, _, _, _],
            [_, _, _, _, D, B, S, S, S, S, B, D, _, _, _, _],
            [_, _, _, _, D, B, S, S, S, S, B, D, _, _, _, _],
            [_, _, _, _, D, B, S, S, S, S, B, D, _, _, _, _],
            [_, _, _, _, D, B, S, S, S, S, B, D, _, _, _, _],
            [_, _, _, _, D, B, S, S, S, S, B, D, _, _, _, _],
            [_, _, _, _, D, B, B, B, B, B, B, D, _, _, _, _],
            [_, _, _, _, _, D, D, D, D, D, D, _, _, _, _, _],
            [_, _, _, _, _, _, D, W, W, D, _, _, _, _, _, _],
            [_, _, _, _, _, _, D, W, W, D, _, _, _, _, _, _],
            [_, _, _, _, _, D, D, D, D, D, D, _, _, _, _, _],
            [_, _, _, _, _, D, _, _, _, _, D, _, _, _, _, _],
            [_, _, _, _, _, D, _, _, _, _, D, _, _, _, _, _],
        ];
    })();

    // PC monitor 16x16 (from Pixel Agents PC_SPRITE)
    const PC = (() => {
        const F = "#555555",
            S = "#3A3A5C",
            B = "#6688CC",
            D = "#444444";
        return [
            [_, _, _, F, F, F, F, F, F, F, F, F, F, _, _, _],
            [_, _, _, F, S, S, S, S, S, S, S, S, F, _, _, _],
            [_, _, _, F, S, B, B, B, B, B, B, S, F, _, _, _],
            [_, _, _, F, S, B, B, B, B, B, B, S, F, _, _, _],
            [_, _, _, F, S, B, B, B, B, B, B, S, F, _, _, _],
            [_, _, _, F, S, B, B, B, B, B, B, S, F, _, _, _],
            [_, _, _, F, S, B, B, B, B, B, B, S, F, _, _, _],
            [_, _, _, F, S, B, B, B, B, B, B, S, F, _, _, _],
            [_, _, _, F, S, S, S, S, S, S, S, S, F, _, _, _],
            [_, _, _, F, F, F, F, F, F, F, F, F, F, _, _, _],
            [_, _, _, _, _, _, _, D, D, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, D, D, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, D, D, D, D, _, _, _, _, _, _],
            [_, _, _, _, _, D, D, D, D, D, D, _, _, _, _, _],
            [_, _, _, _, _, D, D, D, D, D, D, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ];
    })();

    // Plant 16x24 (from Pixel Agents PLANT_SPRITE)
    const PLANT = (() => {
        const G = "#3D8B37",
            D = "#2D6B27",
            T = "#6B4E0A",
            P = "#B85C3A",
            R = "#8B4422";
        return [
            [_, _, _, _, _, _, G, G, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, G, G, G, G, _, _, _, _, _, _, _],
            [_, _, _, _, G, G, D, G, G, G, _, _, _, _, _, _],
            [_, _, _, G, G, D, G, G, D, G, G, _, _, _, _, _],
            [_, _, G, G, G, G, G, G, G, G, G, G, _, _, _, _],
            [_, G, G, D, G, G, G, G, G, G, D, G, G, _, _, _],
            [_, G, G, G, G, D, G, G, D, G, G, G, G, _, _, _],
            [_, _, G, G, G, G, G, G, G, G, G, G, _, _, _, _],
            [_, _, _, G, G, G, D, G, G, G, G, _, _, _, _, _],
            [_, _, _, _, G, G, G, G, G, G, _, _, _, _, _, _],
            [_, _, _, _, _, G, G, G, G, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, T, T, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, T, T, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, R, R, R, R, R, _, _, _, _, _, _],
            [_, _, _, _, R, P, P, P, P, P, R, _, _, _, _, _],
            [_, _, _, _, R, P, P, P, P, P, R, _, _, _, _, _],
            [_, _, _, _, R, P, P, P, P, P, R, _, _, _, _, _],
            [_, _, _, _, R, P, P, P, P, P, R, _, _, _, _, _],
            [_, _, _, _, R, P, P, P, P, P, R, _, _, _, _, _],
            [_, _, _, _, _, R, P, P, P, R, _, _, _, _, _, _],
            [_, _, _, _, _, _, R, R, R, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ];
    })();

    // Lamp 16x16 (from Pixel Agents LAMP_SPRITE)
    const LAMP = (() => {
        const Y = "#FFDD55",
            L = "#FFEE88",
            D = "#888888",
            B = "#555555",
            G = "#FFFFCC";
        return [
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, G, G, G, G, _, _, _, _, _, _],
            [_, _, _, _, _, G, Y, Y, Y, Y, G, _, _, _, _, _],
            [_, _, _, _, G, Y, Y, L, L, Y, Y, G, _, _, _, _],
            [_, _, _, _, Y, Y, L, L, L, L, Y, Y, _, _, _, _],
            [_, _, _, _, Y, Y, L, L, L, L, Y, Y, _, _, _, _],
            [_, _, _, _, _, Y, Y, Y, Y, Y, Y, _, _, _, _, _],
            [_, _, _, _, _, _, D, D, D, D, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, D, D, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, D, D, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, D, D, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, D, D, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, D, D, D, D, _, _, _, _, _, _],
            [_, _, _, _, _, B, B, B, B, B, B, _, _, _, _, _],
            [_, _, _, _, _, B, B, B, B, B, B, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ];
    })();

    // Desk surface 48x16 (simplified wood desk)
    function drawDeskSurface(ctx, x, y, scale) {
        const s = scale || 2;
        const W = "#8B6914",
            S = "#B8922E",
            D = "#6B4E0A";
        const w = 48,
            h = 16;
        ctx.fillStyle = D;
        ctx.fillRect(x, y, w * s, h * s);
        ctx.fillStyle = S;
        ctx.fillRect(x + s, y + s, (w - 2) * s, (h - 4) * s);
        ctx.strokeStyle = W;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, w * s, h * s);
    }

    global.FurnitureSprites = {
        CHAIR,
        PC,
        PLANT,
        LAMP,
        drawSprite,
        drawDeskSurface,
    };
})(typeof window !== "undefined" ? window : this);

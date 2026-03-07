(function (global) {
    "use strict";

    const TILE_W = 16;
    const TILE_H = 24;
    const DIR_DOWN = 0; // use front-facing row only

    const WALK_COLS = [0, 1, 2, 1];
    const TYPE_COLS = [3, 4];
    const READ_COLS = [5, 6];

    const imagePaths = [
        "public/char_0.png",
        "public/char_1.png",
        "public/char_2.png",
        "public/char_3.png",
        "public/char_4.png",
        "public/char_5.png",
    ];

    const images = [];
    let loadedCount = 0;
    const total = imagePaths.length;

    function loadAll() {
        for (let i = 0; i < imagePaths.length; i++) {
            const img = new Image();
            img.onload = () => {
                loadedCount++;
            };
            img.onerror = () => {
                loadedCount++;
            };
            img.src = imagePaths[i];
            images[i] = img;
        }
    }

    loadAll();

    function isReady() {
        return loadedCount >= total;
    }

    function getFrame(paletteIndex, mode, frameIndex) {
        const idx = ((paletteIndex % images.length) + images.length) % images.length;
        const img = images[idx];
        if (!img || !img.complete) return null;

        let cols;
        if (mode === "walk") {
            cols = WALK_COLS;
        } else if (mode === "typing") {
            cols = TYPE_COLS;
        } else {
            cols = READ_COLS;
        }
        const col = cols[frameIndex % cols.length];
        const sx = col * TILE_W;
        const sy = DIR_DOWN * TILE_H;

        return {
            image: img,
            sx,
            sy,
            sw: TILE_W,
            sh: TILE_H,
        };
    }

    global.CharacterPngSprites = {
        isReady,
        getFrame,
        TILE_W,
        TILE_H,
    };
})(typeof window !== "undefined" ? window : this);


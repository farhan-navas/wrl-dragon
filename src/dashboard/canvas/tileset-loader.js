// Loads PNG sprite sheets (Room Builder + Interiors) and draws tiles from them
const TilesetLoader = {
    sheets: {},
    _promises: [],
    _loaded: false,

    loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => {
                console.warn("Failed to load tileset:", url);
                resolve(null);
            };
            img.src = url;
        });
    },

    init() {
        const base = "public/Modern tiles_Free";
        this._promises.push(
            this.loadImage(`${base}/Interiors_free/16x16/Room_Builder_free_16x16.png`)
                .then(img => { this.sheets.roomBuilder = img; })
        );
        this._promises.push(
            this.loadImage(`${base}/Interiors_free/16x16/Interiors_free_16x16.png`)
                .then(img => { this.sheets.interiors = img; })
        );
    },

    ready() {
        return Promise.all(this._promises).then(() => { this._loaded = true; });
    },

    isReady() {
        return this._loaded;
    },

    // Draw a single 16x16 tile from a sheet
    drawTile(ctx, sheetName, col, row, destX, destY, scale) {
        const sheet = this.sheets[sheetName];
        if (!sheet) return;
        const s = scale || 2;
        ctx.drawImage(
            sheet,
            col * 16, row * 16, 16, 16,
            Math.round(destX), Math.round(destY),
            Math.round(16 * s), Math.round(16 * s)
        );
    },

    // Draw a multi-tile region from a sheet
    drawMultiTile(ctx, sheetName, col, row, tilesW, tilesH, destX, destY, scale) {
        const sheet = this.sheets[sheetName];
        if (!sheet) return;
        const s = scale || 2;
        ctx.drawImage(
            sheet,
            col * 16, row * 16, tilesW * 16, tilesH * 16,
            Math.round(destX), Math.round(destY),
            Math.round(tilesW * 16 * s), Math.round(tilesH * 16 * s)
        );
    }
};

TilesetLoader.init();

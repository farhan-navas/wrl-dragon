// Character sprite loader for Modern tiles_Free character strips
const CharacterPngSprites = {
    TILE_W: 16,
    TILE_H: 32,

    CHARACTERS: ["Alex", "Amelia", "Bob", "Adam"],
    STRIPS: ["idle", "idle_anim", "run", "sit", "sit2", "sit3", "phone"],

    // Frame counts for the down-facing direction
    // idle has 1 frame per direction (4 total) — use only frame 0
    // idle_anim has 6 frames per direction (24 total) — use first 6 for down
    // run/sit/sit2 have 6 per direction (24 total) — first 6 = down
    // sit3 has 6 per direction (12 total) — first 6 = down
    // phone has 9 frames (single direction or 3x3)
    FRAME_COUNTS: {
        idle: 1,        // 4 total but each is a different direction — only use frame 0
        idle_anim: 6,   // 24 total, first 6 = down-facing breathing
        run: 6,         // 24 total, first 6 = down-facing walk
        sit: 6,         // 24 total, first 6 = down-facing typing
        sit2: 6,        // 24 total, first 6 = down-facing alt
        sit3: 6,        // 12 total, first 6 = down-facing gesture
        phone: 9,       // 9 frames
    },

    images: {},  // { Alex: { idle: Image, run: Image, ... }, ... }
    _loadedCount: 0,
    _totalCount: 0,

    init() {
        const base = "public/Modern tiles_Free/Characters_free";
        this._totalCount = this.CHARACTERS.length * this.STRIPS.length;

        for (const name of this.CHARACTERS) {
            this.images[name] = {};
            for (const strip of this.STRIPS) {
                const img = new Image();
                img.onload = () => { this._loadedCount++; };
                img.onerror = () => {
                    console.warn(`Failed to load: ${name}_${strip}`);
                    this._loadedCount++;
                };
                img.src = `${base}/${name}_${strip}_16x16.png`;
                this.images[name][strip] = img;
            }
        }
    },

    isReady() {
        return this._loadedCount >= this._totalCount;
    },

    getFrame(characterName, stripName, frameIndex) {
        const charImages = this.images[characterName];
        if (!charImages) return null;
        const img = charImages[stripName];
        if (!img || !img.complete || !img.naturalWidth) return null;

        const maxFrames = this.FRAME_COUNTS[stripName] || 4;
        const frame = frameIndex % maxFrames;

        return {
            image: img,
            sx: frame * this.TILE_W,
            sy: 0,
            sw: this.TILE_W,
            sh: this.TILE_H,
        };
    }
};

CharacterPngSprites.init();

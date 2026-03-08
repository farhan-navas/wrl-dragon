// Character sprite loader for Modern tiles_Free character strips
const CharacterPngSprites = {
    TILE_W: 16,
    TILE_H: 32,

    CHARACTERS: ["Alex", "Amelia", "Bob", "Adam"],
    STRIPS: ["idle", "idle_anim", "run", "sit", "sit2", "sit3", "phone"],

    // Frame counts for the down-facing direction
    // Strips at 16px frame width: idle, idle_anim, run, phone
    // Strips at 32px frame width: sit, sit2, sit3 (wider sitting poses)
    FRAME_COUNTS: {
        idle: 1,        // 4 frames@16px, 1 per direction
        idle_anim: 6,   // 24 frames@16px, 6 per direction — first 6 = down
        run: 6,         // 24 frames@16px, 6 per direction — first 6 = down
        sit: 3,         // 12 frames@32px, 3 per direction — first 3 = down
        sit2: 3,        // 12 frames@32px, 3 per direction — first 3 = down
        sit3: 3,        // 6 frames@32px, 3 per direction — first 3 = down
        phone: 9,       // 9 frames@16px, single direction
    },

    // Frame pixel width per strip (default TILE_W=16 for most, 32 for sit strips)
    FRAME_WIDTHS: {
        sit: 32,
        sit2: 32,
        sit3: 32,
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
        const frameW = this.FRAME_WIDTHS[stripName] || this.TILE_W;

        return {
            image: img,
            sx: frame * frameW,
            sy: 0,
            sw: frameW,
            sh: this.TILE_H,
        };
    }
};

CharacterPngSprites.init();

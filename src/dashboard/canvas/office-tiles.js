// Tile-based office backgrounds using Room Builder Office + Modern Office Shadowless
// Each tier gets a unique wall style, floor style, and furniture layout
const OfficeTiles = {
    SCALE: 2,
    TILE: 16,

    get ST() { return this.TILE * this.SCALE; }, // 32px scaled tile

    // Room Builder Office wall/floor tile coordinates
    // Wall rows come in pairs: top row (full wall) + bottom row (baseboard)
    // Floor cols: 10-12 = style A, 13-15 = style B
    tiers: {
        ceo: {
            wallTopRow: 9,      // Warm beige wall upper
            wallBotRow: 10,     // Warm beige wall lower/baseboard
            floorCols: [13, 14, 15], floorRow: 9,  // Brown wood floor
            altFloorCols: [13, 14, 15], altFloorRow: 10, // Wood variant
        },
        coder: {
            wallTopRow: 7,      // Grey wall upper
            wallBotRow: 8,      // Grey wall lower
            floorCols: [10, 11, 12], floorRow: 7,  // Dark grey floor
            altFloorCols: [10, 11, 12], altFloorRow: 8,
        },
        qa: {
            wallTopRow: 5,      // Lavender wall upper
            wallBotRow: 6,      // Lavender wall lower
            floorCols: [13, 14, 15], floorRow: 5,  // Light grey floor
            altFloorCols: [13, 14, 15], altFloorRow: 6,
        },
    },

    // Modern Office Shadowless furniture coordinates (col, row, tilesW, tilesH)
    furniture: {
        // Desk partitions (long counters with dividers)
        deskBeige:   { col: 0,  row: 1,  tw: 5, th: 2 },  // Warm wooden desk
        deskLtBeige: { col: 5,  row: 1,  tw: 5, th: 2 },  // Lighter wooden desk
        deskGrey:    { col: 10, row: 1,  tw: 5, th: 2 },  // Grey desk partition

        // Couches (5 tiles wide x 2 tall)
        couchBeige:  { col: 0,  row: 5,  tw: 5, th: 2 },  // Beige couch
        couchPurple: { col: 5,  row: 5,  tw: 5, th: 2 },  // Purple/grey couch

        // Chairs (2x3 each from singles grid)
        chairBlue1:  { col: 0,  row: 8,  tw: 2, th: 3 },  // Blue office chair
        chairBlue2:  { col: 2,  row: 8,  tw: 2, th: 3 },  // Grey cabinet/shelf
        plantGreen:  { col: 6,  row: 8,  tw: 2, th: 3 },  // Green plant

        // Orange chairs
        chairOrange: { col: 0,  row: 10, tw: 2, th: 3 },  // Orange/red chair
        plantTeal:   { col: 6,  row: 10, tw: 2, th: 3 },  // Teal plant

        // Wall art & decorations
        artPink:     { col: 0,  row: 12, tw: 2, th: 3 },  // Pink painting
        artGreen:    { col: 2,  row: 12, tw: 2, th: 3 },  // Frame/art
        coffeeMaker: { col: 4,  row: 12, tw: 2, th: 3 },  // Coffee/vending
        plantLg:     { col: 6,  row: 12, tw: 2, th: 3 },  // Large plant

        // Monitors & computers
        monitor1:    { col: 8,  row: 12, tw: 2, th: 3 },  // Monitor setup
        computer1:   { col: 10, row: 12, tw: 2, th: 3 },  // Computer
        screenWall:  { col: 12, row: 12, tw: 2, th: 3 },  // Wall screen

        // Shelving
        shelfDark1:  { col: 0,  row: 16, tw: 2, th: 3 },  // Dark shelf
        shelfDark2:  { col: 2,  row: 16, tw: 2, th: 3 },  // Dark shelf variant

        // Small desks/tables
        tableBeige1: { col: 0,  row: 34, tw: 2, th: 3 },  // Small beige table
        tableBeige3: { col: 0,  row: 36, tw: 3, th: 3 },  // Wider beige table
        tableGrey1:  { col: 0,  row: 46, tw: 2, th: 3 },  // Grey table
        tableGrey3:  { col: 0,  row: 48, tw: 3, th: 3 },  // Wider grey table

        // Larger desks
        deskLBeige:  { col: 5,  row: 18, tw: 3, th: 3 },  // L-shaped beige desk
        deskWood:    { col: 13, row: 19, tw: 2, th: 3 },  // Small wood desk

        // Whiteboard / screen
        whiteboard:  { col: 11, row: 14, tw: 2, th: 3 },  // Whiteboard
        acUnit:      { col: 3,  row: 14, tw: 2, th: 3 },  // AC/thermostat
    },

    // Render walls and floor for a single tier
    renderTierBackground(ctx, tierKey, y0, height) {
        if (!TilesetLoader.isReady()) return;

        const st = this.ST;
        const tilesW = Math.ceil(800 / st);  // 25 tiles
        const tier = this.tiers[tierKey];

        // Row 0: Wall (single tile row for compact tiers)
        for (let c = 0; c < tilesW; c++) {
            TilesetLoader.drawTile(ctx, "rbOffice", c % 4, tier.wallTopRow, c * st, y0, this.SCALE);
        }

        // Row 1: Baseboard transition (lower wall blending into floor)
        for (let c = 0; c < tilesW; c++) {
            TilesetLoader.drawTile(ctx, "rbOffice", c % 4, tier.wallBotRow, c * st, y0 + st, this.SCALE);
        }

        // Rows 2+: Floor tiles
        const floorStartY = y0 + st * 2;
        const floorRows = Math.ceil((height - st * 2) / st);

        for (let r = 0; r < floorRows; r++) {
            const fCols = r % 2 === 0 ? tier.floorCols : tier.altFloorCols;
            const fRow = r % 2 === 0 ? tier.floorRow : tier.altFloorRow;
            for (let c = 0; c < tilesW; c++) {
                const fc = fCols[c % fCols.length];
                TilesetLoader.drawTile(ctx, "rbOffice", fc, fRow, c * st, floorStartY + r * st, this.SCALE);
            }
        }
    },

    // Draw a furniture item from the Modern Office sheet
    drawFurniture(ctx, itemName, destX, destY) {
        const item = this.furniture[itemName];
        if (!item) return;
        TilesetLoader.drawMultiTile(
            ctx, "office",
            item.col, item.row, item.tw, item.th,
            destX, destY, this.SCALE
        );
    },

    // Render furniture for CEO tier (executive suite - warm/prestigious)
    renderCEOFurniture(ctx, y0) {
        const st = this.ST;
        const floorY = y0 + st * 2;  // Where floor starts (y0 + 64)

        // CEO desk (left side) - wooden desk partition centered on agent x=250
        this.drawFurniture(ctx, "deskBeige", 170, floorY - 8);

        // Analyst desk (right side) - centered on agent x=550
        this.drawFurniture(ctx, "deskLtBeige", 470, floorY - 8);

        // Plants flanking the room (on baseboard row)
        this.drawFurniture(ctx, "plantGreen", 16, y0 + st - 16);
        this.drawFurniture(ctx, "plantGreen", 740, y0 + st - 16);

        // Wall decorations (positioned on wall)
        this.drawFurniture(ctx, "artPink", 60, y0 - 20);
        this.drawFurniture(ctx, "screenWall", 650, y0 - 20);
    },

    // Render furniture for Coder tier (dev floor - grey/blue/utilitarian)
    renderCoderFurniture(ctx, y0) {
        const st = this.ST;
        const floorY = y0 + st * 2;

        // Row of grey desk partitions - centered on agent positions (150, 400, 650)
        this.drawFurniture(ctx, "deskGrey", 70, floorY - 8);
        this.drawFurniture(ctx, "deskGrey", 320, floorY - 8);
        this.drawFurniture(ctx, "deskGrey", 570, floorY - 8);

        // Plants at edges
        this.drawFurniture(ctx, "plantTeal", 10, y0 + st - 16);
        this.drawFurniture(ctx, "plantTeal", 750, y0 + st - 16);

        // Shelving on wall
        this.drawFurniture(ctx, "shelfDark1", 370, y0 - 20);
    },

    // Render furniture for QA tier (testing lab - lavender/light)
    renderQAFurniture(ctx, y0) {
        const st = this.ST;
        const floorY = y0 + st * 2;

        // QA desks - lighter desk partitions, centered on agent positions
        this.drawFurniture(ctx, "deskLtBeige", 70, floorY - 8);
        this.drawFurniture(ctx, "deskLtBeige", 320, floorY - 8);
        this.drawFurniture(ctx, "deskLtBeige", 570, floorY - 8);

        // Bookshelves on wall
        this.drawFurniture(ctx, "shelfDark2", 16, y0 - 20);
        this.drawFurniture(ctx, "shelfDark2", 730, y0 - 20);

        // Coffee/vending area
        this.drawFurniture(ctx, "coffeeMaker", 370, y0 - 20);
    },

    // Main render entry point - draws all 3 tiers
    renderAll(ctx, floors) {
        if (!TilesetLoader.isReady()) return false;

        for (const [tierKey, floor] of Object.entries(floors)) {
            this.renderTierBackground(ctx, tierKey, floor.y, floor.height);
        }

        // Furniture per tier
        this.renderCEOFurniture(ctx, floors.ceo.y);
        this.renderCoderFurniture(ctx, floors.coder.y);
        this.renderQAFurniture(ctx, floors.qa.y);

        return true;
    },
};

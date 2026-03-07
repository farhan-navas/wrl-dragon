// Simple canvas-based reward curve chart
const RewardChart = {
    canvas: null,
    ctx: null,

    init() {
        this.canvas = document.getElementById("reward-chart");
        this.ctx = this.canvas.getContext("2d");
    },

    draw(rewards) {
        if (!this.ctx) this.init();
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const pad = { top: 20, right: 10, bottom: 25, left: 40 };

        ctx.clearRect(0, 0, w, h);

        if (!rewards || rewards.length === 0) {
            this.drawEmpty();
            return;
        }

        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        const maxStep = rewards[rewards.length - 1].step || 1;
        const maxCum = Math.max(...rewards.map(r => r.cumulative), 1);

        // Background
        ctx.fillStyle = "#0a0a15";
        ctx.fillRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = "#1a1a2e";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (plotH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }

        // Cumulative reward line
        ctx.strokeStyle = "#6bff6b";
        ctx.lineWidth = 2;
        ctx.beginPath();
        rewards.forEach((r, i) => {
            const x = pad.left + (r.step / maxStep) * plotW;
            const y = pad.top + plotH - (r.cumulative / maxCum) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Per-step reward as dots
        ctx.fillStyle = "#ffb86b44";
        rewards.forEach(r => {
            const x = pad.left + (r.step / maxStep) * plotW;
            const maxR = Math.max(...rewards.map(r => Math.abs(r.reward)), 1);
            const y = pad.top + plotH / 2 - (r.reward / maxR) * (plotH / 4);
            ctx.fillRect(x - 1, y - 1, 2, 2);
        });

        // Axes labels
        ctx.fillStyle = "#666";
        ctx.font = "6px 'Press Start 2P', monospace";
        ctx.textAlign = "center";
        ctx.fillText("Steps", w / 2, h - 5);

        ctx.save();
        ctx.translate(8, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Reward", 0, 0);
        ctx.restore();

        // Title
        ctx.fillStyle = "#888";
        ctx.textAlign = "center";
        ctx.fillText("Cumulative Reward", w / 2, 12);
        ctx.textAlign = "left";

        // Max value
        ctx.fillStyle = "#555";
        ctx.font = "6px 'Press Start 2P', monospace";
        ctx.fillText(maxCum.toFixed(0), 2, pad.top + 4);
    },

    drawEmpty() {
        if (!this.ctx) this.init();
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fillStyle = "#0a0a15";
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fillStyle = "#444";
        ctx.font = "8px 'Press Start 2P', monospace";
        ctx.textAlign = "center";
        ctx.fillText("No reward data", this.canvas.width / 2, this.canvas.height / 2);
        ctx.textAlign = "left";
    },

    addPoint(step, reward, cumulative) {
        // For live updates - append to existing chart
        // Simplified: just redraw with accumulated data
        if (!this._liveData) this._liveData = [];
        this._liveData.push({ step, reward, cumulative });
        this.draw(this._liveData);
    },

    resetLiveData() {
        this._liveData = [];
    }
};

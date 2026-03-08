// Training view for Phase 2 meta-training visualization
const TrainingView = {
    container: null,
    rewardCanvas: null,
    lossCanvas: null,
    _rewardData: {},   // env -> [{iteration, reward}]
    _lossData: [],     // [{iteration, loss}]
    _syntaxData: [],   // [{iteration, rate}]
    _bestCode: "",
    _totalIterations: 0,
    _currentIteration: 0,

    init() {
        this.container = document.getElementById("training-container");
        this.rewardCanvas = document.getElementById("training-reward-chart");
        this.lossCanvas = document.getElementById("training-loss-chart");
    },

    show() {
        if (!this.container) this.init();
        this.container.classList.remove("hidden");
        document.getElementById("canvas-container").classList.add("dimmed");
    },

    hide() {
        if (!this.container) this.init();
        this.container.classList.add("hidden");
        document.getElementById("canvas-container").classList.remove("dimmed");
    },

    onTrainingStarted(event) {
        this.show();
        this._totalIterations = event.iterations || 0;
        this._rewardData = {};
        this._lossData = [];
        this._syntaxData = [];
        this._bestCode = "";

        document.getElementById("training-model").textContent = event.model_id || "?";
        document.getElementById("training-envs").textContent = (event.envs || []).join(", ");
        document.getElementById("training-total").textContent = this._totalIterations;
        document.getElementById("training-current").textContent = "0";
        document.getElementById("training-status-text").textContent = "Training...";

        this._drawRewardChart();
        this._drawLossChart();
    },

    onTrainingStep(event) {
        this._currentIteration = event.iteration || 0;
        document.getElementById("training-current").textContent = this._currentIteration;

        // Accumulate reward data per env
        if (event.mean_rewards) {
            for (const [env, reward] of Object.entries(event.mean_rewards)) {
                if (!this._rewardData[env]) this._rewardData[env] = [];
                this._rewardData[env].push({ iteration: this._currentIteration, reward });
            }
        }

        // Loss
        if (event.loss != null) {
            this._lossData.push({ iteration: this._currentIteration, loss: event.loss });
        }

        // Syntax rate
        if (event.syntax_rate != null) {
            this._syntaxData.push({ iteration: this._currentIteration, rate: event.syntax_rate });
            document.getElementById("training-syntax-rate").textContent =
                (event.syntax_rate * 100).toFixed(0) + "%";
        }

        // Best code
        if (event.best_code) {
            this._bestCode = event.best_code;
            document.getElementById("training-best-code").textContent = this._bestCode;
        }

        this._drawRewardChart();
        this._drawLossChart();
    },

    onTrainingCheckpoint(event) {
        const el = document.getElementById("training-status-text");
        el.textContent = `Checkpoint saved (step ${event.iteration})`;
        setTimeout(() => { el.textContent = "Training..."; }, 3000);
    },

    onTrainingCompleted(event) {
        document.getElementById("training-status-text").textContent = "Complete!";
        document.getElementById("training-current").textContent = event.iterations || this._currentIteration;

        if (event.elapsed_min != null) {
            document.getElementById("training-elapsed").textContent =
                event.elapsed_min.toFixed(1) + " min";
        }
        if (event.loss != null) {
            document.getElementById("training-final-loss").textContent = event.loss.toFixed(4);
        }
    },

    // ---- Charts ----

    _COLORS: ["#6bff6b", "#6bb5ff", "#ffb86b", "#ff6bff", "#ffff6b", "#ff6b6b"],

    _drawRewardChart() {
        if (!this.rewardCanvas) this.init();
        const ctx = this.rewardCanvas.getContext("2d");
        const w = this.rewardCanvas.width;
        const h = this.rewardCanvas.height;
        const pad = { top: 25, right: 15, bottom: 25, left: 45 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#0a0a15";
        ctx.fillRect(0, 0, w, h);

        const envNames = Object.keys(this._rewardData);
        if (envNames.length === 0) {
            ctx.fillStyle = "#444";
            ctx.font = "8px 'Press Start 2P', monospace";
            ctx.textAlign = "center";
            ctx.fillText("Waiting for data...", w / 2, h / 2);
            ctx.textAlign = "left";
            return;
        }

        // Compute ranges
        let maxIter = 1, minR = 0, maxR = 1;
        for (const pts of Object.values(this._rewardData)) {
            for (const p of pts) {
                maxIter = Math.max(maxIter, p.iteration);
                minR = Math.min(minR, p.reward);
                maxR = Math.max(maxR, p.reward);
            }
        }
        const rRange = maxR - minR || 1;

        // Grid
        ctx.strokeStyle = "#1a1a2e";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (plotH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }

        // Draw each env series
        envNames.forEach((env, idx) => {
            const pts = this._rewardData[env];
            const color = this._COLORS[idx % this._COLORS.length];
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            pts.forEach((p, i) => {
                const x = pad.left + (p.iteration / maxIter) * plotW;
                const y = pad.top + plotH - ((p.reward - minR) / rRange) * plotH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Legend
            ctx.fillStyle = color;
            ctx.font = "6px 'Press Start 2P', monospace";
            const lx = pad.left + 5;
            const ly = pad.top + 10 + idx * 10;
            ctx.fillRect(lx, ly - 3, 6, 6);
            ctx.fillText(env, lx + 10, ly + 3);
        });

        // Title
        ctx.fillStyle = "#888";
        ctx.font = "7px 'Press Start 2P', monospace";
        ctx.textAlign = "center";
        ctx.fillText("Mean Reward / Iteration", w / 2, 12);
        ctx.textAlign = "left";

        // Axis values
        ctx.fillStyle = "#555";
        ctx.font = "6px 'Press Start 2P', monospace";
        ctx.fillText(maxR.toFixed(2), 2, pad.top + 4);
        ctx.fillText(minR.toFixed(2), 2, pad.top + plotH);
        ctx.textAlign = "center";
        ctx.fillText(String(maxIter), w - pad.right, h - 5);
        ctx.fillText("0", pad.left, h - 5);
        ctx.textAlign = "left";
    },

    _drawLossChart() {
        if (!this.lossCanvas) this.init();
        const ctx = this.lossCanvas.getContext("2d");
        const w = this.lossCanvas.width;
        const h = this.lossCanvas.height;
        const pad = { top: 25, right: 15, bottom: 25, left: 45 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#0a0a15";
        ctx.fillRect(0, 0, w, h);

        if (this._lossData.length === 0) {
            ctx.fillStyle = "#444";
            ctx.font = "8px 'Press Start 2P', monospace";
            ctx.textAlign = "center";
            ctx.fillText("No loss data", w / 2, h / 2);
            ctx.textAlign = "left";
            return;
        }

        let maxIter = 1, minL = Infinity, maxL = 0;
        for (const p of this._lossData) {
            maxIter = Math.max(maxIter, p.iteration);
            minL = Math.min(minL, p.loss);
            maxL = Math.max(maxL, p.loss);
        }
        const lRange = maxL - minL || 1;

        // Grid
        ctx.strokeStyle = "#1a1a2e";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (plotH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }

        // Loss line
        ctx.strokeStyle = "#ff6b6b";
        ctx.lineWidth = 2;
        ctx.beginPath();
        this._lossData.forEach((p, i) => {
            const x = pad.left + (p.iteration / maxIter) * plotW;
            const y = pad.top + ((p.loss - minL) / lRange) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Syntax rate overlay (green dots)
        if (this._syntaxData.length > 0) {
            ctx.fillStyle = "#6bff6b88";
            this._syntaxData.forEach(p => {
                const x = pad.left + (p.iteration / maxIter) * plotW;
                const y = pad.top + plotH - p.rate * plotH;
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, Math.PI * 2);
                ctx.fill();
            });
        }

        // Title
        ctx.fillStyle = "#888";
        ctx.font = "7px 'Press Start 2P', monospace";
        ctx.textAlign = "center";
        ctx.fillText("Loss (red) / Syntax Rate (green)", w / 2, 12);
        ctx.textAlign = "left";

        // Axis values
        ctx.fillStyle = "#555";
        ctx.font = "6px 'Press Start 2P', monospace";
        ctx.fillText(maxL.toFixed(3), 2, pad.top + 4);
        ctx.fillText(minL.toFixed(3), 2, pad.top + plotH);
        ctx.textAlign = "left";
    },
};

// Side panel for viewing rollout details
const RolloutViewer = {
    currentAgent: null,
    rollouts: [],

    async open(agent) {
        this.currentAgent = agent;
        const panel = document.getElementById("side-panel");
        panel.classList.remove("hidden");
        document.getElementById("panel-title").textContent = `${agent.displayName || agent.id}`;

        await this.loadRollouts(agent.id, agent.env);
    },

    close() {
        this.currentAgent = null;
        document.getElementById("side-panel").classList.add("hidden");
    },

    async loadRollouts(agentId, env) {
        try {
            const url = env
                ? `/api/rollouts/${agentId}?env=${encodeURIComponent(env)}`
                : `/api/rollouts/${agentId}`;
            const resp = await fetch(url);
            this.rollouts = await resp.json();
        } catch {
            this.rollouts = [];
        }

        this.populateEpisodeSelector();
        if (this.rollouts.length > 0) {
            this.selectEpisode(0);
        } else {
            document.getElementById("log-entries").innerHTML =
                '<div class="log-entry">No rollouts yet</div>';
        }
    },

    populateEpisodeSelector() {
        const select = document.getElementById("episode-selector");
        select.innerHTML = "";
        this.rollouts.forEach((r, i) => {
            const opt = document.createElement("option");
            opt.value = i;
            opt.textContent = `Rollout ${i + 1} | R:${r.total_reward.toFixed(1)} | ${r.steps} steps`;
            select.appendChild(opt);
        });
        select.onchange = () => this.selectEpisode(parseInt(select.value));
    },

    async selectEpisode(index) {
        const rollout = this.rollouts[index];
        if (!rollout) return;

        // Load video
        const video = document.getElementById("rollout-video");
        video.src = rollout.video_url;

        // Load reward data
        try {
            const resp = await fetch(`/api/rewards/${rollout.run_id}`);
            const rewards = await resp.json();
            RewardChart.draw(rewards);
        } catch {
            RewardChart.drawEmpty();
        }

        // Load step log
        await this.loadStepLog(rollout);
    },

    async loadStepLog(rollout) {
        const container = document.getElementById("log-entries");
        container.innerHTML = "";

        try {
            const resp = await fetch(`/api/rewards/${rollout.run_id}`);
            const steps = await resp.json();

            for (const step of steps) {
                const div = document.createElement("div");
                div.className = `log-entry${step.reward === 0 ? " done" : ""}`;
                div.textContent = `S${step.step} | R:${step.reward.toFixed(2)} | C:${step.cumulative.toFixed(1)}`;
                container.appendChild(div);
            }
        } catch {
            container.innerHTML = '<div class="log-entry">Could not load log</div>';
        }
    },

    init() {
        document.getElementById("close-panel").addEventListener("click", () => this.close());
    }
};

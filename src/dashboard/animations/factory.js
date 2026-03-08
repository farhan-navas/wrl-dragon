/**
 * Animation Factory — centralized registry of stage/event animations.
 * Maps keys to animation logic; both real WebSocket events and demo mode use run(key, context).
 */
const AnimationFactory = {
    _registry: {},

    register(key, fn) {
        this._registry[key] = fn;
    },

    run(key, context = {}) {
        const fn = this._registry[key];
        if (fn) {
            console.log(`[AnimationFactory] running "${key}"`, context);
            try {
                fn(context);
            } catch (e) {
                console.warn(`AnimationFactory.run("${key}"):`, e);
            }
        } else {
            console.warn(`[AnimationFactory] no handler for "${key}"`);
        }
    },

    _init() {
        // ── Real event animations (match event.type from WebSocket) ──

        this.register("agent_spawned", (ctx) => {
            const agentId = ctx.agent_id;
            if (!Renderer.getAgent(agentId)) {
                Renderer.addAgent({
                    id: agentId,
                    tier: ctx.tier,
                    env: ctx.env,
                    name: ctx.name || agentId,
                });
            }
            Renderer.showSpeechBubble(agentId, "Ready!", 2000);
        });

        this.register("task_assigned", (ctx) => {
            const fromId = ctx.from;
            const toId = ctx.to;

            Renderer.setAgentState(fromId, "assigning");
            Renderer.showSpeechBubble(fromId, ctx.task || "New task", 3000);
            Renderer.addTaskLine(fromId, toId);

            const toAgent = Renderer.getAgent(toId);
            if (toAgent) {
                if (toAgent.tier === "coder") {
                    Renderer.setAgentState(toId, "coding");
                } else {
                    Renderer.setAgentState(toId, "thinking");
                }
                Renderer.showSpeechBubble(toId, "On it!", 2000);
            }

            setTimeout(() => Renderer.setAgentState(fromId, "thinking"), 3000);
        });

        this.register("rollout_started", (ctx) => {
            Renderer.setAgentState(ctx.agent_id, "running");
            Renderer.showSpeechBubble(ctx.agent_id, `Run: ${ctx.run_id || "?"}`, 2000);
            if (typeof RewardChart !== "undefined") RewardChart.resetLiveData();
        });

        this.register("rollout_completed", (ctx) => {
            Renderer.setAgentState(ctx.agent_id, "reporting");
            Renderer.showSpeechBubble(
                ctx.agent_id,
                `R:${ctx.total_reward?.toFixed(0) || "?"} / ${ctx.steps ?? "?"}s`,
                4000
            );
            setTimeout(() => Renderer.setAgentState(ctx.agent_id, "idle"), 4000);
        });

        this.register("reward_update", (ctx) => {
            if (typeof RewardChart !== "undefined") {
                RewardChart.addPoint(ctx.step, ctx.reward, ctx.cumulative);
            }
        });

        // ── Demo-only animations (scripted sequence) ──

        this.register("demo_ceo_thinking", () => {
            Renderer.setAgentState("ceo-1", "thinking");
            Renderer.showSpeechBubble("ceo-1", "Analyzing env...", 3000);
            Renderer.setAgentState("analyst-1", "thinking");
            Renderer.showSpeechBubble("analyst-1", "Checking metrics", 3000);
        });

        this.register("demo_ceo_assigns_coder", () => {
            Renderer.setAgentState("ceo-1", "assigning");
            Renderer.showSpeechBubble("ceo-1", "Generate policy", 3000);
            Renderer.addTaskLine("ceo-1", "coder-1");
            Renderer.setAgentState("coder-1", "thinking");
            Renderer.showSpeechBubble("coder-1", "On it!", 2000);
            setTimeout(() => Renderer.setAgentState("ceo-1", "thinking"), 3000);
        });

        this.register("demo_coders_coding", () => {
            Renderer.setAgentState("ceo-1", "thinking");
            Renderer.setAgentState("coder-1", "coding");
            Renderer.showSpeechBubble("coder-1", "Writing code...", 3000);
            Renderer.setAgentState("coder-2", "coding");
            Renderer.showSpeechBubble("coder-2", "Reward fn...", 3000);
        });

        this.register("demo_coder_assigns_qa", (ctx) => {
            const fromId = ctx.from || "coder-1";
            const toId = ctx.to || "qa-1";
            const isFirst = toId === "qa-1";

            Renderer.setAgentState(fromId, "assigning");
            Renderer.addTaskLine(fromId, toId);
            Renderer.setAgentState(toId, "thinking");
            Renderer.showSpeechBubble(toId, isFirst ? "Received!" : "Starting...", 2000);
            if (fromId === "coder-2") {
                setTimeout(() => Renderer.setAgentState(fromId, "idle"), 2000);
            }
        });

        this.register("demo_qa_running", () => {
            Renderer.setAgentState("coder-1", "idle");
            Renderer.setAgentState("qa-1", "running");
            Renderer.showSpeechBubble("qa-1", "Rollout ep 1/5", 3000);
            Renderer.setAgentState("qa-2", "running");
            Renderer.showSpeechBubble("qa-2", "Rollout ep 1/5", 3000);
        });

        this.register("demo_qa_running_mid", () => {
            Renderer.showSpeechBubble("qa-1", "Rollout ep 3/5", 3000);
            Renderer.showSpeechBubble("qa-2", "Rollout ep 2/5", 3000);
        });

        this.register("demo_qa_reports", () => {
            Renderer.setAgentState("qa-1", "reporting");
            Renderer.showSpeechBubble("qa-1", "R:142 / 500s", 3000);
            Renderer.addTaskLine("qa-1", "ceo-1");

            Renderer.setAgentState("qa-2", "reporting");
            Renderer.showSpeechBubble("qa-2", "R:87 / 312s", 3000);
            Renderer.addTaskLine("qa-2", "analyst-1");
            Renderer.setAgentState("analyst-1", "thinking");
            Renderer.showSpeechBubble("analyst-1", "Analyzing...", 3000);
        });

        this.register("demo_reset", () => {
            Renderer.setAgentState("ceo-1", "idle");
            Renderer.setAgentState("analyst-1", "idle");
            Renderer.setAgentState("coder-1", "idle");
            Renderer.setAgentState("coder-2", "idle");
            Renderer.setAgentState("qa-1", "idle");
            Renderer.setAgentState("qa-2", "idle");
        });
    },
};

// Initialize registry when DOM is ready (Renderer must exist)
if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => AnimationFactory._init());
    } else {
        AnimationFactory._init();
    }
}

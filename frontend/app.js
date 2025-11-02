const form = document.getElementById("analysis-form");
const statusEl = document.getElementById("status");
const resetButton = document.getElementById("reset-button");
const submitButton = form.querySelector("button[type='submit']");

const ideaInput = document.getElementById("idea");
const hunchesInput = document.getElementById("hunches");
const contextInput = document.getElementById("context");
const constraintsInput = document.getElementById("constraints");
const moatConceptInput = document.getElementById("moat-concept");
const moatTriggersInput = document.getElementById("moat-triggers");
const judgeSummaryInput = document.getElementById("judge-summary");

const telemetryLog = [];

function setStatus(message, state = "idle") {
    statusEl.textContent = message;
    if (state) {
        statusEl.dataset.state = state;
    } else {
        delete statusEl.dataset.state;
    }
}

function parseLines(value) {
    return (value || "")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);
}

function parseJSON(value) {
    if (!value || !value.trim()) {
        return {};
    }
    try {
        return JSON.parse(value);
    } catch (err) {
        throw new Error("Jobs context must be valid JSON");
    }
}

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    let data;
    try {
        data = await response.json();
    } catch (err) {
        throw new Error(`Failed to parse response from ${url}`);
    }

    if (!response.ok) {
        const detail = data?.detail || data?.error || response.statusText;
        throw new Error(detail);
    }

    return data;
}

function renderAssumptions(assumptions) {
    const list = document.getElementById("assumptions-list");
    const placeholder = document.getElementById("assumptions-empty");
    list.innerHTML = "";

    if (!Array.isArray(assumptions) || assumptions.length === 0) {
        list.hidden = true;
        placeholder.hidden = false;
        return;
    }

    placeholder.hidden = true;
    list.hidden = false;

    for (const assumption of assumptions) {
        const item = document.createElement("li");
        item.className = "assumption-item";

        const header = document.createElement("header");
        const title = document.createElement("h3");
        title.textContent = assumption.text || "Untitled assumption";

        const level = document.createElement("span");
        level.className = "level-pill";
        const levelValue = typeof assumption.level === "number" ? assumption.level : "?";
        level.innerHTML = `<strong>L${levelValue}</strong>`;

        header.appendChild(title);
        header.appendChild(level);
        item.appendChild(header);

        const confidence = document.createElement("div");
        confidence.className = "confidence";
        const confidenceValue = typeof assumption.confidence === "number" ? (assumption.confidence * 100).toFixed(0) : "--";
        confidence.textContent = `Confidence: ${confidenceValue}%`;
        item.appendChild(confidence);

        if (Array.isArray(assumption.evidence) && assumption.evidence.length) {
            const evidenceList = document.createElement("ul");
            evidenceList.className = "evidence-list";
            for (const evidence of assumption.evidence) {
                const li = document.createElement("li");
                li.textContent = evidence;
                evidenceList.appendChild(li);
            }
            item.appendChild(evidenceList);
        }

        list.appendChild(item);
    }
}

function renderJobs(jobs) {
    const container = document.getElementById("jobs-list");
    const placeholder = document.getElementById("jobs-empty");
    container.innerHTML = "";

    if (!Array.isArray(jobs) || jobs.length === 0) {
        container.hidden = true;
        placeholder.hidden = false;
        return;
    }

    container.hidden = false;
    placeholder.hidden = true;

    jobs.forEach((job, index) => {
        const card = document.createElement("div");
        card.className = "job-card";

        const title = document.createElement("h3");
        const prefix = job.job_id ? `${index + 1}. ` : "";
        title.textContent = `${prefix}${job.statement || "Job statement"}`;
        card.appendChild(title);

        const forces = document.createElement("div");
        forces.className = "forces-grid";
        const buckets = ["push", "pull", "anxiety", "inertia"];
        buckets.forEach((bucket) => {
            const values = (job.forces && job.forces[bucket]) || [];
            if (!Array.isArray(values) || values.length === 0) {
                return;
            }
            const force = document.createElement("div");
            force.className = "force";
            const label = document.createElement("strong");
            label.textContent = bucket;
            force.appendChild(label);
            const list = document.createElement("ul");
            list.style.margin = "0.35rem 0 0";
            list.style.paddingLeft = "1rem";
            values.forEach((item) => {
                const li = document.createElement("li");
                li.textContent = item;
                list.appendChild(li);
            });
            force.appendChild(list);
            forces.appendChild(force);
        });

        if (forces.children.length) {
            card.appendChild(forces);
        }

        container.appendChild(card);
    });
}

function renderMoat(layers) {
    const container = document.getElementById("moat-list");
    const placeholder = document.getElementById("moat-empty");
    container.innerHTML = "";

    if (!Array.isArray(layers) || layers.length === 0) {
        container.hidden = true;
        placeholder.hidden = false;
        return;
    }

    container.hidden = false;
    placeholder.hidden = true;

    layers.forEach((layer) => {
        const block = document.createElement("div");
        block.className = "moat-layer";

        const heading = document.createElement("h3");
        heading.textContent = layer.type || "Innovation Layer";
        block.appendChild(heading);

        if (layer.trigger) {
            const trigger = document.createElement("p");
            trigger.innerHTML = `<strong>Trigger:</strong> ${layer.trigger}`;
            block.appendChild(trigger);
        }

        if (layer.effect) {
            const effect = document.createElement("p");
            effect.innerHTML = `<strong>Effect:</strong> ${layer.effect}`;
            block.appendChild(effect);
        }

        container.appendChild(block);
    });
}

function renderScorecard(scorecard) {
    const container = document.getElementById("scorecard");
    const placeholder = document.getElementById("judge-empty");
    container.innerHTML = "";

    if (!scorecard || !Array.isArray(scorecard.criteria) || scorecard.criteria.length === 0) {
        container.hidden = true;
        placeholder.hidden = false;
        return;
    }

    container.hidden = false;
    placeholder.hidden = true;

    const totalRow = document.createElement("div");
    totalRow.className = "score-row";
    const totalLabel = document.createElement("h3");
    totalLabel.textContent = "Total";
    const totalValue = document.createElement("span");
    totalValue.className = "score-total";
    const total = typeof scorecard.total === "number" ? scorecard.total.toFixed(1) : scorecard.total;
    totalValue.textContent = total ?? "--";
    totalRow.appendChild(totalLabel);
    totalRow.appendChild(totalValue);
    container.appendChild(totalRow);

    scorecard.criteria.forEach((criterion) => {
        const row = document.createElement("div");
        row.className = "score-row";

        const name = document.createElement("h3");
        name.textContent = criterion.name || "Criterion";
        row.appendChild(name);

        const meta = document.createElement("div");
        meta.style.textAlign = "right";

        const value = document.createElement("div");
        value.className = "score-value";
        value.textContent = typeof criterion.score === "number" ? criterion.score.toFixed(1) : criterion.score;
        meta.appendChild(value);

        if (criterion.rationale) {
            const rationale = document.createElement("p");
            rationale.style.margin = "0.35rem 0 0";
            rationale.style.color = "rgba(226, 232, 240, 0.75)";
            rationale.textContent = criterion.rationale;
            meta.appendChild(rationale);
        }

        row.appendChild(meta);
        container.appendChild(row);
    });
}

function renderTelemetry() {
    const summary = document.getElementById("telemetry-summary");
    const placeholder = document.getElementById("telemetry-empty");
    const details = document.getElementById("telemetry-details");
    const recentList = document.getElementById("telemetry-recent");

    summary.innerHTML = "";
    recentList.innerHTML = "";

    if (telemetryLog.length === 0) {
        summary.hidden = true;
        placeholder.hidden = false;
        details.hidden = true;
        details.open = false;
        return;
    }

    placeholder.hidden = true;
    summary.hidden = false;

    const aggregate = telemetryLog.reduce(
        (acc, entry) => {
            const usage = entry.usage || {};
            const totalCalls = usage.total_calls || 0;
            const errorCount = usage.error_count || 0;
            acc.totalCalls += totalCalls;
            acc.errors += errorCount;

            const byTool = usage.by_tool || {};
            Object.keys(byTool).forEach((tool) => {
                const stats = byTool[tool];
                const current = acc.byTool.get(tool) || { count: 0, errors: 0, latencySum: 0 };
                const count = stats.count || 0;
                current.count += count;
                current.errors += stats.errors || 0;
                if (typeof stats.avg_latency_s === "number" && count > 0) {
                    current.latencySum += stats.avg_latency_s * count;
                }
                acc.byTool.set(tool, current);
            });

            const recent = usage.recent_calls || [];
            recent.forEach((call) => {
                acc.recent.push({
                    tool: call.tool || entry.tool,
                    latency: call.latency_s,
                    error: call.error,
                    result: call.result_preview,
                });
            });

            return acc;
        },
        { totalCalls: 0, errors: 0, byTool: new Map(), recent: [] }
    );

    const toolNames = Array.from(aggregate.byTool.keys());

    const appendStat = (label, value) => {
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = value;
        summary.appendChild(dt);
        summary.appendChild(dd);
    };

    appendStat("Total tool calls", aggregate.totalCalls);
    appendStat("Errors", aggregate.errors);
    appendStat("Tools invoked", toolNames.length ? toolNames.join(", ") : "—");

    aggregate.byTool.forEach((stats, tool) => {
        const average = stats.count > 0 ? (stats.latencySum / stats.count).toFixed(2) : "0.00";
        const label = `${tool} avg latency`;
        const detail = `${average}s • ${stats.count} call${stats.count === 1 ? "" : "s"}` +
            (stats.errors ? ` • ${stats.errors} errors` : "");
        appendStat(label, detail);
    });

    const trimmedRecent = aggregate.recent.slice(-8).reverse();
    if (trimmedRecent.length) {
        trimmedRecent.forEach((call) => {
            const li = document.createElement("li");
            const latency = typeof call.latency === "number" ? `${call.latency.toFixed(2)}s` : "--";
            const header = document.createElement("div");
            header.innerHTML = `<strong>${call.tool}</strong> • ${latency}`;
            li.appendChild(header);

            if (call.error) {
                const error = document.createElement("div");
                error.style.color = "var(--error)";
                error.textContent = call.error;
                li.appendChild(error);
            } else if (call.result) {
                const preview = document.createElement("div");
                preview.style.color = "rgba(226, 232, 240, 0.7)";
                preview.textContent = call.result;
                li.appendChild(preview);
            }

            recentList.appendChild(li);
        });
        details.hidden = false;
    } else {
        details.hidden = true;
        details.open = false;
    }
}

function resetResults(options = { announce: true }) {
    telemetryLog.length = 0;
    renderAssumptions([]);
    renderJobs([]);
    renderMoat([]);
    renderScorecard(null);
    renderTelemetry();
    if (options.announce) {
        setStatus("Results cleared.", "success");
    } else {
        setStatus("Ready when you are.");
    }
}

async function runAnalysis(event) {
    event.preventDefault();
    const idea = ideaInput.value.trim();
    if (!idea) {
        setStatus("Please provide an idea description before running the analysis.", "error");
        return;
    }

    let context;
    try {
        context = parseJSON(contextInput.value);
    } catch (err) {
        setStatus(err.message, "error");
        return;
    }

    if (!context || typeof context !== "object") {
        context = {};
    }

    if (!context.prompt && idea) {
        context.prompt = idea;
    }

    const constraints = parseLines(constraintsInput.value);
    const hunches = parseLines(hunchesInput.value);
    const moatConcept = moatConceptInput.value.trim() || idea.slice(0, 80);
    const moatTriggers = moatTriggersInput.value.trim();
    const judgeSummary = judgeSummaryInput.value.trim() || idea;

    telemetryLog.length = 0;
    renderTelemetry();

    submitButton.disabled = true;
    setStatus("Running deconstruction…", "loading");

    try {
        const deconstruct = await postJSON("/deconstruct", {
            idea,
            hunches,
        });
        renderAssumptions(deconstruct.assumptions || []);
        if (deconstruct._telemetry?.tool_usage) {
            telemetryLog.push({ tool: "deconstruct", usage: deconstruct._telemetry.tool_usage });
        }
        renderTelemetry();
        setStatus("Generating JTBD statements…", "loading");

        const jobs = await postJSON("/jobs", {
            context,
            constraints,
        });
        renderJobs(jobs.jobs || []);
        if (jobs._telemetry?.tool_usage) {
            telemetryLog.push({ tool: "jobs", usage: jobs._telemetry.tool_usage });
        }
        renderTelemetry();
        setStatus("Recommending moat layers…", "loading");

        const moat = await postJSON("/moat", {
            concept: moatConcept,
            triggers: moatTriggers,
        });
        renderMoat(moat.layers || []);
        if (moat._telemetry?.tool_usage) {
            telemetryLog.push({ tool: "moat", usage: moat._telemetry.tool_usage });
        }
        renderTelemetry();
        setStatus("Scoring the concept…", "loading");

        const judge = await postJSON("/judge", {
            summary: judgeSummary,
        });
        renderScorecard(judge.scorecard || null);
        if (judge._telemetry?.tool_usage) {
            telemetryLog.push({ tool: "judge", usage: judge._telemetry.tool_usage });
        }
        renderTelemetry();

        setStatus("Analysis complete.", "success");
    } catch (err) {
        console.error(err);
        setStatus(err.message || "Analysis failed.", "error");
    } finally {
        submitButton.disabled = false;
    }
}

form.addEventListener("submit", runAnalysis);
resetButton.addEventListener("click", resetResults);

// Initialize with placeholders hidden
resetResults({ announce: false });

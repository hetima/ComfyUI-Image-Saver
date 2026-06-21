import { app } from '../../scripts/app.js'

// Picker UX for the Workflow Metadata Resolver node.
//
// Instead of hand-typing `field: #6.text` bindings, right-click any node and
// "Send to Metadata Resolver" to capture one of its fields, or use the node's
// "Auto-fill from sampler" button to bulk-populate the common generation params.
// The resolver reads these pointers from the live PROMPT at save time.

const RESOLVER_NODE = "Workflow Metadata Resolver (Image Saver)";
const BINDINGS_WIDGET = "bindings";

// Suggested field names for common inputs, so captured bindings read cleanly.
const FIELD_ALIASES = {
    ckpt_name: "model",
    unet_name: "model",
    model_name: "model",
    sampler_name: "sampler",
    scheduler: "scheduler",
    noise_seed: "seed",
};

// ---- binding-string helpers -------------------------------------------------

function separatorIndex(line) {
    const candidates = [":", "="].map(c => line.indexOf(c)).filter(i => i >= 0);
    return candidates.length ? Math.min(...candidates) : -1;
}

function bindingsWidget(resolver) {
    return resolver.widgets?.find(w => w.name === BINDINGS_WIDGET) ?? null;
}

/** Field names already bound on a resolver, as a Set. */
function boundFields(resolver) {
    const widget = bindingsWidget(resolver);
    const fields = new Set();
    if (!widget) return fields;
    for (const raw of String(widget.value ?? "").split("\n")) {
        const line = raw.trim();
        if (!line || line.startsWith("//") || (line.startsWith("#") && separatorIndex(line) < 0)) continue;
        const idx = separatorIndex(line);
        if (idx >= 0) fields.add(line.slice(0, idx).trim());
    }
    return fields;
}

/** Add or replace the binding for `field`, pointing at `pointer` (e.g. `#6.text`). */
function upsertBinding(resolver, field, pointer) {
    const widget = bindingsWidget(resolver);
    if (!widget) return;

    const lines = String(widget.value ?? "").split("\n");
    const kept = lines.filter(raw => {
        const line = raw.trim();
        if (!line) return false;                                  // drop blanks; re-tidied below
        if (line.startsWith("//") || (line.startsWith("#") && separatorIndex(line) < 0)) return true;
        const idx = separatorIndex(line);
        return idx < 0 || line.slice(0, idx).trim() !== field;    // drop a prior binding for this field
    });

    kept.push(`${field}: ${pointer}`);
    widget.value = kept.join("\n");
    widget.callback?.(widget.value);
    resolver.graph?.setDirtyCanvas(true, true);
}

// ---- graph helpers ----------------------------------------------------------

function nodeClass(node) {
    return node?.type ?? node?.comfyClass ?? "";
}

/** Union of a node's widget names and input names. */
function fieldNames(node) {
    const names = new Set();
    for (const w of node?.widgets ?? []) if (w?.name) names.add(w.name);
    for (const i of node?.inputs ?? []) if (i?.name) names.add(i.name);
    return names;
}

/** The node feeding `inputName` of `node`, or null if unlinked. */
function inputSource(graph, node, inputName) {
    const input = (node?.inputs ?? []).find(i => i?.name === inputName);
    if (!input || input.link == null) return null;
    const link = graph.links?.[input.link];
    if (!link) return null;
    return graph.getNodeById(link.origin_id) ?? null;
}

function findSampler(graph) {
    return (graph?._nodes ?? []).find(n => /KSampler|SamplerCustom/.test(nodeClass(n))) ?? null;
}

/** Walk an input wire (through same-named passthroughs) until a node owns one of `targets`. */
function traceToWidget(graph, node, inputName, targets, depth = 0) {
    if (depth > 16) return null;
    const src = inputSource(graph, node, inputName);
    if (!src) return null;
    const names = fieldNames(src);
    for (const t of targets) if (names.has(t)) return { node: src, widget: t };
    if (names.has(inputName)) return traceToWidget(graph, src, inputName, targets, depth + 1);
    return null;
}

/** Walk a latent wire until a node carries both width and height. */
function traceToSize(graph, node, inputName, depth = 0) {
    if (depth > 16) return null;
    const src = inputSource(graph, node, inputName);
    if (!src) return null;
    const names = fieldNames(src);
    if (names.has("width") && names.has("height")) return src;
    for (const passthrough of ["latent_image", "samples", "latent"]) {
        if (names.has(passthrough)) {
            const found = traceToSize(graph, src, passthrough, depth + 1);
            if (found) return found;
        }
    }
    return null;
}

/** Trace a sampler-centred graph into [field, pointer] bindings (mirrors the gallery parser). */
function autoFillBindings(graph) {
    const sampler = findSampler(graph);
    if (!sampler) return [];

    const sid = sampler.id;
    const names = fieldNames(sampler);
    const out = [];

    for (const [field, input] of [["steps", "steps"], ["cfg", "cfg"], ["sampler", "sampler_name"], ["scheduler", "scheduler"], ["denoise", "denoise"]]) {
        if (names.has(input)) out.push([field, `#${sid}.${input}`]);
    }
    if (names.has("seed")) out.push(["seed", `#${sid}.seed`]);
    else if (names.has("noise_seed")) out.push(["seed", `#${sid}.noise_seed`]);

    // Prompts: point at the sampler's conditioning inputs — the backend follows the link to the text node.
    if (names.has("positive")) out.push(["positive", `#${sid}.positive`]);
    if (names.has("negative")) out.push(["negative", `#${sid}.negative`]);

    const loader = traceToWidget(graph, sampler, "model", ["ckpt_name", "unet_name"]);
    if (loader) out.push(["model", `#${loader.node.id}.${loader.widget}`]);

    const latent = traceToSize(graph, sampler, "latent_image");
    if (latent) {
        out.push(["width", `#${latent.id}.width`]);
        out.push(["height", `#${latent.id}.height`]);
    }
    return out;
}

// ---- capture flow -----------------------------------------------------------

/** Capturable fields of a node: its widgets, plus any linked inputs (resolvable via wire-follow). */
function collectCaptures(node) {
    const seen = new Set();
    const captures = [];
    for (const w of node.widgets ?? []) {
        if (!w?.name || w.type === "button" || seen.has(w.name)) continue;
        seen.add(w.name);
        captures.push({ label: w.name, inputName: w.name });
    }
    for (const input of node.inputs ?? []) {
        if (!input?.name || input.link == null || seen.has(input.name)) continue;
        seen.add(input.name);
        captures.push({ label: `→ ${input.name}`, inputName: input.name });
    }
    return captures;
}

function suggestFieldName(inputName, resolver) {
    if (inputName === "text") {
        const bound = boundFields(resolver);
        if (!bound.has("positive")) return "positive";
        if (!bound.has("negative")) return "negative";
        return "prompt";
    }
    return FIELD_ALIASES[inputName] ?? inputName;
}

function bindToResolver(sourceNode, capture, resolver) {
    const suggested = suggestFieldName(capture.inputName, resolver);
    let field = suggested;
    if (typeof window !== "undefined" && typeof window.prompt === "function") {
        const answer = window.prompt(`Metadata field name for #${sourceNode.id}.${capture.inputName}:`, suggested);
        if (answer === null) return;            // cancelled
        field = answer.trim();
    }
    if (!field) return;
    upsertBinding(resolver, field, `#${sourceNode.id}.${capture.inputName}`);
}

function sendCapture(sourceNode, capture, event) {
    const graph = sourceNode.graph;
    const resolvers = (graph?._nodes ?? []).filter(n => nodeClass(n) === RESOLVER_NODE);

    if (resolvers.length === 0) {
        const resolver = LiteGraph.createNode(RESOLVER_NODE);
        if (!resolver) return;
        graph.add(resolver);
        resolver.pos = [sourceNode.pos[0] + (sourceNode.size?.[0] ?? 200) + 40, sourceNode.pos[1]];
        bindToResolver(sourceNode, capture, resolver);
    } else if (resolvers.length === 1) {
        bindToResolver(sourceNode, capture, resolvers[0]);
    } else {
        // Multiple resolvers — let the user choose which to bind into.
        new LiteGraph.ContextMenu(
            resolvers.map(r => ({
                content: r.title || `${RESOLVER_NODE} #${r.id}`,
                callback: () => bindToResolver(sourceNode, capture, r),
            })),
            { event, title: "Choose Metadata Resolver" }
        );
    }
}

// ---- extension registration -------------------------------------------------

app.registerExtension({
    name: "ComfyUI-Image-Saver.MetadataResolverPicker",

    beforeRegisterNodeDef(nodeType, nodeData) {
        // 1) "Send to Metadata Resolver" on every node's right-click menu.
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            getExtraMenuOptions?.apply(this, arguments);

            if (nodeClass(this) === RESOLVER_NODE) return;
            const captures = collectCaptures(this);
            if (!captures.length) return;

            const node = this;
            options.push({
                content: "Send to Metadata Resolver",
                has_submenu: true,
                submenu: {
                    options: captures.map(capture => ({
                        content: capture.label,
                        callback: (value, opts, event) => sendCapture(node, capture, event),
                    })),
                },
            });
        };

        // 2) "Auto-fill from sampler" button on the resolver node itself.
        if (nodeData?.name === RESOLVER_NODE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.addWidget("button", "Auto-fill from sampler", null, () => {
                    const bindings = autoFillBindings(this.graph);
                    if (!bindings.length) {
                        console.warn("ComfyUI-Image-Saver: Auto-fill found no sampler to trace.");
                        return;
                    }
                    for (const [field, pointer] of bindings) upsertBinding(this, field, pointer);
                });
            };
        }
    },
});

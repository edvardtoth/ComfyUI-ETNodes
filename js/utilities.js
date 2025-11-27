import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "ETNodes.utils.TextPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ETNodes-Text-Preview") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Add a text widget for preview
                // Use ComfyWidgets to ensure correct multiline behavior
                const widgetFactory = ComfyWidgets["STRING"];
                if (widgetFactory) {
                    const w = widgetFactory(this, "preview_text", ["STRING", { multiline: true }], app).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = this.properties.preview_text || "";
                } else {
                    // Fallback if ComfyWidgets is not available (older versions?)
                    const w = this.addWidget("text", "preview_text", this.properties.preview_text || "", (value) => { }, { multiline: true });
                    if (w.inputEl) {
                        w.inputEl.readOnly = true;
                        w.inputEl.style.opacity = 0.6;
                    }
                }
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.text) {
                    const previewWidget = this.widgets.find((w) => w.name === "preview_text");
                    if (previewWidget) {
                        previewWidget.value = message.text[0];
                        this.properties.preview_text = previewWidget.value;

                        // Auto-resize the node to fit the new text content
                        requestAnimationFrame(() => {
                            const sz = this.computeSize();
                            if (sz[0] < this.size[0]) {
                                sz[0] = this.size[0];
                            }
                            if (sz[1] < this.size[1]) {
                                sz[1] = this.size[1];
                            }
                            this.onResize?.(sz);
                            app.graph.setDirtyCanvas(true, false);
                        });
                    }
                }
            };
        }
    },
});

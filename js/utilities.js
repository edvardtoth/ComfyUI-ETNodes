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

app.registerExtension({
    name: "ETNodes.utils.NodeColors",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ETNodes-Gemini-API-Text" || nodeData.name === "ETNodes-Gemini-API-Image" || nodeData.name === "ETNodes-Gemini-API-Video") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.color = "#ae1016";
                this.bgcolor = "#1c1112";
                this.properties = this.properties || {};
                this.properties.cached = false;
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                const isCached = message?.cached?.[0] || false;
                this.properties.cached = isCached;
                
                // Get clean title
                let baseTitle = this.title || nodeData.displayName || nodeData.name;
                if (baseTitle.endsWith(" [CACHING ACTIVE]")) {
                    baseTitle = baseTitle.substring(0, baseTitle.length - 17);
                }
                
                if (isCached) {
                    this.title = baseTitle + " [CACHING ACTIVE]";
                } else {
                    this.title = baseTitle;
                }
                
                console.log("ETNodes Gemini Caching Status:", isCached);
                this.setDirtyCanvas?.(true, true);
            };

            // Clean up the [CACHING ACTIVE] suffix when saving the workflow JSON
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function (o) {
                onSerialize?.apply(this, arguments);
                if (o.title && o.title.endsWith(" [CACHING ACTIVE]")) {
                    o.title = o.title.substring(0, o.title.length - 17);
                }
            };
        }
    },
});

app.registerExtension({
    name: "ETNodes.GeminiImage.ModelControls",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ETNodes-Gemini-API-Image") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const modelWidget = this.widgets.find(w => w.name === "model");
                const ratioWidget = this.widgets.find(w => w.name === "aspect_ratio");
                const resolutionWidget = this.widgets.find(w => w.name === "resolution");
                
                if (modelWidget && ratioWidget) {
                    const updateModelControls = () => {
                        const modelValue = modelWidget.value || "";
                        
                        // 1. Aspect Ratios
                        const isFlash = modelValue.includes("flash");
                        const allRatios = ["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1"];
                        const proRatios = ["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "9:16", "16:9", "21:9"];
                        
                        const newRatios = isFlash ? allRatios : proRatios;
                        ratioWidget.options.values = newRatios;
                        
                        if (!newRatios.includes(ratioWidget.value)) {
                            ratioWidget.value = "auto";
                        }
                        
                        // 2. Resolutions
                        if (resolutionWidget) {
                            const isLite = modelValue.includes("lite");
                            const newResolutions = isLite ? ["1K"] : ["1K", "2K", "4K"];
                            resolutionWidget.options.values = newResolutions;
                            
                            if (!newResolutions.includes(resolutionWidget.value)) {
                                resolutionWidget.value = "1K";
                            }
                        }
                    };
                    
                    const originalCallback = modelWidget.callback;
                    modelWidget.callback = function (value) {
                        const res = originalCallback ? originalCallback.apply(this, arguments) : undefined;
                        updateModelControls();
                        return res;
                    };
                    
                    setTimeout(updateModelControls, 1);
                }
                
                return r;
            };
        }
    }
});

app.registerExtension({
    name: "ETNodes.GeminiVideo.ModelControls",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ETNodes-Gemini-API-Video") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const modelWidget = this.widgets.find(w => w.name === "model");
                const ratioWidget = this.widgets.find(w => w.name === "aspect_ratio");
                const resolutionWidget = this.widgets.find(w => w.name === "resolution");
                const durationWidget = this.widgets.find(w => w.name === "duration_seconds");
                
                if (modelWidget) {
                    const updateModelControls = () => {
                        const modelValue = modelWidget.value || "";
                        
                        // 1. Resolution constraints
                        if (resolutionWidget) {
                            const isLiteOrFastVeo = modelValue.includes("lite") || modelValue.includes("fast");
                            const allowedResolutions = isLiteOrFastVeo ? ["720p", "1080p"] : ["720p", "1080p", "4K"];
                            
                            resolutionWidget.options.values = allowedResolutions;
                            if (!allowedResolutions.includes(resolutionWidget.value)) {
                                resolutionWidget.value = "720p";
                            }
                        }
                        
                        // 2. Duration constraints
                        if (durationWidget) {
                            const isVeo = modelValue.includes("veo");
                            const maxDuration = isVeo ? 8 : 10;
                            
                            durationWidget.options.max = maxDuration;
                            if (durationWidget.value > maxDuration) {
                                durationWidget.value = maxDuration;
                            }
                        }

                        // 3. Aspect Ratio constraints
                        if (ratioWidget) {
                            const isOmni = modelValue.includes("omni");
                            const allowedRatios = isOmni ? ["16:9", "9:16"] : ["16:9", "9:16", "1:1"];
                            
                            ratioWidget.options.values = allowedRatios;
                            if (!allowedRatios.includes(ratioWidget.value)) {
                                ratioWidget.value = "16:9";
                            }
                        }
                    };
                    
                    const originalCallback = modelWidget.callback;
                    modelWidget.callback = function (value) {
                        const res = originalCallback ? originalCallback.apply(this, arguments) : undefined;
                        updateModelControls();
                        return res;
                    };
                    
                    setTimeout(updateModelControls, 1);
                }
                
                return r;
            };
        }
    }
});




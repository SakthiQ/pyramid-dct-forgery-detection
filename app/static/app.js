document.addEventListener("DOMContentLoaded", () => {
    // Structural Elements mapping
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadSection = document.getElementById("upload-section");
    const loadingSection = document.getElementById("loading-section");
    const resultsSection = document.getElementById("results-section");
    const resetBtn = document.getElementById("reset-btn");

    // Value Node Bindings
    const badge = document.getElementById("classification-badge");
    const confFill = document.getElementById("confidence-fill");
    const confVal = document.getElementById("confidence-val");
    const pVal = document.getElementById("pvalue-val");
    const origImg = document.getElementById("orig-img");
    const heatmapImg = document.getElementById("heatmap-img");
    const maskImg = document.getElementById("mask-img");

    let originalObjectUrl = null;

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    resetBtn.addEventListener("click", () => {
        // Destroy existing visual caches
        resultsSection.classList.add("hidden");
        uploadSection.classList.remove("hidden");
        if(originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
        fileInput.value = "";
    });

    async function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Security Warning: Upload schema unsupported. Acceptable bounds: [image/*]");
            return;
        }

        // Cache original mapping instantly to native component
        originalObjectUrl = URL.createObjectURL(file);
        origImg.src = originalObjectUrl;

        // Animate state switch bounds
        uploadSection.classList.add("hidden");
        loadingSection.classList.remove("hidden");

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Push binary stream dynamically into our main Python Pipeline Route Layer
            const response = await fetch("/api/v1/analyze", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errJson = await response.json();
                throw new Error(errJson.detail || `Pipeline Failed: ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            alert("Orchestrator Failure: " + error.message);
            loadingSection.classList.add("hidden");
            uploadSection.classList.remove("hidden");
        }
    }

    function renderResults(data) {
        loadingSection.classList.add("hidden");
        resultsSection.classList.remove("hidden");

        // Classification Parsing
        const isFake = data.classification === "FAKE";
        
        badge.textContent = data.classification;
        badge.className = "badge " + (isFake ? "fake" : "authentic");

        // Format raw floats safely into percentages
        const confPercent = (data.confidence * 100).toFixed(1);
        confVal.textContent = `${confPercent}%`;
        
        // Dynamic gradient structural shifts based off FAKE bounds
        confFill.style.background = isFake ? "linear-gradient(to right, #fb7185, #ef4444)" : "linear-gradient(to right, #34d399, #10b981)";
        
        // Dispatch micro-animation fill update natively
        setTimeout(() => {
            confFill.style.width = `${confPercent}%`;
        }, 50);

        // Exponential smoothing for complex stat matrices
        pVal.textContent = data.p_value.toExponential(3);

        // Native injection mappings tracing the root /outputs FastApi static path.
        // Cache bust enforces dynamic structural reload between iterations
        const cacheBust = new Date().getTime();
        heatmapImg.src = `/${data.heatmap_path}?cb=${cacheBust}`;
        maskImg.src = `/${data.mask_path}?cb=${cacheBust}`;
    }
});

// WebGPU point primitives have fixed size. Request WebGL through Babylon's
// public viewer API so the point-size control works in Chromium and Arc too.
const root = (window.gradio_config.root || location.origin).replace(/\/$/, "");
// Fetch dependencies together rather than discovering them over serial requests.
const preloaded = new Set(Array.from(
    document.querySelectorAll('link[rel="modulepreload"]'), link => link.href,
));
for (const name of __VIEWER_DEPENDENCIES__) {
    const url = `${root}/assets/${name}`;
    if (preloaded.has(url)) continue;
    const link = document.createElement("link");
    link.rel = "modulepreload";
    link.href = url;
    document.head.appendChild(link);
    preloaded.add(url);
}
const library = import(`${root}/assets/` + __VIEWER_LIBRARY__);
const canvas = element.querySelector("canvas");
const notice = element.querySelector(".notice");
const download = element.querySelector(".download");
let details;
let pointSize = Number(document.getElementById("click3r-point-size")?.dataset.size || __DEFAULT_POINT_SIZE__);
let version = 0;
let loadedURL;
let initialCamera;
let disposed = false;

function applyPointSize() {
    if (!details) return;
    for (const material of details.scene.materials) {
        if (material.pointsCloud) material.pointSize = pointSize;
    }
    // Schedule a complete frame through the viewer, including begin/end-frame.
    details.markSceneMutated();
}

function applyCamera(position) {
    if (!details || !position) return;
    const camera = details.camera;
    if (position[0] != null) camera.alpha = position[0] * Math.PI / 180;
    if (position[1] != null) camera.beta = position[1] * Math.PI / 180;
    if (position[2] != null) camera.radius = position[2];
    // Match Gradio's orbit controls.
    camera.lowerRadiusLimit = 0.1;
    camera.wheelPrecision = 250 / camera.radius;
    camera.panningSensibility = 10000 / camera.radius;
    camera.storeState();
    details.markSceneMutated();
}

async function cameraPosition(blob) {
    const header = new DataView(await blob.slice(0, 20).arrayBuffer());
    if (header.getUint32(0, true) !== 0x46546c67) return null;
    const length = header.getUint32(12, true);
    const gltf = JSON.parse(await blob.slice(20, 20 + length).text());
    return gltf.scenes?.[gltf.scene || 0]?.extras?.camera_position;
}

// Initialize while the user selects images; avoid first-use renderer startup.
const ready = library.then(async ({CreateViewerForCanvas}) => {
    const viewer = await CreateViewerForCanvas(canvas, {
        engine: "WebGL",
        clearColor: [1, 1, 1, 1],
        useRightHandedSystem: true,
        cameraAutoOrbit: {enabled: false},
        onInitialized: value => { details = value; },
    });
    return viewer;
});

async function showModel() {
    const current = ++version;
    const file = props.value;
    if (!file) {
        canvas.style.visibility = "hidden";
        notice.hidden = false;
        notice.textContent = "Run reconstruction to view the scene.";
        download.removeAttribute("href");
        return;
    }
    const url = file.url;
    try {
        const viewer = await ready;
        if (disposed || current !== version) return;
        if (url !== loadedURL) {
            notice.hidden = false;
            notice.textContent = "Loading reconstruction…";
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Model download failed: ${response.status}`);
            const blob = await response.blob();
            if (disposed || current !== version) return;
            initialCamera = await cameraPosition(blob);
            await viewer.loadModel(new File([blob], file.orig_name || "reconstruction.glb"));
            if (disposed || current !== version) return;
            loadedURL = url;
        }
        canvas.style.visibility = "visible";
        notice.hidden = true;
        download.href = url;
        download.download = file.orig_name || "reconstruction.glb";
        applyPointSize();
        applyCamera(initialCamera);
    } catch (error) {
        notice.hidden = false;
        notice.textContent = "The 3D view could not load. Please try reconstructing again.";
        console.error("Reconstruction viewer:", error);
    }
}

function updatePointSize(event) {
    pointSize = Number(event.detail);
    applyPointSize();
}
document.addEventListener("click3r:point-size", updatePointSize);
element.querySelector(".reset").addEventListener("click", () => {
    details?.camera.restoreState();
    details?.markSceneMutated();
});
element.querySelector(".fullscreen").addEventListener("click", () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void element.querySelector(".viewer").requestFullscreen();
});
watch("value", () => { void showModel(); });
void showModel();
const cleanup = setInterval(() => {
    if (element.isConnected) return;
    disposed = true;
    clearInterval(cleanup);
    document.removeEventListener("click3r:point-size", updatePointSize);
    void ready.then(viewer => { viewer.dispose(); });
}, 1000);

import {
    HandLandmarker, FilesetResolver
}
    from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");

// math functions
function distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function getMeasurements(points) {
    const middleFingerLength = distance(points[9], points[12]);
    const palmWidth = distance(points[5], points[17]);
    const handHeight = distance(points[0], points[12]);
    const indexLength = distance(points[5], points[8]);
    const ringLength = distance(points[13], points[16]);
    const pinkyLength = distance(points[17], points[20]);
    const thumbLength = distance(points[1], points[4]);

    const tipSpread = distance(points[8], points[12]); // index to middle fingertip distance
    const tipToMiddle = tipSpread / middleFingerLength; // how spread the fingers

    return {
        ratio1: middleFingerLength / palmWidth, // finger-palm ratio
        ratio2: palmWidth / handHeight, // palm ratio
        ratio3: indexLength / palmWidth, // index-palm
        ratio4: ringLength / palmWidth, // ring-palm
        ratio5: pinkyLength / palmWidth, // pinky-palm
        ratio6: thumbLength / palmWidth, // thumb-palm
        ratio7: tipToMiddle, // fingetip chunkiness
    };
}

// classifying based off of a PMC report but i am not sure if i want to change it later
function classifyAge(r) {
    const r1Score = (r.ratio1 - 1.55) / (1.85 - 1.55);
    const r2Score = (0.33 - r.ratio2) / (0.33 - 0.25);
    const r3Score = (r.ratio3 - 1.40) / (1.70 - 1.40);
    const r4Score = (r.ratio4 - 1.38) / (1.68 - 1.38);
    const r5Score = (r.ratio5 - 0.95) / (1.20 - 0.95);
    const r6Score = (r.ratio6 - 0.80) / (1.05 - 0.80);
    const r7Score = (0.45 - r.ratio7) / (0.45 - 0.30);
     
    const confidence = (r1Score * 0.30) + (r2Score * 0.20) + (r3Score * 0.15) + (r4Score * 0.15) + (r5Score * 0.10) + (r6Score * 0.05) + (r7Score * 0.05);
    const clamped = Math.max(0, Math.min(1, confidence));

    return{
        result: clamped >= 0.5 ? "likely 13+" : "likely under 13",
        confidence: Math.round(clamped * 100),
        raw: clamped
    };
}

// important: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js
// lines that connect the 21 landmarks
// https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // index finger
    [0, 9], [9, 10], [10, 11], [11, 12], // middle finger
    [0, 13], [13, 14], [14, 15], [15, 16], // ring finger
    [0, 17], [17, 18], [18, 19], [19, 20], // pinky
    [5, 9], [9, 13], [13, 17] // knuckle row across palm
];

let landmarker;

// load the mediapipe model
try {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );
    landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    status.textContent = "model done. click start camera.";
    document.getElementById("start-btn").style.display = "inline";

} catch (e) {
    status.textContent = "Error: " + e.message;
    console.error(e);
}

// turn on webcam
window.startCamera = async function () {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });
    video.srcObject = stream;
    video.setAttribute("playsinline", true); // force load on some browsers
    video.play();
    video.onloadedmetadata = () => {
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        status.textContent = "show your hand, palm face towards screen";
        document.getElementById("placeholder").style.display = "none";
        document.getElementById("capture-btn").disabled = false;
        document.getElementById("start-btn").disabled = true;
    };
}

// run detection 
let lastTime = -1;
window.captureAndAnalyze = function () {
    const results = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks.length === 0) {
        status.textContent = "no hand detected";
        return;
    }

    const points = results.landmarks[0];

    // check mediapipe confidence 
    const detectionConfidence = results.handednesses[0][0].score;
    document.getElementById("detection-score").textContent = Math.round(detectionConfidence * 100) + "%";
    
    if (detectionConfidence < 0.7) {
        status.textContent = "bad detection — try again with better lighting";
        return;
    }

    // draw skeleton
    ctx.strokeStyle = "rgba(37, 99, 235, 0.6)";
    ctx.lineWidth = 2;
    for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(points[a].x * canvas.width, points[a].y * canvas.height);
        ctx.lineTo(points[b].x * canvas.width, points[b].y * canvas.height);
        ctx.stroke();
    }

    // draw dots
    for (const point of points) {
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#2563eb";
        ctx.fill();
    }

    // measurements
    const m = getMeasurements(points);
    document.getElementById("ratio1").textContent = m.ratio1.toFixed(3);
    document.getElementById("ratio2").textContent = m.ratio2.toFixed(3);
    document.getElementById("ratio3").textContent = m.ratio3.toFixed(3);
    document.getElementById("ratio4").textContent = m.ratio4.toFixed(3);
    document.getElementById("ratio5").textContent = m.ratio5.toFixed(3);
    document.getElementById("ratio6").textContent = m.ratio6.toFixed(3);
    document.getElementById("ratio7").textContent = m.ratio7.toFixed(3);

    const verdict = classifyAge(m);
    document.getElementById("result").textContent = verdict.result;
    document.getElementById("confidence").textContent = verdict.confidence + "%";
    document.getElementById("bar").style.width = verdict.confidence + "%";
    status.textContent = "Done";
}

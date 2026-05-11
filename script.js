import {
    HandLandmarker, FilesetResolver
}
    from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const ratio1El = document.getElementById("ratio1");
const ratio2El = document.getElementById("ratio2");

// math functions
function distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}
function getMeasurements(points) {
    const middleFingerLength = distance(points[9], points[12]);
    const palmWidth = distance(points[5], points[17]);
    const handHeight = distance(points[0], points[12]);

    return {
        ratio1: middleFingerLength / palmWidth, // finger-palm ratio
        ratio2: palmWidth / handHeight, // palm ratio
    };
}

// classifying based off of a PMC report but i am not sure if i want to change it later
function classifyAge(ratio1, ratio2) {
    const r1Score = (ratio1 - 1.55) / (1.85 - 1.55) // 0 = child, 1 = adult
    const r2Score = (0.33 - ratio2) / (0.33 - 0.25) // goes down as you get older 
    const confidence = (r1Score * 0.65) + (r2Score * 0.35); // ratio 1 is more reliable
    const clamped = Math.max(0, Math.min(1, confidence)); // clamp to [0, 1]
    return{
        result: clamped >= 0.5 ? "likely 13+" : "likely under 13",
        confidence: Math.round(clamped * 100)
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
    video.onloadedmetadata = () => {
        status.textContent = "show your hand, palm face towards screen";
        document.getElementById("capture-btn").style.display = "inline"; // show capture button
    };
}

// run detection 
let lastTime = -1;
window.captureAndAnalyze = function () {
    const results = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, 640, 480);

    if (results.landmarks.length === 0) {
        status.textContent = "no hand detected";
        return;
    }

    const points = results.landmarks[0];

    // draw skeleton
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(points[a].x * 640, points[a].y * 480);
        ctx.lineTo(points[b].x * 640, points[b].y * 480);
        ctx.stroke();
    }
    for (const point of points) {
        ctx.beginPath();
        ctx.arc(point.x * 640, point.y * 480, 4, 0, Math.PI * 2);
        ctx.fillStyle = "red";
        ctx.fill();
    }

    // media pipe confidence
    const detectionConfidence = results.handednesses[0][0].score;
    if (detectionConfidence < 0.7) {
        status.textContent = "bad detection — try again with better lighting";
        return;
    }

    // show measurements
    const m = getMeasurements(points);
    ratio1El.textContent = m.ratio1.toFixed(3);
    ratio2El.textContent = m.ratio2.toFixed(3);
    status.textContent = "done!";
    const verdict = classifyAge(m.ratio1, m.ratio2);
    document.getElementById("result").textContent = verdict.result;
    document.getElementById("confidence").textContent = verdict.confidence + "%";   
}

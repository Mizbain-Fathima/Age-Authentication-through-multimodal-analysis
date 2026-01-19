/**
 * Age Authentication System - Frontend Application
 * Handles webcam capture, audio recording, and API interactions
 */

// =====================
// Configuration
// =====================

const API_BASE = window.location.origin + '/api';
const MAX_RECORDING_TIME = 15; // seconds
const FRAME_CAPTURE_INTERVAL = 100; // ms

// =====================
// State Management
// =====================

const state = {
    currentStep: 1,
    captchaId: null,
    captchaText: null,
    captchaExpiry: null,
    isRecording: false,
    mediaStream: null,
    audioContext: null,
    analyser: null,
    recordedFrames: [],
    audioChunks: [],
    mediaRecorder: null,
    recordingStartTime: null,
    recordingTimer: null,
    captchaTimer: null
};

// =====================
// DOM Elements
// =====================

const elements = {
    // Steps
    steps: document.querySelectorAll('.step'),
    stepConnectors: document.querySelectorAll('.step-connector'),
    stepSections: document.querySelectorAll('.step-section'),
    
    // Step 1
    btnGetCaptcha: document.getElementById('btn-get-captcha'),
    
    // Step 2
    captchaText: document.getElementById('captcha-text'),
    captchaTimer: document.getElementById('captcha-timer'),
    videoPreview: document.getElementById('video-preview'),
    videoOverlay: document.getElementById('video-overlay'),
    videoStatus: document.getElementById('video-status'),
    audioCanvas: document.getElementById('audio-canvas'),
    audioLevelBar: document.getElementById('audio-level-bar'),
    btnRecord: document.getElementById('btn-record'),
    recordingTimer: document.getElementById('recording-timer'),
    
    // Step 3
    procFace: document.getElementById('proc-face'),
    procVoice: document.getElementById('proc-voice'),
    procLiveness: document.getElementById('proc-liveness'),
    procAge: document.getElementById('proc-age'),
    
    // Step 4
    resultIcon: document.getElementById('result-icon'),
    resultTitle: document.getElementById('result-title'),
    resultBadge: document.getElementById('result-badge'),
    ageNumber: document.getElementById('age-number'),
    ageGroup: document.getElementById('age-group'),
    faceAge: document.getElementById('face-age'),
    voiceAge: document.getElementById('voice-age'),
    faceLiveness: document.getElementById('face-liveness'),
    faceLivenessBar: document.getElementById('face-liveness-bar'),
    voiceLiveness: document.getElementById('voice-liveness'),
    voiceLivenessBar: document.getElementById('voice-liveness-bar'),
    lipSync: document.getElementById('lip-sync'),
    lipSyncBar: document.getElementById('lip-sync-bar'),
    confidenceFill: document.getElementById('confidence-fill'),
    confidenceText: document.getElementById('confidence-text'),
    adultStatus: document.getElementById('adult-status'),
    btnRestart: document.getElementById('btn-restart')
};

// =====================
// Initialization
// =====================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Event listeners
    elements.btnGetCaptcha.addEventListener('click', getCaptcha);
    elements.btnRecord.addEventListener('click', toggleRecording);
    elements.btnRestart.addEventListener('click', restart);
    
    console.log('Age Authentication System initialized');
}

// =====================
// Step Navigation
// =====================

function goToStep(stepNumber) {
    state.currentStep = stepNumber;
    
    // Update step indicators
    elements.steps.forEach((step, index) => {
        const stepNum = index + 1;
        step.classList.remove('active', 'completed');
        
        if (stepNum < stepNumber) {
            step.classList.add('completed');
        } else if (stepNum === stepNumber) {
            step.classList.add('active');
        }
    });
    
    // Update connectors
    elements.stepConnectors.forEach((connector, index) => {
        connector.classList.toggle('completed', index < stepNumber - 1);
    });
    
    // Show/hide sections
    elements.stepSections.forEach(section => {
        section.classList.remove('active');
    });
    
    const sectionId = ['step-captcha', 'step-recording', 'step-processing', 'step-results'][stepNumber - 1];
    document.getElementById(sectionId).classList.add('active');
}

// =====================
// Step 1: Get Captcha
// =====================

async function getCaptcha() {
    try {
        elements.btnGetCaptcha.disabled = true;
        elements.btnGetCaptcha.innerHTML = '<span class="btn-icon">⏳</span> Loading...';
        
        const response = await fetch(`${API_BASE}/captcha?complexity=medium`);
        
        if (!response.ok) {
            throw new Error('Failed to get captcha');
        }
        
        const data = await response.json();
        
        state.captchaId = data.captcha_id;
        state.captchaText = data.sentence;
        state.captchaExpiry = Date.now() + (data.expires_in * 1000);
        
        // Display captcha
        elements.captchaText.textContent = data.sentence;
        
        // Start expiry timer
        startCaptchaTimer();
        
        // Initialize media
        await initializeMedia();
        
        // Move to step 2
        goToStep(2);
        
    } catch (error) {
        console.error('Error getting captcha:', error);
        alert('Failed to get verification phrase. Please try again.');
    } finally {
        elements.btnGetCaptcha.disabled = false;
        elements.btnGetCaptcha.innerHTML = '<span class="btn-icon">⚡</span> Generate Verification Phrase';
    }
}

function startCaptchaTimer() {
    if (state.captchaTimer) {
        clearInterval(state.captchaTimer);
    }
    
    state.captchaTimer = setInterval(() => {
        const remaining = Math.max(0, state.captchaExpiry - Date.now());
        const minutes = Math.floor(remaining / 60000);
        const seconds = Math.floor((remaining % 60000) / 1000);
        
        elements.captchaTimer.textContent = 
            `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        if (remaining <= 0) {
            clearInterval(state.captchaTimer);
            alert('Verification phrase expired. Please get a new one.');
            restart();
        }
    }, 1000);
}

// =====================
// Step 2: Media Handling
// =====================

async function initializeMedia() {
    try {
        // Get user media
        state.mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: true
        });
        
        // Set up video preview
        elements.videoPreview.srcObject = state.mediaStream;
        
        // Set up audio visualization
        setupAudioVisualization();
        
        // Update status
        elements.videoStatus.querySelector('.status-text').textContent = 'Camera Ready';
        
    } catch (error) {
        console.error('Error accessing media:', error);
        alert('Unable to access camera/microphone. Please grant permissions and try again.');
        throw error;
    }
}

function setupAudioVisualization() {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 256;
    
    const source = state.audioContext.createMediaStreamSource(state.mediaStream);
    source.connect(state.analyser);
    
    visualizeAudio();
}

function visualizeAudio() {
    if (!state.analyser) return;
    
    const canvas = elements.audioCanvas;
    const ctx = canvas.getContext('2d');
    const bufferLength = state.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        requestAnimationFrame(draw);
        
        state.analyser.getByteFrequencyData(dataArray);
        
        // Clear canvas
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw frequency bars
        const barWidth = canvas.width / bufferLength * 2.5;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            
            // Gradient color
            const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
            gradient.addColorStop(0, '#00f5ff');
            gradient.addColorStop(1, '#ff00aa');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
            
            x += barWidth;
        }
        
        // Update level bar
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        const levelPercent = (average / 255) * 100;
        elements.audioLevelBar.style.width = `${levelPercent}%`;
    }
    
    draw();
}

function toggleRecording() {
    if (state.isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    state.isRecording = true;
    state.recordedFrames = [];
    state.audioChunks = [];
    state.recordingStartTime = Date.now();
    
    // Update UI
    elements.btnRecord.classList.add('recording');
    elements.btnRecord.querySelector('.record-text').textContent = 'Stop Recording';
    elements.recordingTimer.classList.add('active');
    
    // Start audio recording
    const audioTracks = state.mediaStream.getAudioTracks();
    const audioStream = new MediaStream(audioTracks);
    state.mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: 'audio/webm'
    });
    
    state.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            state.audioChunks.push(event.data);
        }
    };
    
    state.mediaRecorder.start(100);
    
    // Start video frame capture
    captureFrames();
    
    // Start recording timer
    state.recordingTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        
        elements.recordingTimer.textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        // Auto-stop at max time
        if (elapsed >= MAX_RECORDING_TIME) {
            stopRecording();
        }
    }, 1000);
}

function captureFrames() {
    if (!state.isRecording) return;
    
    const canvas = document.createElement('canvas');
    const video = elements.videoPreview;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Store frame as base64
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    state.recordedFrames.push(frameData);
    
    // Continue capturing
    setTimeout(captureFrames, FRAME_CAPTURE_INTERVAL);
}

function stopRecording() {
    state.isRecording = false;
    
    // Update UI
    elements.btnRecord.classList.remove('recording');
    elements.btnRecord.querySelector('.record-text').textContent = 'Start Recording';
    elements.recordingTimer.classList.remove('active');
    
    // Stop timers
    if (state.recordingTimer) {
        clearInterval(state.recordingTimer);
    }
    
    // Stop media recorder
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
    
    // Wait for final data and submit
    setTimeout(() => {
        submitVerification();
    }, 500);
}

// =====================
// Step 3: Process Verification
// =====================

async function submitVerification() {
    goToStep(3);
    
    // Reset processing indicators
    resetProcessingSteps();
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('captcha_id', state.captchaId);
        
        // Add video frames
        if (state.recordedFrames.length > 0) {
            // Send frames as JSON array
            formData.append('video_frames', JSON.stringify(state.recordedFrames));
        }
        
        // Add audio
        if (state.audioChunks.length > 0) {
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            formData.append('audio', audioBlob, 'recording.webm');
        }
        
        // Update processing steps
        await simulateProcessingStep('proc-face', 800);
        await simulateProcessingStep('proc-voice', 600);
        await simulateProcessingStep('proc-liveness', 700);
        
        // Submit to API
        const response = await fetch(`${API_BASE}/verify`, {
            method: 'POST',
            body: formData
        });
        
        await simulateProcessingStep('proc-age', 500);
        
        if (!response.ok) {
            throw new Error('Verification request failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Verification error:', error);
        displayError(error.message);
    } finally {
        cleanup();
    }
}

function resetProcessingSteps() {
    ['proc-face', 'proc-voice', 'proc-liveness', 'proc-age'].forEach(id => {
        const step = document.getElementById(id);
        step.classList.remove('completed');
        step.querySelector('.proc-icon').textContent = '⏳';
    });
}

async function simulateProcessingStep(stepId, delay) {
    return new Promise(resolve => {
        setTimeout(() => {
            const step = document.getElementById(stepId);
            step.classList.add('completed');
            step.querySelector('.proc-icon').textContent = '✓';
            resolve();
        }, delay);
    });
}

// =====================
// Step 4: Display Results
// =====================

function displayResults(result) {
    goToStep(4);
    
    const { success, liveness, age, overall_confidence } = result;
    
    // Main result
    if (success) {
        elements.resultIcon.textContent = '✅';
        elements.resultTitle.textContent = 'Verification Successful';
        elements.resultBadge.className = 'result-badge verified';
        elements.resultBadge.innerHTML = '<span class="badge-icon">✓</span><span class="badge-text">VERIFIED</span>';
    } else {
        elements.resultIcon.textContent = '❌';
        elements.resultTitle.textContent = 'Verification Failed';
        elements.resultBadge.className = 'result-badge failed';
        elements.resultBadge.innerHTML = '<span class="badge-icon">✗</span><span class="badge-text">FAILED</span>';
    }
    
    // Age display
    const estimatedAge = Math.round(age.estimated_age);
    elements.ageNumber.textContent = estimatedAge || '--';
    
    // Age group
    const groupBadge = elements.ageGroup.querySelector('.group-badge');
    groupBadge.className = `group-badge ${age.age_group}`;
    groupBadge.textContent = capitalizeFirst(age.age_group);
    
    // Detailed metrics
    elements.faceAge.textContent = age.face_age ? Math.round(age.face_age) : '--';
    elements.voiceAge.textContent = age.voice_age ? Math.round(age.voice_age) : '--';
    
    // Liveness scores
    const faceLiveScore = Math.round(liveness.face_liveness_score * 100);
    const voiceLiveScore = Math.round(liveness.voice_liveness_score * 100);
    const lipSyncScore = Math.round(liveness.lip_sync_score * 100);
    const confidenceScore = Math.round(overall_confidence * 100);
    
    animateProgress('face-liveness-bar', 'face-liveness', faceLiveScore);
    animateProgress('voice-liveness-bar', 'voice-liveness', voiceLiveScore);
    animateProgress('lip-sync-bar', 'lip-sync', lipSyncScore);
    
    // Overall confidence
    setTimeout(() => {
        elements.confidenceFill.style.width = `${confidenceScore}%`;
        elements.confidenceText.textContent = `${confidenceScore}%`;
    }, 500);
    
    // Adult status
    if (age.is_adult) {
        elements.adultStatus.className = 'adult-status verified';
        elements.adultStatus.innerHTML = `
            <span class="status-icon">✓</span>
            <span class="status-text">Age 18+ Verified (${Math.round(age.adult_confidence * 100)}% confidence)</span>
        `;
    } else {
        elements.adultStatus.className = 'adult-status failed';
        elements.adultStatus.innerHTML = `
            <span class="status-icon">✗</span>
            <span class="status-text">Under 18 Detected</span>
        `;
    }
}

function animateProgress(barId, textId, value) {
    const bar = document.getElementById(barId);
    const text = document.getElementById(textId);
    
    setTimeout(() => {
        bar.style.width = `${value}%`;
        text.textContent = `${value}%`;
    }, 300);
}

function displayError(message) {
    goToStep(4);
    
    elements.resultIcon.textContent = '⚠️';
    elements.resultTitle.textContent = 'Error Occurred';
    elements.resultBadge.className = 'result-badge failed';
    elements.resultBadge.innerHTML = `<span class="badge-icon">!</span><span class="badge-text">ERROR</span>`;
    
    elements.ageNumber.textContent = '--';
    
    // Show error message
    const groupBadge = elements.ageGroup.querySelector('.group-badge');
    groupBadge.className = 'group-badge';
    groupBadge.textContent = message;
}

// =====================
// Cleanup & Restart
// =====================

function cleanup() {
    // Stop media stream
    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(track => track.stop());
        state.mediaStream = null;
    }
    
    // Clear timers
    if (state.captchaTimer) {
        clearInterval(state.captchaTimer);
    }
    if (state.recordingTimer) {
        clearInterval(state.recordingTimer);
    }
    
    // Close audio context
    if (state.audioContext) {
        state.audioContext.close();
        state.audioContext = null;
    }
}

function restart() {
    cleanup();
    
    // Reset state
    state.currentStep = 1;
    state.captchaId = null;
    state.captchaText = null;
    state.captchaExpiry = null;
    state.isRecording = false;
    state.recordedFrames = [];
    state.audioChunks = [];
    
    // Reset UI
    elements.recordingTimer.textContent = '00:00';
    elements.btnRecord.classList.remove('recording');
    elements.btnRecord.querySelector('.record-text').textContent = 'Start Recording';
    
    goToStep(1);
}

// =====================
// Utilities
// =====================

function capitalizeFirst(str) {
    return str ? str.charAt(0).toUpperCase() + str.slice(1) : '';
}


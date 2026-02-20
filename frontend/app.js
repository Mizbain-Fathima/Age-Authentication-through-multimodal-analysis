/**
 * Multimodal Age Authentication - Frontend
 * Simplified version for /api/process endpoint
 */

// =====================
// Configuration
// =====================

const API_BASE = 'http://localhost:8000';
const MAX_RECORDING_TIME = 15; // seconds

// =====================
// State Management
// =====================

const state = {
    videoStream: null,
    audioStream: null,
    mediaStream: null, // Combined for preview
    videoRecorder: null,
    audioContext: null,
    audioSource: null,
    audioProcessor: null,
    audioSamples: [], // PCM Float32Array samples
    videoChunks: [],
    isRecording: false,
    recordingStartTime: null,
    recordingTimer: null,
    captchaId: null,
    captchaSentence: null,
    captchaExpiry: null,
    captchaTimer: null,
    uiState: 'idle', // idle, camera_ready, recording, processing, success, failed
    MIN_RECORDING_DURATION: 5 // seconds
};

// =====================
// DOM Elements
// =====================

const elements = {
    videoBlur: document.getElementById('videoBlur'),
    videoSharp: document.getElementById('videoSharp'),
    videoStatus: document.getElementById('videoStatus'),
    btnStart: document.getElementById('btnStart'),
    btnStop: document.getElementById('btnStop'),
    captchaSection: document.getElementById('captchaSection'),
    captchaText: document.getElementById('captchaText'),
    captchaTimer: document.getElementById('captchaTimer'),
    statusSection: document.getElementById('statusSection'),
    statusIcon: document.getElementById('statusIcon'),
    statusText: document.getElementById('statusText'),
    resultsSection: document.getElementById('resultsSection'),
    resultCard: document.getElementById('resultCard'),
    resultBadge: document.getElementById('resultBadge'),
    badgeIcon: document.getElementById('badgeIcon'),
    badgeText: document.getElementById('badgeText'),
    ageNumber: document.getElementById('ageNumber'),
    ageNote: document.getElementById('ageNote'),
    adultStatus: document.getElementById('adultStatus'),
    adultIcon: document.getElementById('adultIcon'),
    adultText: document.getElementById('adultText'),
    confidenceBar: document.getElementById('confidenceBar'),
    confidenceValue: document.getElementById('confidenceValue'),
    faceLivenessBar: document.getElementById('faceLivenessBar'),
    faceLivenessValue: document.getElementById('faceLivenessValue'),
    voiceLivenessBar: document.getElementById('voiceLivenessBar'),
    voiceLivenessValue: document.getElementById('voiceLivenessValue'),
        lipSyncBar: document.getElementById('lipSyncBar'),
        lipSyncValue: document.getElementById('lipSyncValue'),
        eyeBlinkBar: document.getElementById('eyeBlinkBar'),
        eyeBlinkValue: document.getElementById('eyeBlinkValue'),
    errorMessage: document.getElementById('errorMessage'),
    btnRestart: document.getElementById('btnRestart')
};

// =====================
// Initialization
// =====================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Set up event listeners
    elements.btnStart.addEventListener('click', startVerification);
    elements.btnStop.addEventListener('click', stopAndVerify);
    elements.btnRestart.addEventListener('click', restart);
    
    // Initialize camera and microphone
    try {
        await initializeMedia();
    } catch (error) {
        console.error('Failed to initialize media:', error);
        showError('Unable to access camera/microphone. Please grant permissions and refresh the page.');
    }
}

// =====================
// Media Initialization
// =====================

async function initializeMedia() {
    // 1. Validate video elements exist BEFORE any operations
    if (!elements.videoBlur || !elements.videoSharp) {
        const error = new Error('Video elements not found in DOM');
        console.error('Video elements not found:', {
            videoBlur: !!elements.videoBlur,
            videoSharp: !!elements.videoSharp
        });
        throw error;
    }
    
    try {
        // 3. Call getUserMedia ONLY once with simplified constraints
        state.mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' },
            audio: true
        });
        
        // Verify stream has tracks
        if (!state.mediaStream || state.mediaStream.getVideoTracks().length === 0) {
            throw new Error('No video tracks in stream');
        }
        
        // 5. Split tracks ONLY for recording (not for preview)
        state.videoStream = new MediaStream(state.mediaStream.getVideoTracks());
        state.audioStream = new MediaStream(state.mediaStream.getAudioTracks());
        
        // 2. Force autoplay compatibility - set properties BEFORE play()
        elements.videoBlur.muted = true;
        elements.videoSharp.muted = true;
        elements.videoBlur.playsInline = true;
        elements.videoSharp.playsInline = true;
        elements.videoBlur.autoplay = true;
        elements.videoSharp.autoplay = true;
        
        // 4. Assign the SAME stream to BOTH video elements
        elements.videoBlur.srcObject = state.mediaStream;
        elements.videoSharp.srcObject = state.mediaStream;
        
        // 4. Call play safely - never block UI if it fails
        await Promise.all([
            elements.videoBlur.play().catch((err) => {
                console.warn('videoBlur.play() failed (autoplay policy):', err);
            }),
            elements.videoSharp.play().catch((err) => {
                console.warn('videoSharp.play() failed (autoplay policy):', err);
            })
        ]);
        
        // 7. Only update status if stream exists and has video tracks
        if (state.mediaStream && state.mediaStream.getVideoTracks().length > 0) {
            // 1. Mark camera as ready
            state.uiState = 'camera_ready';
            updateVideoStatus('Camera Ready', 'ready');
            updateUIState();
            console.log('Media initialized successfully', {
                videoTracks: state.mediaStream.getVideoTracks().length,
                audioTracks: state.mediaStream.getAudioTracks().length
            });
        }
    } catch (error) {
        // 6. Improved error diagnostics - NO generic errors
        console.error('Camera init failed:', error.name, error.message, error);
        
        // 7. Do NOT show error if stream exists and has video tracks
        if (state.mediaStream && state.mediaStream.getVideoTracks().length > 0) {
            console.log('Stream exists, ignoring play() error');
            return; // Stream is valid, just autoplay failed
        }
        
        // Map specific errors
        let errorMessage;
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMessage = 'Permission denied. Please grant camera and microphone access.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            errorMessage = 'Camera is already in use by another application.';
        } else if (error.name === 'OverconstrainedError') {
            errorMessage = 'Camera constraints not supported by your device.';
        } else {
            errorMessage = error.message || 'Unable to access camera/microphone.';
        }
        
        alert(errorMessage);
        throw error;
    }
}

function updateVideoStatus(text, status) {
    const statusText = elements.videoStatus.querySelector('.status-text');
    const statusDot = elements.videoStatus.querySelector('.status-dot');
    
    if (statusText) statusText.textContent = text;
    if (statusDot) {
        statusDot.className = 'status-dot';
        statusDot.classList.add(`status-${status}`);
    }
}

// =====================
// Recording Logic
// =====================

async function startVerification() {
    if (!state.mediaStream) {
        showError('Camera/microphone not available. Please refresh the page.');
        return;
    }
    
    try {
        // Disable button
        elements.btnStart.disabled = true;
        elements.btnStart.innerHTML = '<span class="btn-icon">⏳</span> Getting captcha...';
        
        // Step 1: Get captcha from backend
        const formData = new FormData();
        formData.append('action', 'start');
        
        const response = await fetch(`${API_BASE}/api/process`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.action !== 'start' || !result.captcha_sentence) {
            throw new Error('Failed to get captcha sentence');
        }
        
        // Store captcha (valid until user stops recording and submits verify)
        state.captchaId = result.captcha_id;
        state.captchaSentence = result.captcha_sentence;
        state.captchaExpiry = null; // no time-based expiry; expires when verify is submitted
        
        // Display captcha
        elements.captchaText.textContent = result.captcha_sentence;
        elements.captchaSection.style.display = 'block';
        elements.captchaTimer.textContent = 'Valid until you stop and verify';
        elements.captchaTimer.style.display = 'block';
        
        // Step 2: Start recording
        await startRecording();
        
        // Update button
        elements.btnStart.disabled = true;
        elements.btnStart.innerHTML = '<span class="btn-icon">▶</span> Recording...';
        
        console.log('Verification started with captcha');
    } catch (error) {
        console.error('Error starting verification:', error);
        showError(`Failed to start verification: ${error.message}`);
        elements.btnStart.disabled = false;
        elements.btnStart.innerHTML = '<span class="btn-icon">▶</span> Start Verification';
    }
}

async function startRecording() {
    try {
        // Reset previous recording
        state.videoChunks = [];
        state.audioSamples = [];
        
        // Create video recorder
        const videoOptions = {
            mimeType: 'video/webm;codecs=vp8',
            videoBitsPerSecond: 2500000
        };
        if (!MediaRecorder.isTypeSupported(videoOptions.mimeType)) {
            videoOptions.mimeType = 'video/webm';
        }
        
        state.videoRecorder = new MediaRecorder(state.videoStream, videoOptions);
        state.videoRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                state.videoChunks.push(event.data);
            }
        };
        
        // Setup Web Audio API for PCM capture
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        state.audioContext = new AudioContext({ sampleRate: 16000 });
        
        // Create source from audio stream
        state.audioSource = state.audioContext.createMediaStreamSource(state.audioStream);
        
        // Create script processor for PCM capture (4096 buffer size)
        const bufferSize = 4096;
        state.audioProcessor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        state.audioProcessor.onaudioprocess = (e) => {
            if (state.isRecording) {
                const inputData = e.inputBuffer.getChannelData(0);
                // Copy Float32Array samples
                const samples = new Float32Array(inputData.length);
                samples.set(inputData);
                state.audioSamples.push(samples);
            }
        };
        
        // Connect audio pipeline
        state.audioSource.connect(state.audioProcessor);
        state.audioProcessor.connect(state.audioContext.destination);
        
        // Start video recorder
        state.videoRecorder.start(100);
        state.isRecording = true;
        state.recordingStartTime = Date.now();
        
        // 3. Set recording state
        state.uiState = 'recording';
        updateUIState('Recording...');
        updateVideoStatus('Recording...', 'recording');
        
        // Start recording timer
        startRecordingTimer();
        
        console.log('Recording started (video + Web Audio API)');
    } catch (error) {
        console.error('Error starting recording:', error);
        throw error;
    }
}

function startCaptchaTimer() {
    // Captcha is valid until user stops recording and submits; no countdown
    if (state.captchaTimer) {
        clearInterval(state.captchaTimer);
        state.captchaTimer = null;
    }
    elements.captchaTimer.textContent = 'Valid until you stop and verify';
    elements.captchaTimer.style.display = 'block';
}

function stopAndVerify() {
    if (!state.isRecording) {
        return;
    }
    
    // Check minimum duration
    const duration = (Date.now() - state.recordingStartTime) / 1000;
    if (duration < state.MIN_RECORDING_DURATION) {
        const remaining = (state.MIN_RECORDING_DURATION - duration).toFixed(1);
        showError(`Please record for at least ${state.MIN_RECORDING_DURATION} seconds. ${remaining}s remaining.`);
        return;
    }
    
    try {
        // Stop recording
        state.isRecording = false;
        
        // Stop video recorder
        if (state.videoRecorder && state.videoRecorder.state !== 'inactive') {
            state.videoRecorder.stop();
        }
        
        // Stop audio processing
        if (state.audioProcessor) {
            state.audioProcessor.disconnect();
        }
        if (state.audioSource) {
            state.audioSource.disconnect();
        }
        
        // Stop timer
        if (state.recordingTimer) {
            clearInterval(state.recordingTimer);
            state.recordingTimer = null;
        }
        
        // Update UI state
        state.uiState = 'processing';
        elements.btnStart.disabled = true;
        elements.btnStop.disabled = true;
        updateVideoStatus('Processing...', 'processing');
        
        // Show processing status
        showProcessingStatus('Decoding media...');
        
        // Wait for video recorder to finish, then process
        if (state.videoRecorder) {
            state.videoRecorder.onstop = () => {
                processRecording();
            };
        } else {
            processRecording();
        }
        
    } catch (error) {
        console.error('Error stopping recording:', error);
        state.uiState = 'failed';
        showError('Failed to stop recording. Please try again.');
        hideProcessingStatus();
        elements.btnStart.disabled = false;
        elements.btnStop.disabled = false;
    }
}

function startRecordingTimer() {
    if (state.recordingTimer) {
        clearInterval(state.recordingTimer);
    }
    
    state.recordingTimer = setInterval(() => {
        const elapsed = (Date.now() - state.recordingStartTime) / 1000;
        const remaining = MAX_RECORDING_TIME - Math.floor(elapsed);
        const minRemaining = Math.max(0, state.MIN_RECORDING_DURATION - elapsed);
        
        if (remaining <= 0) {
            // Auto-stop after max time
            stopAndVerify();
        } else if (minRemaining > 0) {
            updateVideoStatus(`Recording... (min ${minRemaining.toFixed(1)}s)`, 'recording');
            elements.btnStop.disabled = true;
        } else {
            updateVideoStatus(`Recording... ${remaining}s remaining`, 'recording');
            elements.btnStop.disabled = false;
        }
    }, 100);
}

// =====================
// WAV Encoding
// =====================

function encodeWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // PCM format
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // 16-bit
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    // Convert Float32 to Int16
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

// =====================
// API Communication
// =====================

async function processRecording() {
    if (state.videoChunks.length === 0 && state.audioSamples.length === 0) {
        state.uiState = 'failed';
        showError('No recording data available. Please try again.');
        hideProcessingStatus();
        updateUIState();
        return;
    }
    
    if (!state.captchaId || !state.captchaSentence) {
        state.uiState = 'failed';
        showError('Captcha not available. Please start a new verification.');
        hideProcessingStatus();
        updateUIState();
        return;
    }
    
    try {
        // Create video blob
        const videoBlob = state.videoChunks.length > 0 
            ? new Blob(state.videoChunks, { type: 'video/webm' })
            : null;
        
        // Convert PCM samples to WAV
        let audioBlob = null;
        if (state.audioSamples.length > 0) {
            // Concatenate all Float32Array samples
            const totalLength = state.audioSamples.reduce((sum, arr) => sum + arr.length, 0);
            const concatenated = new Float32Array(totalLength);
            let offset = 0;
            for (const samples of state.audioSamples) {
                concatenated.set(samples, offset);
                offset += samples.length;
            }
            audioBlob = encodeWAV(concatenated, 16000);
        }
        
        if (!videoBlob && !audioBlob) {
            state.uiState = 'failed';
            showError('No valid recording data. Please try again.');
            hideProcessingStatus();
            updateUIState();
            return;
        }
        
        // Create FormData
        const formData = new FormData();
        formData.append('action', 'verify');
        formData.append('captcha_id', state.captchaId);
        if (videoBlob) {
            formData.append('video_file', videoBlob, 'video.webm');
        }
        if (audioBlob) {
            formData.append('audio_file', audioBlob, 'audio.wav');
        }
        formData.append('sample_rate', '16000');
        
        // Calculate duration
        const duration = (Date.now() - state.recordingStartTime) / 1000;
        formData.append('duration', duration.toString());
        
        console.log('Sending verification request to API...');
        
        // 3. Set processing state BEFORE API call
        state.uiState = 'processing';
        updateProcessingStatus('Analyzing face...');
        updateUIState('Processing...');
        
        // Send to API
        const response = await fetch(`${API_BASE}/api/process`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        updateProcessingStatus('Processing voice...');
        const result = await response.json();
        
        console.log('API Response:', result);
        
        // 3. Transition UI state based on result - ALWAYS transition out of processing
        if (result.success === true) {
            state.uiState = 'success';
            updateProcessingStatus('Finalizing result...');
            await new Promise(resolve => setTimeout(resolve, 300));
            hideProcessingStatus();
            clearCaptcha();
            displayResults(result);
        } else {
            state.uiState = 'failed';
            hideProcessingStatus();
            showError(result.message || 'Verification failed');
        }
        
        updateUIState();
        
    } catch (error) {
        console.error('Error processing recording:', error);
        state.uiState = 'failed';
        hideProcessingStatus();
        showError(`Failed to process verification: ${error.message}`);
        updateUIState();
    }
}

function clearCaptcha() {
    state.captchaId = null;
    state.captchaSentence = null;
    state.captchaExpiry = null;
    if (state.captchaTimer) {
        clearInterval(state.captchaTimer);
        state.captchaTimer = null;
    }
    elements.captchaSection.style.display = 'none';
    elements.captchaText.textContent = '';
    elements.captchaTimer.textContent = '';
    elements.captchaTimer.style.display = 'none';
}

// =====================
// UI Updates
// =====================

function showProcessingStatus(message = 'Processing verification...') {
    elements.statusSection.style.display = 'block';
    elements.statusIcon.textContent = '⏳';
    elements.statusText.textContent = message;
    elements.resultsSection.style.display = 'none';
}

function updateProcessingStatus(message) {
    if (elements.statusSection.style.display !== 'none') {
        elements.statusText.textContent = message;
    }
}

function updateUIState(message = null) {
    // Update button states based on UI state
    if (state.uiState === 'idle' || state.uiState === 'camera_ready') {
        elements.btnStart.disabled = false;
        elements.btnStop.disabled = true;
    } else if (state.uiState === 'recording') {
        elements.btnStart.disabled = true;
        elements.btnStop.disabled = false;
    } else if (state.uiState === 'processing') {
        elements.btnStart.disabled = true;
        elements.btnStop.disabled = true;
    } else if (state.uiState === 'success' || state.uiState === 'failed') {
        elements.btnStart.disabled = false;
        elements.btnStop.disabled = true;
    }
    
    // Update status text if message provided
    if (message && elements.statusText) {
        elements.statusText.textContent = message;
    }
    
    // Update status banner color
    const statusCard = elements.statusSection.querySelector('.status-card');
    if (statusCard) {
        statusCard.className = 'status-card';
        if (state.uiState === 'processing') {
            statusCard.classList.add('status-processing');
            elements.statusSection.style.display = 'block';
        } else if (state.uiState === 'success') {
            statusCard.classList.add('status-success');
        } else if (state.uiState === 'failed') {
            statusCard.classList.add('status-failed');
        }
    }
}

function hideProcessingStatus() {
    elements.statusSection.style.display = 'none';
}

function displayResults(result) {
    // Show results section
    elements.resultsSection.style.display = 'block';
    
    // Hide error message initially
    elements.errorMessage.style.display = 'none';
    
    // Determine success/failure
    const success = result.success === true;
    const checks = result.checks || {};
    
    // Update badge
    if (success) {
        elements.resultBadge.className = 'result-badge success';
        elements.badgeIcon.textContent = '✓';
        elements.badgeText.textContent = 'VERIFIED';
    } else {
        elements.resultBadge.className = 'result-badge failure';
        elements.badgeIcon.textContent = '✗';
        elements.badgeText.textContent = 'VERIFICATION FAILED';
        // Show error message if present
        if (result.message) {
            elements.errorMessage.textContent = result.message;
            elements.errorMessage.style.display = 'block';
        }
    }
    
    // Update age
    const estimatedAge = safeNumber(result.estimated_age);
    if (estimatedAge !== null) {
        elements.ageNumber.textContent = Math.round(estimatedAge);
    } else {
        elements.ageNumber.textContent = '--';
    }
    // Update age source note
    const ageSourceLabels = {
        fusion: 'Based on face + voice (fusion model)',
        combined: 'Based on face + voice (combined)',
        face: 'Based on facial appearance',
        voice: 'Based on voice'
    };
    if (elements.ageNote) {
        elements.ageNote.textContent = ageSourceLabels[result.age_source] || 'Based on face + voice';
    }
    
    // Update adult status
    const isAdult = result.is_adult === true;
    if (isAdult) {
        elements.adultStatus.className = 'adult-status verified';
        elements.adultIcon.textContent = '✓';
        elements.adultText.textContent = 'Age 18+ Verified';
    } else {
        elements.adultStatus.className = 'adult-status not-verified';
        elements.adultIcon.textContent = '✗';
        elements.adultText.textContent = 'Under 18';
    }
    
    // Update confidence
    const confidence = safeNumber(result.confidence) || 0;
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceBar.style.width = `${confidencePercent}%`;
    elements.confidenceValue.textContent = `${confidencePercent}%`;
    
    // Update face liveness
    const faceLiveness = safeNumber(checks.face_liveness) || 0;
    const faceLivenessPercent = Math.round(faceLiveness * 100);
    elements.faceLivenessBar.style.width = `${faceLivenessPercent}%`;
    elements.faceLivenessValue.textContent = `${faceLivenessPercent}%`;
    
    // Update voice liveness
    const voiceLiveness = safeNumber(checks.voice_liveness) || 0;
    const voiceLivenessPercent = Math.round(voiceLiveness * 100);
    elements.voiceLivenessBar.style.width = `${voiceLivenessPercent}%`;
    elements.voiceLivenessValue.textContent = `${voiceLivenessPercent}%`;
    
    // Update lip sync
    const lipSync = safeNumber(checks.lip_sync);
    if (lipSync !== null && lipSync !== undefined) {
        const lipSyncPercent = Math.round(lipSync * 100);
        elements.lipSyncBar.style.width = `${lipSyncPercent}%`;
        elements.lipSyncValue.textContent = `${lipSyncPercent}%`;
    } else {
        elements.lipSyncBar.style.width = '0%';
        elements.lipSyncValue.textContent = '--';
    }
    
    // Update eye blink
    const eyeBlink = safeNumber(checks.eye_blink);
    if (eyeBlink !== null && eyeBlink !== undefined) {
        const eyeBlinkPercent = Math.round(eyeBlink * 100);
        elements.eyeBlinkBar.style.width = `${eyeBlinkPercent}%`;
        elements.eyeBlinkValue.textContent = `${eyeBlinkPercent}%`;
    } else {
        elements.eyeBlinkBar.style.width = '0%';
        elements.eyeBlinkValue.textContent = '--';
    }
    
    // Show error message if present
    if (result.message && !success) {
        elements.errorMessage.textContent = result.message;
        elements.errorMessage.style.display = 'block';
    }
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.style.display = 'block';
    elements.resultsSection.style.display = 'block';
    
    // Set failure state
    elements.resultBadge.className = 'result-badge failure';
    elements.badgeIcon.textContent = '✗';
    elements.badgeText.textContent = 'ERROR';
}

function restart() {
    // Reset state
    state.videoChunks = [];
    state.audioSamples = [];
    state.isRecording = false;
    // Reset to camera_ready if media stream exists, otherwise idle
    state.uiState = (state.mediaStream && state.mediaStream.getVideoTracks().length > 0) ? 'camera_ready' : 'idle';
    
    if (state.recordingTimer) {
        clearInterval(state.recordingTimer);
        state.recordingTimer = null;
    }
    
    // Stop any active recorders
    if (state.videoRecorder && state.videoRecorder.state !== 'inactive') {
        state.videoRecorder.stop();
    }
    
    // Stop audio processing
    if (state.audioProcessor) {
        state.audioProcessor.disconnect();
    }
    if (state.audioSource) {
        state.audioSource.disconnect();
    }
    if (state.audioContext && state.audioContext.state !== 'closed') {
        state.audioContext.close();
    }
    
    // Clear captcha
    clearCaptcha();
    
    // Reset UI
    hideProcessingStatus();
    elements.resultsSection.style.display = 'none';
    elements.errorMessage.style.display = 'none';
    updateVideoStatus('Camera Ready', 'ready');
    updateUIState();
    
    elements.btnStart.innerHTML = '<span class="btn-icon">▶</span> Start Verification';
    
    // Reset video preview if needed
    if (state.mediaStream) {
        if (elements.videoBlur.srcObject !== state.mediaStream) {
            elements.videoBlur.srcObject = state.mediaStream;
        }
        if (elements.videoSharp.srcObject !== state.mediaStream) {
            elements.videoSharp.srcObject = state.mediaStream;
        }
    }
}

// =====================
// Utility Functions
// =====================

function safeNumber(value) {
    if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
        return null;
    }
    return parseFloat(value);
}

// =====================
// Cleanup
// =====================

window.addEventListener('beforeunload', () => {
    // Stop all tracks
    if (state.videoStream) {
        state.videoStream.getTracks().forEach(track => track.stop());
    }
    if (state.audioStream) {
        state.audioStream.getTracks().forEach(track => track.stop());
    }
    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(track => track.stop());
    }
    
    // Stop recorders
    if (state.videoRecorder && state.videoRecorder.state !== 'inactive') {
        state.videoRecorder.stop();
    }
    
    // Stop audio processing
    if (state.audioProcessor) {
        state.audioProcessor.disconnect();
    }
    if (state.audioSource) {
        state.audioSource.disconnect();
    }
    if (state.audioContext && state.audioContext.state !== 'closed') {
        state.audioContext.close();
    }
    
    // Clear timers
    if (state.recordingTimer) {
        clearInterval(state.recordingTimer);
    }
    if (state.captchaTimer) {
        clearInterval(state.captchaTimer);
    }
});


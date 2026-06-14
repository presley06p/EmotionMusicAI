/* ═══════════════════════════════════════════════════════════════════════════
   EmoTune AI — detect.js
   Webcam emotion detection, text analysis, music recommendations
═══════════════════════════════════════════════════════════════════════════ */

// ── Emotion config ────────────────────────────────────────────────────────
const EMOTION_COLORS = {
  happy:    '#1DB954',
  sad:      '#1E90FF',
  angry:    '#FF4444',
  fear:     '#9B59B6',
  love:     '#FF6B9D',
  surprise: '#FFD700',
  neutral:  '#00D4AA',
  disgust:  '#FF8C00',
  no_face:  '#888888',
};

const EMOTION_EMOJIS = {
  happy:'😊', sad:'😢', angry:'😠', fear:'😨',
  love:'❤️', surprise:'😮', neutral:'😐',
  disgust:'🤢', no_face:'🚫',
};

// ── State ─────────────────────────────────────────────────────────────────
let videoStream    = null;
let autoInterval   = null;
let lastEmotion    = null;
let lastTracks     = [];

// ── Tab switching ─────────────────────────────────────────────────────────
function switchTab(tab) {
  document.getElementById('panelFace').style.display = tab === 'face' ? '' : 'none';
  document.getElementById('panelText').style.display = tab === 'text' ? '' : 'none';

  document.getElementById('tabFace').classList.toggle('dtab--active', tab === 'face');
  document.getElementById('tabText').classList.toggle('dtab--active', tab === 'text');
}

// ════════════════════════════════════════════════════════════════════════
//  WEBCAM
// ════════════════════════════════════════════════════════════════════════
async function startCamera() {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });

    const video    = document.getElementById('videoEl');
    const overlay  = document.getElementById('camOverlay');
    const dot      = document.getElementById('camDot');
    const statusTx = document.getElementById('camStatusText');
    const btnStart = document.getElementById('btnStartCam');
    const btnStop  = document.getElementById('btnStopCam');
    const btnCap   = document.getElementById('btnCapture');

    video.srcObject = videoStream;
    video.style.display = 'block';
    overlay.style.display = 'none';

    dot.classList.add('active');
    statusTx.textContent = 'Camera live';
    btnStart.style.display = 'none';
    btnStop.style.display  = '';
    btnCap.disabled = false;

    // Auto-detect listener
    document.getElementById('autoDetect').addEventListener('change', toggleAutoDetect);

  } catch (err) {
    alert('Could not access webcam: ' + err.message);
  }
}

function stopCamera() {
  if (videoStream) {
    videoStream.getTracks().forEach(t => t.stop());
    videoStream = null;
  }
  clearInterval(autoInterval);
  autoInterval = null;

  const video    = document.getElementById('videoEl');
  const overlay  = document.getElementById('camOverlay');
  const dot      = document.getElementById('camDot');
  const statusTx = document.getElementById('camStatusText');
  const btnStart = document.getElementById('btnStartCam');
  const btnStop  = document.getElementById('btnStopCam');
  const btnCap   = document.getElementById('btnCapture');

  video.srcObject = null;
  video.style.display = 'none';
  overlay.style.display = 'flex';
  dot.classList.remove('active');
  statusTx.textContent = 'Camera off';
  btnStart.style.display = '';
  btnStop.style.display  = 'none';
  btnCap.disabled = true;

  document.getElementById('autoDetect').checked = false;
}

function toggleAutoDetect() {
  const checked = document.getElementById('autoDetect').checked;
  if (checked) {
    autoInterval = setInterval(captureAndAnalyze, 3000);
  } else {
    clearInterval(autoInterval);
    autoInterval = null;
  }
}

async function captureAndAnalyze() {
  const video  = document.getElementById('videoEl');
  const canvas = document.getElementById('canvasEl');

  if (!videoStream) return;

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  // Mirror to match mirrored video display
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);

  const b64 = canvas.toDataURL('image/jpeg', 0.8);

  showLoading('Detecting facial emotion…');

  try {
    const resp = await fetch('/api/analyze-face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64 }),
    });
    const data = await resp.json();
    hideLoading();

    if (data.emotion === 'no_face' || data.error) {
      showTemporaryMsg('No face detected. Try again.');
      return;
    }

    renderFaceResult(data);

    if (data.tracks && data.tracks.length) {
      lastEmotion = data.emotion;
      lastTracks  = data.tracks;
      renderMusicSection(data.emotion, data.tracks);
    }

  } catch (err) {
    hideLoading();
    console.error(err);
    showTemporaryMsg('Analysis failed. Please try again.');
  }
}

function renderFaceResult(data) {
  const { emotion, confidence, all_scores, face_box, mock } = data;
  const color = EMOTION_COLORS[emotion] || '#888';

  document.getElementById('faceResult').style.display      = '';
  document.getElementById('facePlaceholder').style.display = 'none';

  document.getElementById('faceEmoji').textContent      = EMOTION_EMOJIS[emotion] || '😐';
  document.getElementById('faceEmotionName').textContent = capitalize(emotion);

  // Method badge
  const methodLabels = {'claude-vision':'Claude Vision', 'demo':'Demo Mode', 'fer':'FER Model'};
  document.getElementById('faceMethodBadge').textContent = methodLabels[data.method] || (mock ? 'Demo' : 'AI');
  document.getElementById('faceEmotionName').style.color = color;

  // Show explanation from Claude Vision if available
  const faceExp = document.getElementById('faceExplanation');
  if (faceExp) {
    if (data.explanation) {
      faceExp.textContent = data.explanation;
      faceExp.style.display = '';
    } else {
      faceExp.style.display = 'none';
    }
  }

  // Confidence ring
  const pct = Math.round(confidence * 100);
  const circumference = 213.6;
  const offset = circumference - (pct / 100) * circumference;
  const ring = document.getElementById('faceConfRing');
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = color;
  document.getElementById('faceConfPct').textContent = pct + '%';

  // Face box overlay
  if (face_box) {
    const video = document.getElementById('videoEl');
    const vw = video.offsetWidth  || 640;
    const vh = video.offsetHeight || 480;
    const [bx, by, bw, bh] = face_box;
    // Mirror x
    const mirroredX = vw - bx - bw;
    const faceBoxEl = document.getElementById('faceBox');
    faceBoxEl.style.display  = 'block';
    faceBoxEl.style.left     = (mirroredX / (video.videoWidth || 640) * 100) + '%';
    faceBoxEl.style.top      = (by / (video.videoHeight || 480) * 100) + '%';
    faceBoxEl.style.width    = (bw / (video.videoWidth || 640) * 100) + '%';
    faceBoxEl.style.height   = (bh / (video.videoHeight || 480) * 100) + '%';
    faceBoxEl.style.borderColor = color;
    faceBoxEl.style.boxShadow   = `0 0 12px ${color}66`;
  }

  // All-score bars
  renderEmoBars('faceEmoBars', all_scores);
}

// ════════════════════════════════════════════════════════════════════════
//  TEXT ANALYSIS
// ════════════════════════════════════════════════════════════════════════
function setPrompt(text) {
  document.getElementById('textInput').value = text;
  updateCharCount();
}

function clearText() {
  document.getElementById('textInput').value = '';
  updateCharCount();
}

function updateCharCount() {
  const val = document.getElementById('textInput').value;
  document.getElementById('charCount').textContent = `${val.length} / 500`;
}

document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('textInput');
  if (ta) {
    ta.addEventListener('input', updateCharCount);
    ta.addEventListener('keydown', e => {
      if (e.key === 'Enter' && e.ctrlKey) analyzeText();
    });
  }
});

async function analyzeText() {
  const text = (document.getElementById('textInput').value || '').trim();
  if (!text) {
    showTemporaryMsg('Please enter some text first.');
    return;
  }

  showLoading('Analysing your text emotion…');
  const btn = document.getElementById('btnAnalyzeText');
  btn.disabled = true;

  try {
    const resp = await fetch('/api/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json();
    hideLoading();
    btn.disabled = false;

    if (data.error) {
      showTemporaryMsg(data.error);
      return;
    }

    renderTextResult(data);

    if (data.tracks && data.tracks.length) {
      lastEmotion = data.emotion;
      lastTracks  = data.tracks;
      renderMusicSection(data.emotion, data.tracks);
    }

  } catch (err) {
    hideLoading();
    btn.disabled = false;
    console.error(err);
    showTemporaryMsg('Analysis failed. Please try again.');
  }
}

function renderTextResult(data) {
  const { emotion, confidence, all_scores, explanation, method } = data;
  const color = EMOTION_COLORS[emotion] || '#888';

  document.getElementById('textResult').style.display      = '';
  document.getElementById('textPlaceholder').style.display = 'none';

  document.getElementById('textEmoji').textContent      = EMOTION_EMOJIS[emotion] || '😐';
  document.getElementById('textEmotionName').textContent = capitalize(emotion);
  const textMethodLabels = {'claude-ai':'Claude AI', 'transformer':'Transformer NLP', 'keyword':'Keyword NLP'};
  document.getElementById('textMethodBadge').textContent = textMethodLabels[method] || 'NLP';
  document.getElementById('textEmotionName').style.color = color;

  if (explanation) {
    document.getElementById('textExplanation').textContent = explanation;
  }

  const pct = Math.round(confidence * 100);
  const circumference = 213.6;
  const offset = circumference - (pct / 100) * circumference;
  const ring = document.getElementById('textConfRing');
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = color;
  document.getElementById('textConfPct').textContent = pct + '%';

  renderEmoBars('textEmoBars', all_scores);
}

// ── Shared emotion bars renderer ─────────────────────────────────────────
function renderEmoBars(containerId, scores) {
  const container = document.getElementById(containerId);
  if (!container || !scores) return;

  const entries = Object.entries(scores).sort((a,b) => b[1] - a[1]);

  container.innerHTML = entries.map(([emo, val]) => {
    const pct   = Math.round(val * 100);
    const color = EMOTION_COLORS[emo] || '#888';
    return `
      <div class="emo-bar-row">
        <div class="emo-bar-name">${capitalize(emo)}</div>
        <div class="emo-bar-wrap">
          <div class="emo-bar-fill" style="width:${pct}%; background:${color};"></div>
        </div>
        <div class="emo-bar-pct">${pct}%</div>
      </div>
    `;
  }).join('');
}

// ════════════════════════════════════════════════════════════════════════
//  MUSIC SECTION
// ════════════════════════════════════════════════════════════════════════
function renderMusicSection(emotion, tracks) {
  const section = document.getElementById('musicSection');
  const label   = document.getElementById('musicEmotionLabel');
  const grid    = document.getElementById('trackGrid');

  label.textContent = capitalize(emotion);
  grid.innerHTML    = tracks.map(renderTrackCard).join('');
  section.style.display = '';

  // Scroll to music section
  setTimeout(() => {
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
}

function renderTrackCard(track) {
  const hasCover   = track.album_cover && !track.album_cover.includes('placeholder');
  const coverHtml  = hasCover
    ? `<img src="${escHtml(track.album_cover)}" alt="cover" loading="lazy" />`
    : `<div class="track-cover-placeholder"><i class="bi bi-music-note-beamed"></i></div>`;

  const previewBtn = track.preview_url
    ? `<button class="btn-track-action" onclick="playPreview('${escHtml(track.preview_url)}', this)" title="Preview">
         <i class="bi bi-play-fill"></i>
       </button>`
    : '';

  const spotifyBtn = track.spotify_url
    ? `<a href="${escHtml(track.spotify_url)}" target="_blank" class="btn-track-action" title="Open in Spotify">
         <i class="bi bi-spotify"></i>
       </a>`
    : '';

  const mockBadge = track.mock
    ? `<div class="mock-badge">Demo</div>` : '';

  const trackData = encodeURIComponent(JSON.stringify({
    song_name:   track.name,
    artist:      track.artist,
    emotion:     lastEmotion || 'neutral',
    spotify_url: track.spotify_url,
    album_cover: track.album_cover,
  }));

  return `
    <div class="track-card">
      <div class="track-cover">
        ${coverHtml}
        ${mockBadge}
        <div class="track-play-overlay">
          <button class="track-play-btn" onclick="playTrack('${escHtml(track.spotify_url)}')">
            <i class="bi bi-play-fill"></i>
          </button>
        </div>
      </div>
      <div class="track-info">
        <div class="track-name" title="${escHtml(track.name)}">${escHtml(track.name)}</div>
        <div class="track-artist">${escHtml(track.artist)}</div>
        <div class="track-meta">
          <span class="track-duration">${escHtml(track.duration || '')}</span>
          <div class="track-actions">
            ${previewBtn}
            ${spotifyBtn}
            <button class="btn-track-action" onclick="saveSong(this, '${trackData}')" title="Save">
              <i class="bi bi-heart"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  `;
}

function playTrack(url) {
  if (url && !url.includes('mock')) {
    window.open(url, '_blank');
  }
}

let currentAudio = null;
function playPreview(url, btn) {
  if (!url) return;
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
    document.querySelectorAll('.btn-track-action .bi-stop-fill').forEach(i => {
      i.className = 'bi bi-play-fill';
    });
  }
  const icon = btn.querySelector('i');
  if (icon.className.includes('stop')) {
    icon.className = 'bi bi-play-fill';
    return;
  }
  currentAudio = new Audio(url);
  currentAudio.volume = 0.7;
  currentAudio.play().catch(() => {});
  icon.className = 'bi bi-stop-fill';
  currentAudio.onended = () => { icon.className = 'bi bi-play-fill'; };
}

async function saveSong(btn, encodedData) {
  const data = JSON.parse(decodeURIComponent(encodedData));
  try {
    const resp = await fetch('/api/save-song', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const result = await resp.json();
    if (result.success) {
      btn.classList.add('saved');
      btn.querySelector('i').className = 'bi bi-heart-fill';
      showSaveToast();
    }
  } catch (err) {
    console.error(err);
  }
}

async function refreshTracks() {
  if (!lastEmotion) return;
  showLoading('Refreshing recommendations…');
  try {
    const resp = await fetch('/api/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: `I feel ${lastEmotion}` }),
    });
    const data = await resp.json();
    hideLoading();
    if (data.tracks) {
      lastTracks = data.tracks;
      renderMusicSection(lastEmotion, data.tracks);
    }
  } catch (err) {
    hideLoading();
  }
}

// ── Loading helpers ───────────────────────────────────────────────────────
function showLoading(msg) {
  const el = document.getElementById('detectLoading');
  document.getElementById('loadingText').textContent = msg || 'Analysing…';
  el.style.display = 'flex';
}
function hideLoading() {
  document.getElementById('detectLoading').style.display = 'none';
}

// ── Save toast ────────────────────────────────────────────────────────────
function showSaveToast() {
  const toast = document.getElementById('saveToast');
  toast.style.display = 'flex';
  setTimeout(() => {
    toast.style.transition = 'opacity .4s';
    toast.style.opacity = '0';
    setTimeout(() => { toast.style.display = 'none'; toast.style.opacity = '1'; }, 400);
  }, 2500);
}

// ── Temp message ─────────────────────────────────────────────────────────
function showTemporaryMsg(msg) {
  const toast = document.createElement('div');
  toast.className = 'flash-toast flash-warning';
  toast.style.cssText = `
    position:fixed; top:80px; left:50%; transform:translateX(-50%);
    z-index:9999; pointer-events:none;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity .3s';
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ── Utilities ─────────────────────────────────────────────────────────────
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function escHtml(str) {
  return String(str || '')
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

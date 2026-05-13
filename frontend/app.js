const API_BASE = 'https://fraud-detector.eduardo-lalo1999.workers.dev';

const samples = {
  normal: [-0.493230744836423, 0.110827954119709, 0.595539410505494, -2.26755558180812, 0.272888237754473, -1.30143942290887, 0.895946681389707, -0.426423837425371, -0.768733017429244, 0.353873894516519, -1.0819988063087, -1.71981084740664, -1.70604563772243, -0.107881542036573, -0.776354090894807, 0.715709816730862, 0.146290661927997, -1.71307810023531, 0.105160755145993, 0.0558027064522817, 0.00159122772019078, 0.103394070188666, -0.217291925542905, -0.052884702722099, 0.00406510826105916, -0.402567012141426, 0.0151640493823205, -0.0183594738813524, 22.5],
  suspicious: [-2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468, -0.522187864667764, -1.42654531920595, -2.53738730624579, 1.39165724829804, -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473, -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657, -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474, 0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526, 0.0445191674731724, 0.177839798284401, 0.261145002567677, -0.143275874698919, 0]
};

const el = (id) => document.getElementById(id);
const featuresInput = el('featuresInput');
const resultBox = el('resultBox');
const formStatus = el('formStatus');
const scoreRing = el('scoreRing');
const scoreValue = el('scoreValue');
const predictionValue = el('predictionValue');
const thresholdValue = el('thresholdValue');
const modelVersion = el('modelVersion');
const healthText = el('healthText');
const healthPulse = el('healthPulse');

function setSample(name) {
  if (name === 'blank') {
    featuresInput.value = '';
    return;
  }
  featuresInput.value = JSON.stringify(samples[name], null, 2);
}

function parseFeatures() {
  const parsed = JSON.parse(featuresInput.value);
  const features = Array.isArray(parsed) ? parsed : parsed.features;
  if (!Array.isArray(features)) throw new Error('El JSON debe ser un arreglo o un objeto con la propiedad "features".');
  if (features.length !== 29) throw new Error(`Se esperaban 29 valores numericos; recibidos: ${features.length}.`);
  if (!features.every((v) => typeof v === 'number' && Number.isFinite(v))) throw new Error('Todos los features deben ser numeros finitos.');
  return features;
}

function renderPrediction(data) {
  const probability = Number(data.probability ?? 0);
  const pct = Math.max(0, Math.min(100, probability * 100));
  scoreRing.style.setProperty('--score', String(pct));
  scoreValue.textContent = `${pct.toFixed(probability < 0.01 ? 4 : 2)}%`;
  thresholdValue.textContent = Number(data.threshold ?? 0).toFixed(4);
  modelVersion.textContent = data.model_version ?? 'v1';
  predictionValue.textContent = data.prediction === 1 ? 'Fraude' : 'No fraude';
  predictionValue.style.color = data.prediction === 1 ? '#ff6b7a' : '#60f0ba';
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    healthText.textContent = data.status === 'ok' ? 'API en línea' : `API: ${data.status}`;
    healthPulse.className = `pulse ${data.status === 'ok' ? 'ok' : ''}`;
  } catch (error) {
    healthText.textContent = 'No se pudo verificar la API';
    healthPulse.className = 'pulse bad';
  }
}

async function predict(event) {
  event.preventDefault();
  formStatus.textContent = 'Enviando…';
  resultBox.textContent = '';
  try {
    const features = parseFeatures();
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    if (!response.ok) throw new Error(JSON.stringify(data, null, 2));
    renderPrediction(data);
    resultBox.textContent = JSON.stringify(data, null, 2);
    formStatus.textContent = 'Respuesta recibida.';
  } catch (error) {
    formStatus.textContent = 'Error';
    resultBox.textContent = error.message;
  }
}

document.querySelectorAll('[data-sample]').forEach((button) => {
  button.addEventListener('click', () => setSample(button.dataset.sample));
});
el('predictForm').addEventListener('submit', predict);
setSample('normal');
checkHealth();

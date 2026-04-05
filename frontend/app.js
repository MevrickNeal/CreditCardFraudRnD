
const API_URL = 'http://localhost:8000/api';


let networkLatency = [];
let rocAucHistory = [];
let chartInstance = null;
let shapChartInstance = null;
let gaugeVae = null;
let gaugeEns = null;
let gaugeRisk = null;


document.addEventListener('DOMContentLoaded', async () => {
    initGauges();
    await loadProfiles();
    startSimulationFeed();

    
    document.getElementById('process-btn').addEventListener('click', processManualTransaction);
});


function initGauges() {
    const gaugeConfig = (color, currentVal = 0) => ({
        type: 'doughnut',
        data: {
            datasets: [{
                data: [currentVal, 100 - currentVal],
                backgroundColor: [color, 'rgba(226, 232, 240, 0.6)'], 
                borderWidth: 0,
                cutout: '80%',
                circumference: 270,
                rotation: 225
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { tooltip: { enabled: false }, legend: { display: false } },
            animation: { animateRotate: true, animateScale: false }
        }
    });

    gaugeRisk = new Chart(document.getElementById('gauge-risk').getContext('2d'), gaugeConfig('#ef4444')); 
    gaugeVae = new Chart(document.getElementById('gauge-vae').getContext('2d'), gaugeConfig('#9333ea')); 
    gaugeEns = new Chart(document.getElementById('gauge-ensemble').getContext('2d'), gaugeConfig('#2563eb')); 
}

function updateGauge(chart, value, color) {
    chart.data.datasets[0].data = [value, 100 - value];
    if (color) chart.data.datasets[0].backgroundColor[0] = color;
    chart.update();
}

async function loadProfiles() {
    try {
        const response = await fetch(`${API_URL}/sandbox/profiles`);
        const profiles = await response.json();
        const select = document.getElementById('profile-select');
        select.innerHTML = '<option value="">-- Choose Card Profile --</option>';

        profiles.forEach(p => {
            select.innerHTML += `<option value="${p.card_number}">[${p.type}] Card ending in ${p.card_number.slice(-4)}</option>`;
        });
    } catch (error) {
        console.error("Failed to load profiles", error);
    }
}

async function processManualTransaction() {
    const card = document.getElementById('profile-select').value;
    const amount = document.getElementById('tx-amount').value;

    if (!card) {
        alert("Please select a profile");
        return;
    }

    const btn = document.getElementById('process-btn');
    btn.innerHTML = `<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Authorize Payment`;
    btn.disabled = true;

    const startTime = performance.now();
    try {
        const response = await fetch(`${API_URL}/process_payment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                card_number: card,
                amount: parseFloat(amount),
                time: new Date().toISOString()
            })
        });

        const data = await response.json();
        const endTime = performance.now();

        updateDashboard(data, endTime - startTime);

    } catch (error) {
        console.error("Inference Error:", error);
    } finally {
        btn.innerHTML = `Authorize Payment`;
        btn.disabled = false;
    }
}

function updateDashboard(data, latency) {
    
    document.getElementById('latency-metric').innerText = `${Math.round(latency)} ms`;
    document.getElementById('auc-metric').innerText = data.system_roc_auc.toFixed(3);

    
    const totalRisk = Math.round(data.risk_score * 100);
    const vaeScore = Math.round(data.vae_anomaly * 100);
    const ensScore = Math.round(data.ensemble_prob * 100);

    document.getElementById('score-total').innerText = `${totalRisk}%`;
    document.getElementById('score-vae').innerText = vaeScore;
    document.getElementById('score-ens').innerText = ensScore;

    
    const riskColor = totalRisk > 60 ? '#ef4444' : (totalRisk > 30 ? '#f59e0b' : '#10b981');
    document.getElementById('score-total').style.color = riskColor;
    updateGauge(gaugeRisk, totalRisk, riskColor);
    updateGauge(gaugeVae, vaeScore);
    updateGauge(gaugeEns, ensScore);

    
    renderShapChart(data.shap_values, data.transaction_id);

    
    prependToFeed(data);

    
    showBanner(data);
}

function renderShapChart(shapData, txId) {
    document.getElementById('shap-tx-id').innerText = txId;

    const labels = Object.keys(shapData);
    const values = Object.values(shapData);
    const colors = values.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)');
    const borders = values.map(v => v > 0 ? 'rgba(220, 38, 38, 1)' : 'rgba(5, 150, 105, 1)');

    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    const ctx = document.getElementById('shapChart').getContext('2d');
    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (Impact on Fraud Risk)',
                data: values,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y', 
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(0,0,0,0.05)' },
                    ticks: { color: '#64748b', font: { family: 'Inter' } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#334155', font: { size: 12, family: 'Inter', weight: '500' } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function prependToFeed(tx) {
    const feed = document.getElementById('transaction-feed');

    
    if (feed.children.length > 0 && feed.children[0].classList.contains('text-center')) {
        feed.innerHTML = '';
    }

    const isFraud = tx.status === 'Declined';
    const isVerify = tx.status === 'Bank Verification Required';

    
    const iconSuccess = `<svg class="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`;
    const iconDanger = `<svg class="w-4 h-4 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`;
    const iconVerify = `<svg class="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"></path></svg>`;

    const icon = isFraud ? iconDanger : (isVerify ? iconVerify : iconSuccess);
    const bgClass = isFraud ? 'bg-red-50 border-red-200' : (isVerify ? 'bg-amber-50 border-amber-200' : 'bg-white border-slate-200 shadow-sm');
    const iconWrapperClass = isFraud ? 'bg-red-100 border-red-200' : (isVerify ? 'bg-amber-100 border-amber-200' : 'bg-emerald-100 border-emerald-200');
    const statusColor = isFraud ? 'text-red-500' : (isVerify ? 'text-amber-600' : 'text-emerald-500');

    const el = document.createElement('div');
    el.className = `p-3 rounded-xl border ${bgClass} flex justify-between items-center fade-in`;

    el.innerHTML = `
        <div class="flex items-center space-x-3">
            <div class="h-8 w-8 rounded-full ${iconWrapperClass} flex items-center justify-center border">
                ${icon}
            </div>
            <div>
                <p class="text-[11px] font-mono font-bold text-slate-600">${tx.transaction_id}</p>
                <p class="text-[10px] text-slate-400 font-medium">CARD ····${tx.card_number}</p>
            </div>
        </div>
        <div class="text-right">
            <p class="text-sm font-bold text-slate-800">$${parseFloat(tx.amount).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            <p class="text-[10px] font-bold uppercase tracking-wide ${statusColor}">${tx.status}</p>
        </div>
    `;

    feed.prepend(el);
    if (feed.children.length > 50) feed.removeChild(feed.lastChild);
}

function showBanner(data) {
    const banner = document.getElementById('decision-banner');
    const bg = document.getElementById('banner-icon-bg');

    banner.classList.remove('hidden');

    const iconWarning = `<svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`;
    const iconCheck = `<svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`;
    const iconPhone = `<svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"></path></svg>`;

    const reason = data.reason || '';

    if (data.status === 'Declined') {
        banner.className = 'glass-panel p-6 border-l-4 border-l-red-500 rounded-r-2xl mt-6 bg-red-50/80 fade-in shadow-lg';
        bg.className = 'rounded-2xl p-3 mr-4 bg-red-500 alert-pulse shadow-md';
        bg.innerHTML = iconWarning;
        document.getElementById('banner-title').innerText = 'Transaction Declined - Fraud Detected';
        document.getElementById('banner-desc').innerText = `${reason}. Risk Score: ${(data.risk_score * 100).toFixed(1)}%. The AI hybrid model has blocked this transaction.`;
        document.getElementById('banner-title').className = 'text-lg font-bold text-red-700';
        document.getElementById('banner-desc').className = 'text-sm font-medium text-red-600/90 leading-relaxed';
    } else if (data.status === 'Bank Verification Required') {
        banner.className = 'glass-panel p-6 border-l-4 border-l-amber-500 rounded-r-2xl mt-6 bg-amber-50/80 fade-in shadow-lg';
        bg.className = 'rounded-2xl p-3 mr-4 bg-amber-500 shadow-md';
        bg.innerHTML = iconPhone;
        document.getElementById('banner-title').innerText = 'Bank Verification Required';
        document.getElementById('banner-desc').innerText = `${reason}. Issuer bank verification call initiated. Transaction held pending cardholder confirmation.`;
        document.getElementById('banner-title').className = 'text-lg font-bold text-amber-700';
        document.getElementById('banner-desc').className = 'text-sm font-medium text-amber-600/90 leading-relaxed';
    } else {
        banner.className = 'glass-panel p-6 border-l-4 border-l-emerald-500 rounded-r-2xl mt-6 bg-emerald-50/80 fade-in shadow-lg';
        bg.className = 'rounded-2xl p-3 mr-4 bg-emerald-500 shadow-md';
        bg.innerHTML = iconCheck;
        document.getElementById('banner-title').innerText = 'Transaction Approved';
        document.getElementById('banner-desc').innerText = `Normal patterns observed. Ensembled Risk is very low (${(data.risk_score * 100).toFixed(1)}%).`;
        document.getElementById('banner-title').className = 'text-lg font-bold text-emerald-700';
        document.getElementById('banner-desc').className = 'text-sm font-medium text-emerald-600/90 leading-relaxed';
    }
}


function startSimulationFeed() {
    setInterval(() => {
        const feed = document.getElementById('transaction-feed');
        if (feed.children.length > 0 && feed.children[0].classList.contains('text-center')) {
            feed.innerHTML = '';
        }

        
        const txId = `TXN-${Math.floor(Math.random() * 90000) + 10000}`;
        const amount = Math.random() * 50 + 5;

        const el = document.createElement('div');
        el.className = `p-2 rounded-xl border bg-white/40 border-slate-100 flex justify-between items-center opacity-60 transition-opacity`;
        el.innerHTML = `
            <div class="flex items-center space-x-3">
                <svg class="w-3 h-3 text-slate-400 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
                <div>
                    <p class="text-[10px] font-mono text-slate-500 font-medium">${txId}</p>
                </div>
            </div>
            <div class="text-right">
                <p class="text-xs font-semibold text-slate-600">$${amount.toFixed(2)}</p>
                <p class="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Simulated</p>
            </div>
        `;

        feed.prepend(el);
        if (feed.children.length > 50) feed.removeChild(feed.lastChild);
    }, 4000);
}

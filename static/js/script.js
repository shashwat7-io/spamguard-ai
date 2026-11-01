/**
 * =============================================================================
 * SPAMGUARD AI – CLIENT-SIDE INFERENCE & INTERACTION CONTROLLER
 * =============================================================================
 * Author: SPAMGUARD AI Engineering Team
 * Objective: Real-time telemetry, async API calls to /api/predict and /api/metrics,
 *            interactive preset loaders, multi-stage loading radar, and dynamic UI.
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements Cache
    const emailInput = document.getElementById('email-input');
    const charCounter = document.getElementById('char-counter');
    const wordCounter = document.getElementById('word-counter');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const analyzeIcon = document.getElementById('analyze-icon');
    const clearInputBtn = document.getElementById('clear-input-btn');
    const pasteClipboardBtn = document.getElementById('paste-clipboard-btn');
    const inputErrorAlert = document.getElementById('input-error-alert');
    const inputErrorMsg = document.getElementById('input-error-msg');

    // Results States
    const stateIdle = document.getElementById('state-idle');
    const stateLoading = document.getElementById('state-loading');
    const stateVerdict = document.getElementById('state-verdict');
    const resultsTimeBadge = document.getElementById('results-time-badge');

    // Verdict Elements
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIconWrap = document.getElementById('verdict-icon-wrap');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictHeading = document.getElementById('verdict-heading');
    const verdictRiskTag = document.getElementById('verdict-risk-tag');
    const verdictConfidence = document.getElementById('verdict-confidence');
    const spamProbNum = document.getElementById('spam-prob-num');
    const spamProbBar = document.getElementById('spam-prob-bar');
    const hamProbNum = document.getElementById('ham-prob-num');
    const hamProbBar = document.getElementById('ham-prob-bar');
    const advisoryText = document.getElementById('advisory-text');
    const signalsList = document.getElementById('signals-list');
    const signalsCountBadge = document.getElementById('signals-count-badge');
    const preprocessedSnippet = document.getElementById('preprocessed-snippet');
    const metaLatency = document.getElementById('meta-latency');
    const metaTokens = document.getElementById('meta-tokens');

    // Telemetry & Metrics Elements
    const heroAcc = document.getElementById('hero-acc');
    const heroPrec = document.getElementById('hero-prec');
    const kpiAccuracy = document.getElementById('kpi-accuracy');
    const kpiPrecision = document.getElementById('kpi-precision');
    const kpiRecall = document.getElementById('kpi-recall');
    const kpiF1 = document.getElementById('kpi-f1');
    const cmTn = document.getElementById('cm-tn');
    const cmFp = document.getElementById('cm-fp');
    const cmFn = document.getElementById('cm-fn');
    const cmTp = document.getElementById('cm-tp');
    const topSpamVocabList = document.getElementById('top-spam-vocab-list');
    const topHamVocabList = document.getElementById('top-ham-vocab-list');

    // Navbar Status Pill
    const systemStatusPill = document.getElementById('system-status-pill');
    const statusLabel = document.getElementById('status-label');

    // Cached Benchmark Samples (Fallback if API is unreachable)
    let benchmarkSamples = {};

    // =========================================================================
    // 1. Textarea Telemetry & Counter Listeners
    // =========================================================================

    function updateTextareaTelemetry() {
        const text = emailInput.value || '';
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;

        charCounter.textContent = chars.toLocaleString();
        wordCounter.textContent = words.toLocaleString();

        if (chars > 0) {
            hideInputError();
        }
    }

    emailInput.addEventListener('input', updateTextareaTelemetry);
    emailInput.addEventListener('keyup', updateTextareaTelemetry);

    // =========================================================================
    // 2. Toolbar Actions (Clear & Paste)
    // =========================================================================

    clearInputBtn.addEventListener('click', () => {
        emailInput.value = '';
        updateTextareaTelemetry();
        hideInputError();
        showState('idle');
        emailInput.focus();
    });

    pasteClipboardBtn.addEventListener('click', async () => {
        try {
            if (navigator.clipboard && navigator.clipboard.readText) {
                const clipText = await navigator.clipboard.readText();
                if (clipText) {
                    emailInput.value = clipText;
                    updateTextareaTelemetry();
                    hideInputError();
                    emailInput.focus();
                } else {
                    showInputError('Clipboard is empty.');
                }
            } else {
                showInputError('Clipboard access is restricted by your browser.');
            }
        } catch (err) {
            showInputError('Could not read clipboard. Please paste manually (Cmd+V / Ctrl+V).');
        }
    });

    // =========================================================================
    // 3. Benchmark Sample Loaders
    // =========================================================================

    async function loadSamples() {
        try {
            const resp = await fetch('/api/samples');
            if (resp.ok) {
                benchmarkSamples = await resp.json();
            }
        } catch (e) {
            console.warn('[SpamGuard] Could not fetch sample list:', e);
        }
    }

    document.querySelectorAll('.preset-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleKey = btn.getAttribute('data-sample');
            if (benchmarkSamples && benchmarkSamples[sampleKey]) {
                emailInput.value = benchmarkSamples[sampleKey].content;
                updateTextareaTelemetry();
                hideInputError();

                // Highlight active button briefly
                btn.style.transform = 'scale(0.97)';
                setTimeout(() => { btn.style.transform = ''; }, 150);

                // Auto-trigger analysis for seamless user experience
                triggerAnalysis();
            }
        });
    });

    // =========================================================================
    // 4. State Management (Idle, Loading, Verdict)
    // =========================================================================

    function showState(state) {
        stateIdle.classList.add('hidden');
        stateLoading.classList.add('hidden');
        stateVerdict.classList.add('hidden');

        if (state === 'idle') {
            stateIdle.classList.remove('hidden');
            resultsTimeBadge.textContent = 'READY';
            resultsTimeBadge.className = 'card-top-badge';
        } else if (state === 'loading') {
            stateLoading.classList.remove('hidden');
            resultsTimeBadge.textContent = 'INFERENCE ACTIVE';
            resultsTimeBadge.className = 'card-top-badge badge-blue';
        } else if (state === 'verdict') {
            stateVerdict.classList.remove('hidden');
        }
    }

    function showInputError(msg) {
        inputErrorMsg.textContent = msg;
        inputErrorAlert.classList.remove('hidden');
    }

    function hideInputError() {
        inputErrorAlert.classList.add('hidden');
    }

    // =========================================================================
    // 5. Prediction Execution & Animation Pipeline
    // =========================================================================

    async function triggerAnalysis() {
        const text = emailInput.value.trim();

        // Client-side Validation
        if (!text) {
            showInputError('Please enter email content before starting analysis.');
            emailInput.focus();
            return;
        }

        if (text.length < 5) {
            showInputError('Email content is too short (minimum 5 characters required).');
            emailInput.focus();
            return;
        }

        hideInputError();

        // Set Loading UI
        analyzeBtn.disabled = true;
        analyzeSpinner.classList.remove('hidden');
        analyzeIcon.classList.add('hidden');
        analyzeBtnText.textContent = 'Analyzing Text Stream...';
        showState('loading');

        // Telemetry pipeline step animations
        const step1 = document.getElementById('pipe-step-1');
        const step2 = document.getElementById('pipe-step-2');
        const step3 = document.getElementById('pipe-step-3');

        step1.className = 'pipeline-step step-active';
        step2.className = 'pipeline-step';
        step3.className = 'pipeline-step';

        setTimeout(() => {
            if (step2) step2.className = 'pipeline-step step-active';
        }, 150);

        setTimeout(() => {
            if (step3) step3.className = 'pipeline-step step-active';
        }, 300);

        const startTime = performance.now();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email_text: text })
            });

            const result = await response.json();
            const totalDuration = Math.round(performance.now() - startTime);

            if (!response.ok) {
                showInputError(result.error || 'Failed to complete analysis.');
                showState('idle');
                return;
            }

            // Render result payload
            renderPredictionVerdict(result, totalDuration);

        } catch (error) {
            console.error('[SpamGuard] Inference error:', error);
            showInputError('Network error connecting to SpamGuard backend. Please check server status.');
            showState('idle');
        } finally {
            analyzeBtn.disabled = false;
            analyzeSpinner.classList.add('hidden');
            analyzeIcon.classList.remove('hidden');
            analyzeBtnText.textContent = 'Analyze Email Threat';
        }
    }

    analyzeBtn.addEventListener('click', triggerAnalysis);

    // =========================================================================
    // 6. Verdict Rendering & Threat Indicators
    // =========================================================================

    function renderPredictionVerdict(data, clientDuration) {
        showState('verdict');

        const isSpam = data.is_spam;
        const confidence = data.confidence;
        const spamProb = data.spam_probability;
        const hamProb = data.ham_probability;
        const riskLevel = data.risk_level;

        // 1. Time Badge
        resultsTimeBadge.textContent = `${clientDuration}ms LATENCY`;
        resultsTimeBadge.className = isSpam ? 'card-top-badge badge-purple' : 'card-top-badge badge-blue';

        // 2. Verdict Banner
        verdictBanner.className = `verdict-banner ${isSpam ? 'verdict-spam' : 'verdict-ham'}`;

        if (isSpam) {
            verdictHeading.textContent = 'MALICIOUS SPAM DETECTED';
            verdictIcon.innerHTML = `
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            `;
        } else {
            verdictHeading.textContent = 'VERIFIED LEGITIMATE (HAM)';
            verdictIcon.innerHTML = `
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <polyline points="9 12 11 14 15 10"/>
            `;
        }

        // 3. Risk Level Tag
        verdictRiskTag.textContent = `${riskLevel} RISK`;
        if (riskLevel === 'CRITICAL') {
            verdictRiskTag.className = 'verdict-risk-tag risk-critical';
        } else if (riskLevel === 'HIGH') {
            verdictRiskTag.className = 'verdict-risk-tag risk-high';
        } else if (riskLevel === 'MEDIUM') {
            verdictRiskTag.className = 'verdict-risk-tag risk-medium';
        } else {
            verdictRiskTag.className = 'verdict-risk-tag risk-safe';
        }

        // 4. Confidence Gauge
        verdictConfidence.textContent = `${confidence.toFixed(1)}%`;

        // 5. Probability Bars with CSS transitions
        spamProbNum.textContent = `${spamProb.toFixed(1)}%`;
        hamProbNum.textContent = `${hamProb.toFixed(1)}%`;

        // Trigger transition by applying width
        setTimeout(() => {
            spamProbBar.style.width = `${spamProb}%`;
            hamProbBar.style.width = `${hamProb}%`;
        }, 50);

        // 6. Advisory Guidance
        advisoryText.textContent = data.security_action || 'No anomalous action needed.';

        // 7. Detected Signals List
        signalsList.innerHTML = '';
        const signals = data.detected_signals || [];
        signalsCountBadge.textContent = `${signals.length} Identified`;

        if (signals.length === 0) {
            signalsList.innerHTML = `
                <div class="no-signals-msg">
                    No deceptive phishing keywords, credential harvest vectors, or suspicious URLs detected.
                </div>
            `;
        } else {
            signals.forEach(sig => {
                const item = document.createElement('div');
                item.className = 'signal-item';
                item.innerHTML = `
                    <span class="signal-label">${escapeHtml(sig.label)}</span>
                    <span class="signal-matched">${escapeHtml(sig.matched)}</span>
                `;
                signalsList.appendChild(item);
            });
        }

        // 8. Preprocessing inspection
        preprocessedSnippet.textContent = data.preprocessed_snippet || '...';
        metaLatency.textContent = `${data.metrics?.inference_time_ms || clientDuration} ms`;
        metaTokens.textContent = data.metrics?.preprocessed_tokens || 0;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // =========================================================================
    // 7. Empirical Metrics Fetcher & Hydration
    // =========================================================================

    async function loadMetrics() {
        try {
            const resp = await fetch('/api/metrics');
            if (!resp.ok) return;

            const data = await resp.json();
            const metrics = data.metrics || {};
            const cm = data.confusion_matrix || {};

            // Update Hero stats
            if (heroAcc && metrics.accuracy_percent) heroAcc.textContent = `${metrics.accuracy_percent}%`;
            if (heroPrec && metrics.precision_percent) heroPrec.textContent = `${metrics.precision_percent}%`;

            // Update KPI cards
            if (kpiAccuracy && metrics.accuracy_percent) kpiAccuracy.textContent = `${metrics.accuracy_percent}%`;
            if (kpiPrecision && metrics.precision_percent) kpiPrecision.textContent = `${metrics.precision_percent}%`;
            if (kpiRecall && metrics.recall_percent) kpiRecall.textContent = `${metrics.recall_percent}%`;
            if (kpiF1 && metrics.f1_percent) kpiF1.textContent = `${metrics.f1_percent}%`;

            // Update Confusion Matrix
            if (cmTn && cm.true_negatives !== undefined) cmTn.textContent = cm.true_negatives.toLocaleString();
            if (cmFp && cm.false_positives !== undefined) cmFp.textContent = cm.false_positives.toLocaleString();
            if (cmFn && cm.false_negatives !== undefined) cmFn.textContent = cm.false_negatives.toLocaleString();
            if (cmTp && cm.true_positives !== undefined) cmTp.textContent = cm.true_positives.toLocaleString();

            // Render dynamic top spam tokens
            if (topSpamVocabList && data.top_spam_keywords) {
                topSpamVocabList.innerHTML = '';
                data.top_spam_keywords.slice(0, 10).forEach(kw => {
                    const span = document.createElement('span');
                    span.className = 'vocab-pill spam-token';
                    span.innerHTML = `${escapeHtml(kw.word)} <em>+${kw.log_ratio}</em>`;
                    topSpamVocabList.appendChild(span);
                });
            }

            // Render dynamic top ham tokens
            if (topHamVocabList && data.top_ham_keywords) {
                topHamVocabList.innerHTML = '';
                data.top_ham_keywords.slice(0, 10).forEach(kw => {
                    const span = document.createElement('span');
                    span.className = 'vocab-pill ham-token';
                    span.innerHTML = `${escapeHtml(kw.word)} <em>${kw.log_ratio}</em>`;
                    topHamVocabList.appendChild(span);
                });
            }

        } catch (e) {
            console.warn('[SpamGuard] Could not load live metrics:', e);
        }
    }

    // =========================================================================
    // 8. Health Check Verification
    // =========================================================================

    async function checkHealth() {
        try {
            const resp = await fetch('/api/health');
            if (resp.ok) {
                const health = await resp.json();
                if (health.model_loaded) {
                    statusLabel.textContent = 'ML Engine Ready';
                    systemStatusPill.className = 'system-status-chip';
                } else {
                    statusLabel.textContent = 'Model Degraded';
                    systemStatusPill.style.color = '#F59E0B';
                }
            }
        } catch (e) {
            statusLabel.textContent = 'Offline';
            systemStatusPill.style.color = '#FF0055';
        }
    }

    // =========================================================================
    // 9. Smooth Scroll & Keyboard Shortcuts
    // =========================================================================

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });

                // Update active nav link
                document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
                if (this.classList.contains('nav-link')) {
                    this.classList.add('active');
                }
            }
        });
    });

    // Keyboard shortcut: Ctrl+Enter / Cmd+Enter inside textarea submits analysis
    emailInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            triggerAnalysis();
        }
    });

    // Initialize System Data
    loadSamples();
    loadMetrics();
    checkHealth();
    updateTextareaTelemetry();
});

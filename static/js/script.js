/**
 * =============================================================================
 * SPAM EMAIL DETECTION – THREAT DETECTION & ANALYTICS DASHBOARD CONTROLLER
 * =============================================================================
 * Author: Spam Email Detection Engineering Team
 * Objective: Multi-model telemetry, asynchronous API calls to /api/predict
 *            and /api/metrics, interactive presets, URL threat visualizations,
 *            distinct phishing heuristics, and interactive confusion matrices.
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    // Input Mode Elements
    const tabStructured = document.getElementById('tab-structured');
    const tabRaw = document.getElementById('tab-raw');
    const containerStructured = document.getElementById('container-structured');
    const containerRaw = document.getElementById('container-raw');

    // Input Fields
    const emailSubject = document.getElementById('email-subject');
    const emailSender = document.getElementById('email-sender');
    const emailBody = document.getElementById('email-body');
    const emailInput = document.getElementById('email-input');

    // Model Selector Elements
    const chooseBestModelBtn = document.getElementById('choose-best-model');
    const chooseBaselineModelBtn = document.getElementById('choose-baseline-model');
    let selectedModel = 'best'; // 'best' (Linear SVM) or 'baseline' (Multinomial NB)

    // Telemetry Elements
    const charCounter = document.getElementById('char-counter');
    const wordCounter = document.getElementById('word-counter');
    const urlCounter = document.getElementById('url-counter');

    // Buttons & Alerts
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

    // ML Verdict Banner Elements
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictModelUsed = document.getElementById('verdict-model-used');
    const verdictIconWrap = document.getElementById('verdict-icon-wrap');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictHeading = document.getElementById('verdict-heading');
    const verdictRiskTag = document.getElementById('verdict-risk-tag');
    const verdictConfidence = document.getElementById('verdict-confidence');
    const spamProbNum = document.getElementById('spam-prob-num');
    const spamProbBar = document.getElementById('spam-prob-bar');
    const hamProbNum = document.getElementById('ham-prob-num');
    const hamProbBar = document.getElementById('ham-prob-bar');

    // Phishing & Threat Assessment Elements
    const phishingScoreNum = document.getElementById('phishing-score-num');
    const phishingVerdictText = document.getElementById('phishing-verdict-text');
    const phishingSeverityTag = document.getElementById('phishing-severity-tag');
    const urlThreatBadge = document.getElementById('url-threat-badge');
    const urlFindingsList = document.getElementById('url-findings-list');
    const senderFindingsText = document.getElementById('sender-findings-text');
    const signalsList = document.getElementById('signals-list');
    const signalsCountBadge = document.getElementById('signals-count-badge');
    const disclaimerText = document.getElementById('disclaimer-text');

    // Preprocessed Preview Elements
    const preprocessedSnippet = document.getElementById('preprocessed-snippet');
    const metaLatency = document.getElementById('meta-latency');
    const metaTokens = document.getElementById('meta-tokens');
    const metaModelName = document.getElementById('meta-model-name');

    // Analytics Dashboard & Telemetry Elements
    const heroAcc = document.getElementById('hero-acc');
    const heroF1 = document.getElementById('hero-f1');
    const heroPrec = document.getElementById('hero-prec');
    const kpiAccuracy = document.getElementById('kpi-accuracy');
    const kpiPrecision = document.getElementById('kpi-precision');
    const kpiRecall = document.getElementById('kpi-recall');
    const kpiF1 = document.getElementById('kpi-f1');
    const modelComparisonTbody = document.getElementById('model-comparison-tbody');

    // Confusion Matrix Elements
    const matrixModelSelector = document.getElementById('matrix-model-selector');
    const matrixModelSubtitle = document.getElementById('matrix-model-subtitle');
    const cmTn = document.getElementById('cm-tn');
    const cmFp = document.getElementById('cm-fp');
    const cmFn = document.getElementById('cm-fn');
    const cmTp = document.getElementById('cm-tp');
    const cmSpecificity = document.getElementById('cm-specificity');
    const cmFpRate = document.getElementById('cm-fp-rate');
    const cmFnRate = document.getElementById('cm-fn-rate');
    const cmSensitivity = document.getElementById('cm-sensitivity');

    // Dataset Telemetry
    const dsTotal = document.getElementById('ds-total');
    const dsHam = document.getElementById('ds-ham');
    const dsSpam = document.getElementById('ds-spam');
    const dsTrain = document.getElementById('ds-train');
    const dsTest = document.getElementById('ds-test');
    const dsVocab = document.getElementById('ds-vocab');
    const topSpamVocabList = document.getElementById('top-spam-vocab-list');
    const topHamVocabList = document.getElementById('top-ham-vocab-list');

    // Navbar Status Pill
    const statusLabel = document.getElementById('status-label');

    // Cached Data
    let benchmarkSamples = {};
    let cachedMetrics = null;
    let currentInputMode = 'structured'; // 'structured' or 'raw'

    // =========================================================================
    // 1. Input Mode Tabs Switching
    // =========================================================================

    function setInputMode(mode) {
        currentInputMode = mode;
        if (mode === 'structured') {
            tabStructured.classList.add('active');
            tabStructured.setAttribute('aria-selected', 'true');
            tabRaw.classList.remove('active');
            tabRaw.setAttribute('aria-selected', 'false');

            containerStructured.classList.remove('hidden');
            containerRaw.classList.add('hidden');
        } else {
            tabRaw.classList.add('active');
            tabRaw.setAttribute('aria-selected', 'true');
            tabStructured.classList.remove('active');
            tabStructured.setAttribute('aria-selected', 'false');

            containerRaw.classList.remove('hidden');
            containerStructured.classList.add('hidden');

            // Synchronize structured inputs into raw if raw is empty
            if (!emailInput.value.trim() && (emailSubject.value || emailBody.value)) {
                const parts = [];
                if (emailSubject.value.trim()) parts.push(`Subject: ${emailSubject.value.trim()}`);
                if (emailSender.value.trim()) parts.push(`From: ${emailSender.value.trim()}`);
                if (emailBody.value.trim()) parts.push(emailBody.value.trim());
                emailInput.value = parts.join('\n\n');
            }
        }
        updateTelemetry();
    }

    tabStructured.addEventListener('click', () => setInputMode('structured'));
    tabRaw.addEventListener('click', () => setInputMode('raw'));

    // =========================================================================
    // 2. Model Selection Toggle (Linear SVM Best vs Naive Bayes Baseline)
    // =========================================================================

    function selectModel(modelKey) {
        selectedModel = modelKey;
        if (modelKey === 'best') {
            chooseBestModelBtn.classList.add('active');
            chooseBaselineModelBtn.classList.remove('active');
            if (statusLabel) statusLabel.textContent = 'Active: Linear SVM (99.29% F1)';
        } else {
            chooseBaselineModelBtn.classList.add('active');
            chooseBestModelBtn.classList.remove('active');
            if (statusLabel) statusLabel.textContent = 'Active: Multinomial NB (97.99% F1)';
        }
    }

    chooseBestModelBtn.addEventListener('click', () => selectModel('best'));
    chooseBaselineModelBtn.addEventListener('click', () => selectModel('baseline'));

    // =========================================================================
    // 3. Telemetry Counter Calculations
    // =========================================================================

    function getCombinedText() {
        if (currentInputMode === 'structured') {
            const parts = [];
            if (emailSubject.value.trim()) parts.push(emailSubject.value.trim());
            if (emailBody.value.trim()) parts.push(emailBody.value.trim());
            return parts.join(' ');
        }
        return emailInput.value || '';
    }

    function updateTelemetry() {
        const text = getCombinedText();
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        const urlMatches = text.match(/https?:\/\/[^\s<>"]+|www\.[^\s<>"]+/gi) || [];

        charCounter.textContent = chars.toLocaleString();
        wordCounter.textContent = words.toLocaleString();
        urlCounter.textContent = urlMatches.length.toString();

        if (chars > 0) {
            hideInputError();
        }
    }

    [emailSubject, emailSender, emailBody, emailInput].forEach(elem => {
        if (elem) {
            elem.addEventListener('input', updateTelemetry);
            elem.addEventListener('keyup', updateTelemetry);
        }
    });

    // =========================================================================
    // 4. Toolbar Actions (Clear & Paste)
    // =========================================================================

    clearInputBtn.addEventListener('click', () => {
        emailSubject.value = '';
        emailSender.value = '';
        emailBody.value = '';
        emailInput.value = '';
        updateTelemetry();
        hideInputError();
        showState('idle');
        if (currentInputMode === 'structured') {
            emailSubject.focus();
        } else {
            emailInput.focus();
        }
    });

    pasteClipboardBtn.addEventListener('click', async () => {
        try {
            if (navigator.clipboard && navigator.clipboard.readText) {
                const clipText = await navigator.clipboard.readText();
                if (clipText) {
                    if (currentInputMode === 'structured') {
                        // Check if clipboard contains a Subject line
                        const subjMatch = clipText.match(/^Subject:\s*(.+)$/m);
                        if (subjMatch) {
                            emailSubject.value = subjMatch[1].trim();
                            emailBody.value = clipText.replace(/^Subject:\s*.+$/m, '').trim();
                        } else {
                            emailBody.value = clipText;
                        }
                    } else {
                        emailInput.value = clipText;
                    }
                    updateTelemetry();
                    hideInputError();
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
    // 5. Benchmark Sample Loaders
    // =========================================================================

    async function loadSamples() {
        try {
            const resp = await fetch('/api/samples');
            if (resp.ok) {
                benchmarkSamples = await resp.json();
            }
        } catch (e) {
            console.warn('[Spam Email Detection] Could not fetch sample list:', e);
        }
    }

    document.querySelectorAll('.preset-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleKey = btn.getAttribute('data-sample');
            if (benchmarkSamples && benchmarkSamples[sampleKey]) {
                const sample = benchmarkSamples[sampleKey];

                // Populate both structured and raw fields
                emailSubject.value = sample.subject || '';
                emailSender.value = sample.sender || '';
                emailBody.value = sample.body || '';
                emailInput.value = sample.content || '';

                updateTelemetry();
                hideInputError();

                // Highlight preset button briefly
                btn.style.transform = 'scale(0.97)';
                setTimeout(() => { btn.style.transform = ''; }, 150);

                // Auto-trigger analysis
                triggerAnalysis();
            }
        });
    });

    // =========================================================================
    // 6. State Management
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
            resultsTimeBadge.textContent = 'THREAT SCAN ACTIVE';
            resultsTimeBadge.className = 'card-top-badge badge-blue';
        } else if (state === 'verdict') {
            stateVerdict.classList.remove('hidden');
            resultsTimeBadge.textContent = 'EVALUATED';
            resultsTimeBadge.className = 'card-top-badge badge-purple';
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
    // 7. Prediction & Threat Inspection Execution
    // =========================================================================

    async function triggerAnalysis() {
        let payload = {};

        if (currentInputMode === 'structured') {
            const subject = emailSubject.value.trim();
            const sender = emailSender.value.trim();
            const body = emailBody.value.trim();

            if (!body && !subject) {
                showInputError('Please enter email subject or body before starting analysis.');
                emailBody.focus();
                return;
            }

            if ((subject + body).length < 5) {
                showInputError('Email content is too short (minimum 5 characters required).');
                emailBody.focus();
                return;
            }

            payload = {
                subject: subject,
                sender: sender,
                body: body,
                model: selectedModel
            };
        } else {
            const rawText = emailInput.value.trim();

            if (!rawText) {
                showInputError('Please enter email text before starting analysis.');
                emailInput.focus();
                return;
            }

            if (rawText.length < 5) {
                showInputError('Email content is too short (minimum 5 characters required).');
                emailInput.focus();
                return;
            }

            payload = {
                email_text: rawText,
                model: selectedModel
            };
        }

        hideInputError();

        // UI Loading state
        analyzeBtn.disabled = true;
        analyzeSpinner.classList.remove('hidden');
        analyzeIcon.classList.add('hidden');
        analyzeBtnText.textContent = 'Scanning Message Telemetry...';
        showState('loading');

        // Pipeline animation markers
        const step1 = document.getElementById('pipe-step-1');
        const step2 = document.getElementById('pipe-step-2');
        const step3 = document.getElementById('pipe-step-3');

        step1.className = 'pipeline-step step-active';
        step2.className = 'pipeline-step';
        step3.className = 'pipeline-step';

        setTimeout(() => { if (step2) step2.className = 'pipeline-step step-active'; }, 150);
        setTimeout(() => { if (step3) step3.className = 'pipeline-step step-active'; }, 300);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                showInputError(result.error || 'Failed to complete threat analysis.');
                showState('idle');
                return;
            }

            renderThreatVerdict(result);
            showState('verdict');

        } catch (err) {
            console.error('[Spam Email Detection] Analysis error:', err);
            showInputError('Network error connecting to Spam Email Detection backend.');
            showState('idle');
        } finally {
            analyzeBtn.disabled = false;
            analyzeSpinner.classList.add('hidden');
            analyzeIcon.classList.remove('hidden');
            analyzeBtnText.textContent = 'Execute Multi-Vector Analysis';
        }
    }

    analyzeBtn.addEventListener('click', triggerAnalysis);

    // =========================================================================
    // 8. Render Verdict & Threat Intelligence Output
    // =========================================================================

    function renderThreatVerdict(data) {
        const isSpam = data.is_spam;
        const modelName = data.model_classification ? data.model_classification.model_name : 'Linear SVM';
        const spamPct = data.spam_probability;
        const hamPct = data.ham_probability;
        const confidence = data.confidence;
        const riskLevel = data.risk_level;

        // 1. Model Pretitle
        if (verdictModelUsed) {
            verdictModelUsed.textContent = `CLASSIFIER :: ${modelName.toUpperCase()}`;
        }

        // 2. Verdict Banner & Risk Tag
        if (isSpam) {
            verdictBanner.className = 'verdict-banner verdict-spam';
            verdictHeading.textContent = 'SPAM DETECTED';
            verdictHeading.className = 'verdict-heading text-danger';

            verdictRiskTag.textContent = `${riskLevel} RISK`;
            verdictRiskTag.className = 'verdict-risk-tag tag-critical';

            verdictConfidence.textContent = `${confidence.toFixed(1)}%`;
            verdictIconWrap.className = 'verdict-icon-wrap icon-spam';
            verdictIcon.innerHTML = `
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            `;
        } else {
            verdictBanner.className = 'verdict-banner verdict-ham';
            verdictHeading.textContent = 'UNLIKELY SPAM';
            verdictHeading.className = 'verdict-heading text-success';

            verdictRiskTag.textContent = 'LOW RISK';
            verdictRiskTag.className = 'verdict-risk-tag tag-low';

            verdictConfidence.textContent = `${confidence.toFixed(1)}%`;
            verdictIconWrap.className = 'verdict-icon-wrap icon-ham';
            verdictIcon.innerHTML = `
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <polyline points="9 12 11 14 15 10"/>
            `;
        }

        // 3. Probability Bars Animation
        spamProbNum.textContent = `${spamPct.toFixed(1)}%`;
        hamProbNum.textContent = `${hamPct.toFixed(1)}%`;

        setTimeout(() => {
            spamProbBar.style.width = `${Math.min(100, Math.max(0, spamPct))}%`;
            hamProbBar.style.width = `${Math.min(100, Math.max(0, hamPct))}%`;
        }, 50);

        // 4. Phishing Assessment Section
        const phishing = data.phishing_assessment || {};
        const phishingScore = phishing.phishing_score || 0;
        const phishingRisk = phishing.phishing_risk_level || 'LOW';

        if (phishingScoreNum) phishingScoreNum.textContent = phishingScore;
        if (phishingVerdictText) phishingVerdictText.textContent = phishing.phishing_verdict || 'Standard analysis complete';

        if (phishingSeverityTag) {
            phishingSeverityTag.textContent = `${phishingRisk} RISK`;
            phishingSeverityTag.className = `phishing-severity-tag tag-${phishingRisk.toLowerCase()}`;
        }

        // 5. URL Findings List
        const urlAnalysis = phishing.url_analysis || {};
        const urls = urlAnalysis.all_urls || [];
        if (urlThreatBadge) {
            urlThreatBadge.textContent = `${urls.length} Link${urls.length === 1 ? '' : 's'}`;
        }

        if (urlFindingsList) {
            if (urls.length === 0) {
                urlFindingsList.innerHTML = `<div class="finding-empty">No external hyperlinks found in email content.</div>`;
            } else {
                urlFindingsList.innerHTML = urls.map(u => {
                    const isCrit = u.severity === 'CRITICAL';
                    const isWarn = u.is_suspicious;
                    const borderClass = isCrit ? 'item-critical' : (isWarn ? 'item-warn' : '');
                    const reasonsHtml = u.reasons.map(r => `<div>• ${r}</div>`).join('');
                    return `
                        <div class="url-item ${borderClass}">
                            <div class="url-item-head">
                                <span class="url-link">${escapeHtml(u.url)}</span>
                                <span class="phishing-severity-tag tag-${u.severity.toLowerCase()}">${u.severity}</span>
                            </div>
                            <div class="url-reasons-wrap">${reasonsHtml}</div>
                        </div>
                    `;
                }).join('');
            }
        }

        // 6. Sender Inspection Display
        const senderEval = phishing.sender_analysis || {};
        if (senderFindingsText) {
            if (!senderEval.provided) {
                senderFindingsText.innerHTML = `<span class="text-muted">No sender email address provided.</span>`;
            } else if (senderEval.is_suspicious) {
                const flagsHtml = senderEval.flags.map(f => `<span class="text-danger">⚠️ ${f}</span>`).join('<br>');
                senderFindingsText.innerHTML = `<div><strong>${escapeHtml(senderEval.address)}</strong></div>${flagsHtml}`;
            } else {
                senderFindingsText.innerHTML = `<span class="text-success">✓ ${escapeHtml(senderEval.address)} (Standard RFC address)</span>`;
            }
        }

        // 7. Threat Signals & Keywords List
        const signals = phishing.threat_signals || [];
        if (signalsCountBadge) {
            signalsCountBadge.textContent = `${signals.length} Found`;
        }

        if (signalsList) {
            if (signals.length === 0) {
                signalsList.innerHTML = `<div class="finding-empty">Zero deceptive phishing or social engineering keywords found.</div>`;
            } else {
                signalsList.innerHTML = signals.map(sig => `
                    <div class="signal-item">
                        <span class="signal-dot dot-${sig.severity ? sig.severity.toLowerCase() : 'medium'}"></span>
                        <div class="signal-content">
                            <div class="signal-label">${escapeHtml(sig.label)} <span class="signal-cat">(${escapeHtml(sig.category || 'Threat')})</span></div>
                            <div class="signal-matched">Matched: <code class="inline-code">"${escapeHtml(sig.matched)}"</code></div>
                        </div>
                    </div>
                `).join('');
            }
        }

        // 8. Safety Disclaimer Notice
        if (disclaimerText && data.safety_disclaimer) {
            disclaimerText.innerHTML = `<strong>Statistical Assessment Only:</strong> ${escapeHtml(data.safety_disclaimer)}`;
        }

        // 9. Preprocessed Tokens Preview
        if (preprocessedSnippet) {
            preprocessedSnippet.textContent = data.preprocessed_snippet || 'None';
        }
        if (metaLatency) {
            metaLatency.textContent = `${data.metrics ? data.metrics.inference_time_ms : 0} ms`;
        }
        if (metaTokens) {
            metaTokens.textContent = data.metrics ? data.metrics.preprocessed_tokens : 0;
        }
        if (metaModelName) {
            metaModelName.textContent = modelName;
        }
    }

    // =========================================================================
    // 9. Fetch & Render Analytics Dashboard Metrics (/api/metrics)
    // =========================================================================

    async function fetchMetrics() {
        try {
            const resp = await fetch('/api/metrics');
            if (!resp.ok) return;

            const data = await resp.json();
            cachedMetrics = data;

            // 1. Hero Ticker & KPI Cards (Best Model)
            const m = data.metrics || {};
            if (heroAcc) heroAcc.textContent = `${m.accuracy_percent || 99.29}%`;
            if (heroF1) heroF1.textContent = `${m.f1_percent || 99.15}%`;
            if (heroPrec) heroPrec.textContent = `${m.precision_percent || 99.71}%`;

            if (kpiAccuracy) kpiAccuracy.textContent = `${m.accuracy_percent || 99.29}%`;
            if (kpiPrecision) kpiPrecision.textContent = `${m.precision_percent || 99.71}%`;
            if (kpiRecall) kpiRecall.textContent = `${m.recall_percent || 98.58}%`;
            if (kpiF1) kpiF1.textContent = `${m.f1_percent || 99.15}%`;

            // 2. Render 4-Model Comparison Table
            renderModelComparisonTable(data.model_comparison || []);

            // 3. Render Confusion Matrix for Best Model initially
            const bestModelName = data.best_model_name || 'Linear SVM';
            renderConfusionMatrixForModel(bestModelName);

            // 4. Render Dataset Statistics
            if (dsTotal) dsTotal.textContent = (data.total_dataset_samples || 4235).toLocaleString();
            if (dsHam) {
                const ham = data.ham_samples || 2471;
                const tot = data.total_dataset_samples || 4235;
                dsHam.textContent = `${ham.toLocaleString()} (${((ham/tot)*100).toFixed(1)}%)`;
            }
            if (dsSpam) {
                const spam = data.spam_samples || 1764;
                const tot = data.total_dataset_samples || 4235;
                dsSpam.textContent = `${spam.toLocaleString()} (${((spam/tot)*100).toFixed(1)}%)`;
            }
            if (dsTrain) dsTrain.textContent = `${(data.train_samples || 3388).toLocaleString()} (80%)`;
            if (dsTest) dsTest.textContent = `${(data.test_samples || 847).toLocaleString()} (20%)`;
            if (dsVocab) dsVocab.textContent = (data.vocabulary_size || 15000).toLocaleString();

            // 5. Render Explainable AI Top Vocabulary
            renderTopVocabulary(data.top_spam_keywords || [], data.top_ham_keywords || []);

        } catch (e) {
            console.warn('[Spam Email Detection] Could not load empirical metrics:', e);
        }
    }

    // =========================================================================
    // 10. Model Comparison Table Renderer
    // =========================================================================

    function renderModelComparisonTable(models) {
        if (!modelComparisonTbody || !models.length) return;

        modelComparisonTbody.innerHTML = models.map(m => {
            const isBest = m.is_best;
            const isBaseline = m.is_baseline;
            const rowClass = isBest ? 'model-row-best' : '';

            let badgeHtml = `<span class="badge-cand">Evaluated</span>`;
            if (isBest) {
                badgeHtml = `<span class="badge-best">★ Best Performer</span>`;
            } else if (isBaseline) {
                badgeHtml = `<span class="badge-base">Baseline</span>`;
            }

            return `
                <tr class="${rowClass}">
                    <td><strong>${escapeHtml(m.model_name)}</strong></td>
                    <td><span class="text-secondary">${escapeHtml(m.family || 'ML Classifier')}</span></td>
                    <td><strong class="text-cyan">${m.accuracy_percent.toFixed(2)}%</strong></td>
                    <td><span class="text-success">${m.precision_percent.toFixed(2)}%</span></td>
                    <td><span class="text-secondary">${m.recall_percent.toFixed(2)}%</span></td>
                    <td><strong class="${isBest ? 'text-cyan' : 'text-primary'}">${m.f1_percent.toFixed(2)}%</strong></td>
                    <td><code class="inline-code">${m.fit_time_seconds ? m.fit_time_seconds.toFixed(3) : 0}s</code></td>
                    <td>${badgeHtml}</td>
                </tr>
            `;
        }).join('');
    }

    // =========================================================================
    // 11. Interactive Confusion Matrix Renderer
    // =========================================================================

    function renderConfusionMatrixForModel(modelName) {
        if (!cachedMetrics || !cachedMetrics.model_comparison) return;

        const model = cachedMetrics.model_comparison.find(m => m.model_name === modelName);
        if (!model || !model.confusion_matrix) return;

        const cm = model.confusion_matrix;
        const tn = cm.true_negatives;
        const fp = cm.false_positives;
        const fn = cm.false_negatives;
        const tp = cm.true_positives;

        const specificity = (tn + fp) > 0 ? ((tn / (tn + fp)) * 100).toFixed(1) : 0;
        const fpRate = (tn + fp) > 0 ? ((fp / (tn + fp)) * 100).toFixed(1) : 0;
        const fnRate = (fn + tp) > 0 ? ((fn / (fn + tp)) * 100).toFixed(1) : 0;
        const sensitivity = (fn + tp) > 0 ? ((tp / (fn + tp)) * 100).toFixed(1) : 0;

        if (cmTn) cmTn.textContent = tn;
        if (cmFp) cmFp.textContent = fp;
        if (cmFn) cmFn.textContent = fn;
        if (cmTp) cmTp.textContent = tp;

        if (cmSpecificity) cmSpecificity.textContent = `${specificity}% Specificity`;
        if (cmFpRate) cmFpRate.textContent = `${fpRate}% False Alarm Rate`;
        if (cmFnRate) cmFnRate.textContent = `${fnRate}% Miss Rate`;
        if (cmSensitivity) cmSensitivity.textContent = `${sensitivity}% Catch Rate`;

        if (matrixModelSubtitle) {
            const role = model.is_best ? '(Best Performer)' : (model.is_baseline ? '(Baseline)' : '(Candidate)');
            matrixModelSubtitle.innerHTML = `Showing independent test set verification for <strong>${escapeHtml(modelName)}</strong> ${role}:`;
        }

        // Update active pill
        document.querySelectorAll('.matrix-pill').forEach(pill => {
            if (pill.getAttribute('data-cm-model') === modelName) {
                pill.classList.add('active');
            } else {
                pill.classList.remove('active');
            }
        });
    }

    // Confusion Matrix Pill Listeners
    if (matrixModelSelector) {
        matrixModelSelector.addEventListener('click', (e) => {
            const pill = e.target.closest('.matrix-pill');
            if (pill) {
                const modelName = pill.getAttribute('data-cm-model');
                renderConfusionMatrixForModel(modelName);
            }
        });
    }

    // =========================================================================
    // 12. Top Vocabulary Pills Renderer
    // =========================================================================

    function renderTopVocabulary(spamKeywords, hamKeywords) {
        if (topSpamVocabList && spamKeywords.length) {
            topSpamVocabList.innerHTML = spamKeywords.slice(0, 10).map(k => {
                const score = k.weight !== undefined ? k.weight : k.log_ratio;
                const sign = score > 0 ? '+' : '';
                return `<span class="vocab-pill spam-token">${escapeHtml(k.word)} <em>${sign}${score.toFixed(2)}</em></span>`;
            }).join('');
        }

        if (topHamVocabList && hamKeywords.length) {
            topHamVocabList.innerHTML = hamKeywords.slice(0, 10).map(k => {
                const score = k.weight !== undefined ? k.weight : k.log_ratio;
                const sign = score > 0 ? '+' : '';
                return `<span class="vocab-pill ham-token">${escapeHtml(k.word)} <em>${sign}${score.toFixed(2)}</em></span>`;
            }).join('');
        }
    }

    // Helper: Escape HTML
    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // =========================================================================
    // 13. Initialization
    // =========================================================================
    loadSamples();
    fetchMetrics();
    updateTelemetry();
});

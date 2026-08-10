document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const ticketText = document.getElementById('ticketText');
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');

    // Results elements
    const queueResult = document.getElementById('queueResult');
    const priorityResult = document.getElementById('priorityResult');
    const priorityIcon = document.getElementById('priorityIcon');
    const confidenceResult = document.getElementById('confidenceResult');
    const confidenceFill = document.getElementById('confidenceFill');
    const responseText = document.getElementById('responseText');

    // Actions
    const copyBtn = document.getElementById('copyBtn');
    const regenerateBtn = document.getElementById('regenerateBtn');
    const editBtn = document.getElementById('editBtn');

    // Navigation logic
    const navItems = document.querySelectorAll('.nav-item[data-view]');
    const viewSections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            // Remove active state from all tabs
            navItems.forEach(nav => nav.classList.remove('active'));
            // Hide all views
            viewSections.forEach(view => view.classList.add('hidden'));
            
            // Activate clicked tab
            item.classList.add('active');
            // Show corresponding view
            const viewId = item.getAttribute('data-view');
            document.getElementById(viewId).classList.remove('hidden');
        });
    });

    // Auto-resize textarea function
    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = el.scrollHeight + 'px';
    }

    ticketText.addEventListener('input', () => autoResize(ticketText));
    responseText.addEventListener('input', () => autoResize(responseText));

    analyzeBtn.addEventListener('click', async () => {
        const text = ticketText.value.trim();
        if (!text) {
            alert('Please enter a ticket first!');
            return;
        }

        // UI State: Loading
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticket_text: text })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to process ticket');
            }

            const data = await response.json();

            // Populate Metrics
            queueResult.textContent = data.queue;
            priorityResult.textContent = data.priority;

            // Priority color coding
            priorityResult.className = '';
            priorityIcon.className = 'metric-icon';
            if (data.priority.toLowerCase() === 'high') {
                priorityResult.classList.add('priority-high');
                priorityIcon.classList.add('red');
            } else if (data.priority.toLowerCase() === 'medium') {
                priorityResult.classList.add('priority-medium');
                priorityIcon.style.background = '#fef3c7';
                priorityIcon.style.color = '#f59e0b';
            } else {
                priorityResult.classList.add('priority-low');
                priorityIcon.classList.add('green');
            }

            const confPercent = Math.round(data.confidence * 100);
            confidenceResult.textContent = `${confPercent}%`;

            // UI State: Show Results
            loadingState.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            analyzeBtn.disabled = false;

            // Animate progress bar
            setTimeout(() => {
                confidenceFill.style.width = `${confPercent}%`;
                // Color grading for confidence
                if (confPercent > 85) confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
                else if (confPercent > 60) confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
                else confidenceFill.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
            }, 100);

            // Typewriter effect for response
            typeWriter(data.suggested_response, responseText);

        } catch (error) {
            alert(error.message);
            loadingState.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    // Typewriter utility
    function typeWriter(text, element, i = 0) {
        if (i === 0) element.value = '';
        if (i < text.length) {
            element.value += text.charAt(i);
            autoResize(element);
            setTimeout(() => typeWriter(text, element, i + 1), 10);
        }
    }

    // Action Buttons
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(responseText.value);
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✅ Copied!';
        setTimeout(() => copyBtn.textContent = originalText, 2000);
    });

    regenerateBtn.addEventListener('click', () => {
        analyzeBtn.click(); // Simple re-trigger for now
    });

    editBtn.addEventListener('click', () => {
        const isReadonly = responseText.hasAttribute('readonly');
        if (isReadonly) {
            responseText.removeAttribute('readonly');
            responseText.focus();
            editBtn.textContent = '💾 Save';
            responseText.style.border = '1px solid #60a5fa';
            responseText.style.background = 'rgba(15, 23, 42, 0.8)';
        } else {
            responseText.setAttribute('readonly', true);
            editBtn.textContent = '✏️ Edit';
            responseText.style.border = 'none';
            responseText.style.background = 'transparent';
        }
    });
});

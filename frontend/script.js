/**
 * AI Study Assistant — Frontend Logic
 * 
 * Handles: file upload, document management, chat, and UI updates.
 * Communicates with the FastAPI backend via fetch().
 */

const API_BASE = '/api';

// ── State ────────────────────────────────────────────────────────
let chatHistory = [];

// ── DOM Elements ────────────────────────────────────────────────
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const docList = document.getElementById('docList');
const scopeSelect = document.getElementById('scopeSelect');
const modeSelect = document.getElementById('modeSelect');
const chatContainer = document.getElementById('chatContainer');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const healthDot = document.getElementById('healthDot');

// ── Initialize ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadDocuments();
    setupEventListeners();
});

function setupEventListeners() {
    // File upload
    fileInput.addEventListener('change', handleFileUpload);

    // Send question
    sendBtn.addEventListener('click', sendQuestion);

    // Enter to send (Shift+Enter for new line)
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    // Auto-resize textarea
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 150) + 'px';
    });
}

// ── Health Check ────────────────────────────────────────────────
async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();

        if (data.ollama?.ollama_running && data.ollama?.model_available) {
            healthDot.className = 'health-dot ok';
            healthDot.title = `Connected — Model: ${data.ollama.configured_model}`;
        } else if (data.ollama?.ollama_running) {
            healthDot.className = 'health-dot warn';
            healthDot.title = `Ollama running but model "${data.ollama.configured_model}" not found. Available: ${data.ollama.available_models?.join(', ') || 'none'}`;
        } else {
            healthDot.className = 'health-dot err';
            healthDot.title = 'Ollama not running. Start with: ollama serve';
        }
    } catch {
        healthDot.className = 'health-dot err';
        healthDot.title = 'Cannot connect to backend server.';
    }
}

// ── File Upload ─────────────────────────────────────────────────
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Validate
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Only PDF files are supported.');
        fileInput.value = '';
        return;
    }

    // Show progress
    uploadProgress.hidden = false;
    progressFill.style.width = '10%';
    progressText.textContent = 'Uploading...';
    sendBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', file);

        progressFill.style.width = '30%';
        progressText.textContent = 'Processing document...';

        const resp = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });

        progressFill.style.width = '90%';

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await resp.json();

        progressFill.style.width = '100%';
        progressText.textContent = `✓ ${data.message}`;

        // Refresh document list
        await loadDocuments();

        // Add system message
        addMessage(
            `📄 **${file.name}** uploaded successfully!\n\n` +
            `• ${data.document.page_count} pages\n` +
            `• ${data.document.chunk_count} chunks created\n\n` +
            `You can now ask questions about this document.`,
            'assistant'
        );

    } catch (err) {
        progressFill.style.width = '100%';
        progressFill.style.background = 'var(--danger)';
        progressText.textContent = `✗ ${err.message}`;
    } finally {
        sendBtn.disabled = false;
        fileInput.value = '';

        // Hide progress after delay
        setTimeout(() => {
            uploadProgress.hidden = true;
            progressFill.style.width = '0%';
            progressFill.style.background = '';
        }, 3000);
    }
}

// ── Load Documents ──────────────────────────────────────────────
async function loadDocuments() {
    try {
        const resp = await fetch(`${API_BASE}/documents`);
        const data = await resp.json();
        const docs = data.documents || [];

        // Update document list
        if (docs.length === 0) {
            docList.innerHTML = '<p class="empty-msg">No documents uploaded yet.</p>';
        } else {
            docList.innerHTML = docs.map(doc => `
                <div class="doc-item" data-id="${doc.document_id}">
                    <div class="doc-info">
                        <div class="doc-name" title="${doc.document_name}">${doc.document_name}</div>
                        <div class="doc-meta">${doc.page_count} pages · ${doc.chunk_count} chunks · ${doc.file_size_mb}MB</div>
                    </div>
                    <button class="doc-delete" onclick="deleteDocument('${doc.document_id}', '${doc.document_name.replace(/'/g, "\\'")}')"
                            title="Delete document">🗑️</button>
                </div>
            `).join('');
        }

        // Update scope selector
        const currentScope = scopeSelect.value;
        scopeSelect.innerHTML = '<option value="">All Documents</option>';
        docs.forEach(doc => {
            const option = document.createElement('option');
            option.value = doc.document_id;
            option.textContent = doc.document_name;
            scopeSelect.appendChild(option);
        });
        scopeSelect.value = currentScope;

    } catch (err) {
        console.error('Failed to load documents:', err);
    }
}

// ── Delete Document ─────────────────────────────────────────────
async function deleteDocument(docId, docName) {
    if (!confirm(`Delete "${docName}"? This will remove the document and all its data.`)) {
        return;
    }

    try {
        const resp = await fetch(`${API_BASE}/documents/${docId}`, {
            method: 'DELETE',
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Delete failed');
        }

        const data = await resp.json();
        addMessage(`🗑️ Deleted "${docName}" (${data.chunks_deleted} chunks removed).`, 'assistant');
        await loadDocuments();

    } catch (err) {
        alert(`Failed to delete: ${err.message}`);
    }
}

// ── Send Question ───────────────────────────────────────────────
async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    // Add user message to chat
    addMessage(question, 'user');

    // Clear input
    questionInput.value = '';
    questionInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Show loading
    const loadingId = addLoadingMessage();

    try {
        const body = {
            question: question,
            document_id: scopeSelect.value || null,
            answer_mode: modeSelect.value,
            chat_history: chatHistory.slice(-6),
        };

        const resp = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        // Remove loading
        removeMessage(loadingId);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Failed to get answer');
        }

        const data = await resp.json();

        // Add to chat history for conversational context
        chatHistory.push({ role: 'user', content: question });
        chatHistory.push({ role: 'assistant', content: data.answer });

        // Display answer with sources
        addAnswerMessage(data);

    } catch (err) {
        removeMessage(loadingId);
        addMessage(`❌ Error: ${err.message}`, 'assistant');
    } finally {
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

// ── Message Display ─────────────────────────────────────────────
function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `message ${role}-msg`;
    div.innerHTML = `
        <div class="msg-content">
            ${formatMessage(text)}
        </div>
    `;
    chatContainer.appendChild(div);
    scrollToBottom();
    return div;
}

function addAnswerMessage(data) {
    const div = document.createElement('div');
    div.className = 'message assistant-msg';

    // Grounding badge
    const badge = data.was_grounded
        ? '<span class="grounding-badge grounded">✓ Grounded in documents</span>'
        : '<span class="grounding-badge not-grounded">⚠ No relevant content found</span>';

    // Sources
    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
        const tags = data.sources.map(s =>
            `<span class="source-tag">
                📖 ${s.document_name} — Page ${s.page_number}
                <span class="source-score">${Math.round(s.relevance_score * 100)}%</span>
            </span>`
        ).join('');

        sourcesHtml = `
            <div class="sources">
                <div class="sources-title">Sources</div>
                ${tags}
            </div>
        `;
    }

    div.innerHTML = `
        <div class="msg-content">
            ${badge}
            ${formatMessage(data.answer)}
            ${sourcesHtml}
        </div>
    `;

    chatContainer.appendChild(div);
    scrollToBottom();
}

let loadingCounter = 0;

function addLoadingMessage() {
    const id = `loading-${++loadingCounter}`;
    const div = document.createElement('div');
    div.className = 'message assistant-msg loading-msg';
    div.id = id;
    div.innerHTML = `
        <div class="msg-content">
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
            <span style="color: var(--text-secondary)">Thinking...</span>
        </div>
    `;
    chatContainer.appendChild(div);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── Text Formatting ─────────────────────────────────────────────
function formatMessage(text) {
    // Very basic markdown-like formatting
    return text
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Inline code
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Bullet points
        .replace(/^[•\-\*]\s+(.*)$/gm, '<li>$1</li>')
        // Wrap consecutive <li> in <ul>
        .replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
        // Numbered lists
        .replace(/^\d+\.\s+(.*)$/gm, '<li>$1</li>')
        // Newlines to paragraphs
        .split('\n\n')
        .map(p => p.trim() ? (p.startsWith('<') ? p : `<p>${p}</p>`) : '')
        .join('')
        // Single newlines within paragraphs
        .replace(/(?<!<\/li>)\n(?!<)/g, '<br>');
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cstring>
#include "rada_engine.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "shell32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <signal.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif

// Embedded Minimalist Single-File Web UI (HTML, CSS, JS) - Zero external dependencies, 100% offline
const char* EMBEDDED_HTML = R"html(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rada.cpp | Hetman-2.0B</title>
    <style>
        :root {
            --bg: #090b10;
            --surface: #111420;
            --surface-elevated: #181c2d;
            --surface-hover: #1f253b;
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.16);
            --accent-gold: #facc15;
            --accent-gold-dim: rgba(250, 204, 21, 0.15);
            --accent-blue: #3b82f6;
            --accent-blue-dim: rgba(59, 130, 246, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif; }
        body { background: var(--bg); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; -webkit-font-smoothing: antialiased; }

        /* Custom Minimalist Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }

        /* Navigation Header */
        header {
            background: rgba(17, 20, 32, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .brand-logo { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; display: flex; align-items: center; gap: 8px; }
        .brand-logo span { color: var(--accent-gold); }
        .model-pill {
            background: var(--accent-gold-dim);
            color: var(--accent-gold);
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 12px;
            border: 1px solid rgba(250, 204, 21, 0.25);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .live-dot {
            width: 8px; height: 8px; background: #22c55e; border-radius: 50%; display: inline-block;
            box-shadow: 0 0 10px #22c55e;
            animation: pulse 2s infinite ease-in-out;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }

        .telemetry { display: flex; align-items: center; gap: 16px; font-size: 13px; color: var(--text-muted); }
        .telemetry-item { display: flex; align-items: center; gap: 6px; }
        .telemetry-item b { color: #38bdf8; font-weight: 600; }
        .header-actions { display: flex; align-items: center; gap: 8px; }
        .icon-btn {
            background: transparent; border: 1px solid var(--border); color: var(--text-muted);
            padding: 6px 12px; border-radius: var(--radius-sm); font-size: 12px; cursor: pointer;
            transition: all 0.2s; display: flex; align-items: center; gap: 6px;
        }
        .icon-btn:hover { background: var(--surface-hover); color: var(--text-main); border-color: var(--border-hover); }

        /* Main Chat Window */
        #chat-container {
            flex: 1; overflow-y: auto; padding: 24px 20px; display: flex; flex-direction: column;
            gap: 20px; max-width: 860px; width: 100%; margin: 0 auto;
        }

        /* Hero Welcome Screen */
        .welcome-card {
            background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
            padding: 28px 24px; text-align: center; margin: 40px auto 20px auto; max-width: 640px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }
        .welcome-title { font-size: 22px; font-weight: 700; margin-bottom: 8px; letter-spacing: -0.3px; }
        .welcome-sub { color: var(--text-muted); font-size: 14px; line-height: 1.5; margin-bottom: 24px; }
        .quick-prompts { display: flex; flex-direction: column; gap: 10px; text-align: left; }
        .prompt-chip {
            background: var(--surface-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);
            padding: 12px 16px; font-size: 13px; color: var(--text-main); cursor: pointer;
            transition: all 0.2s; display: flex; align-items: center; justify-content: space-between;
        }
        .prompt-chip:hover {
            background: var(--surface-hover); border-color: var(--accent-gold);
            transform: translateY(-1px);
        }
        .prompt-chip span.arrow { color: var(--accent-gold); font-size: 16px; }

        /* Messages */
        .message-row { display: flex; gap: 14px; width: 100%; animation: fadeIn 0.25s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .message-row.user { justify-content: flex-end; }
        .avatar {
            width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center;
            justify-content: center; font-size: 16px; flex-shrink: 0; margin-top: 2px;
        }
        .assistant .avatar { background: linear-gradient(135deg, #854d0e, #ca8a04); box-shadow: 0 2px 10px rgba(202, 138, 4, 0.25); }
        .user .avatar { background: linear-gradient(135deg, #1d4ed8, #2563eb); }

        .content-box { max-width: 85%; display: flex; flex-direction: column; gap: 6px; }
        .user .content-box { align-items: flex-end; }
        .sender-name { font-size: 12px; font-weight: 600; color: var(--text-dim); padding-left: 2px; }

        .bubble {
            background: var(--surface); border: 1px solid var(--border); padding: 14px 18px;
            border-radius: var(--radius-md); font-size: 15px; line-height: 1.65; color: var(--text-main);
            word-break: break-word; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .user .bubble {
            background: #1e3a8a; border-color: rgba(59, 130, 246, 0.4); color: #ffffff;
            border-bottom-right-radius: 4px;
        }
        .assistant .bubble { border-bottom-left-radius: 4px; }

        /* Thinking Accordion (DeepSeek/Claude style) */
        .thinking-box {
            background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: var(--radius-sm); margin-bottom: 12px; overflow: hidden;
        }
        .thinking-summary {
            padding: 8px 12px; font-size: 13px; font-weight: 600; color: var(--text-muted);
            cursor: pointer; display: flex; align-items: center; gap: 8px; user-select: none;
            background: rgba(255, 255, 255, 0.02); transition: background 0.2s;
        }
        .thinking-summary:hover { background: rgba(255, 255, 255, 0.05); color: var(--text-main); }
        .thinking-body {
            padding: 10px 14px; font-size: 13px; color: var(--text-muted); line-height: 1.6;
            border-top: 1px solid rgba(255, 255, 255, 0.05); font-style: italic; white-space: pre-wrap;
        }
        .thought-summary-badge {
            background: rgba(202, 138, 4, 0.1); border: 1px solid rgba(202, 138, 4, 0.25);
            border-radius: var(--radius-sm); padding: 8px 12px; margin-bottom: 12px;
            font-size: 13px; color: #fef08a; line-height: 1.5;
        }

        /* Code Blocks & Copying */
        .code-container {
            background: #06070a; border: 1px solid var(--border); border-radius: var(--radius-sm);
            margin: 12px 0; overflow: hidden;
        }
        .code-header {
            background: rgba(255, 255, 255, 0.03); padding: 6px 12px; font-size: 12px;
            color: var(--text-dim); display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05); font-family: ui-monospace, monospace;
        }
        .copy-btn {
            background: transparent; border: none; color: var(--text-muted); cursor: pointer;
            font-size: 12px; padding: 2px 6px; border-radius: 4px; transition: 0.2s;
        }
        .copy-btn:hover { color: var(--text-main); background: rgba(255, 255, 255, 0.08); }
        pre { padding: 12px 14px; overflow-x: auto; font-size: 13.5px; line-height: 1.5; font-family: ui-monospace, SFMono-Regular, "Cascadia Code", Menlo, monospace; color: #e2e8f0; }
        code.inline {
            background: rgba(255, 255, 255, 0.08); padding: 2px 6px; border-radius: 4px;
            font-family: ui-monospace, monospace; font-size: 13.5px; color: #facc15;
        }

        .cursor { display: inline-block; width: 6px; height: 15px; background: var(--accent-gold); margin-left: 4px; vertical-align: middle; animation: blink 0.9s infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

        /* Input Controls Area */
        footer {
            background: rgba(17, 20, 32, 0.92); backdrop-filter: blur(12px);
            border-top: 1px solid var(--border); padding: 14px 20px 20px 20px;
        }
        .controls-wrapper { max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 10px; }
        
        .style-selector { display: flex; gap: 8px; align-items: center; }
        .style-label { font-size: 12px; color: var(--text-dim); font-weight: 600; margin-right: 4px; }
        .style-pill {
            background: var(--surface-elevated); border: 1px solid var(--border); color: var(--text-muted);
            padding: 5px 12px; border-radius: var(--radius-lg); font-size: 12px; font-weight: 500;
            cursor: pointer; transition: all 0.2s; user-select: none;
        }
        .style-pill:hover { background: var(--surface-hover); color: var(--text-main); }
        .style-pill.active {
            background: var(--accent-gold-dim); border-color: rgba(250, 204, 21, 0.4);
            color: var(--accent-gold); font-weight: 600; box-shadow: 0 0 12px rgba(250, 204, 21, 0.15);
        }

        .input-bar {
            background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
            padding: 8px 12px 8px 16px; display: flex; gap: 10px; align-items: flex-end;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .input-bar:focus-within {
            border-color: rgba(250, 204, 21, 0.4); box-shadow: 0 0 0 3px rgba(250, 204, 21, 0.1);
        }
        textarea {
            flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 15px;
            line-height: 1.5; resize: none; max-height: 180px; min-height: 24px; outline: none; padding: 4px 0;
        }
        textarea::placeholder { color: var(--text-dim); }

        .send-btn {
            background: var(--accent-gold); color: #000; border: none; width: 36px; height: 36px;
            border-radius: 10px; font-size: 16px; font-weight: 700; cursor: pointer;
            display: flex; align-items: center; justify-content: center; transition: all 0.2s; flex-shrink: 0;
        }
        .send-btn:hover { background: #fde047; transform: scale(1.04); }
        .send-btn:disabled { background: var(--surface-hover); color: var(--text-dim); cursor: not-allowed; transform: none; }

        .shortcut-hint { font-size: 11px; color: var(--text-dim); text-align: center; margin-top: 2px; }

        @media (max-width: 640px) {
            header { padding: 10px 16px; }
            .telemetry { display: none; }
            #chat-container { padding: 16px 12px; }
            .content-box { max-width: 92%; }
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="live-dot" title="Local CUDA engine active"></div>
            <div class="brand-logo">⚔️ <span>RADA</span>.cpp</div>
            <div class="model-pill">Hetman-2.0B</div>
        </div>
        <div class="telemetry">
            <div class="telemetry-item">Context: <b>256k</b></div>
            <div class="telemetry-item">VRAM: <b>2.00 GB</b></div>
            <div class="telemetry-item">Speed: <b>~215 tok/s</b></div>
        </div>
        <div class="header-actions">
            <button class="icon-btn" onclick="clearChat()">🗑️ Clear</button>
        </div>
    </header>

    <div id="chat-container">
        <div class="welcome-card" id="welcome-card">
            <div style="font-size: 40px; margin-bottom: 10px;">⚡</div>
            <h1 class="welcome-title">Hetman-2.0B</h1>
            <p class="welcome-sub">High-speed, autonomous edge AI assistant powered by BitNet b1.58 and Gated DeltaNet with 256k context locally on GPUs with &ge;6 GB VRAM.</p>
            <div class="quick-prompts">
                <div class="prompt-chip" onclick="usePrompt('Write an optimized parallel data processing function in Python or C++')">
                    <div><b>💻 Engineering:</b> Write clean code, diagnose bugs, or build an optimized routine</div>
                    <span class="arrow">→</span>
                </div>
                <div class="prompt-chip" onclick="usePrompt('Explain how DeltaNet linear recurrent attention works in simple terms')">
                    <div><b>📚 Conceptual Breakdown:</b> Clear analysis of complex architectural and algorithmic concepts</div>
                    <span class="arrow">→</span>
                </div>
                <div class="prompt-chip" onclick="usePrompt('Outline an execution plan for high-efficiency model training and deployment')">
                    <div><b>✍️ Analysis & Planning:</b> Structured workflows, project planning, and formal analysis</div>
                    <span class="arrow">→</span>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <div class="controls-wrapper">
            <div class="style-selector">
                <span class="style-label">Mode:</span>
                <div class="style-pill active" onclick="setStyle('general', this)">💬 General (Chat)</div>
                <div class="style-pill" onclick="setStyle('tech', this)">💻 Engineering</div>
                <div class="style-pill" onclick="setStyle('formal', this)">💼 Formal / Analytical</div>
            </div>
            <div class="input-bar">
                <textarea id="prompt" rows="1" placeholder="Ask Hetman-2.0B anything... (Enter to send)" oninput="autoResize(this)" onkeydown="handleKey(event)"></textarea>
                <button class="send-btn" id="send" onclick="sendMessage()" title="Send">↑</button>
            </div>
            <div class="shortcut-hint">Press <b>Enter ↵</b> to send • <b>Shift+Enter</b> for new line • 100% Local & Private</div>
        </div>
    </footer>

    <script>
        let currentStyle = 'general';
        let isGenerating = false;

        function setStyle(style, el) {
            currentStyle = style;
            document.querySelectorAll('.style-pill').forEach(p => p.classList.remove('active'));
            el.classList.add('active');
        }

        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 180) + 'px';
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function usePrompt(text) {
            const input = document.getElementById('prompt');
            input.value = text;
            autoResize(input);
            sendMessage();
        }

        function clearChat() {
            const chat = document.getElementById('chat-container');
            chat.innerHTML = `
                <div class="welcome-card" id="welcome-card">
                    <div style="font-size: 40px; margin-bottom: 10px;">⚡</div>
                    <h1 class="welcome-title">Hetman-2.0B</h1>
                    <p class="welcome-sub">High-speed, autonomous edge AI assistant powered by BitNet b1.58 and Gated DeltaNet with 256k context locally on GPUs with &ge;6 GB VRAM.</p>
                    <div class="quick-prompts">
                        <div class="prompt-chip" onclick="usePrompt('Write an optimized parallel data processing function in Python or C++')">
                            <div><b>💻 Engineering:</b> Write clean code, diagnose bugs, or build an optimized routine</div>
                            <span class="arrow">→</span>
                        </div>
                        <div class="prompt-chip" onclick="usePrompt('Explain how DeltaNet linear recurrent attention works in simple terms')">
                            <div><b>📚 Conceptual Breakdown:</b> Clear analysis of complex architectural and algorithmic concepts</div>
                            <span class="arrow">→</span>
                        </div>
                        <div class="prompt-chip" onclick="usePrompt('Outline an execution plan for high-efficiency model training and deployment')">
                            <div><b>✍️ Analysis & Planning:</b> Structured workflows, project planning, and formal analysis</div>
                            <span class="arrow">→</span>
                        </div>
                    </div>
                </div>`;
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        function formatMarkdown(text) {
            let processed = text;

            // Extract closed <think> ... </think> into collapsible block
            processed = processed.replace(/<think>([\s\S]*?)<\/think>/gi, function(match, content) {
                return `<details class="thinking-box">
                    <summary class="thinking-summary">🧠 Завершено міркування (Associative Trace)</summary>
                    <div class="thinking-body">${escapeHtml(content.trim())}</div>
                </details>`;
            });

            // Handle unclosed <think> during live streaming
            if (processed.includes("<think>")) {
                const parts = processed.split("<think>");
                const before = parts[0];
                const activeThought = parts.slice(1).join("<think>");
                processed = before + `<details class="thinking-box" open>
                    <summary class="thinking-summary">🧠 <span class="live-dot" style="display:inline-block; margin-right:4px;"></span> Гетьман міркує над завданням...</summary>
                    <div class="thinking-body">${escapeHtml(activeThought.trim())}</div>
                </details>`;
            }

            // Extract <thought_summary> ... </thought_summary>
            processed = processed.replace(/<thought_summary>([\s\S]*?)<\/thought_summary>/gi, function(match, content) {
                return `<div class="thought-summary-badge">⚡ <b>Підсумок міркувань:</b> ${escapeHtml(content.trim())}</div>`;
            });

            // Code blocks ```lang ... ```
            processed = processed.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, function(match, lang, code) {
                const language = lang || 'code';
                const safeCode = escapeHtml(code.trim());
                return `<div class="code-container">
                    <div class="code-header">
                        <span>${language}</span>
                        <button class="copy-btn" onclick="copySnippet(this)">Copy</button>
                    </div>
                    <pre><code>${safeCode}</code></pre>
                </div>`;
            });

            // Inline code `code`
            processed = processed.replace(/`([^`]+)`/g, '<code class="inline">$1</code>');

            // Bold text **text**
            processed = processed.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');

            // Line breaks
            processed = processed.replace(/\n/g, '<br>');

            return processed;
        }

        function copySnippet(btn) {
            const pre = btn.closest('.code-container').querySelector('pre code');
            if (pre) {
                navigator.clipboard.writeText(pre.innerText).then(() => {
                    btn.innerText = 'Copied!';
                    setTimeout(() => { btn.innerText = 'Copy'; }, 1500);
                });
            }
        }

        async function sendMessage() {
            if (isGenerating) return;
            const input = document.getElementById('prompt');
            const text = input.value.trim();
            if (!text) return;

            const chat = document.getElementById('chat-container');
            const welcome = document.getElementById('welcome-card');
            if (welcome) welcome.remove();

            // Append User message
            const userRow = document.createElement('div');
            userRow.className = 'message-row user';
            userRow.innerHTML = `
                <div class="content-box">
                    <span class="sender-name">User</span>
                    <div class="bubble">${escapeHtml(text)}</div>
                </div>
                <div class="avatar">👤</div>`;
            chat.appendChild(userRow);

            input.value = '';
            input.style.height = 'auto';

            // Append Assistant placeholder
            const assistRow = document.createElement('div');
            assistRow.className = 'message-row assistant';
            assistRow.innerHTML = `
                <div class="avatar">🏛️</div>
                <div class="content-box">
                    <span class="sender-name">Hetman-2.0B</span>
                    <div class="bubble"><span class="cursor"></span></div>
                </div>`;
            chat.appendChild(assistRow);
            chat.scrollTop = chat.scrollHeight;

            const bubble = assistRow.querySelector('.bubble');
            const sendBtn = document.getElementById('send');
            isGenerating = true;
            sendBtn.disabled = true;

            let fullText = '';

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: text, style: currentStyle })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let renderScheduled = false;
                const scheduleRender = () => {
                    if (!renderScheduled) {
                        renderScheduled = true;
                        requestAnimationFrame(() => {
                            bubble.innerHTML = formatMarkdown(fullText) + '<span class="cursor"></span>';
                            chat.scrollTop = chat.scrollHeight;
                            renderScheduled = false;
                        });
                    }
                };

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, { stream: true });
                    fullText += chunk;
                    scheduleRender();
                }
                bubble.innerHTML = formatMarkdown(fullText);
            } catch (err) {
                bubble.innerHTML = `<span style="color: #ef4444;">Connection error with Rada.cpp engine: ${escapeHtml(err.message)}</span>`;
            } finally {
                isGenerating = false;
                sendBtn.disabled = false;
                chat.scrollTop = chat.scrollHeight;
            }
        }
    </script>
</body>
</html>)html";

void open_browser(const std::string& url) {
#ifdef _WIN32
    ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
#elif __APPLE__
    std::string cmd = "open " + url;
    system(cmd.c_str());
#else
    std::string cmd = "xdg-open " + url + " >/dev/null 2>&1 &";
    system(cmd.c_str());
#endif
}

static std::string escape_json_string(const std::string& in) {
    std::string out;
    out.reserve(in.length() + 8);
    for (char c : in) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static std::string unescape_json_string(const std::string& in) {
    std::string out;
    out.reserve(in.length());
    for (size_t i = 0; i < in.length(); ++i) {
        if (in[i] == '\\' && i + 1 < in.length()) {
            char next = in[++i];
            if (next == 'n') { out += '\n'; }
            else if (next == 'r') { out += '\r'; }
            else if (next == 't') { out += '\t'; }
            else if (next == '"') { out += '"'; }
            else if (next == '\\') { out += '\\'; }
            else if (next == 'u' && i + 4 < in.length()) {
                std::string hex_str = in.substr(i + 1, 4);
                try {
                    uint32_t cp = std::stoul(hex_str, nullptr, 16);
                    i += 4;
                    // Encode Unicode codepoint into UTF-8 bytes
                    if (cp < 0x80) {
                        out += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | (cp >> 6));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (cp >> 12));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                } catch (...) {
                    out += "\\u" + hex_str;
                }
            } else {
                out += next;
            }
        } else {
            out += in[i];
        }
    }
    return out;
}

static std::string extract_json_prompt(const std::string& body) {
    size_t p_pos = body.find("\"prompt\"");
    size_t key_len = 8;
    if (p_pos == std::string::npos) {
        // Fallback to OpenAI API format: {"messages": [{"role": "user", "content": "..."}]}
        // Search for the last "content" field in the request payload
        size_t last_content = body.rfind("\"content\"");
        if (last_content != std::string::npos) {
            p_pos = last_content;
            key_len = 9;
        } else {
            return "Привіт!";
        }
    }
    size_t colon = body.find(':', p_pos + key_len);
    if (colon == std::string::npos) return "Привіт!";
    size_t quote_start = body.find('"', colon + 1);
    if (quote_start == std::string::npos) return "Привіт!";
    
    // Find unescaped end quote
    size_t quote_end = quote_start + 1;
    while (quote_end < body.length()) {
        if (body[quote_end] == '"') {
            size_t bs = 0;
            size_t k = quote_end;
            while (k > quote_start && body[k - 1] == '\\') {
                bs++;
                k--;
            }
            if (bs % 2 == 0) break; // Even backslashes -> actual delimiter quote
        }
        quote_end++;
    }
    if (quote_end <= body.length()) {
        return unescape_json_string(body.substr(quote_start + 1, quote_end - quote_start - 1));
    }
    return "Hello!";
}

static bool send_all(SOCKET fd, const char* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        int res = send(fd, data + total_sent, static_cast<int>(len - total_sent), 0);
        if (res <= 0) {
            return false;
        }
        total_sent += static_cast<size_t>(res);
    }
    return true;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#else
    signal(SIGPIPE, SIG_IGN);
#endif

    std::string model_path = "models/hetman-2.0b-ternary.rada";
    int port = 8080;
    int gpu_id = 0;
    size_t ctx_len = 262144;
    int threads = 4;
    bool auto_open = true;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "-m" || arg == "--model") && i + 1 < argc) model_path = argv[++i];
            else if ((arg == "-p" || arg == "--port") && i + 1 < argc) port = std::stoi(argv[++i]);
            else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) ctx_len = std::stoull(argv[++i]);
            else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) gpu_id = std::stoi(argv[++i]);
            else if (arg == "--no-browser") auto_open = false;
            else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: rada_web [-m model.rada] [-p port] [-c ctx_len] [-t threads] [--no-browser]\n";
                return 0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Invalid parameter: " << e.what() << "\n";
        return 1;
    }

    // Automatic fallback if default path not found
    std::ifstream test_f(model_path, std::ios::binary);
    if (!test_f.good()) {
        std::ifstream f_mini("models/hetman_mini.rada", std::ios::binary);
        if (f_mini.good()) model_path = "models/hetman_mini.rada";
        else {
            std::ifstream f_root("hetman_mini.rada", std::ios::binary);
            if (f_root.good()) model_path = "hetman_mini.rada";
        }
    }

    std::cout << "\033[1;33m=======================================================================\033[0m\n";
    std::cout << "\033[1;33m       RADA WEB: MINIMALIST CHAT SERVER FOR HETMAN-2.0B               \033[0m\n";
    std::cout << "\033[1;33m=======================================================================\033[0m\n";
    std::cout << "[RADA WEB] Loading model: " << model_path << "\n";

    rada::RadaEngine engine;
    if (!engine.load_model(model_path, gpu_id, ctx_len, threads)) {
        std::cerr << "[ERROR] Failed to load model at: " << model_path << "\n";
        return 1;
    }

#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "[ERROR] Failed to create socket\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // Strictly bind to 127.0.0.1 for local security
    address.sin_port = htons(static_cast<u_short>(port));

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "[ERROR] Failed to bind to port " << port << " on 127.0.0.1\n";
        closesocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    listen(server_fd, 10);
    std::string url = "http://127.0.0.1:" + std::to_string(port);
    std::cout << "\033[1;32m[RADA WEB] Server running securely at:\033[0m " << url << "\n";

    if (auto_open) {
        std::cout << "[RADA WEB] Launching chat page in default web browser...\n";
        open_browser(url);
    }

    std::cout << "[RADA WEB] Awaiting incoming connections (Press Ctrl+C to stop)...\n\n";

    while (true) {
        SOCKET client_fd = accept(server_fd, NULL, NULL);
        if (client_fd == INVALID_SOCKET) continue;

        // Set socket receive and send timeouts (10 seconds)
#ifdef _WIN32
        DWORD timeout_ms = 10000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
        struct timeval tv{};
        tv.tv_sec = 10;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const void*)&tv, sizeof(tv));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, (const void*)&tv, sizeof(tv));
#endif

        try {
            std::string request;
            char buffer[4096];
            size_t content_length = 0;
            bool headers_done = false;

            while (true) {
                int bytes_read = recv(client_fd, buffer, sizeof(buffer), 0);
                if (bytes_read <= 0) break;
                if (request.length() + bytes_read > 16 * 1024 * 1024) break; // 16 MB max payload ceiling
                request.append(buffer, bytes_read);

                if (!headers_done) {
                    size_t header_end = request.find("\r\n\r\n");
                    if (header_end != std::string::npos) {
                        headers_done = true;
                        std::string headers_lower = request.substr(0, header_end);
                        for (auto& c : headers_lower) c = (char)::tolower((unsigned char)c);
                        size_t cl_pos = headers_lower.find("content-length:");
                        if (cl_pos != std::string::npos) {
                            size_t val_start = request.find_first_not_of(" \t", cl_pos + 15);
                            size_t val_end = request.find("\r\n", val_start);
                            if (val_start != std::string::npos && val_end != std::string::npos) {
                                try {
                                    content_length = std::stoull(request.substr(val_start, val_end - val_start));
                                } catch (...) { content_length = 0; }
                            }
                        }
                    }
                }

                if (headers_done) {
                    size_t body_start = request.find("\r\n\r\n") + 4;
                    if (request.length() - body_start >= content_length) {
                        break;
                    }
                }
            }

            if (request.empty()) {
                closesocket(client_fd);
                continue;
            }

            // Handle OPTIONS preflight request (CORS support for Web IDEs/Extensions)
            if (request.rfind("OPTIONS ", 0) == 0) {
                std::string header = "HTTP/1.1 204 No Content\r\n"
                                     "Access-Control-Allow-Origin: *\r\n"
                                     "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                                     "Access-Control-Allow-Headers: *\r\n"
                                     "Connection: close\r\n\r\n";
                send_all(client_fd, header.c_str(), header.length());
            }
            // Handle GET / -> Serve embedded HTML UI
            else if (request.rfind("GET / ", 0) == 0 || request.rfind("GET /index.html", 0) == 0) {
                std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                send_all(client_fd, header.c_str(), header.length());
                send_all(client_fd, EMBEDDED_HTML, strlen(EMBEDDED_HTML));
            }
            // Handle GET /api/health -> Return engine health status
            else if (request.rfind("GET /api/health", 0) == 0) {
                std::string body = "{\"status\":\"healthy\",\"engine\":\"rada.cpp\",\"model\":\"hetman-2.0b\"}\n";
                std::string header = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " +
                                     std::to_string(body.length()) + "\r\nConnection: close\r\n\r\n";
                send_all(client_fd, header.c_str(), header.length());
                send_all(client_fd, body.c_str(), body.length());
            }
            // Handle GET /api/stats -> Return runtime statistics
            else if (request.rfind("GET /api/stats", 0) == 0) {
                auto stats = engine.get_stats();
                std::string body = "{\"status\":\"ok\",\"model\":\"" + escape_json_string(model_path) +
                                   "\",\"context_length\":" + std::to_string(ctx_len) +
                                   ",\"threads\":" + std::to_string(threads) +
                                   ",\"vram_usage_mb\":" + std::to_string(stats.vram_usage_bytes / (1024 * 1024)) +
                                   ",\"last_speed_tok_s\":" + std::to_string(stats.last_speed_tok_s) + "}\n";
                std::string header = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " +
                                     std::to_string(body.length()) + "\r\nConnection: close\r\n\r\n";
                send_all(client_fd, header.c_str(), header.length());
                send_all(client_fd, body.c_str(), body.length());
            }
            // Handle POST /v1/chat/completions -> Stream text chunks or SSE
            else if (request.rfind("POST /v1/chat/completions", 0) == 0) {
                size_t body_pos = request.find("\r\n\r\n");
                std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : "";
                std::string prompt = extract_json_prompt(body);

                bool is_sse = (request.find("text/event-stream") != std::string::npos) ||
                              (body.find("\"stream\":true") != std::string::npos) ||
                              (body.find("\"stream\": true") != std::string::npos);

                rada::GenerationParams params;
                if (body.find("\"style\":\"formal\"") != std::string::npos || body.find("\"style\": \"formal\"") != std::string::npos ||
                    body.find("\"style\":\"legal\"") != std::string::npos || body.find("\"style\": \"legal\"") != std::string::npos) {
                    params.style = rada::StylePreset::FORMAL;
                } else if (body.find("\"style\":\"tech\"") != std::string::npos || body.find("\"style\": \"tech\"") != std::string::npos ||
                           body.find("\"style\":\"engineering\"") != std::string::npos || body.find("\"style\": \"engineering\"") != std::string::npos) {
                    params.style = rada::StylePreset::ENGINEERING;
                } else {
                    params.style = rada::StylePreset::GENERAL;
                }

                if (is_sse) {
                    std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n";
                    send_all(client_fd, header.c_str(), header.length());

                    bool client_alive = true;
                    engine.generate_stream(prompt, params, [&](const std::string& chunk) {
                        if (chunk.empty()) return true;
                        std::string sse_msg = "data: {\"choices\":[{\"delta\":{\"content\":\"" + escape_json_string(chunk) + "\"}}]}\n\n";
                        if (!send_all(client_fd, sse_msg.c_str(), sse_msg.length())) {
                            client_alive = false;
                            return false;
                        }
                        return true;
                    });
                    if (client_alive) {
                        std::string done_msg = "data: [DONE]\n\n";
                        send_all(client_fd, done_msg.c_str(), done_msg.length());
                    }
                } else {
                    std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nTransfer-Encoding: chunked\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n";
                    send_all(client_fd, header.c_str(), header.length());

                    bool client_alive = true;
                    engine.generate_stream(prompt, params, [&](const std::string& chunk) {
                        if (chunk.empty()) return true;
                        std::stringstream hex_len;
                        hex_len << std::hex << chunk.length() << "\r\n";
                        std::string chunk_data = hex_len.str() + chunk + "\r\n";
                        if (!send_all(client_fd, chunk_data.c_str(), chunk_data.length())) {
                            client_alive = false;
                            return false; // Client disconnected, terminate GPU generation early
                        }
                        return true;
                    });

                    if (client_alive) {
                        // Send final zero chunk to terminate chunked encoding cleanly
                        std::string end_chunk = "0\r\n\r\n";
                        send_all(client_fd, end_chunk.c_str(), end_chunk.length());
                    }
                }
            } else {
                std::string not_found = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
                send_all(client_fd, not_found.c_str(), not_found.length());
            }
        } catch (const std::exception& e) {
            std::cerr << "[RADA WEB] Request error: " << e.what() << "\n";
        }

        closesocket(client_fd);
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

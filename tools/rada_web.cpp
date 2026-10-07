#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <chrono>
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
            <div style="font-size: 40px; margin-bottom: 10px;">🏛️</div>
            <h1 class="welcome-title">Council of Free Compatriots</h1>
            <p class="welcome-sub">Hetman-2.0B is online. Powered by 28 Gated DeltaNet recurrent memory layers, BitNet b1.58 ternary weights, and an 8.39M-slot Hopfield associative core.</p>
            <div class="quick-prompts">
                <div class="prompt-chip" onclick="usePrompt('Analyze statutory jurisprudence on commercial obligations under the laws of Ukraine')">
                    <div><b>⚖️ Statutory Jurisprudence:</b> Commercial contract obligations under Ukrainian law</div>
                    <span class="arrow">→</span>
                </div>
                <div class="prompt-chip" onclick="usePrompt('Explain how Gated DeltaNet linear attention achieves O(1) state without KV-cache explosion')">
                    <div><b>💻 Systems Architecture:</b> How DeltaNet maintains O(1) 128 KB memory state</div>
                    <span class="arrow">→</span>
                </div>
                <div class="prompt-chip" onclick="usePrompt('Give deliberate counsel upon courage, honor, and strategic perseverance')">
                    <div><b>⚔️ Deliberative Counsel:</b> Seek wisdom upon perseverance and honor</div>
                    <span class="arrow">→</span>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <div class="controls-wrapper">
            <div class="style-selector">
                <span class="style-label">Style:</span>
                <div class="style-pill active" onclick="setStyle('kozak', this)">⚔️ Cossack (Wisdom)</div>
                <div class="style-pill" onclick="setStyle('legal', this)">⚖️ Legal (Statutory)</div>
                <div class="style-pill" onclick="setStyle('tech', this)">💻 Engineering (Tech)</div>
            </div>
            <div class="input-bar">
                <textarea id="prompt" rows="1" placeholder="Consult with Hetman-2.0B... (Press Enter to send)" oninput="autoResize(this)" onkeydown="handleKey(event)"></textarea>
                <button class="send-btn" id="send" onclick="sendMessage()" title="Send prompt">↑</button>
            </div>
            <div class="shortcut-hint">Press <b>Enter ↵</b> to send • <b>Shift+Enter</b> for new line • 100% Offline</div>
        </div>
    </footer>

    <script>
        let currentStyle = 'kozak';
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
                    <div style="font-size: 40px; margin-bottom: 10px;">🏛️</div>
                    <h1 class="welcome-title">Council of Free Compatriots</h1>
                    <p class="welcome-sub">Hetman-2.0B is online. Powered by 28 Gated DeltaNet recurrent memory layers, BitNet b1.58 ternary weights, and an 8.39M-slot Hopfield associative core.</p>
                    <div class="quick-prompts">
                        <div class="prompt-chip" onclick="usePrompt('Analyze statutory jurisprudence on commercial obligations under the laws of Ukraine')">
                            <div><b>⚖️ Statutory Jurisprudence:</b> Commercial contract obligations under Ukrainian law</div>
                            <span class="arrow">→</span>
                        </div>
                        <div class="prompt-chip" onclick="usePrompt('Explain how Gated DeltaNet linear attention achieves O(1) state without KV-cache explosion')">
                            <div><b>💻 Systems Architecture:</b> How DeltaNet maintains O(1) 128 KB memory state</div>
                            <span class="arrow">→</span>
                        </div>
                        <div class="prompt-chip" onclick="usePrompt('Give deliberate counsel upon courage, honor, and strategic perseverance')">
                            <div><b>⚔️ Deliberative Counsel:</b> Seek wisdom upon perseverance and honor</div>
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

            // Extract <think> ... </think> into collapsible block
            processed = processed.replace(/<think>([\s\S]*?)<\/think>/gi, function(match, content) {
                return `<details class="thinking-box" open>
                    <summary class="thinking-summary">🧠 Thought Process (Associative Trace)</summary>
                    <div class="thinking-body">${escapeHtml(content.trim())}</div>
                </details>`;
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

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, { stream: true });
                    fullText += chunk;
                    bubble.innerHTML = formatMarkdown(fullText) + '<span class="cursor"></span>';
                    chat.scrollTop = chat.scrollHeight;
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

static std::string unescape_json_string(const std::string& in) {
    std::string out;
    out.reserve(in.length());
    for (size_t i = 0; i < in.length(); ++i) {
        if (in[i] == '\\' && i + 1 < in.length()) {
            char next = in[i + 1];
            if (next == 'n') { out += '\n'; ++i; }
            else if (next == 'r') { out += '\r'; ++i; }
            else if (next == 't') { out += '\t'; ++i; }
            else if (next == '"') { out += '"'; ++i; }
            else if (next == '\\') { out += '\\'; ++i; }
            else { out += in[i]; }
        } else {
            out += in[i];
        }
    }
    return out;
}

int main(int argc, char* argv[]) {
#ifndef _WIN32
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
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Invalid parameter: " << e.what() << "\n";
        return 1;
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
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // Strictly bind to 127.0.0.1 for local security
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "[ERROR] Failed to bind to port " << port << " on 127.0.0.1\n";
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

        char buffer[4096] = {0};
        int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (bytes_read <= 0) {
            closesocket(client_fd);
            continue;
        }

        std::string request(buffer, bytes_read);

        // Handle GET / -> Serve embedded HTML UI
        if (request.rfind("GET / ", 0) == 0 || request.rfind("GET /index.html", 0) == 0) {
            std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n";
            send(client_fd, header.c_str(), (int)header.length(), 0);
            send(client_fd, EMBEDDED_HTML, (int)strlen(EMBEDDED_HTML), 0);
        }
        // Handle POST /v1/chat/completions -> Stream text chunks
        else if (request.rfind("POST /v1/chat/completions", 0) == 0) {
            std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
            send(client_fd, header.c_str(), (int)header.length(), 0);

            // Parser for prompt and style from body
            size_t body_pos = request.find("\r\n\r\n");
            std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : "";
            
            std::string prompt = "Hello";
            size_t p_pos = body.find("\"prompt\":");
            if (p_pos != std::string::npos) {
                size_t p_start = body.find("\"", p_pos + 9);
                size_t p_end = body.find("\"", p_start + 1);
                if (p_start != std::string::npos && p_end != std::string::npos) {
                    prompt = unescape_json_string(body.substr(p_start + 1, p_end - p_start - 1));
                }
            }

            rada::GenerationParams params;
            if (body.find("\"style\":\"legal\"") != std::string::npos) params.style = rada::StylePreset::LEGAL;
            else if (body.find("\"style\":\"tech\"") != std::string::npos) params.style = rada::StylePreset::ENGINEERING;
            else params.style = rada::StylePreset::KOZAK;

            engine.generate_stream(prompt, params, [&](const std::string& chunk) {
                std::stringstream hex_len;
                hex_len << std::hex << chunk.length() << "\r\n";
                std::string chunk_data = hex_len.str() + chunk + "\r\n";
                int res = send(client_fd, chunk_data.c_str(), (int)chunk_data.length(), 0);
                if (res <= 0) return false; // Client closed connection, stop GPU compute immediately!
                return true;
            });

            // Send zero chunk to terminate HTTP chunked transfer
            std::string end_chunk = "0\r\n\r\n";
            send(client_fd, end_chunk.c_str(), (int)end_chunk.length(), 0);
        } else {
            std::string not_found = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
            send(client_fd, not_found.c_str(), (int)not_found.length(), 0);
        }

        closesocket(client_fd);
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

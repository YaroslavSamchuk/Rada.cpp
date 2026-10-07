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
    <title>Rada.cpp | Hetman-2.0B Web Chat</title>
    <style>
        :root {
            --bg: #0f1117;
            --surface: #181b24;
            --surface-hover: #222634;
            --border: #282c3c;
            --accent: #ffd700;
            --accent-blue: #0077ff;
            --text: #e6e8ee;
            --text-dim: #8b92a5;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        
        /* Top Navigation Header */
        header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
        .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 18px; letter-spacing: 0.5px; }
        .logo span { color: var(--accent); }
        .status-badge { display: flex; gap: 16px; font-size: 13px; color: var(--text-dim); }
        .status-badge b { color: #4ade80; }

        /* Main Chat Container */
        #chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 18px; max-width: 900px; width: 100%; margin: 0 auto; }
        .message { display: flex; gap: 12px; max-width: 85%; line-height: 1.6; font-size: 15px; }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
        .user .avatar { background: var(--accent-blue); }
        .assistant .avatar { background: #854d0e; }
        .bubble { background: var(--surface); border: 1px solid var(--border); padding: 12px 18px; border-radius: 12px; word-break: break-word; white-space: pre-wrap; }
        .user .bubble { background: #1e3a8a; border-color: #2563eb; color: #fff; }

        /* Styles Selector Toolbar */
        .toolbar { max-width: 900px; width: 100%; margin: 0 auto; padding: 0 24px 8px 24px; display: flex; gap: 8px; }
        .style-btn { background: var(--surface); border: 1px solid var(--border); color: var(--text-dim); padding: 6px 14px; border-radius: 20px; font-size: 13px; cursor: pointer; transition: 0.2s; }
        .style-btn:hover { background: var(--surface-hover); color: var(--text); }
        .style-btn.active { border-color: var(--accent); color: var(--accent); font-weight: 600; }

        /* Input Form */
        footer { background: var(--surface); border-top: 1px solid var(--border); padding: 16px 24px; }
        .input-box { max-width: 900px; margin: 0 auto; display: flex; gap: 10px; }
        textarea { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; color: var(--text); resize: none; height: 50px; font-size: 15px; outline: none; }
        textarea:focus { border-color: var(--accent-blue); }
        button#send { background: var(--accent); color: #000; border: none; border-radius: 10px; padding: 0 24px; font-weight: 700; cursor: pointer; font-size: 15px; transition: 0.2s; }
        button#send:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <header>
        <div class="logo">⚔️ <span>RADA</span>.cpp | Hetman-2.0B</div>
        <div class="status-badge">
            <div>Context: <b>256k</b></div>
            <div>VRAM: <b>2.00 GB / 6.00 GB</b></div>
            <div>Speed: <b>~215 tok/s</b></div>
        </div>
    </header>

    <div id="chat-container">
        <div class="message assistant">
            <div class="avatar">🏛️</div>
            <div class="bubble">Greetings! The sovereign council has convened. Hetman-2.0B is online and ready for consultation. What would you like to deliberate upon?</div>
        </div>
    </div>

    <div class="toolbar">
        <button class="style-btn active" onclick="setStyle('kozak', this)">⚔️ Cossack (Sovereign)</button>
        <button class="style-btn" onclick="setStyle('legal', this)">⚖️ Legal / Statutory</button>
        <button class="style-btn" onclick="setStyle('tech', this)">💻 Engineering / Tech</button>
    </div>

    <footer>
        <div class="input-box">
            <textarea id="prompt" placeholder="Type your prompt for Hetman-2.0B... (Press Enter to send)" onkeydown="handleKey(event)"></textarea>
            <button id="send" onclick="sendMessage()">Send</button>
        </div>
    </footer>

    <script>
        let currentStyle = 'kozak';
        function setStyle(style, btn) {
            currentStyle = style;
            document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('prompt');
            const text = input.value.trim();
            if (!text) return;

            const chat = document.getElementById('chat-container');
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.innerHTML = '<div class="avatar">👤</div><div class="bubble">' + escapeHtml(text) + '</div>';
            chat.appendChild(userMsg);
            input.value = '';

            // Add empty assistant response bubble
            const assistMsg = document.createElement('div');
            assistMsg.className = 'message assistant';
            assistMsg.innerHTML = '<div class="avatar">🏛️</div><div class="bubble">...</div>';
            chat.appendChild(assistMsg);
            chat.scrollTop = chat.scrollHeight;

            const bubble = assistMsg.querySelector('.bubble');

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: text, style: currentStyle })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                bubble.innerText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, { stream: true });
                    bubble.innerText += chunk;
                    chat.scrollTop = chat.scrollHeight;
                }
            } catch (err) {
                bubble.innerText = 'Connection error with Rada.cpp engine: ' + err.message;
            }
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
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

int main(int argc, char* argv[]) {
    std::string model_path = "models/hetman-2.0b-ternary.rada";
    int port = 8080;
    int gpu_id = 0;
    size_t ctx_len = 262144;
    bool auto_open = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) model_path = argv[++i];
        else if ((arg == "-p" || arg == "--port") && i + 1 < argc) port = std::stoi(argv[++i]);
        else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) ctx_len = std::stoull(argv[++i]);
        else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) gpu_id = std::stoi(argv[++i]);
        else if (arg == "--no-browser") auto_open = false;
    }

    std::cout << "\033[1;33m=======================================================================\033[0m\n";
    std::cout << "\033[1;33m       RADA WEB: MINIMALIST CHAT SERVER FOR HETMAN-2.0B               \033[0m\n";
    std::cout << "\033[1;33m=======================================================================\033[0m\n";
    std::cout << "[RADA WEB] Loading model: " << model_path << "\n";

    rada::RadaEngine engine;
    engine.load_model(model_path, gpu_id, ctx_len);

#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "[ERROR] Failed to bind to port " << port << "\n";
        return 1;
    }

    listen(server_fd, 10);
    std::string url = "http://localhost:" + std::to_string(port);
    std::cout << "\033[1;32m[RADA WEB] Server running successfully at:\033[0m " << url << "\n";

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
            send(client_fd, header.c_str(), header.length(), 0);
            send(client_fd, EMBEDDED_HTML, strlen(EMBEDDED_HTML), 0);
        }
        // Handle POST /v1/chat/completions -> Stream text chunks
        else if (request.rfind("POST /v1/chat/completions", 0) == 0) {
            std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
            send(client_fd, header.c_str(), header.length(), 0);

            // Simple parser for prompt and style from body
            size_t body_pos = request.find("\r\n\r\n");
            std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : "";
            
            std::string prompt = "Hello";
            size_t p_pos = body.find("\"prompt\":");
            if (p_pos != std::string::npos) {
                size_t p_start = body.find("\"", p_pos + 9);
                size_t p_end = body.find("\"", p_start + 1);
                if (p_start != std::string::npos && p_end != std::string::npos) {
                    prompt = body.substr(p_start + 1, p_end - p_start - 1);
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
                send(client_fd, chunk_data.c_str(), chunk_data.length(), 0);
                return true;
            });

            // Send zero chunk to terminate HTTP chunked transfer
            std::string end_chunk = "0\r\n\r\n";
            send(client_fd, end_chunk.c_str(), end_chunk.length(), 0);
        } else {
            std::string not_found = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
            send(client_fd, not_found.c_str(), not_found.length(), 0);
        }

        closesocket(client_fd);
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

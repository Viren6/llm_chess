"""ChatGPT subscription transport through the official Codex app server.

Codex owns OAuth credentials and refresh. No API keys or bearer tokens are read by
the benchmark. Each completion uses a fresh, ephemeral thread in an empty cwd.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace


def codex_home():
    return Path(os.environ.get("LLM_CHESS_CODEX_HOME", "~/.local/share/llm-chess/codex")).expanduser().resolve()


def codex_env():
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_ACCESS_TOKEN"):
        env.pop(key, None)
    home = codex_home()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (home / "config.toml").exists():
        raise ValueError("Use a dedicated LLM_CHESS_CODEX_HOME without config.toml; custom Codex settings can alter the benchmark")
    env["CODEX_HOME"] = str(home)
    return env


def codex_command():
    return [os.environ.get("LLM_CHESS_CODEX_BIN", "codex"),
            "-c", 'forced_login_method="chatgpt"', "-c", 'model_provider="openai"']


class AppServer:
    """Single synchronous JSON-RPC session, with bounded waits and process cleanup."""

    def __init__(self, timeout=7200):
        self.timeout = timeout
        self.events = queue.Queue()
        self.pending = deque()
        self.next_id = 0
        self.cwd = tempfile.TemporaryDirectory(prefix="llm-chess-oauth-")
        self.proc = None
        try:
            self.proc = subprocess.Popen(
                codex_command() + ["app-server"], env=codex_env(), cwd=self.cwd.name,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            self.rpc("initialize", {"clientInfo": {"name": "llm_chess", "version": "0.1.0"},
                                    "capabilities": {"experimentalApi": True}})
            self.send({"method": "initialized", "params": {}})
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.events.put(json.loads(line))
        except Exception:
            pass
        finally:
            self.events.put(None)

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def event(self, deadline):
        try:
            message = (self.pending.popleft() if self.pending else
                       self.events.get(timeout=max(0, deadline - time.monotonic())))
        except queue.Empty:
            raise TimeoutError("Codex OAuth request timeout") from None
        if message is None:
            raise RuntimeError("Codex app server exited before completing the request")
        if "method" in message and "id" in message:
            self.send({"id": message["id"], "error": {"code": -32601,
                       "message": "Tools and user interaction are disabled in this benchmark"}})
            raise RuntimeError("Codex requested a tool or user interaction; benchmark aborted")
        return message

    def rpc(self, method, params):
        self.next_id += 1
        request_id = self.next_id
        self.send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + min(self.timeout, 60)
        pending = []
        try:
            while True:
                message = self.event(deadline)
                if message.get("id") == request_id:
                    if "error" in message:
                        raise RuntimeError(f"Codex {method}: {message['error']['message']}")
                    return message["result"]
                pending.append(message)
        finally:
            # Notifications can arrive before the response (including turn/completed).
            self.pending.extendleft(reversed(pending))

    def require_chatgpt(self):
        account = self.rpc("account/read", {"refreshToken": True}).get("account") or {}
        if account.get("type") != "chatgpt":
            raise RuntimeError("ChatGPT OAuth login required: python3 openai_oauth.py login")

    def check_model(self, model, effort):
        cursor = None
        while True:
            result = self.rpc("model/list", {"includeHidden": True, "cursor": cursor})
            for entry in result["data"]:
                if entry.get("model") == model:
                    efforts = [r["reasoningEffort"] for r in entry["supportedReasoningEfforts"]]
                    if effort not in efforts:
                        raise ValueError(f"{model} does not advertise reasoning effort {effort}: {efforts}")
                    return
            cursor = result.get("nextCursor")
            if not cursor:
                raise ValueError(f"{model} is not available in this ChatGPT account's Codex model list")

    def complete(self, model, effort, messages):
        if any(m.get("role") not in ("system", "user") or not isinstance(m.get("content"), str)
               for m in messages):
            raise ValueError("OAuth transport supports the stateless simple harness (text system/user messages)")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        config = {
            "web_search": "disabled", "project_doc_max_bytes": 0,
            "tools.update_plan.enabled": False,
            "tools.experimental_request_user_input.enabled": False,
            "include_apps_instructions": False, "include_collaboration_mode_instructions": False,
            "include_permissions_instructions": False,
            "features.skip_host_skill_discovery": True,
        }
        for feature in ("shell_tool", "unified_exec", "apply_patch_freeform", "apps", "plugins",
                        "multi_agent", "collab", "js_repl", "code_mode", "browser_use", "computer_use",
                        "image_generation", "imagegenext", "view_image", "memory_tool", "memories",
                        "skill_search", "tool_search", "tool_suggest", "goals", "sleep_tool"):
            config[f"features.{feature}"] = False
        started = self.rpc("thread/start", {
            "model": model, "modelProvider": "openai", "cwd": self.cwd.name,
            "approvalPolicy": "never", "sandbox": "read-only", "ephemeral": True,
            "baseInstructions": system, "developerInstructions": "", "personality": "none",
            "config": config,
        })
        if started.get("model", model) != model:
            raise RuntimeError("Codex selected a different model; benchmark aborted")
        tid = started["thread"]["id"]
        turn = self.rpc("turn/start", {"threadId": tid, "model": model, "effort": effort,
                        "input": [{"type": "text", "text": prompt}]})
        turn_id = turn["turn"]["id"]
        deadline = time.monotonic() + self.timeout
        text, reasoning, usage = [], [], {}
        while True:
            event = self.event(deadline)
            method, p = event.get("method"), event.get("params", {})
            if p.get("threadId") != tid:
                continue
            if method in ("item/started", "item/completed"):
                item = p["item"]
                if item["type"] not in ("userMessage", "agentMessage", "reasoning"):
                    raise RuntimeError(f"Unexpected Codex item {item['type']}; tool-free benchmark aborted")
                if method == "item/completed" and item["type"] == "agentMessage":
                    text.append(item["text"])
                if method == "item/completed" and item["type"] == "reasoning":
                    reasoning.extend(item.get("summary", []))
            elif method == "thread/tokenUsage/updated":
                usage = p["tokenUsage"]["total"]
            elif method == "turn/completed" and p["turn"]["id"] == turn_id:
                if p["turn"]["status"] != "completed":
                    error = p["turn"].get("error") or {}
                    raise RuntimeError(f"Codex turn failed: {error.get('message', p['turn']['status'])}")
                break
        if not text:
            raise RuntimeError("Codex returned no assistant text")
        return "\n".join(text), "\n".join(reasoning), usage

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout):
                stream.close()
        self.cwd.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OpenAIOAuthClient:
    """AG2 custom model client; subscription usage is recorded with no API charge."""

    def __init__(self, config, **kwargs):
        self.model = config["model"]
        self.effort = config.get("oauth_reasoning_effort", config.get("reasoning_effort", "max"))
        self.timeout = config.get("timeout", 7200)

    def create(self, params):
        with AppServer(self.timeout) as server:
            server.require_chatgpt()
            content, reasoning, usage = server.complete(self.model, self.effort, params["messages"])
        pt, ct = usage.get("inputTokens", 0), usage.get("outputTokens", 0)
        return SimpleNamespace(model=self.model, cost=0.0,
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=reasoning))],
            usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct))

    def message_retrieval(self, response):
        return [c.message.content for c in response.choices]

    def cost(self, response):
        return 0.0

    @staticmethod
    def get_usage(response):
        return {**vars(response.usage), "cost": 0.0, "model": response.model}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["login", "status", "check"])
    parser.add_argument("--device-auth", action="store_true")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="max")
    args = parser.parse_args()
    if args.action == "login":
        command = codex_command() + ["login"] + (["--device-auth"] if args.device_auth else [])
        raise SystemExit(subprocess.call(command, env=codex_env()))
    with AppServer(timeout=60) as server:
        server.require_chatgpt()
        if args.action == "check":
            server.check_model(args.model, args.reasoning_effort)
            print(f"ChatGPT OAuth ready: {args.model}, reasoning={args.reasoning_effort}")
        else:
            print("ChatGPT OAuth authenticated (subscription access)")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from None

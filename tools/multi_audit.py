#!/usr/bin/env python3
"""Heterogeneous multi-model audit & planning runner via mahu unified OpenAI-compatible API.

Follows the TheoryScout & auto-llm-driving workflow:
- Never rely on a single model self-auditing its own proof or experimental preregistration.
- Dispatches read-only adversarial proof/novelty/preregistration audits across independent
  model families (e.g., gpt-6.1-sol, gpt-6-luna, grok-4.6; deepseek disabled per user instruction).
- Respects mahu gateway per-key concurrency limit (default --max-workers 1) with automatic
  backoff retry on HTTP 429 (`gateway_concurrency_limit`) and SSE streaming (`stream=True`)
  so long reasoning traces never trigger socket read timeouts.
- Reads credentials strictly from untracked `.mahu.env` or environment variables; never prints keys.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import ssl
import sys
import time
import urllib.error
import urllib.request


def load_mahu_env() -> tuple[str, str]:
    root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [root / ".mahu.env", pathlib.Path.home() / ".config" / "mahu" / "mahu.env"]
    vals: dict[str, str] = {}
    for env_file in candidates:
        if env_file.exists():
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('"').strip("'")
            break
    base_url = os.environ.get("MAHU_BASE_URL") or vals.get("MAHU_BASE_URL", "")
    api_key = os.environ.get("MAHU_API_KEY") or vals.get("MAHU_API_KEY", "")
    if not base_url or not api_key:
        raise RuntimeError("Missing MAHU_BASE_URL or MAHU_API_KEY in environment or .mahu.env")
    return base_url.rstrip("/"), api_key


def _stream_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    chunk_timeout: int,
    wall_timeout: int,
) -> tuple[str, str]:
    ctx = ssl._create_unverified_context()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "TheoryScout-MultiAudit/1.1",
            "Authorization": f"Bearer {api_key}",
        },
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    t0 = time.time()
    with urllib.request.urlopen(req, context=ctx, timeout=chunk_timeout) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in ctype:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            msg = data["choices"][0]["message"]
            return msg.get("content") or "", msg.get("reasoning_content") or ""

        for raw_line in resp:
            if time.time() - t0 > wall_timeout:
                raise TimeoutError(f"Wall timeout {wall_timeout}s exceeded during streaming")
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = evt.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
    return "".join(content_parts), "".join(reasoning_parts)


def call_model(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    chunk_timeout: int = 180,
    wall_timeout: int = 900,
    max_retries: int = 4,
) -> dict:
    t0 = time.time()
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Calling {model} (attempt {attempt}/{max_retries})...", flush=True)
            content, reasoning = _stream_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                chunk_timeout=chunk_timeout,
                wall_timeout=wall_timeout,
            )
            if not content.strip() and reasoning.strip():
                content = reasoning
            return {
                "model": model,
                "ok": bool(content.strip()),
                "seconds": round(time.time() - t0, 2),
                "content": content,
                "reasoning_preview": reasoning[:500] if reasoning else "",
                "error": "" if content.strip() else "Empty content returned",
            }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
            last_err = f"HTTP {e.code}: {err_body}"
            if e.code in (429, 502, 503, 504) and attempt < max_retries:
                wait_s = 8 * attempt
                print(f"[{time.strftime('%H:%M:%S')}] {model} got HTTP {e.code}, retrying in {wait_s}s...", flush=True)
                time.sleep(wait_s)
                continue
            break
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries and ("timed out" in last_err.lower() or "reset" in last_err.lower()):
                wait_s = 6 * attempt
                print(f"[{time.strftime('%H:%M:%S')}] {model} transient error ({last_err}), retrying in {wait_s}s...", flush=True)
                time.sleep(wait_s)
                continue
            break

    return {
        "model": model,
        "ok": False,
        "seconds": round(time.time() - t0, 2),
        "error": last_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heterogeneous model audits via mahu API.")
    parser.add_argument("--models", default="gpt-6-luna,grok-4.6", help="Comma-separated models (do not use deepseek)")
    parser.add_argument("--role", default="audit", choices=["audit", "referee", "counterexample"])
    parser.add_argument("--files", nargs="+", required=True, help="Input markdown/code files to inspect")
    parser.add_argument("--out", required=True, help="Output markdown report path")
    parser.add_argument("--extra-instruction", default="", help="Extra focus prompt")
    parser.add_argument("--max-workers", type=int, default=2, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens")
    args = parser.parse_args()

    base_url, api_key = load_mahu_env()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if "deepseek" in m.lower():
            raise ValueError(f"Model {m!r} is disabled per user instruction (do not use deepseek).")

    bundle_parts = []
    for fp in args.files:
        p = pathlib.Path(fp)
        bundle_parts.append(f"===== FILE: {p.name} =====\n{p.read_text(encoding='utf-8')}")
    bundle = "\n\n".join(bundle_parts)

    role_prompts = {
        "audit": (
            "你是 TheoryScout 体系下的独立只读对抗审计员（Adversarial Proof & Preregistration Auditor）。"
            "你的唯一职责是挑刺、找反例、抓量词漏洞、抓平凡包装（把教材线性代数包装成新定理）、"
            "检查实验预注册是否存在作弊空间或不可证伪退路。必须给出明确结论：PASS / CONDITIONAL_PASS / FAIL，"
            "并列出：(1) 数学证明是否有漏洞或量词错误；(2) 核心新颖性是否真正站得住（哪条定理最硬、哪条太水应降级为引理）；"
            "(3) GPU 预注册实验启动前必须修正的漏洞。用严谨简洁的中文回答。"
        ),
        "counterexample": (
            "你是专门构造数学反例与边界崩溃案例的对抗审查员。针对给定定理、不等式与算法陈述，"
            "尝试在允许的参数域内构造反例（如奇异边界、零向量、BF16 饱和、跨层抵消、非对齐块边界等），"
            "检验每条命题的充要条件是否严密。用中文输出。"
        ),
        "referee": (
            "你是 ICML 2027 / NeurIPS 2027 理论与大模型系统交叉赛道的严苛审稿人（高置信度 Confidence=4）。"
            "参照顶会录用门槛（拒绝把平凡恒等式包装成大词、拒绝脱离真实模型的空洞上界），"
            "评估当前工作距离主会 Clear Accept 还差哪一个最关键的理论深化或真实模型机制实验。用中文输出。"
        ),
    }
    sys_prompt = role_prompts[args.role]
    user_prompt = (args.extra_instruction + "\n\n" if args.extra_instruction else "") + bundle

    results = []
    if args.max_workers <= 1:
        for m in models:
            r = call_model(base_url, api_key, m, sys_prompt, user_prompt, max_tokens=args.max_tokens)
            results.append(r)
            print(f"[{time.strftime('%H:%M:%S')}] Finished {m}: ok={r['ok']} ({r['seconds']}s)", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {
                ex.submit(call_model, base_url, api_key, m, sys_prompt, user_prompt, max_tokens=args.max_tokens): m
                for m in models
            }
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

    order = {m: idx for idx, m in enumerate(models)}
    results.sort(key=lambda r: order.get(r["model"], 999))

    out_lines = [
        f"# Heterogeneous Multi-Model Audit Report (`role={args.role}`)",
        "",
        f"- **Generated at**: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        f"- **Models**: `{', '.join(models)}`",
        f"- **Audited files**: `{', '.join(pathlib.Path(f).name for f in args.files)}`",
        "",
    ]
    for r in results:
        out_lines.append("---")
        out_lines.append(f"## Model: `{r['model']}` ({'OK' if r['ok'] else 'ERROR'}, {r['seconds']}s)")
        out_lines.append("")
        if r["ok"]:
            out_lines.append(r["content"].strip())
        else:
            out_lines.append(f"**Error**: `{r['error']}`")
        out_lines.append("")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Saved multi-model report to {out_path}", flush=True)
    for r in results:
        print(f"  - {r['model']}: ok={r['ok']} ({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

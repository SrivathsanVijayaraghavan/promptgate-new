"""Main PromptGate middleware entry point."""

import asyncio
import hashlib
import json
from datetime import datetime, timezone

from promptgate.aggregator import aggregate
from promptgate.detector.intent import IntentClassifier
from promptgate.detector.rule_based import RuleBasedDetector
from promptgate.detector.semantic import SemanticDetector
from promptgate.parser.input_parser import parse_input
from promptgate.policy import evaluate
from promptgate.response import build_response
from promptgate.scorer import score


class PromptGate:
    """AI security middleware that classifies prompt injection risk before LLM access.

    Runs a three-layer detection pipeline:
      1. Rule-based — fast keyword/phrase matching against pattern files.
      2. Semantic   — sentence-embedding similarity against known attack library.
      3. Intent     — fine-tuned DistilBERT classifier for implicit/conversational
                      injections that bypass vocabulary-based detection entirely.

    All layers feed signals into the same accumulation model. Each layer degrades
    gracefully when its optional dependencies are not installed.

    Phase 6 additions:
      - check_batch()  — efficient multi-input processing
      - acheck()       — async wrapper for FastAPI / LangChain integration
      - acheck_batch() — async batch wrapper
      - history        — conversation context for multi-turn attack detection
      - log_mode       — privacy-safe JSONL audit logging
      - on_block / on_flag / on_review / on_allow / on_error — callback hooks
    """

    def __init__(
        self,
        thresholds: dict | None = None,
        skip_semantic: bool = False,
        skip_intent: bool = False,
        semantic_threshold: float = 0.65,
        intent_threshold: float = 0.70,
        log_mode: bool = False,
        log_path: str = "./promptgate_audit.jsonl",
        on_block=None,
        on_flag=None,
        on_review=None,
        on_allow=None,
        on_error=None,
    ) -> None:
        """Initialise PromptGate with optional configuration.

        Args:
            thresholds: Optional dict to override DEFAULT_THRESHOLDS.
                        Accepted keys: block, review, flag.
                        Unspecified keys fall back to DEFAULT_THRESHOLDS.
            skip_semantic: If True, the semantic detector is never called
                           regardless of whether it is installed.
            skip_intent: If True, the intent classifier is never called
                         regardless of whether it is installed.
            semantic_threshold: Cosine similarity cutoff passed to
                                SemanticDetector. Default 0.65.
            intent_threshold: INJECTION probability cutoff passed to
                              IntentClassifier. Default 0.70.
            log_mode: If True, appends a privacy-safe audit record to
                      log_path after every check() call. Raw input is
                      never logged — only its sha256 hash and metadata.
                      Default False.
            log_path: Path to the JSONL audit log file. Only used when
                      log_mode=True. Default './promptgate_audit.jsonl'.
            on_block: Optional callable(result: dict) called when the
                      decision is BLOCK.
            on_flag: Optional callable(result: dict) called when the
                     decision is FLAG.
            on_review: Optional callable(result: dict) called when the
                       decision is REVIEW.
            on_allow: Optional callable(result: dict) called when the
                      decision is ALLOW.
            on_error: Optional callable(exc: Exception) called when any
                      hook raises an exception. If on_error itself raises,
                      the exception is silently swallowed. Hook failures
                      never affect the detection result.
        """
        self.thresholds = thresholds
        self.skip_semantic = skip_semantic
        self.skip_intent = skip_intent
        self.log_mode = log_mode
        self.log_path = log_path
        self.on_block = on_block
        self.on_flag = on_flag
        self.on_review = on_review
        self.on_allow = on_allow
        self.on_error = on_error
        self.rule_detector = RuleBasedDetector()
        self.semantic_detector = SemanticDetector(threshold=semantic_threshold)
        self.intent_detector = IntentClassifier(threshold=intent_threshold)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        cleaned: str,
        intent_input: str,
    ) -> dict:
        """Run detection layers and build response for one pre-parsed input.

        Shared by check(), check_batch(), and the async wrappers. Accepts
        the already-cleaned text and a (possibly history-enriched) intent
        input string so callers control what each layer sees.

        Args:
            cleaned: Normalised lowercase text from the parser. Used by
                     rule-based and semantic layers.
            intent_input: Text passed to the intent classifier. May include
                          prepended conversation history.

        Returns:
            Structured response dict with exactly 7 keys.
        """
        # Layer 1 — Rule-based
        rule_signals = self.rule_detector.detect(cleaned)
        rule_checked = (
            f"rule_based: {len(rule_signals)} pattern{'s' if len(rule_signals) != 1 else ''} matched"
            if rule_signals
            else "rule_based: no injection patterns found"
        )

        # Layer 2 — Semantic
        semantic_signals = []
        if self.skip_semantic:
            semantic_checked = "semantic: skipped by configuration"
        elif not self.semantic_detector.is_available():
            semantic_checked = "semantic: skipped (not installed)"
        else:
            semantic_signals = self.semantic_detector.detect(cleaned)
            semantic_checked = (
                "semantic: similar attack found above threshold"
                if semantic_signals
                else "semantic: no similar attacks found"
            )

        # Layer 3 — Intent
        intent_signals = []
        if self.skip_intent:
            intent_checked = "intent: skipped by configuration"
        elif not self.intent_detector.is_available():
            intent_checked = "intent: skipped (model not trained or not installed)"
        else:
            intent_signals = self.intent_detector.detect(intent_input)
            intent_checked = (
                "intent: injection intent detected above threshold"
                if intent_signals
                else "intent: no injection intent detected"
            )

        all_signals = rule_signals + semantic_signals + intent_signals
        signals_checked = [rule_checked, semantic_checked, intent_checked]

        aggregated = aggregate(all_signals)
        signals = aggregated["signals"]
        threat_categories = aggregated["threat_categories"]

        risk_score = score(signals)
        decision = evaluate(risk_score, self.thresholds)

        return build_response(
            decision=decision,
            risk_score=risk_score,
            threat_categories=threat_categories,
            signals=signals,
            signals_checked=signals_checked,
        )

    def _build_intent_input(self, cleaned: str, history: list[dict] | None) -> str:
        """Prepend the last 3 conversation turns to the cleaned input.

        Passes enriched context to the intent classifier only. Rule-based
        and semantic layers always receive the raw cleaned input — history
        increases noise for pattern matching and embedding comparison.

        DistilBERT truncates at 512 tokens. By prepending history and
        appending the current message last, truncation (when it occurs)
        removes older context rather than the current input. This ensures
        the current message is always represented in the classification.

        Invalid turn dicts (missing 'role' or 'content') are silently
        skipped — malformed history never crashes detection.

        Args:
            cleaned: Normalised current user input.
            history: Optional list of prior conversation turns.
                     Each turn should have 'role' and 'content' keys.

        Returns:
            String to pass to the intent classifier. Equals cleaned when
            history is None or empty.
        """
        if not history:
            return cleaned

        recent = history[-3:]
        context_parts = [
            f"{turn['role']}: {turn['content']}"
            for turn in recent
            if isinstance(turn, dict) and "role" in turn and "content" in turn
        ]
        if not context_parts:
            return cleaned

        context_parts.append(f"user: {cleaned}")
        return " | ".join(context_parts)

    def _log_decision(self, raw_input: str, result: dict) -> None:
        """Append one privacy-safe audit record to the JSONL log file.

        Never logs raw input text. Logs the sha256 hash of raw input so
        identical inputs can be correlated across requests without exposing
        prompt content. All other fields are metadata only.

        Log record fields:
            timestamp       — ISO 8601 UTC timestamp
            input_hash      — "sha256:" + hex digest of raw input
            decision        — ALLOW / FLAG / REVIEW / BLOCK
            confidence      — accumulated risk score [0.0, 1.0]
            risk_level      — minimal / low / medium / high
            threat_categories — list of detected threat categories
            signal_count    — number of signals that fired
            signals_checked — audit strings from each detection layer

        Logging failures (disk full, permission error, etc.) are silently
        swallowed — they must never crash or delay the detection pipeline.

        Args:
            raw_input: Original unmodified user input. Hashed, never stored.
            result: The result dict from the detection pipeline.
        """
        if not self.log_mode:
            return

        input_hash = "sha256:" + hashlib.sha256(
            raw_input.encode("utf-8")
        ).hexdigest()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": input_hash,
            "decision": result["decision"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "threat_categories": result["threat_categories"],
            "signal_count": len(result["signals"]),
            "signals_checked": result["signals_checked"],
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _call_hook(self, hook, payload) -> None:
        """Call a callback hook safely, routing exceptions to on_error.

        Hook exceptions are passed to on_error if set; otherwise silently
        swallowed. If on_error itself raises, that exception is also
        swallowed. Detection results are never affected by hook behavior.

        Args:
            hook: Callable to invoke, or None.
            payload: Argument to pass to the hook.
        """
        if hook is None:
            return
        try:
            hook(payload)
        except Exception as exc:
            if self.on_error is not None:
                try:
                    self.on_error(exc)
                except Exception:
                    pass

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> dict:
        """Run the full three-layer risk classification pipeline.

        Pipeline steps:
          1. parse    — normalise, lowercase, detect encoding anomalies
          2. rule     — keyword/phrase matching against pattern files
          3. semantic — embedding similarity against known attack library
          4. intent   — fine-tuned classifier; receives history context
          5. aggregate — group signals into threat categories
          6. score    — accumulate severities, clamp to [0.0, 1.0]
          7. decide   — map score to ALLOW / FLAG / REVIEW / BLOCK
          8. log      — append audit record if log_mode=True
          9. hook     — call the matching decision callback if set
         10. build    — assemble and return the final structured response

        Args:
            user_input: Raw user prompt text.
            history: Optional list of previous conversation turns for
                     multi-turn attack detection. Each turn is a dict
                     with keys 'role' (str) and 'content' (str).
                     The last 3 turns are prepended to the input before
                     intent classification. Rule-based and semantic layers
                     always analyse the current input only.
                     Invalid turn dicts are silently skipped.

        Returns:
            Structured response dict with exactly 7 keys:
            decision, confidence, risk_level, threat_categories,
            signals, signals_checked, message.
        """
        parsed = parse_input(user_input)
        cleaned = parsed["cleaned_text"]
        intent_input = self._build_intent_input(cleaned, history)

        result = self._run_pipeline(cleaned, intent_input)

        self._log_decision(user_input, result)

        hook_map = {
            "BLOCK":  self.on_block,
            "FLAG":   self.on_flag,
            "REVIEW": self.on_review,
            "ALLOW":  self.on_allow,
        }
        self._call_hook(hook_map.get(result["decision"]), result)

        return result

    def check_batch(self, inputs: list[str]) -> list[dict]:
        """Check multiple prompts efficiently in a single call.

        Runs the full three-layer pipeline across all inputs. The semantic
        detector encodes all inputs in one batch, making this significantly
        faster than calling check() in a loop when the semantic layer is
        active. Rule-based and intent detectors are called per input as
        they have no batching API.

        Log mode and callback hooks apply per result, identical to check().

        Args:
            inputs: List of raw user prompt strings.

        Returns:
            List of result dicts in the same order as inputs. Each dict has
            the same 7-key structure as check(). Returns [] if inputs is empty.
        """
        if not inputs:
            return []

        # Parse all inputs upfront
        parsed_list = [parse_input(inp) for inp in inputs]
        cleaned_list = [p["cleaned_text"] for p in parsed_list]

        # Semantic batch — one model.encode() call for all inputs
        if not self.skip_semantic and self.semantic_detector.is_available():
            semantic_signals_list = self.semantic_detector.detect_batch(cleaned_list)
            # detect_batch returns [] when inputs is non-empty but all are
            # blank — normalise to list of empty lists in that case.
            if not semantic_signals_list:
                semantic_signals_list = [[] for _ in inputs]
        else:
            semantic_signals_list = [[] for _ in inputs]

        results = []
        for i, (raw_input, cleaned) in enumerate(zip(inputs, cleaned_list)):
            # Layer 1 — Rule-based (per input)
            rule_signals = self.rule_detector.detect(cleaned)
            rule_checked = (
                f"rule_based: {len(rule_signals)} pattern{'s' if len(rule_signals) != 1 else ''} matched"
                if rule_signals
                else "rule_based: no injection patterns found"
            )

            # Layer 2 — Semantic (already computed in batch above)
            semantic_signals = semantic_signals_list[i] if i < len(semantic_signals_list) else []
            if self.skip_semantic:
                semantic_checked = "semantic: skipped by configuration"
            elif not self.semantic_detector.is_available():
                semantic_checked = "semantic: skipped (not installed)"
            else:
                semantic_checked = (
                    "semantic: similar attack found above threshold"
                    if semantic_signals
                    else "semantic: no similar attacks found"
                )

            # Layer 3 — Intent (per input, no batch API)
            intent_signals = []
            if self.skip_intent:
                intent_checked = "intent: skipped by configuration"
            elif not self.intent_detector.is_available():
                intent_checked = "intent: skipped (model not trained or not installed)"
            else:
                intent_signals = self.intent_detector.detect(cleaned)
                intent_checked = (
                    "intent: injection intent detected above threshold"
                    if intent_signals
                    else "intent: no injection intent detected"
                )

            all_signals = rule_signals + semantic_signals + intent_signals
            signals_checked = [rule_checked, semantic_checked, intent_checked]

            aggregated = aggregate(all_signals)
            signals = aggregated["signals"]
            threat_categories = aggregated["threat_categories"]

            risk_score = score(signals)
            decision = evaluate(risk_score, self.thresholds)

            result = build_response(
                decision=decision,
                risk_score=risk_score,
                threat_categories=threat_categories,
                signals=signals,
                signals_checked=signals_checked,
            )

            self._log_decision(raw_input, result)

            hook_map = {
                "BLOCK":  self.on_block,
                "FLAG":   self.on_flag,
                "REVIEW": self.on_review,
                "ALLOW":  self.on_allow,
            }
            self._call_hook(hook_map.get(decision), result)

            results.append(result)

        return results

    async def acheck(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> dict:
        """Async version of check().

        Runs the synchronous detection pipeline in a thread pool executor
        to avoid blocking the event loop during model inference. Safe to
        call concurrently from async code.

        NOTE: PyTorch inference holds the GIL. Concurrent acheck() calls
        will serialize rather than run in true parallel. For high-throughput
        concurrent workloads, create multiple PromptGate instances.

        FastAPI usage example::

            app = FastAPI()
            gate = PromptGate()

            @app.post("/chat")
            async def chat(message: str):
                result = await gate.acheck(message)
                if result["decision"] != "ALLOW":
                    raise HTTPException(
                        status_code=403, detail=result["message"]
                    )
                return {"response": await call_your_llm(message)}

        Args:
            user_input: Raw user prompt text.
            history: Optional conversation history (see check() for details).

        Returns:
            Same 7-key result dict as check().
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.check(user_input, history)
        )

    async def acheck_batch(self, inputs: list[str]) -> list[dict]:
        """Async version of check_batch().

        Runs check_batch() in a thread pool executor. Inherits the same
        semantic batching optimisation as check_batch().

        NOTE: PyTorch inference holds the GIL. See acheck() for details
        on concurrency limitations.

        Args:
            inputs: List of raw user prompt strings.

        Returns:
            List of result dicts in the same order as inputs.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check_batch, inputs)
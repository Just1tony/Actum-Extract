"""LLM-backed structured extraction engine with a self-healing retry loop.

The central design idea of this module is that a single failed Pydantic
validation is not a terminal failure — it is *signal*. When the LLM returns
a payload that fails validation (bad arithmetic, malformed NIP, inconsistent
totals), we do not simply retry blindly; we capture the exact
``ValidationError``, translate it into a precise, human-readable correction
instruction, and append it to the conversation before re-invoking the model.
This "self-healing loop" consistently recovers extractions that would
otherwise fail outright, and it is the single highest-leverage reliability
technique available when using LLMs for structured generation.

The loop is implemented on top of two composable layers:

* ``instructor`` patches the OpenAI client so that a Pydantic model can be
  passed directly as ``response_model``; on the client side it validates the
  raw LLM tool-call arguments against that model and raises
  ``pydantic.ValidationError`` on failure, and (natively) supports
  ``max_retries`` for exactly this purpose. We additionally wrap this in our
  own explicit ``tenacity`` loop so we have full control over backoff,
  logging, and prompt-repair strategy at each attempt — useful for cases
  running against providers/instructor versions where we want engine-level
  control rather than relying solely on the library's internal retries.
* ``tenacity`` provides the outer retry policy (attempt budget, exponential
  backoff, and structured logging hooks) around upstream errors such as
  rate limits or transient network failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import instructor
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from core.config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)

_EXTRACTION_SYSTEM_PROMPT = """\
You are a meticulous financial document analyst specializing in structured \
data extraction from commercial invoices, contracts, and financial reports.

Your task is to read the provided document text in full and extract every \
requested field with complete fidelity to the source. Rules you must follow:

1. Never invent, estimate, or hallucinate a value that is not present in \
the source text. If a field is genuinely absent, use null (for optional \
fields) rather than guessing.
2. All monetary and quantity fields must reflect the EXACT numbers printed \
in the document — do not round beyond what the schema requires.
3. Arithmetic fields (line totals, VAT, grand totals) must be internally \
consistent: quantity * unit_price = net amount, net + VAT = gross amount. \
If the source document itself contains an arithmetic error, extract the \
values exactly as printed rather than "correcting" them — but ensure your \
OWN reported net/VAT/gross figures are computed consistently with the \
quantities and rates you extract.
4. Strip formatting artifacts (currency symbols, thousands separators) from \
numeric fields, but preserve exact decimal precision.
5. Extract every line item — do not summarize, merge, or omit any row from \
the source table.
"""


class UpstreamError(Exception):
    """Raised when the LLM provider fails after exhausting all retries.

    This wraps the underlying provider exception (rate limits, timeouts,
    connection failures, 5xx responses) so that API-layer code can catch a
    single, well-defined exception type regardless of which specific OpenAI
    SDK error occurred.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        """Initializes the error with a descriptive message and optional cause.

        Args:
            message: Human-readable description of the upstream failure.
            cause: The original exception raised by the OpenAI SDK, if any.
        """
        super().__init__(message)
        self.cause = cause


class ExtractionValidationError(Exception):
    """Raised when the self-healing loop exhausts all attempts.

    Signals that the LLM was unable to produce a payload satisfying the
    target Pydantic schema even after ``max_retries`` correction attempts.
    """

    def __init__(
        self,
        message: str,
        *,
        last_validation_error: ValidationError,
        attempts_made: int,
    ) -> None:
        """Initializes the error with diagnostic details for the caller.

        Args:
            message: Human-readable summary of the failure.
            last_validation_error: The final ``ValidationError`` raised by
                Pydantic on the last attempt, preserved for API responses
                and debugging.
            attempts_made: Total number of extraction attempts performed.
        """
        super().__init__(message)
        self.last_validation_error = last_validation_error
        self.attempts_made = attempts_made


@dataclass(slots=True)
class ExtractionResult:
    """Container for a successful extraction and its execution metadata.

    Attributes:
        data: The validated, populated Pydantic model instance.
        attempts_made: Total number of LLM calls performed, including the
            initial attempt (1 means it succeeded on the first try).
        self_healed: ``True`` if at least one validation failure occurred
            and was successfully corrected via the self-healing loop.
        total_latency_seconds: Wall-clock time spent across all attempts.
        model_used: The LLM model identifier used for the extraction.
    """

    data: BaseModel
    attempts_made: int
    self_healed: bool
    total_latency_seconds: float
    model_used: str
    validation_errors_encountered: list[str] = field(default_factory=list)


def _format_validation_feedback(error: ValidationError) -> str:
    """Translates a Pydantic ``ValidationError`` into corrective LLM feedback.

    Each individual error is rendered as a precise, actionable instruction
    referencing the exact field path and the reason it failed, so the model
    can target its correction rather than regenerating the entire document
    from scratch with only vague guidance.

    Args:
        error: The validation error raised while parsing the LLM's output
            against the target response model.

    Returns:
        A formatted multi-line string suitable for injection into the next
        user turn of the conversation.
    """
    lines: list[str] = [
        "Your previous response FAILED schema validation with the following "
        f"{error.error_count()} error(s). You MUST fix every one of them in "
        "your next response. Re-emit the COMPLETE, corrected object — do not "
        "omit any fields that were previously correct.",
        "",
    ]
    for issue in error.errors():
        field_path = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"- Field '{field_path}': {issue['msg']}")
    return "\n".join(lines)


class ExtractionEngine:
    """Structured extraction engine wrapping ``instructor`` + ``tenacity``.

    This class owns the async OpenAI client (patched by ``instructor``) and
    exposes a single public entry point, ``extract``, that performs
    self-healing structured extraction against an arbitrary Pydantic
    response model.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initializes the engine and its patched async LLM client.

        Args:
            settings: Optional explicit settings instance, primarily for
                dependency injection in tests. Defaults to the process-wide
                cached settings.
        """
        self._settings = settings or get_settings()

        raw_client = AsyncOpenAI(
            api_key=self._settings.openai_api_key.get_secret_value(),
            base_url=self._settings.openai_base_url,
            timeout=self._settings.llm_request_timeout_seconds,
        )
        # ``instructor.from_openai`` patches ``.chat.completions.create`` so
        # it accepts a ``response_model`` kwarg and returns a validated
        # Pydantic instance directly instead of a raw completion object.
        self._client = instructor.from_openai(
            raw_client, mode=instructor.Mode.TOOLS
        )

    async def extract(
        self,
        *,
        document_text: str,
        response_model: type[ResponseModelT],
        max_self_healing_attempts: int | None = None,
    ) -> ExtractionResult:
        """Extracts structured data from raw document text.

        Implements a two-layer retry strategy:

        1. **Self-healing loop (inner)**: on a Pydantic ``ValidationError``,
           the conversation is extended with the assistant's failed
           (best-effort) output plus a precise correction instruction, and
           the model is called again. This loop runs up to
           ``max_self_healing_attempts`` times.
        2. **Upstream resilience (outer)**: each individual LLM call is
           wrapped in a ``tenacity`` retry policy that handles transient
           provider failures (rate limits, timeouts, connection errors)
           with exponential backoff and jitter, independent of the
           self-healing attempt count.

        Args:
            document_text: The cleaned, extracted plain text of the source
                document to analyze.
            response_model: The Pydantic model class describing the desired
                output structure.
            max_self_healing_attempts: Overrides the configured default
                number of self-healing attempts, primarily for testing.

        Returns:
            An ``ExtractionResult`` wrapping the validated model instance
            and execution metadata.

        Raises:
            ExtractionValidationError: If every self-healing attempt is
                exhausted without producing a schema-valid response.
            UpstreamError: If the LLM provider itself fails (rate limit,
                timeout, connection error, 5xx) after exhausting the
                upstream resilience retry budget.
        """
        attempt_budget = (
            max_self_healing_attempts
            if max_self_healing_attempts is not None
            else self._settings.extraction_max_retries
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract structured data from the following document "
                    f"text:\n\n---BEGIN DOCUMENT---\n{document_text}\n"
                    "---END DOCUMENT---"
                ),
            },
        ]

        start_time = time.monotonic()
        attempts_made = 0
        self_healed = False
        validation_errors_encountered: list[str] = []
        last_validation_error: ValidationError | None = None

        for attempt_index in range(attempt_budget + 1):
            attempts_made += 1
            try:
                result = await self._call_llm_with_resilience(
                    messages=messages, response_model=response_model
                )
                elapsed = time.monotonic() - start_time
                logger.info(
                    "extraction_succeeded",
                    extra={
                        "attempts_made": attempts_made,
                        "self_healed": self_healed,
                        "latency_seconds": round(elapsed, 3),
                        "model": self._settings.extraction_model,
                    },
                )
                return ExtractionResult(
                    data=result,
                    attempts_made=attempts_made,
                    self_healed=self_healed,
                    total_latency_seconds=elapsed,
                    model_used=self._settings.extraction_model,
                    validation_errors_encountered=validation_errors_encountered,
                )
            except ValidationError as validation_error:
                last_validation_error = validation_error
                validation_errors_encountered.append(str(validation_error))
                logger.warning(
                    "extraction_validation_failed",
                    extra={
                        "attempt": attempt_index + 1,
                        "attempt_budget": attempt_budget + 1,
                        "error_count": validation_error.error_count(),
                    },
                )

                if attempt_index >= attempt_budget:
                    break

                self_healed = True
                feedback = _format_validation_feedback(validation_error)
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I attempted to extract the document but my "
                            "output did not satisfy the required schema."
                        ),
                    }
                )
                messages.append({"role": "user", "content": feedback})

        elapsed = time.monotonic() - start_time
        logger.error(
            "extraction_exhausted_self_healing_budget",
            extra={"attempts_made": attempts_made, "latency_seconds": round(elapsed, 3)},
        )
        assert last_validation_error is not None  # guaranteed by loop structure
        raise ExtractionValidationError(
            f"Extraction failed schema validation after {attempts_made} "
            "attempt(s); self-healing budget exhausted.",
            last_validation_error=last_validation_error,
            attempts_made=attempts_made,
        )

    async def _call_llm_with_resilience(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """Performs a single structured extraction call with upstream retries.

        This wraps exactly one logical extraction attempt (which itself may
        involve ``instructor``'s own low-level retries against malformed
        tool-call JSON) in a ``tenacity`` policy that retries on transient
        provider-side failures only. A ``pydantic.ValidationError`` is
        intentionally NOT retried here — it is allowed to propagate up to
        the self-healing loop in ``extract``, which handles it with
        prompt-level correction rather than a blind retry.

        Args:
            messages: The full conversation history to send to the LLM.
            response_model: The Pydantic model class to validate the
                response against.

        Returns:
            The validated Pydantic model instance produced by the LLM.

        Raises:
            ValidationError: Propagated unmodified from ``instructor`` when
                the model's output fails schema validation.
            UpstreamError: If the provider fails with a retryable error type
                across all attempts of the upstream resilience policy.
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10),
                retry=retry_if_exception_type(
                    (RateLimitError, APITimeoutError, APIConnectionError)
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._client.chat.completions.create(
                        model=self._settings.extraction_model,
                        messages=messages,
                        response_model=response_model,
                        temperature=self._settings.extraction_temperature,
                        max_tokens=self._settings.extraction_max_tokens,
                        max_retries=0,  # instructor-internal retries disabled;
                        # we own the self-healing loop explicitly at the
                        # engine layer for full observability and control.
                    )
        except ValidationError:
            raise
        except RetryError as retry_error:
            underlying = retry_error.last_attempt.exception()
            logger.error(
                "upstream_provider_exhausted_retries",
                extra={"underlying_error": str(underlying)},
            )
            raise UpstreamError(
                f"LLM provider failed after retry budget exhausted: {underlying}",
                cause=underlying,
            ) from retry_error
        except APIStatusError as status_error:
            logger.error(
                "upstream_provider_status_error",
                extra={
                    "status_code": status_error.status_code,
                    "response_body": str(status_error.response.text)[:500],
                },
            )
            raise UpstreamError(
                f"LLM provider returned HTTP {status_error.status_code}: "
                f"{status_error.message}",
                cause=status_error,
            ) from status_error

        raise UpstreamError(
            "LLM provider call completed without producing a result; this "
            "indicates an unexpected control-flow state in the resilience loop."
        )

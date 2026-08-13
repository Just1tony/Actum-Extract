"""ActumAI Extract — FastAPI application entry point.

Exposes the public HTTP surface of the Intelligent Document Processing &
Structured Extraction Engine:

* ``POST /api/v1/extract/text`` — structured extraction from raw text.
* ``POST /api/v1/extract/pdf``  — structured extraction from an uploaded PDF.
* ``GET  /health``              — liveness/readiness probe.

Cross-cutting concerns (request ID propagation, structured access logging,
and translation of internal exceptions into clean HTTP error responses) are
implemented once here as middleware and exception handlers, rather than
being duplicated inside every route handler.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from core.config import Settings, get_settings
from core.logger import configure_logging, generate_request_id, get_logger, request_id_ctx
from extractors.engine import (
    ExtractionEngine,
    ExtractionValidationError,
    UpstreamError,
)
from schemas.invoice import InvoiceExtractionResult
from services.document_parser import DocumentParserService, DocumentParsingError

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manages application startup and shutdown lifecycle.

    Configures structured logging exactly once at process startup and
    instantiates singleton service objects, attaching them to
    ``app.state`` so route handlers can access them without re-constructing
    HTTP clients per request.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI for the duration of the application's
        running lifetime.
    """
    configure_logging()
    settings = get_settings()
    app.state.settings = settings
    app.state.extraction_engine = ExtractionEngine(settings=settings)
    app.state.document_parser = DocumentParserService(settings=settings)
    logger.info(
        "application_startup",
        extra={"environment": settings.environment, "extraction_model": settings.extraction_model},
    )
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title="ActumAI Extract",
    description=(
        "Intelligent Document Processing & Structured Extraction Engine. "
        "Transforms unstructured business documents into rigorously "
        "validated, nested JSON using LLMs, Instructor, and Pydantic v2."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Injects a correlation ID and logs request timing for every call.

    A new request ID is generated per inbound request (or reused from an
    ``X-Request-ID`` header if the caller supplied one, enabling end-to-end
    tracing across service boundaries), bound to the ``request_id_ctx``
    context variable so all logging within this request's async task
    automatically carries it, and echoed back on the response for client-side
    correlation.

    Args:
        request: The incoming Starlette/FastAPI request.
        call_next: The next handler in the middleware chain.

    Returns:
        The HTTP response, with an ``X-Request-ID`` header attached and
        request timing logged.
    """
    incoming_request_id = request.headers.get("X-Request-ID")
    request_id = incoming_request_id or generate_request_id()
    token = request_id_ctx.set(request_id)

    start_time = time.monotonic()
    logger.info(
        "request_started",
        extra={"method": request.method, "path": request.url.path},
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)
        logger.exception(
            "request_failed_unhandled",
            extra={"method": request.method, "path": request.url.path, "duration_ms": elapsed_ms},
        )
        raise
    finally:
        request_id_ctx.reset(token)

    elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": elapsed_ms,
        },
    )
    return response


# =============================================================================
# Dependencies
# =============================================================================


def get_extraction_engine(request: Request) -> ExtractionEngine:
    """FastAPI dependency providing the process-wide extraction engine.

    Args:
        request: The current request, used to access ``app.state``.

    Returns:
        The singleton ``ExtractionEngine`` instance created at startup.
    """
    return request.app.state.extraction_engine


def get_document_parser(request: Request) -> DocumentParserService:
    """FastAPI dependency providing the process-wide document parser.

    Args:
        request: The current request, used to access ``app.state``.

    Returns:
        The singleton ``DocumentParserService`` instance created at startup.
    """
    return request.app.state.document_parser


# =============================================================================
# Request / response contracts
# =============================================================================


class TextExtractionRequest(BaseModel):
    """Request body for the raw-text extraction endpoint.

    Attributes:
        document_text: The raw, unstructured text of the business document
            to extract structured data from.
    """

    document_text: str = Field(..., min_length=1, max_length=500_000)


class ExtractionMetadata(BaseModel):
    """Execution metadata returned alongside every successful extraction.

    Attributes:
        attempts_made: Total number of LLM calls performed.
        self_healed: Whether the self-healing retry loop had to correct at
            least one validation failure.
        total_latency_seconds: Wall-clock extraction time.
        model_used: The LLM model identifier used.
    """

    attempts_made: int
    self_healed: bool
    total_latency_seconds: float
    model_used: str


class ExtractionResponse(BaseModel):
    """Top-level response envelope for successful extraction requests.

    Attributes:
        data: The validated, structured extraction result.
        metadata: Execution metadata describing how the result was produced.
    """

    data: InvoiceExtractionResult
    metadata: ExtractionMetadata


class HealthResponse(BaseModel):
    """Response body for the health check endpoint.

    Attributes:
        status: Literal ``"ok"`` when the service is healthy.
        environment: The active deployment environment.
        extraction_model: The configured extraction model identifier.
    """

    status: str
    environment: str
    extraction_model: str


class ErrorResponse(BaseModel):
    """Standardized error envelope returned by all failure paths.

    Attributes:
        error: Short machine-readable error category.
        message: Human-readable description of the failure.
        details: Optional structured details (e.g. per-field validation
            errors) for programmatic consumption.
    """

    error: str
    message: str
    details: list[dict] | None = None


# =============================================================================
# Routes
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Reports service liveness and active configuration.

    Args:
        settings: Injected application settings.

    Returns:
        A ``HealthResponse`` confirming the service is running.
    """
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        extraction_model=settings.extraction_model,
    )


@app.post(
    "/api/v1/extract/text",
    response_model=ExtractionResponse,
    responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    tags=["Extraction"],
)
async def extract_from_text(
    payload: TextExtractionRequest,
    engine: ExtractionEngine = Depends(get_extraction_engine),
    parser: DocumentParserService = Depends(get_document_parser),
) -> ExtractionResponse:
    """Extracts structured invoice/contract data from raw document text.

    Args:
        payload: The request body containing the raw document text.
        engine: Injected extraction engine.
        parser: Injected document parser, used here only for text
            normalization (whitespace cleanup) prior to extraction.

    Returns:
        An ``ExtractionResponse`` wrapping the validated structured data and
        execution metadata.

    Raises:
        HTTPException: Translated by the global exception handlers from
            ``DocumentParsingError``, ``ExtractionValidationError``, or
            ``UpstreamError`` raised by the underlying services.
    """
    cleaned_text = parser.clean_raw_text(payload.document_text)
    result = await engine.extract(
        document_text=cleaned_text, response_model=InvoiceExtractionResult
    )
    return ExtractionResponse(
        data=result.data,
        metadata=ExtractionMetadata(
            attempts_made=result.attempts_made,
            self_healed=result.self_healed,
            total_latency_seconds=result.total_latency_seconds,
            model_used=result.model_used,
        ),
    )


@app.post(
    "/api/v1/extract/pdf",
    response_model=ExtractionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    tags=["Extraction"],
)
async def extract_from_pdf(
    file: UploadFile = File(..., description="A PDF file containing the business document."),
    engine: ExtractionEngine = Depends(get_extraction_engine),
    parser: DocumentParserService = Depends(get_document_parser),
    settings: Settings = Depends(get_settings),
) -> ExtractionResponse:
    """Extracts structured invoice/contract data from an uploaded PDF file.

    Args:
        file: The uploaded PDF document.
        engine: Injected extraction engine.
        parser: Injected document parser used to convert the PDF into
            clean plain text.
        settings: Injected application settings, used to enforce the
            maximum upload size.

    Returns:
        An ``ExtractionResponse`` wrapping the validated structured data and
        execution metadata.

    Raises:
        HTTPException: 400 if the file is not a PDF; 413 if it exceeds the
            configured size limit; other errors translated by the global
            exception handlers.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{file.content_type}'. Only PDF files are accepted.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(file_bytes)} bytes exceeds the maximum "
                f"allowed {settings.max_upload_size_bytes} bytes."
            ),
        )

    parsed = parser.parse_pdf_bytes(file_bytes, source_filename=file.filename or "unknown.pdf")
    result = await engine.extract(
        document_text=parsed.text, response_model=InvoiceExtractionResult
    )
    return ExtractionResponse(
        data=result.data,
        metadata=ExtractionMetadata(
            attempts_made=result.attempts_made,
            self_healed=result.self_healed,
            total_latency_seconds=result.total_latency_seconds,
            model_used=result.model_used,
        ),
    )


# =============================================================================
# Global exception handlers
# =============================================================================


@app.exception_handler(DocumentParsingError)
async def handle_document_parsing_error(
    request: Request, exc: DocumentParsingError
) -> JSONResponse:
    """Translates document parsing failures into a 400 response.

    Args:
        request: The current request.
        exc: The raised document parsing error.

    Returns:
        A JSON response describing the parsing failure.
    """
    logger.warning("document_parsing_error", extra={"error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="document_parsing_error", message=str(exc)
        ).model_dump(),
    )


@app.exception_handler(ExtractionValidationError)
async def handle_extraction_validation_error(
    request: Request, exc: ExtractionValidationError
) -> JSONResponse:
    """Translates exhausted self-healing attempts into a 422 response.

    The underlying Pydantic error details are surfaced in ``details`` so API
    consumers can see exactly which fields the model could not satisfy even
    after correction attempts.

    Args:
        request: The current request.
        exc: The raised extraction validation error.

    Returns:
        A JSON response describing the validation failure with structured
        field-level details.
    """
    logger.error(
        "extraction_validation_error",
        extra={"attempts_made": exc.attempts_made, "error": str(exc)},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="extraction_validation_failed",
            message=(
                f"The document could not be extracted into a valid schema "
                f"after {exc.attempts_made} attempt(s), including "
                f"self-healing corrections."
            ),
            details=exc.last_validation_error.errors(include_url=False),
        ).model_dump(mode="json"),
    )


@app.exception_handler(UpstreamError)
async def handle_upstream_error(request: Request, exc: UpstreamError) -> JSONResponse:
    """Translates LLM provider failures into a 502 Bad Gateway response.

    Args:
        request: The current request.
        exc: The raised upstream provider error.

    Returns:
        A JSON response indicating an upstream dependency failure.
    """
    logger.error("upstream_error", extra={"error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorResponse(
            error="upstream_provider_error",
            message="The upstream LLM provider failed to service this request. Please retry.",
        ).model_dump(),
    )


@app.exception_handler(ValidationError)
async def handle_pydantic_validation_error(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Translates any uncaught Pydantic validation error into a 422 response.

    Args:
        request: The current request.
        exc: The raised Pydantic validation error.

    Returns:
        A JSON response with structured field-level validation details.
    """
    logger.warning("request_validation_error", extra={"error_count": exc.error_count()})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="request_validation_failed",
            message="The request payload failed validation.",
            details=exc.errors(include_url=False),
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler ensuring no unhandled exception leaks a raw traceback.

    Args:
        request: The current request.
        exc: The unhandled exception.

    Returns:
        A generic 500 JSON response; full details are captured in the
        structured logs (with request ID correlation) rather than exposed
        to the client.
    """
    logger.exception("unhandled_exception", extra={"error_type": type(exc).__name__})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred. This has been logged for investigation.",
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

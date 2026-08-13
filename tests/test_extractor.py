"""Test suite for schema validation and the self-healing extraction loop.

The test strategy is layered:

1. **Pure schema tests** exercise ``schemas.invoice`` directly with no
   mocking, verifying that arithmetic, NIP/REGON, and date-ordering
   validators behave correctly in isolation.
2. **Engine tests** monkeypatch ``ExtractionEngine._call_llm_with_resilience``
   (the single seam between our retry/self-healing orchestration and the
   actual network call) to simulate: (a) success on the first attempt,
   (b) a validation failure followed by a successful correction, and
   (c) exhausting the self-healing budget entirely. This isolates the
   *orchestration logic* under test from the LLM provider itself, which is
   exactly the boundary we want to verify deterministically and offline.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from core.config import Settings
from extractors.engine import (
    ExtractionEngine,
    ExtractionValidationError,
    UpstreamError,
)
from schemas.invoice import (
    CompanyParty,
    Currency,
    DocumentType,
    InvoiceExtractionResult,
    InvoiceLineItem,
    PaymentDetails,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_seller() -> CompanyParty:
    """A structurally valid Polish seller entity with a checksum-correct NIP."""
    return CompanyParty(
        legal_name="Acme Manufacturing Sp. z o.o.",
        tax_id_nip="1234563218",  # valid checksum digit
        national_business_registry_id_regon="123456785",
        city="Warszawa",
        country="Poland",
    )


@pytest.fixture
def valid_buyer() -> CompanyParty:
    """A structurally valid buyer entity with no Polish tax identifiers."""
    return CompanyParty(legal_name="Global Imports GmbH", country="Germany")


@pytest.fixture
def valid_line_item() -> InvoiceLineItem:
    """A single arithmetically consistent line item: 10 * 100.00 @ 23% VAT."""
    return InvoiceLineItem(
        position_number=1,
        description="Industrial widget, model X-200",
        quantity=Decimal("10"),
        unit_of_measure="szt.",
        unit_price_net=Decimal("100.00"),
        vat_rate_percent=Decimal("23"),
        net_amount=Decimal("1000.00"),
        vat_amount=Decimal("230.00"),
        gross_amount=Decimal("1230.00"),
    )


@pytest.fixture
def valid_payment_details() -> PaymentDetails:
    """Payment terms with a due date strictly after the issue date."""
    return PaymentDetails(
        payment_method="bank transfer",
        issue_date=date(2024, 9, 1),
        payment_due_date=date(2024, 9, 15),
        payment_terms_days=14,
    )


@pytest.fixture
def valid_extraction_result(
    valid_seller: CompanyParty,
    valid_buyer: CompanyParty,
    valid_line_item: InvoiceLineItem,
    valid_payment_details: PaymentDetails,
) -> InvoiceExtractionResult:
    """A fully valid, end-to-end consistent extraction result."""
    return InvoiceExtractionResult(
        document_type=DocumentType.COMMERCIAL_INVOICE,
        document_number="FV/2024/09/00123",
        currency=Currency.PLN,
        seller=valid_seller,
        buyer=valid_buyer,
        line_items=[valid_line_item],
        payment=valid_payment_details,
        total_net_amount=Decimal("1000.00"),
        total_vat_amount=Decimal("230.00"),
        total_gross_amount=Decimal("1230.00"),
    )


@pytest.fixture
def test_settings() -> Settings:
    """Application settings tuned for fast, deterministic test execution."""
    return Settings(
        openai_api_key="sk-test-key-not-real",
        extraction_model="gpt-4o-2024-08-06",
        extraction_max_retries=2,
    )


@pytest.fixture
def engine(test_settings: Settings) -> ExtractionEngine:
    """An ``ExtractionEngine`` instance backed by test settings."""
    return ExtractionEngine(settings=test_settings)


# =============================================================================
# Schema validation tests
# =============================================================================


class TestInvoiceLineItemValidation:
    """Tests for arithmetic and structural validation on line items."""

    def test_valid_line_item_passes(self, valid_line_item: InvoiceLineItem) -> None:
        """A correctly computed line item should validate without error."""
        assert valid_line_item.net_amount == Decimal("1000.00")

    def test_net_amount_mismatch_raises(self) -> None:
        """A net_amount inconsistent with quantity * unit_price must fail."""
        with pytest.raises(ValidationError, match="arithmetic mismatch"):
            InvoiceLineItem(
                position_number=1,
                description="Mismatched item",
                quantity=Decimal("10"),
                unit_price_net=Decimal("100.00"),
                vat_rate_percent=Decimal("23"),
                net_amount=Decimal("500.00"),  # should be 1000.00
                vat_amount=Decimal("115.00"),
                gross_amount=Decimal("615.00"),
            )

    def test_vat_amount_mismatch_raises(self) -> None:
        """A vat_amount inconsistent with net_amount * vat_rate must fail."""
        with pytest.raises(ValidationError, match="VAT mismatch"):
            InvoiceLineItem(
                position_number=1,
                description="Bad VAT item",
                quantity=Decimal("1"),
                unit_price_net=Decimal("100.00"),
                vat_rate_percent=Decimal("23"),
                net_amount=Decimal("100.00"),
                vat_amount=Decimal("999.00"),  # should be 23.00
                gross_amount=Decimal("123.00"),
            )

    def test_gross_amount_mismatch_raises(self) -> None:
        """A gross_amount not equal to net + vat must fail."""
        with pytest.raises(ValidationError, match="gross amount mismatch"):
            InvoiceLineItem(
                position_number=1,
                description="Bad gross item",
                quantity=Decimal("1"),
                unit_price_net=Decimal("100.00"),
                vat_rate_percent=Decimal("23"),
                net_amount=Decimal("100.00"),
                vat_amount=Decimal("23.00"),
                gross_amount=Decimal("50.00"),  # should be 123.00
            )

    def test_rounding_within_tolerance_passes(self) -> None:
        """Sub-cent rounding drift within tolerance should not fail validation."""
        item = InvoiceLineItem(
            position_number=1,
            description="Fractional quantity item",
            quantity=Decimal("3"),
            unit_price_net=Decimal("33.33"),
            vat_rate_percent=Decimal("23"),
            net_amount=Decimal("99.99"),
            vat_amount=Decimal("23.00"),
            gross_amount=Decimal("122.99"),
        )
        assert item.net_amount == Decimal("99.99")

    def test_zero_quantity_rejected(self) -> None:
        """Quantity must be strictly positive."""
        with pytest.raises(ValidationError):
            InvoiceLineItem(
                position_number=1,
                description="Zero quantity",
                quantity=Decimal("0"),
                unit_price_net=Decimal("10.00"),
                vat_rate_percent=Decimal("23"),
                net_amount=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                gross_amount=Decimal("0.00"),
            )


class TestCompanyPartyValidation:
    """Tests for Polish NIP/REGON identifier validation."""

    def test_valid_nip_with_separators_is_normalized(self) -> None:
        """A NIP supplied with dashes should be stripped and validated."""
        party = CompanyParty(legal_name="Test Sp. z o.o.", tax_id_nip="123-456-32-18")
        assert party.tax_id_nip == "1234563218"

    def test_invalid_nip_checksum_raises(self) -> None:
        """A structurally 10-digit NIP with a wrong checksum digit must fail."""
        with pytest.raises(ValidationError, match="checksum"):
            CompanyParty(legal_name="Bad NIP Co", tax_id_nip="1234563219")

    def test_nip_wrong_length_raises(self) -> None:
        """A NIP that is not exactly 10 digits must fail."""
        with pytest.raises(ValidationError, match="10 digits"):
            CompanyParty(legal_name="Short NIP Co", tax_id_nip="12345")

    def test_invalid_regon_length_raises(self) -> None:
        """A REGON that is neither 9 nor 14 digits must fail."""
        with pytest.raises(ValidationError, match="9 or 14 digits"):
            CompanyParty(
                legal_name="Bad REGON Co",
                national_business_registry_id_regon="12345",
            )

    def test_null_tax_identifiers_allowed_for_foreign_entity(
        self, valid_buyer: CompanyParty
    ) -> None:
        """Foreign entities without Polish identifiers should validate fine."""
        assert valid_buyer.tax_id_nip is None


class TestPaymentDetailsValidation:
    """Tests for date-ordering validation on payment terms."""

    def test_due_date_before_issue_date_raises(self) -> None:
        """A payment_due_date earlier than issue_date must fail."""
        with pytest.raises(ValidationError, match="cannot be earlier"):
            PaymentDetails(
                issue_date=date(2024, 9, 15),
                payment_due_date=date(2024, 9, 1),
            )

    def test_due_date_equal_to_issue_date_allowed(self) -> None:
        """Same-day payment terms (due on receipt) should be valid."""
        details = PaymentDetails(issue_date=date(2024, 9, 1), payment_due_date=date(2024, 9, 1))
        assert details.payment_due_date == details.issue_date


class TestInvoiceExtractionResultValidation:
    """Tests for document-level total reconciliation against line items."""

    def test_valid_document_passes(
        self, valid_extraction_result: InvoiceExtractionResult
    ) -> None:
        """A fully consistent document should validate end to end."""
        assert valid_extraction_result.total_gross_amount == Decimal("1230.00")

    def test_total_net_mismatch_raises(
        self,
        valid_seller: CompanyParty,
        valid_buyer: CompanyParty,
        valid_line_item: InvoiceLineItem,
        valid_payment_details: PaymentDetails,
    ) -> None:
        """A total_net_amount inconsistent with the summed line items must fail."""
        with pytest.raises(ValidationError, match="total_net_amount"):
            InvoiceExtractionResult(
                document_type=DocumentType.COMMERCIAL_INVOICE,
                document_number="FV/001",
                currency=Currency.PLN,
                seller=valid_seller,
                buyer=valid_buyer,
                line_items=[valid_line_item],
                payment=valid_payment_details,
                total_net_amount=Decimal("9999.00"),  # should be 1000.00
                total_vat_amount=Decimal("230.00"),
                total_gross_amount=Decimal("1230.00"),
            )

    def test_empty_line_items_rejected(
        self,
        valid_seller: CompanyParty,
        valid_buyer: CompanyParty,
        valid_payment_details: PaymentDetails,
    ) -> None:
        """A document with zero line items must fail min_length validation."""
        with pytest.raises(ValidationError):
            InvoiceExtractionResult(
                document_type=DocumentType.COMMERCIAL_INVOICE,
                document_number="FV/002",
                currency=Currency.PLN,
                seller=valid_seller,
                buyer=valid_buyer,
                line_items=[],
                payment=valid_payment_details,
                total_net_amount=Decimal("0"),
                total_vat_amount=Decimal("0"),
                total_gross_amount=Decimal("0"),
            )


# =============================================================================
# Self-healing extraction engine tests
# =============================================================================


class TestSelfHealingExtractionLoop:
    """Tests verifying the engine's validation-failure recovery behavior."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(
        self,
        engine: ExtractionEngine,
        valid_extraction_result: InvoiceExtractionResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the LLM returns valid data immediately, one attempt suffices."""
        call_count = 0

        async def fake_call(
            *, messages: list[dict[str, Any]], response_model: type
        ) -> InvoiceExtractionResult:
            nonlocal call_count
            call_count += 1
            return valid_extraction_result

        monkeypatch.setattr(engine, "_call_llm_with_resilience", fake_call)

        result = await engine.extract(
            document_text="dummy invoice text", response_model=InvoiceExtractionResult
        )

        assert call_count == 1
        assert result.attempts_made == 1
        assert result.self_healed is False
        assert result.data is valid_extraction_result

    @pytest.mark.asyncio
    async def test_self_heals_after_one_validation_failure(
        self,
        engine: ExtractionEngine,
        valid_extraction_result: InvoiceExtractionResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The engine should recover when the first attempt fails validation
        but the second (post-feedback) attempt succeeds.
        """
        call_count = 0

        def _build_validation_error() -> ValidationError:
            try:
                InvoiceLineItem(
                    position_number=1,
                    description="Broken item",
                    quantity=Decimal("10"),
                    unit_price_net=Decimal("100.00"),
                    vat_rate_percent=Decimal("23"),
                    net_amount=Decimal("1.00"),  # intentionally wrong
                    vat_amount=Decimal("230.00"),
                    gross_amount=Decimal("1230.00"),
                )
            except ValidationError as exc:
                return exc
            raise AssertionError("Expected ValidationError was not raised")

        async def fake_call(
            *, messages: list[dict[str, Any]], response_model: type
        ) -> InvoiceExtractionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _build_validation_error()
            # Verify the corrective feedback was actually injected into the
            # conversation before the second attempt.
            assert any(
                "FAILED schema validation" in message["content"]
                for message in messages
                if message["role"] == "user"
            )
            return valid_extraction_result

        monkeypatch.setattr(engine, "_call_llm_with_resilience", fake_call)

        result = await engine.extract(
            document_text="dummy invoice text", response_model=InvoiceExtractionResult
        )

        assert call_count == 2
        assert result.attempts_made == 2
        assert result.self_healed is True
        assert len(result.validation_errors_encountered) == 1
        assert result.data is valid_extraction_result

    @pytest.mark.asyncio
    async def test_exhausts_self_healing_budget_and_raises(
        self, engine: ExtractionEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every attempt fails validation, the engine raises after the
        configured retry budget (max_self_healing_attempts=2 -> 3 total calls).
        """
        call_count = 0

        def _build_validation_error() -> ValidationError:
            try:
                CompanyParty(legal_name="X", tax_id_nip="0000000000")
            except ValidationError as exc:
                return exc
            raise AssertionError("Expected ValidationError was not raised")

        async def always_failing_call(
            *, messages: list[dict[str, Any]], response_model: type
        ) -> InvoiceExtractionResult:
            nonlocal call_count
            call_count += 1
            raise _build_validation_error()

        monkeypatch.setattr(engine, "_call_llm_with_resilience", always_failing_call)

        with pytest.raises(ExtractionValidationError) as exc_info:
            await engine.extract(
                document_text="dummy invoice text",
                response_model=InvoiceExtractionResult,
                max_self_healing_attempts=2,
            )

        assert call_count == 3  # initial attempt + 2 self-healing retries
        assert exc_info.value.attempts_made == 3
        assert exc_info.value.last_validation_error is not None

    @pytest.mark.asyncio
    async def test_upstream_error_propagates_without_self_healing(
        self, engine: ExtractionEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuine upstream provider failure should propagate as
        ``UpstreamError`` without being treated as a self-healable validation
        failure.
        """

        async def failing_call(
            *, messages: list[dict[str, Any]], response_model: type
        ) -> InvoiceExtractionResult:
            raise UpstreamError("simulated provider outage")

        monkeypatch.setattr(engine, "_call_llm_with_resilience", failing_call)

        with pytest.raises(UpstreamError, match="simulated provider outage"):
            await engine.extract(
                document_text="dummy invoice text", response_model=InvoiceExtractionResult
            )

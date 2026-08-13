"""Structured extraction schema for commercial invoices and contracts.

These Pydantic v2 models serve a dual purpose that is central to the whole
system:

1. **Runtime validation** — every field enforced here (arithmetic
   consistency, currency codes, Polish tax identifiers, date ordering) is a
   hard guarantee about the shape of data leaving this service. Downstream
   systems (ERP, accounting, data warehouses) can trust this payload without
   re-validating it.
2. **LLM instruction surface** — with the ``instructor`` library, every
   ``Field(description=...)`` and docstring below is compiled into the JSON
   Schema handed to the LLM as part of the function/tool-calling contract.
   Precise, unambiguous field descriptions materially improve extraction
   accuracy; this file should be read as prompt engineering as much as data
   modeling.

All monetary values are modeled as ``Decimal`` to avoid floating point
rounding artifacts when validating ``quantity * unit_price == line_total``.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator

# --- Reusable low-level validation primitives -------------------------------

_NIP_PATTERN = re.compile(r"^\d{10}$")
_REGON_PATTERN = re.compile(r"^(\d{9}|\d{14})$")
_ISO_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")

# Tolerance for floating-point-style rounding drift introduced by LLM
# arithmetic (e.g. the model reporting 123.45 instead of the mathematically
# exact 123.449999...). One cent is a deliberately tight but forgiving bound.
_ARITHMETIC_TOLERANCE = Decimal("0.01")


def _validate_polish_nip_checksum(nip: str) -> bool:
    """Validates a Polish NIP (Tax Identification Number) checksum digit.

    The NIP checksum algorithm multiplies each of the first nine digits by a
    fixed weight vector, sums the products, and reduces modulo 11; the
    result must equal the tenth (final) digit.

    Args:
        nip: A 10-digit numeric string with no separators.

    Returns:
        ``True`` if the checksum digit is valid, ``False`` otherwise.
    """
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    digits = [int(char) for char in nip]
    checksum = sum(weight * digit for weight, digit in zip(weights, digits)) % 11
    return checksum == digits[9]


class Currency(str, Enum):
    """Whitelist of supported ISO 4217 currency codes.

    Restricting to an enum (rather than a free-form validated string) gives
    the LLM a closed, unambiguous set of choices, which meaningfully reduces
    the odds of it inventing a non-standard currency token.
    """

    PLN = "PLN"
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    CZK = "CZK"


class DocumentType(str, Enum):
    """High-level classification of the source business document."""

    COMMERCIAL_INVOICE = "commercial_invoice"
    PROFORMA_INVOICE = "proforma_invoice"
    CREDIT_NOTE = "credit_note"
    CONTRACT = "contract"
    PURCHASE_ORDER = "purchase_order"


class CompanyParty(BaseModel):
    """A legal business entity acting as either seller or buyer.

    Attributes:
        legal_name: Full registered legal name of the company, exactly as
            printed on the document letterhead.
        tax_id_nip: Polish NIP tax identifier, exactly 10 digits with no
            separators (dashes/spaces must be stripped before assignment).
        national_business_registry_id_regon: Polish REGON identifier, 9 or
            14 digits. Optional, as not all documents disclose it.
        vat_id_eu: EU-format VAT identifier (e.g. ``PL1234567890``), used
            for intra-community transactions. Optional.
        street_address: Street name and building/apartment number.
        postal_code: Postal code in the format used by the entity's country.
        city: City or town name.
        country: Full country name (not ISO code) as it would appear on an
            invoice, e.g. ``"Poland"``.
        bank_account_iban: IBAN of the settlement bank account, if disclosed
            on the document.
    """

    legal_name: str = Field(
        ...,
        min_length=2,
        description=(
            "The full registered legal name of the company, copied verbatim "
            "from the document, including legal form suffixes such as "
            "'Sp. z o.o.', 'S.A.', 'GmbH', 'Ltd.'."
        ),
    )
    tax_id_nip: str | None = Field(
        default=None,
        description=(
            "Polish NIP tax identification number. Extract ONLY the digits, "
            "stripping any dashes or spaces (e.g. '123-456-32-18' becomes "
            "'1234563218'). Must be exactly 10 digits. Set to null if the "
            "party is not a Polish entity or the NIP is not present."
        ),
    )
    national_business_registry_id_regon: str | None = Field(
        default=None,
        description=(
            "Polish REGON registry identifier, 9 or 14 digits, digits only. "
            "Set to null if not present on the document."
        ),
    )
    vat_id_eu: str | None = Field(
        default=None,
        description=(
            "EU VAT identification number including the two-letter country "
            "prefix, e.g. 'PL1234563218' or 'DE123456789'. Null if absent."
        ),
    )
    street_address: str | None = Field(default=None)
    postal_code: str | None = Field(default=None)
    city: str | None = Field(default=None)
    country: str | None = Field(
        default=None,
        description="Full country name as commonly written in English, e.g. 'Poland', 'Germany'.",
    )
    bank_account_iban: str | None = Field(
        default=None,
        description=(
            "IBAN of the party's settlement bank account, digits and letters "
            "only, no spaces. Null if not disclosed."
        ),
    )

    @model_validator(mode="after")
    def _validate_tax_identifiers(self) -> Self:
        """Cross-validates the format and checksum of tax identifiers.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If ``tax_id_nip`` is present but is not a 10-digit
                string with a valid checksum digit, or if
                ``national_business_registry_id_regon`` is present but is
                not 9 or 14 digits.
        """
        if self.tax_id_nip is not None:
            normalized = re.sub(r"[\s-]", "", self.tax_id_nip)
            if not _NIP_PATTERN.match(normalized):
                raise ValueError(
                    f"tax_id_nip must be exactly 10 digits after stripping "
                    f"separators, got '{self.tax_id_nip}' (normalized: "
                    f"'{normalized}')."
                )
            if not _validate_polish_nip_checksum(normalized):
                raise ValueError(
                    f"tax_id_nip '{normalized}' failed the NIP checksum "
                    f"validation; this is not a structurally valid Polish NIP."
                )
            self.tax_id_nip = normalized

        if self.national_business_registry_id_regon is not None:
            normalized_regon = re.sub(
                r"[\s-]", "", self.national_business_registry_id_regon
            )
            if not _REGON_PATTERN.match(normalized_regon):
                raise ValueError(
                    "national_business_registry_id_regon must be 9 or 14 "
                    f"digits, got '{self.national_business_registry_id_regon}'."
                )
            self.national_business_registry_id_regon = normalized_regon

        return self


class InvoiceLineItem(BaseModel):
    """A single billable line item on an invoice.

    Attributes:
        position_number: 1-indexed sequential position of this line on the
            source document.
        description: Free-text description of the goods or service billed.
        quantity: Numeric quantity billed, may be fractional (e.g. hours,
            kilograms).
        unit_of_measure: Unit abbreviation such as 'szt.' (pieces), 'kg',
            'h' (hours), 'm2'.
        unit_price_net: Net (pre-tax) price per single unit.
        vat_rate_percent: VAT rate applied to this line, expressed as a
            whole-number percentage (e.g. 23 for 23%, 0 for zero-rated).
        net_amount: Total net amount for this line (quantity * unit_price_net).
        vat_amount: Total VAT amount for this line.
        gross_amount: Total gross (tax-inclusive) amount for this line.
    """

    position_number: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    quantity: Decimal = Field(..., gt=Decimal("0"))
    unit_of_measure: str = Field(
        default="szt.",
        description="Unit of measure abbreviation as printed on the document, e.g. 'szt.', 'kg', 'h', 'm2'.",
    )
    unit_price_net: Decimal = Field(..., ge=Decimal("0"))
    vat_rate_percent: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="VAT rate as a whole percentage number, e.g. 23, 8, 5, 0.",
    )
    net_amount: Decimal = Field(..., ge=Decimal("0"))
    vat_amount: Decimal = Field(..., ge=Decimal("0"))
    gross_amount: Decimal = Field(..., ge=Decimal("0"))

    @model_validator(mode="after")
    def _validate_line_item_arithmetic(self) -> Self:
        """Verifies that quantity, price, and totals are mutually consistent.

        This is the enforcement point for the classic
        ``quantity * unit_price == value`` invariant, plus VAT and gross
        amount consistency. A small tolerance absorbs LLM/source-document
        rounding noise without masking genuine extraction errors.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If the net amount does not reconcile with
                quantity * unit_price, if the VAT amount does not reconcile
                with net_amount * vat_rate, or if gross_amount does not equal
                net_amount + vat_amount, beyond the allowed tolerance.
        """
        expected_net = (self.quantity * self.unit_price_net).quantize(
            _ARITHMETIC_TOLERANCE, rounding=ROUND_HALF_UP
        )
        if abs(expected_net - self.net_amount) > _ARITHMETIC_TOLERANCE:
            raise ValueError(
                f"Line item #{self.position_number} arithmetic mismatch: "
                f"quantity ({self.quantity}) * unit_price_net "
                f"({self.unit_price_net}) = {expected_net}, but net_amount "
                f"was reported as {self.net_amount}."
            )

        expected_vat = (
            self.net_amount * (self.vat_rate_percent / Decimal("100"))
        ).quantize(_ARITHMETIC_TOLERANCE, rounding=ROUND_HALF_UP)
        if abs(expected_vat - self.vat_amount) > _ARITHMETIC_TOLERANCE:
            raise ValueError(
                f"Line item #{self.position_number} VAT mismatch: "
                f"net_amount ({self.net_amount}) * vat_rate_percent "
                f"({self.vat_rate_percent}%) = {expected_vat}, but "
                f"vat_amount was reported as {self.vat_amount}."
            )

        expected_gross = self.net_amount + self.vat_amount
        if abs(expected_gross - self.gross_amount) > _ARITHMETIC_TOLERANCE:
            raise ValueError(
                f"Line item #{self.position_number} gross amount mismatch: "
                f"net_amount ({self.net_amount}) + vat_amount "
                f"({self.vat_amount}) = {expected_gross}, but gross_amount "
                f"was reported as {self.gross_amount}."
            )

        return self


class PaymentDetails(BaseModel):
    """Payment terms and instructions extracted from the document.

    Attributes:
        payment_method: Free-text payment method, e.g. 'bank transfer',
            'cash', 'credit card'.
        payment_due_date: Calendar date by which payment is due.
        issue_date: Calendar date the document was issued.
        sale_date: Calendar date the underlying sale/service was performed;
            may differ from issue_date.
        payment_terms_days: Number of days between issue_date and
            payment_due_date, if explicitly stated or derivable.
    """

    payment_method: str | None = Field(default=None)
    payment_due_date: date | None = Field(default=None)
    issue_date: date = Field(
        ...,
        description="The date the document/invoice was issued, in ISO 8601 format.",
    )
    sale_date: date | None = Field(
        default=None,
        description="The date the underlying goods/service transaction occurred, if different from issue_date.",
    )
    payment_terms_days: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_date_ordering(self) -> Self:
        """Ensures payment due date does not precede the issue date.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If ``payment_due_date`` is earlier than ``issue_date``.
        """
        if self.payment_due_date is not None and self.payment_due_date < self.issue_date:
            raise ValueError(
                f"payment_due_date ({self.payment_due_date}) cannot be "
                f"earlier than issue_date ({self.issue_date})."
            )
        return self


class InvoiceExtractionResult(BaseModel):
    """Top-level structured extraction result for a business document.

    This is the root schema handed to ``instructor`` as the target response
    model. It aggregates document classification, both counterparties, all
    line items, payment terms, and document-level financial totals, with a
    final cross-validator reconciling the sum of line items against the
    reported totals.

    Attributes:
        document_type: Classification of the source document.
        document_number: The invoice/document number as printed (e.g.
            'FV/2024/09/00123').
        currency: ISO 4217 currency code used throughout the document.
        seller: The issuing/selling party.
        buyer: The receiving/purchasing party.
        line_items: All billable line items found on the document, in the
            order they appear.
        payment: Payment terms and relevant dates.
        total_net_amount: Sum of all line items' net amounts.
        total_vat_amount: Sum of all line items' VAT amounts.
        total_gross_amount: Sum of all line items' gross amounts; the final
            payable amount.
        notes: Any additional free-text remarks, footnotes, or special terms
            present on the document that don't fit other fields.
        extraction_confidence_notes: Optional model-authored notes about any
            fields it was uncertain about or had to infer, for human review.
    """

    document_type: DocumentType = Field(
        ...,
        description="Classify the overall nature of this document based on its content and layout.",
    )
    document_number: str = Field(
        ...,
        min_length=1,
        description="The unique invoice or document reference number exactly as printed on the source.",
    )
    currency: Currency = Field(
        ...,
        description="The three-letter ISO 4217 currency code used for all monetary amounts in this document.",
    )
    seller: CompanyParty = Field(
        ..., description="The company issuing the invoice / providing goods or services."
    )
    buyer: CompanyParty = Field(
        ..., description="The company or entity receiving the invoice / purchasing goods or services."
    )
    line_items: list[InvoiceLineItem] = Field(
        ...,
        min_length=1,
        description="Every billable line item on the document, in the order printed.",
    )
    payment: PaymentDetails = Field(...)
    total_net_amount: Decimal = Field(..., ge=Decimal("0"))
    total_vat_amount: Decimal = Field(..., ge=Decimal("0"))
    total_gross_amount: Decimal = Field(..., ge=Decimal("0"))
    notes: str | None = Field(
        default=None,
        description="Any additional remarks, special terms, or footnotes present on the document.",
    )
    extraction_confidence_notes: str | None = Field(
        default=None,
        description=(
            "If any field's value was ambiguous, illegible, or had to be "
            "inferred rather than read directly, briefly note that here for "
            "human reviewer attention. Null if extraction was unambiguous."
        ),
    )

    @model_validator(mode="after")
    def _validate_document_totals(self) -> Self:
        """Reconciles document-level totals against the sum of line items.

        This is the highest-value validator in the schema: it is the final
        gate that catches cases where individual line items are internally
        consistent but the LLM miscalculated (or hallucinated) the header
        totals — a common failure mode in long, multi-page invoices.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If the sum of line item net/VAT/gross amounts does
                not reconcile with the corresponding document-level total
                fields, beyond the allowed per-line rounding tolerance.
        """
        line_count = len(self.line_items)
        aggregate_tolerance = _ARITHMETIC_TOLERANCE * max(line_count, 1)

        summed_net = sum((item.net_amount for item in self.line_items), Decimal("0"))
        summed_vat = sum((item.vat_amount for item in self.line_items), Decimal("0"))
        summed_gross = sum((item.gross_amount for item in self.line_items), Decimal("0"))

        if abs(summed_net - self.total_net_amount) > aggregate_tolerance:
            raise ValueError(
                f"total_net_amount ({self.total_net_amount}) does not match "
                f"the sum of line item net amounts ({summed_net})."
            )
        if abs(summed_vat - self.total_vat_amount) > aggregate_tolerance:
            raise ValueError(
                f"total_vat_amount ({self.total_vat_amount}) does not match "
                f"the sum of line item VAT amounts ({summed_vat})."
            )
        if abs(summed_gross - self.total_gross_amount) > aggregate_tolerance:
            raise ValueError(
                f"total_gross_amount ({self.total_gross_amount}) does not "
                f"match the sum of line item gross amounts ({summed_gross})."
            )

        expected_total_gross = self.total_net_amount + self.total_vat_amount
        if abs(expected_total_gross - self.total_gross_amount) > aggregate_tolerance:
            raise ValueError(
                f"total_gross_amount ({self.total_gross_amount}) does not "
                f"equal total_net_amount + total_vat_amount "
                f"({expected_total_gross})."
            )

        return self

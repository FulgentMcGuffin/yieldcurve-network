"""Tests for the citation registry.

The registry exists so the reasoning behind the model set cannot rot. These
tests are what stop it: they fail if an implemented model loses its citation, or
if the evaluated-but-rejected models and consulted implementations are dropped.
"""

from __future__ import annotations

import pytest

from ycn.analysis.af_models import FITTERS, ResidualModel
from ycn.analysis.af_references import (
    REFERENCES,
    Reference,
    ReferenceStatus,
    format_bibliography,
    references_for,
    references_frame,
)


def test_keys_match_dict_keys():
    """Each entry is filed under its own key."""
    for key, ref in REFERENCES.items():
        assert ref.key == key


def test_every_registered_model_has_an_implemented_reference():
    """A model you can fit is a model you can cite."""
    for model in FITTERS:
        implemented = [
            r for r in references_for(model) if r.status is ReferenceStatus.IMPLEMENTED
        ]
        assert implemented, f"{model.value} has no IMPLEMENTED reference"


@pytest.mark.parametrize(
    "model", [ResidualModel.AFNS, ResidualModel.DTAFNS, ResidualModel.NEURAL_HJM]
)
def test_planned_models_are_cited_before_they_are_built(model):
    """The three arbitrage-free models carry their citation from the start."""
    implemented = [
        r for r in references_for(model) if r.status is ReferenceStatus.IMPLEMENTED
    ]
    assert implemented, f"{model.value} has no IMPLEMENTED reference"


@pytest.mark.parametrize(
    "key",
    ["kim_wright2005_ea03", "christensen_rudebusch2015_shadow_afns", "cdr2009_afgns"],
)
def test_rejected_models_are_retained(key):
    """Models considered and not built stay on record, with the reason."""
    ref = REFERENCES[key]
    assert ref.status is ReferenceStatus.EVALUATED_NOT_CHOSEN
    assert ref.notes, f"{key} must record why it was not chosen"


def test_kim_wright_records_the_pdf_misnomer():
    """The local PDF is Kim-Wright, not AFNS -- record it so nobody re-confuses them."""
    ref = REFERENCES["kim_wright2005_ea03"]
    assert "3factorArbitrageFree.pdf" in ref.local_path
    assert "Kim-Wright" in ref.notes


@pytest.mark.parametrize("key", ["snejens_arbfree_dns", "werleycordeiro_dnss_kalman"])
def test_reference_implementations_are_retained(key):
    """Consulted-but-not-vendored code keeps its URL."""
    ref = REFERENCES[key]
    assert ref.status is ReferenceStatus.REFERENCE_IMPLEMENTATION
    assert ref.url.startswith("https://github.com/")


def test_neural_reference_flags_its_own_limitations():
    """The experimental model must carry its caveat next to its citation."""
    notes = REFERENCES["gao_hyndman2025_neural_hjm"].notes
    assert "EXPERIMENTAL" in notes
    assert "subset" in notes.lower()


def test_every_reference_is_well_formed():
    """No blank fields where a citation needs content."""
    for ref in REFERENCES.values():
        assert isinstance(ref, Reference)
        assert ref.title and ref.authors and ref.venue and ref.identifier
        assert ref.url.startswith("http")
        assert 1980 <= ref.year <= 2030


def test_references_frame_round_trips():
    """The frame carries every entry, and filtering narrows it."""
    everything = references_frame()
    assert everything.height == len(REFERENCES)
    assert set(everything.columns) >= {"key", "status", "year", "url", "models"}

    implemented = references_frame(ReferenceStatus.IMPLEMENTED)
    assert 0 < implemented.height < everything.height
    assert set(implemented.get_column("status").unique().to_list()) == {"implemented"}


def test_format_bibliography_covers_all_statuses():
    """Every group renders, and each entry shows key, url and year."""
    text = format_bibliography()
    for ref in REFERENCES.values():
        assert f"[{ref.key}]" in text
        assert ref.url in text
    for status in ReferenceStatus:
        assert f"{status.label}:" in text


def test_format_bibliography_can_filter():
    """Filtering to one status drops the others."""
    text = format_bibliography(ReferenceStatus.REFERENCE_IMPLEMENTATION)
    assert "github.com/snejens" in text
    assert "cdr2011_afns" not in text

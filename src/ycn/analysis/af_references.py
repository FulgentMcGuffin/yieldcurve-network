"""Citation registry for the arbitrage-free residual framework.

Kept as typed, importable data rather than prose so it can be tested and cannot
silently drift from the code. ``tests/test_af_references.py`` asserts that every
implemented model carries a citation and that the evaluated-but-rejected models
and reference implementations survive.

Deliberately records models that were **considered and not built**, and external
implementations that were **consulted but not vendored**, so that the reasoning
behind the current model set stays discoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import polars as pl

from .af_models import ResidualModel


class ReferenceStatus(str, Enum):
    """Why a reference is in the registry."""

    IMPLEMENTED = "implemented"
    EVALUATED_NOT_CHOSEN = "evaluated_not_chosen"
    REFERENCE_IMPLEMENTATION = "reference_implementation"
    BACKGROUND = "background"

    @property
    def label(self) -> str:
        return {
            ReferenceStatus.IMPLEMENTED: "Implemented",
            ReferenceStatus.EVALUATED_NOT_CHOSEN: "Evaluated, not chosen",
            ReferenceStatus.REFERENCE_IMPLEMENTATION: "Reference implementation",
            ReferenceStatus.BACKGROUND: "Background",
        }[self]


@dataclass(frozen=True)
class Reference:
    """One paper or code repository.

    Args:
        key: Stable identifier used in code comments and docstrings.
        title: Full title.
        authors: Author list, as cited.
        year: Publication year.
        venue: Journal, working-paper series, or host.
        identifier: DOI, working-paper number, or arXiv id.
        url: Canonical link.
        status: Why it is here.
        models: Which registry models it underpins; empty for background.
        notes: What it contributes, and any caveat.
        local_path: Local copy, if one exists on this machine.
    """

    key: str
    title: str
    authors: str
    year: int
    venue: str
    identifier: str
    url: str
    status: ReferenceStatus
    models: tuple[ResidualModel, ...] = ()
    notes: str = ""
    local_path: str = ""


REFERENCES: dict[str, Reference] = {
    ref.key: ref
    for ref in (
        # -- Implemented -----------------------------------------------------
        Reference(
            key="cdr2011_afns",
            title="The affine arbitrage-free class of Nelson-Siegel term structure models",
            authors="Christensen, J.H.E., Diebold, F.X. & Rudebusch, G.D.",
            year=2011,
            venue="Journal of Econometrics 164(1), 4-20",
            identifier="doi:10.1016/j.jeconom.2011.02.011",
            url="https://www.sas.upenn.edu/~fdiebold/papers/paper78/cdr.pdf",
            status=ReferenceStatus.IMPLEMENTED,
            models=(ResidualModel.AFNS,),
            notes=(
                "Proposition 1 gives the AFNS yield function; section 2.3 gives the "
                "closed-form yield-adjustment term and shows only six combinations of "
                "Sigma are identified. Source for af_loadings.afns_yield_adjustment."
            ),
        ),
        Reference(
            key="eghbalzadeh2024_dtafns",
            title=(
                "The discrete-time arbitrage-free Nelson-Siegel model: a closed-form "
                "solution and applications to mixed funds representation"
            ),
            authors="Eghbalzadeh, R., Godin, F. & Gaillardetz, P.",
            year=2024,
            venue="Annals of Actuarial Science 18(2), 310-341",
            identifier="doi:10.1017/S1748499523000234",
            url=(
                "https://www.cambridge.org/core/journals/annals-of-actuarial-science/"
                "article/discretetime-arbitragefree-nelsonsiegel-model-a-closedform-"
                "solution-and-applications-to-mixed-funds-representation/"
                "30FCE20F397E859D153A3BF3E2E266AE"
            ),
            status=ReferenceStatus.IMPLEMENTED,
            models=(ResidualModel.DTAFNS,),
            notes=(
                "Derives DTAFNS directly in discrete time, so loadings and the "
                "adjustment A_tau are closed-form geometric sums with no integrals "
                "and no matrix logarithm. Slightly modifies Hong et al. (2019)."
            ),
        ),
        Reference(
            key="gao_hyndman2025_neural_hjm",
            title=(
                "Arbitrage-Free Bond and Yield Curve Forecasting with Neural Filters "
                "under HJM Constraints"
            ),
            authors="Gao, X. & Hyndman, C.",
            year=2025,
            venue="arXiv preprint",
            identifier="arXiv:2511.17892",
            url="https://arxiv.org/abs/2511.17892",
            status=ReferenceStatus.IMPLEMENTED,
            models=(ResidualModel.NEURAL_HJM,),
            notes=(
                "EXPERIMENTAL, PARTIAL. No code or hyperparameters are published, so "
                "ycn implements a documented subset: DNS forward-rate parameterisation, "
                "an LSTM latent state, a learned-gain correction and the HJM drift "
                "restriction as a soft penalty. Omits the EKF, the particle filter and "
                "the CLSTM. Its residuals are filtered in-sample residuals, a different "
                "statistical object from the cross-sectional residuals of the other "
                "models, and cannot be validated against published numbers."
            ),
        ),
        # -- Evaluated, not chosen -------------------------------------------
        Reference(
            key="kim_wright2005_ea03",
            title=(
                "An Arbitrage-Free Three-Factor Term Structure Model and the Recent "
                "Behavior of Long-Term Yields and Distant-Horizon Forward Rates"
            ),
            authors="Kim, D.H. & Wright, J.H.",
            year=2005,
            venue="Federal Reserve Board, Finance and Economics Discussion Series",
            identifier="FEDS 2005-33",
            url="https://www.federalreserve.gov/pubs/feds/2005/200533/200533pap.pdf",
            status=ReferenceStatus.EVALUATED_NOT_CHOSEN,
            notes=(
                "The model behind the Fed's published Kim-Wright term premium series. "
                "A Gaussian essentially-affine Duffee A0(3) model with three LATENT "
                "factors -- no Nelson-Siegel structure, no decay parameter and no "
                "yield-adjustment term, so its residuals are not decomposable against "
                "the existing NS framework. Estimation needs Blue Chip survey rows to "
                "break the K versus K^Q = K - Sigma*Phi identification problem, and no "
                "such survey data exists for these issuers. "
                "NOTE: the local PDF is named '3factorArbitrageFree.pdf' and is often "
                "mistaken for AFNS; it is Kim-Wright, and predates AFNS by two years."
            ),
            local_path=r"D:\Code\synthai\synthai-papers\3factorArbitrageFree.pdf",
        ),
        Reference(
            key="christensen_rudebusch2015_shadow_afns",
            title="Estimating Shadow-Rate Term Structure Models with Near-Zero Yields",
            authors="Christensen, J.H.E. & Rudebusch, G.D.",
            year=2015,
            venue="Journal of Financial Econometrics 13(2), 226-259",
            identifier="FRBSF Working Paper 2013-39",
            url="https://www.frbsf.org/wp-content/uploads/wp2013-39.pdf",
            status=ReferenceStatus.EVALUATED_NOT_CHOSEN,
            notes=(
                "AFNS with a Black (1995) shadow short rate so fitted yields respect "
                "the zero lower bound. Relevant to euro-area panels spanning the "
                "negative-rate era, where a Gaussian model can fit impossible yields "
                "and contaminate the residuals. Deferred: needs an option-style "
                "correction that the closed-form offset machinery cannot express."
            ),
        ),
        Reference(
            key="cdr2009_afgns",
            title="An arbitrage-free generalized Nelson-Siegel term structure model",
            authors="Christensen, J.H.E., Diebold, F.X. & Rudebusch, G.D.",
            year=2009,
            venue="The Econometrics Journal 12(3), C33-C64",
            identifier="FRBSF Working Paper 2008-07",
            url="https://www.frbsf.org/wp-content/uploads/wp08-07bk.pdf",
            status=ReferenceStatus.EVALUATED_NOT_CHOSEN,
            notes=(
                "Five-factor arbitrage-free generalisation, the AF analogue of "
                "Svensson -- no AF model matches Svensson's own loadings, but pairing "
                "the extra curvature factor with a second slope factor does. Better "
                "long-end fit. Deferred: two more factors for a marginal gain on a "
                "ten-tenor grid."
            ),
        ),
        # -- Reference implementations ---------------------------------------
        Reference(
            key="snejens_arbfree_dns",
            title="arbfree_dynamic_nelson_siegel",
            authors="snejens",
            year=2021,
            venue="GitHub",
            identifier="github:snejens/arbfree_dynamic_nelson_siegel",
            url="https://github.com/snejens/arbfree_dynamic_nelson_siegel",
            status=ReferenceStatus.REFERENCE_IMPLEMENTATION,
            models=(ResidualModel.AFNS,),
            notes=(
                "Working Python AFNS/AFGNS with Kalman-filter maximum likelihood. "
                "Not vendored. The port source for the correlated-Sigma specification "
                "and for cross-checking the state-space estimator."
            ),
        ),
        Reference(
            key="werleycordeiro_dnss_kalman",
            title="Dynamic_Nelson_Siegel_Svensson_Kalman_Filter",
            authors="Cordeiro, W.",
            year=2020,
            venue="GitHub",
            identifier="github:werleycordeiro/Dynamic_Nelson_Siegel_Svensson_Kalman_Filter",
            url="https://github.com/werleycordeiro/Dynamic_Nelson_Siegel_Svensson_Kalman_Filter",
            status=ReferenceStatus.REFERENCE_IMPLEMENTATION,
            notes=(
                "Dynamic Nelson-Siegel-Svensson fitting and forecasting with a Kalman "
                "filter. Not vendored. Consulted for the state-space layout."
            ),
        ),
        # -- Background ------------------------------------------------------
        Reference(
            key="hong2019_dtafns",
            title="Forecasting interest rates with a discrete-time arbitrage-free model",
            authors="Hong, Z., Niu, L. & Zeng, G.",
            year=2019,
            venue="Working paper / Wang Yanan Institute",
            identifier="hong2019",
            url="https://www.cambridge.org/core/journals/annals-of-actuarial-science",
            status=ReferenceStatus.BACKGROUND,
            models=(ResidualModel.DTAFNS,),
            notes=(
                "The original discrete-time AFNS specification, enforcing exact "
                "Nelson-Siegel loadings. Eghbalzadeh et al. (2024) modify it for "
                "tractability and are the specification actually implemented here."
            ),
        ),
        Reference(
            key="diebold_li2006",
            title="Forecasting the term structure of government bond yields",
            authors="Diebold, F.X. & Li, C.",
            year=2006,
            venue="Journal of Econometrics 130(2), 337-364",
            identifier="doi:10.1016/j.jeconom.2005.03.005",
            url="https://www.sas.upenn.edu/~fdiebold/papers/paper49/Diebold-Li.pdf",
            status=ReferenceStatus.BACKGROUND,
            notes=(
                "The two-step estimator: fix the decay, fit factors cross-sectionally "
                "by least squares per date, then model the factor series. The default "
                "estimator here follows this, with the yield adjustment added to the "
                "cross-sectional step. Also the source of the errors-in-variables "
                "caveat on the resulting Sigma."
            ),
        ),
        Reference(
            key="dra2006_macro_finance",
            title="The macroeconomy and the yield curve: a dynamic latent factor approach",
            authors="Diebold, F.X., Rudebusch, G.D. & Aruoba, S.B.",
            year=2006,
            venue="Journal of Econometrics 131(1-2), 309-338",
            identifier="doi:10.1016/j.jeconom.2005.01.011",
            url="https://www.sas.upenn.edu/~fdiebold/papers/paper61/temp-dra.pdf",
            status=ReferenceStatus.BACKGROUND,
            notes="State-space DNS with correlated factors; the correlated-factor baseline.",
        ),
        Reference(
            key="nelson_siegel1987",
            title="Parsimonious modeling of yield curves",
            authors="Nelson, C.R. & Siegel, A.F.",
            year=1987,
            venue="The Journal of Business 60(4), 473-489",
            identifier="doi:10.1086/296409",
            url="https://www.jstor.org/stable/2352957",
            status=ReferenceStatus.IMPLEMENTED,
            models=(ResidualModel.NS,),
            notes=(
                "The original three-factor curve, implemented in "
                "yield_curve_factors.ns_basis and fitted per date by "
                "fit_nelson_siegel. Not arbitrage-free: it is the baseline whose "
                "missing convexity correction the AFNS adjustment supplies."
            ),
        ),
        Reference(
            key="hjm1992",
            title=(
                "Bond pricing and the term structure of interest rates: a new "
                "methodology for contingent claims valuation"
            ),
            authors="Heath, D., Jarrow, R. & Morton, A.",
            year=1992,
            venue="Econometrica 60(1), 77-105",
            identifier="doi:10.2307/2951677",
            url="https://www.jstor.org/stable/2951677",
            status=ReferenceStatus.BACKGROUND,
            models=(ResidualModel.NEURAL_HJM,),
            notes=(
                "The forward-rate drift restriction the neural model penalises "
                "departures from."
            ),
        ),
    )
}


def references_frame(status: ReferenceStatus | None = None) -> pl.DataFrame:
    """Registry as a Polars frame, optionally filtered by status.

    Args:
        status: Keep only references with this status; ``None`` keeps all.

    Returns:
        One row per reference, sorted by status then year.
    """
    rows = [
        {
            "key": ref.key,
            "status": ref.status.value,
            "year": ref.year,
            "authors": ref.authors,
            "title": ref.title,
            "venue": ref.venue,
            "identifier": ref.identifier,
            "url": ref.url,
            "models": ",".join(m.value for m in ref.models),
            "notes": ref.notes,
            "local_path": ref.local_path,
        }
        for ref in REFERENCES.values()
        if status is None or ref.status is status
    ]
    if not rows:
        return pl.DataFrame(
            schema={
                "key": pl.Utf8,
                "status": pl.Utf8,
                "year": pl.Int64,
                "authors": pl.Utf8,
                "title": pl.Utf8,
                "venue": pl.Utf8,
                "identifier": pl.Utf8,
                "url": pl.Utf8,
                "models": pl.Utf8,
                "notes": pl.Utf8,
                "local_path": pl.Utf8,
            }
        )
    return pl.DataFrame(rows).sort("status", "year")


def references_for(model: ResidualModel) -> list[Reference]:
    """Every reference underpinning one model, implemented ones first."""
    matched = [ref for ref in REFERENCES.values() if model in ref.models]
    return sorted(
        matched, key=lambda r: (r.status is not ReferenceStatus.IMPLEMENTED, r.year)
    )


def format_bibliography(status: ReferenceStatus | None = None) -> str:
    """Human-readable bibliography, grouped by status.

    Args:
        status: Render only this status; ``None`` renders every group.

    Returns:
        A plain-text bibliography suitable for a CLI banner or a doc appendix.
    """
    groups = [status] if status is not None else list(ReferenceStatus)
    lines: list[str] = []
    for group in groups:
        refs = sorted(
            (r for r in REFERENCES.values() if r.status is group),
            key=lambda r: (r.year, r.authors),
        )
        if not refs:
            continue
        lines.append(f"{group.label}:")
        for ref in refs:
            lines.append(f"  [{ref.key}] {ref.authors} ({ref.year}). {ref.title}.")
            lines.append(f"      {ref.venue}. {ref.identifier}")
            lines.append(f"      {ref.url}")
            if ref.local_path:
                lines.append(f"      local: {ref.local_path}")
        lines.append("")
    return "\n".join(lines).rstrip()

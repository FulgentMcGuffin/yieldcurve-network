# YieldCurve-Network

## Overview

The goal of this project is to expose the **structure of a sovereign / issuer yield-curve panel as a multi-layer network**. A curve panel is inherently two-dimensional — every observation belongs to both an *issuer* and a *maturity term* — and collapsing it into a single correlation network throws one of those dimensions away. This tool keeps both: it slices the panel along one axis into a stack of network *layers*, links the shared nodes across those layers, and analyses the stack as a single multiplex.

The system reads a **wide** curve table: one row per `(date, issuer)` and one numeric column per maturity term. This is the natural storage layout for curve data and the shape of the `par_rates` / `zero_rates` tables it was built against. The wide→long reshape, and the decision about which column plays which role, are handled automatically — there are no node-name or series-value pickers to get wrong.

A curve panel admits exactly two layerings, and the **Network Type** dropdown chooses between them:

| Network type | Layers are… | Nodes are… | Answers |
|---|---|---|---|
| **Issuer Network by Term** | maturity terms (0.5Y, 1Y, … 30Y) | issuers (DE, FR, IT, …) | *Which issuers co-move, and does that grouping change along the curve?* |
| **Term Network by Issuer** | issuers | maturity terms | *How does the curve hang together, and does its internal structure differ by issuer?* |

Inter-layer edges connect the same node across every layer it appears in, so the multiplex captures both within-layer co-movement and the cross-layer identity of a node.

Transformations (can be user supplied in the API) and/or filtering through SQL allow for the user to analyze transformed data and time/sector slices in complex ways.

If you are looking for a more generic network based exploration of datasets (i.e. not focussed on yield curves) with a similar functionality check out [https://github.com/FulgentMcGuffin/tgraphportfolio](https://github.com/FulgentMcGuffin/tgraphportfolio).


### Methods Covered

1. **Automatic panel detection**: Infers the issuer column and the set of term columns from the table schema; parses maturity labels (`0.5Y`, `10Y`, `6M`, and the zero-padded `Y000p5` form) so every axis in every plot is ordered by maturity rather than alphabetically.
2. **Pluggable Data Transforms**: Node-wise preprocessing — currently daily simple returns — applied *per layer*, which is the only correct order for a per-node operation.
3. **Selectable Pairwise Connection Measures**:
   - **Distance Correlation**: Captures both linear and non-linear associations without assuming monotonic relationships.
   - **Pearson Correlation**: Evaluates standard linear correlation.
   - **Spearman Correlation**: Measures monotonic rank-based relationships.
   - **Kendall Tau**: Rank-based correlation robust to ties and small samples.
   - **Shrinkage Correlation (Ledoit-Wolf)**: Denoised Pearson correlation via Random Matrix Theory; stabilizes estimates when samples ≈ nodes — the common case for a short window over a curve panel.
   - **Conditional Correlation**: Correlation computed only on high-magnitude move days; captures stress-regime linkage distinct from calm-period correlation.
   - **Mutual Information**: Non-linear, non-monotonic dependence detector; entropy-based measure of shared information.
   - **Chatterjee ξ**: Rank-based test for any dependence; computationally cheap alternative to distance correlation.
   - **Maximal Correlation (ACE)** *(optional, see below)*: Alternating Conditional Expectations, finding nonparametric transformations that maximize correlation.
4. **Graph Construction and Thresholding**: Prunes weak connections using a user-defined independence threshold to build weighted `NetworkX` graphs, one per layer.
5. **Multiplex Assembly**: Stacks the per-layer graphs into one graph whose vertices are `(node, layer)` pairs, joining each node to itself across every pair of layers it occurs in.
6. **Manual Cell Selection**: A mouse-driven term × issuer grid for including or excluding individual `(term, issuer)` series before the networks are built — ragged coverage is the norm in curve data, and this makes it explicit rather than implicit.
7. **Multi-Layer Metrics and Community Detection**: Per-layer intra/inter edge composition, a node × layer centrality heatmap, and per-layer community detection (ASE + KMeans) with Jaccard alignment so a community ID means the same group of nodes in every layer.
8. **Interactive 3D Multiplex Visualization**: A rotatable Plotly stack, one plane per layer, with layer toggling and click-through to a node table.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync
uv run ycn-gui
```

Point the sidebar at a DuckDB or SQLite file. If the database contains a table called `par_rates` it is selected automatically, as is a column called `date`. Otherwise, choose your relevant columns. Choose a **Network Type**, optionally narrow the data, then click **Build network**.

| Issuer Network by Term | Term Network by Issuer |
|:---:|:---:|
| ![Issuer Network by Term](rsrc/images/term_issuer_mln.png) | ![Term Network by Issuer](rsrc/images/issuer_term_mln.png) |

| Issuer Network Centrality  | Issuer Network Communities |
|:---:|:---:|
| ![Issuer Network Centrality](rsrc/images/term_issuer_centrality.png) | ![Issuer Network Communities](rsrc/images/term_issuer_community.png) |



### Test data

No curve database to hand? Generate a synthetic one:

```bash
uv run python scripts/make_fake_par_rates.py
# -> data/ycs_fake.duckdb
```

It writes a `par_rates` table in the same wide shape: 15 issuers × 10 terms × ~1250 business days, built from Nelson-Siegel curves whose level/slope/curvature factors follow bloc-correlated random walks, so the networks have genuine community structure rather than noise. Coverage is deliberately ragged — several issuers omit the long or short end, and two start late — which gives the User Filter grid real holes to display.

> The figures below were produced from that synthetic panel (Spearman, 2018, threshold 0.33), so they illustrate the views rather than any real market.

| Per-layer metrics — Issuer Network by Term | Communities — Issuer Network by Term |
|:---:|:---:|
| ![Edge composition and issuer × term eigenvector centrality](rsrc/images/mln_metrics_issuer_by_term.png) | ![Jaccard-aligned communities across term layers](rsrc/images/mln_community_issuer_by_term.png) |

| Per-layer metrics — Term Network by Issuer | Communities — Term Network by Issuer |
|:---:|:---:|
| ![Edge composition and term × issuer eigenvector centrality](rsrc/images/mln_metrics_term_by_issuer.png) | ![Jaccard-aligned communities across issuer layers](rsrc/images/mln_community_term_by_issuer.png) |

### Optional: Alternating Conditional Expectations (ACE)

The "Maximal correlation (ACE)" measure depends on [`ace_cream`](https://github.com/FulgentMcGuffin/ace_cream), which compiles a Fortran extension and therefore requires a Fortran compiler (`gfortran`) plus a C compiler at install time. It is kept optional:

```bash
uv sync --extra ace
```

If it is not installed, or its compiled extension fails to import, the GUI omits "Maximal correlation (ACE)" from the connection-measure dropdown and says so in the process log.

## The Data Model

### Expected table shape

```text
par_rates
┌────────────┬────────┬─────────┬─────────┬─────┬─────────┐
│ date       │ source │ 0.5Y    │ 1Y      │ ... │ 30Y     │
├────────────┼────────┼─────────┼─────────┼─────┼─────────┤
│ 2020-01-02 │ DE     │ -0.0061 │ -0.0059 │ ... │  0.0042 │
│ 2020-01-02 │ FR     │ -0.0048 │ -0.0041 │ ... │  0.0089 │
└────────────┴────────┴─────────┴─────────┴─────┴─────────┘
```

### Role inference

Only the **date column** is chosen by hand. Everything else is derived:

- **Term columns** are the numeric columns whose *name* parses as a maturity. Recognised spellings are `0.5Y` / `10Y` / `2.5y`, `6M`, `4W`, `30D`, and the zero-padded `Y000p5` … `Y030p0` form. A text column whose name looks like a term does not qualify — a term column must hold rates.
- **The issuer column** is the first remaining text column (falling back to the first remaining column of any type).

The sidebar reports what was found (`Issuer column: source · 10 terms: 0.5Y, 1Y, 2Y, …`). If a table yields no issuer column or fewer than two term columns it is not a curve panel, **Build network** is disabled, and the reason is stated.

Internally the panel is unpivoted to long `(date, issuer, term, rate)`, and the two network types are expressed purely as a choice of which of those columns supplies nodes and which supplies layers. Nothing in the network or measure code knows about yield curves.

### Maturity ordering

Curve axes sort by maturity everywhere — layer stacks, bar charts, heatmap axes, and the User Filter grid. This matters more than it sounds: alphabetically, `10Y` precedes `1Y` and `0.5Y` precedes `30Y`, so a plain sort renders any curve plot unreadable. Labels that do not parse as a maturity fall back to alphabetical order, so issuer axes are unaffected.

## Filtering the Panel

Three filters compose, applied in this order:

1. **Optional Filter** — tick **WHERE clause** and give a raw SQL boolean expression (e.g. `"source" <> 'GRC'`). It is pushed down into the load query, so the database does the work and the rows never reach Python. Untick it to disable the filter without losing what you typed.
2. **Date range** — seeded from the actual bounds of the chosen date column.
3. **User Filter** — a manual term × issuer cell selection, applied last.

### The User Filter

Curve panels are ragged: an issuer may not quote the short end, may have dropped the 20Y, or may only start part-way through the history. **User Filter** opens a grid of the data that survives the first two filters — terms as rows, issuers as columns — and lets you pick exactly which series take part.

- A cell exists only where the filtered data does. Combinations with no data render as inert blanks, visually distinct from an unchecked cell, and cannot be clicked.
- **Click anywhere in a cell** to toggle it.
- **Click a row or column header** to clear that whole term or issuer — or fill it, if it is already completely empty. The affected row/column flashes so the extent of the change is visible.
- **Shift+click** another header repeats that action across the contiguous range; **Ctrl+click** repeats it on scattered rows/columns. Both propagate the direction of the preceding plain click rather than flipping each section independently, so a group always ends up uniform.
- **Every header shows a `checked/available` count**, so a partially selected row or column is obvious rather than having to be read off the grid.
- **Check all / Uncheck all / Invert** act on every available cell.
- **Term order** and **Issuer order** can be reversed independently, to bring a distant cell within reach on a large panel.
- Everything defaults to checked, so the dialog is opt-out rather than opt-in.

Note that filling a row necessarily re-checks cells you had removed column-by-column (and vice versa) — the header counts are there to make that immediately visible.

Only checked cells reach the network. The selection persists between openings, but is **discarded whenever the table, date column, date range or Optional Filter changes** — the selection is a set of labels, not row identities, so it stops being meaningful once the underlying data moves. The process log says when this happens.


## Multi-Layer Network (MLN) Analysis

### Anatomy of the Multiplex

- **Nodes** are `(node, layer)` pairs. `DE` at `2Y` and `DE` at `10Y` are distinct vertices.
- **Intra-layer edges** come from the connection network computed *within* that layer, pruned by the independence threshold. Edge strength is the connection measure itself.
- **Inter-layer edges** join the same node across every pair of layers it appears in. Maturities have a natural order, but the multiplex does not restrict these to adjacent layers — a node present in eight terms is linked across all 28 pairs.

**Transforms are applied per layer, not globally.** `daily_returns` computes `pct_change` grouped by node; since a node appears in many layers, transforming the pooled frame would compute returns across interleaved rows from different layers. The MLN subsets by layer first, then transforms, then pivots.

Layers may share few nodes, or none — that is a property of the data, not a fault. When no node appears in more than one layer the process log says so explicitly, so an empty inter-layer set never reads as a bug.

### MLN Settings

| Setting | Default | Meaning |
|---------|---------|---------|
| **Centrality measure** | eigenvector | Centrality shown in the node × layer heatmap: `eigenvector`, `betweenness`, or `degree` |
| **Jaccard similarity** | 0.60 | Minimum member overlap for two per-layer communities to be treated as the *same* community across layers |
| **Community method** | fixed | k-selection strategy per layer (see below) |
| **Max communities** | 10 | For `fixed`: exact k per layer. For the optimisation methods: upper bound of the search. |

Alongside it, the main sidebar carries the **connection measure** (plus measure-specific **Edge Settings**, e.g. the stress-regime quantile for conditional correlation), the **transforms**, the **date range**, and the **independence threshold** (default 0.33 — keep edges where the measure ≥ threshold).

### Community Detection Methods

Communities are assigned within each layer using **Adjacency Spectral Embedding (ASE)** followed by **KMeans clustering**. Five strategies determine the number of communities (*k*) independently per layer:

1. **FIXED** — a fixed *k* for every layer. Simplest and most reproducible; useful for enforcing a prior ("partition every term into 4 blocs").
2. **SILHOUETTE** — maximizes the average silhouette coefficient in the latent (ASE) embedding, balancing intra-cluster cohesion against inter-cluster separation.
3. **MODULARITY** — maximizes modularity in the **original network** rather than latent space; directly optimizes a classical graph-partitioning objective.
4. **DAVIES-BOULDIN** — minimizes the Davies-Bouldin index in latent space, penalizing overlapping or poorly-separated clusters.
5. **CALINSKI-HARABASZ** — maximizes the ratio of between-cluster to within-cluster variance; tends to favour balanced partitions.

#### Cross-Layer Community Alignment

Detection is **independent within each layer** — no information crosses the layer boundary, which is what makes per-layer structure honest. But it also means cluster label `0` in the 2Y layer has nothing to do with label `0` in the 10Y layer.

The **Jaccard threshold** repairs this. After detection, each layer's communities are matched against those already seen using Hungarian assignment on Jaccard overlap of their member sets. Two communities in different layers receive the same global ID when their overlap reaches the threshold; anything below it, and anything unmatched, gets a fresh ID.

The consequence is that **a colour means the same group of nodes in every layer**, which is the only thing that makes the community heatmap readable across columns. Raise the threshold to demand stronger evidence before declaring two communities equivalent (yielding more, finer communities); lower it to merge more aggressively.

### The Three Tabs

#### MLN — Interactive 3D Multiplex

An interactive Plotly view: one translucent plane per layer, nodes arranged on a shared circle so a node keeps the same angular position in every layer, intra-layer edges coloured by connection strength, and vertical inter-layer links where nodes are shared.

- **Rotate, pan and zoom** the stack directly.
- **Hover** any node or edge for its identity, layer and weight.
- **Visible layers** checklist toggles layers in and out. Re-rendering reuses the already-computed multiplex, so toggling is immediate and never recomputes the networks.
- **Node table** beside the view lists every `(node, layer)` pair; clicking a node in the 3D graph selects and scrolls to its row.

#### MLN: Metrics

A composite figure: the top third carries per-layer **edge counts** (intra versus inter) and the **intra/inter composition**, the lower two-thirds a **node × layer centrality heatmap** using the centrality chosen in MLN Settings. Cells for a node absent from a layer are greyed rather than drawn as a low value, so absence and low centrality stay visually distinct.

An inter-layer edge touches two layers and is therefore counted under both in the per-layer chart; the figure states this beneath the composition panel.

#### MLN: Community

The node × layer community heatmap, coloured by the **Jaccard-aligned global community ID**. Reading across a row shows whether a node keeps its community across layers; reading down a column shows how a layer partitions.

Every tab has an **eye button** in the top-right that opens the underlying table — Excel-style filters, `Ctrl+C` copy, and CSV/Parquet export of the displayed data.

### Execution Model

The multiplex is built on a background thread, so the window stays responsive and the rest of the sidebar can be edited while a build runs; only **Build network** is blocked. Progress is reported per node pair and logged with an `MLN:` prefix.

**Cancel Render** is checked on every node pair, so a runaway build unwinds within one pair rather than running to completion. If a worker is momentarily between checkpoints the GUI detaches from it instead of blocking, and its results are discarded when it finally exits.

### Performance Considerations

- Cost is `L × O(n_layer²)` measure evaluations for `L` layers. Layering is usually *cheaper* than one pooled network, since `(Σnᵢ)² > Σnᵢ²`.
- **Issuer Network by Term** is the cheaper direction on a typical panel: ~10–15 term layers over ~15–40 issuer nodes. **Term Network by Issuer** inverts that — many small layers — and the log warns past 12 layers.
- Expensive measures (distance correlation, mutual information) multiply across layers. Prefer Spearman, Kendall Tau or Chatterjee ξ while exploring, then re-run with the expensive one.
- **Smoke test first**: run on a truncated date range before committing to full history.

## Network Evolution — Next

Evolution analysis extends the multiplex into a **time series of multiplexes**, applying a rolling or expanding window so structural shifts, regime changes and drifting communities become visible along the curve's history.

This is **the next piece of work and is not yet wired up**. The machinery is present and retained — `EvolutionConfig`, the evolution worker, per-window centrality and community metrics, and the rolling/expanding window schedule — and the sidebar's **Evolution** group still collects its settings (window size, step, minimum nodes per window, centrality measure, community method). The **Run Evolution** checkbox is deliberately disabled until the MLN-evolution tab exists, rather than left enabled and silently inert.

## References

### Project Foundations & Libraries

* **graspologic (Python 3.13 Compatible Fork)**: Graph statistical algorithms optimized for modern Python and dependency stacks: [GitHub Repository](https://github.com/FulgentMcGuffin/graspologic).
* **NetworkX**: Network analysis and graph structures: [Official Site](https://networkx.org/) | [GitHub](https://github.com/networkx/networkx).
* **Polars**: High-performance, multi-threaded dataframe execution engine: [Documentation](https://docs.pola.rs/) | [GitHub](https://github.com/pola-rs/polars).
* **DuckDB**: In-process analytical database used as the primary backend: [Documentation](https://duckdb.org/docs/) | [GitHub](https://github.com/duckdb/duckdb).
* **Plotly**: Interactive, browser-rendered 3D graphing used for the multiplex view: [Documentation](https://plotly.com/python/) | [GitHub](https://github.com/plotly/plotly.py).
* **MultiLayerNetViz**: 3D multiplex visualization this project's MLN view is derived from: [GitHub Repository](https://github.com/FulgentMcGuffin/MultiLayerNetViz).

### Yield Curves & Term Structure

* **Yield Curve**: The term structure of interest rates: [Wikipedia](https://en.wikipedia.org/wiki/Yield_curve).
* **Par Yield / Par Rate**: The coupon rate at which a bond prices at par, and the curve built from it: [Wikipedia](https://en.wikipedia.org/wiki/Par_yield).
* **Nelson-Siegel Model**: Parsimonious level/slope/curvature parameterisation of the yield curve, used by this project's synthetic data generator: [Wikipedia](https://en.wikipedia.org/wiki/Fixed-income_attribution#Modeling_the_yield_curve).
  - **Foundational Paper**: Nelson, C. R., & Siegel, A. F. (1987). *"Parsimonious Modeling of Yield Curves."* The Journal of Business, 60(4), 473-489: [DOI (JSTOR)](https://www.jstor.org/stable/2352957).
  - **Dynamic Extension**: Diebold, F. X., & Li, C. (2006). *"Forecasting the term structure of government bond yields."* Journal of Econometrics, 130(2), 337-364: [DOI (Elsevier)](https://doi.org/10.1016/j.jeconom.2005.03.005).

### Statistical Relationships & Correlation

* **Distance Correlation**: Capturing linear and non-linear association: [Wikipedia](https://en.wikipedia.org/wiki/Distance_correlation).
* **Pearson Correlation Coefficient**: Evaluating linear correlation: [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
* **Spearman's Rank Correlation Coefficient**: Monotonic relationship strength: [Wikipedia](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).
* **Kendall's Rank Correlation Coefficient (τ)**: Rank-based concordance measure robust to ties: [Wikipedia](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient).
* **Mutual Information**: Entropy-based measure of shared information between variables: [Wikipedia](https://en.wikipedia.org/wiki/Mutual_information).
* **Chatterjee's ξ (Xi) Correlation**: Rank-based correlation coefficient detecting any association: [arXiv](https://arxiv.org/abs/1909.10140).
* **Alternating Conditional Expectations (ACE)**: Nonparametric maximal-correlation transformation, per the "Bivariate case": [Wikipedia](https://en.wikipedia.org/wiki/Alternating_conditional_expectations).
  - **Implementation**: [ace_cream (Python 3.13 Compatible Fork)](https://github.com/FulgentMcGuffin/ace_cream).
  - **Foundational Paper**: Breiman, L., & Friedman, J. H. (1985). *"Estimating Optimal Transformations for Multiple Regression and Correlation."* Journal of the American Statistical Association, 80(391), 580-598: [DOI (Taylor & Francis)](https://doi.org/10.1080/01621459.1985.10478157).

### Robust and Specialized Correlation Methods

* **Shrinkage Correlation (Ledoit-Wolf)**: Denoised correlation via Random Matrix Theory for high-dimensional, short-window settings: [Wikipedia](https://en.wikipedia.org/wiki/Shrinkage_(statistics)).
  - **Reference**: Ledoit, O., & Wolf, M. (2004). *"Honey, I shrunk the sample covariance matrix."* Journal of Portfolio Management, 30(4), 110-119.
* **Conditional / Exceedance Correlation**: Correlation measured only during stress regimes (extreme moves) versus calm periods; captures tail co-movement and crisis linkage.

### Community Detection & Clustering

* **Silhouette Coefficient**: Average silhouette width for cluster cohesion and separation: [Wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering)).
* **Modularity Optimization**: Maximizing community structure in networks: [Wikipedia](https://en.wikipedia.org/wiki/Modularity_(networks)).
  - **Reference**: Newman, M. E. (2006). *"Modularity and community structure in networks."* Proceedings of the National Academy of Sciences, 103(23), 8577-8582: [DOI (PNAS)](https://doi.org/10.1073/pnas.0601602103).
* **Davies-Bouldin Index**: Average similarity ratio between each cluster and its most similar cluster: [Wikipedia](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index).
  - **Reference**: Davies, D. L., & Bouldin, D. W. (1979). *"A Cluster Separation Measure."* IEEE Transactions on Pattern Analysis and Machine Intelligence, 1(4), 224-227: [DOI (IEEE)](https://doi.org/10.1109/TPAMI.1979.4766909).
* **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance: [Wikipedia](https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index).
  - **Reference**: Caliński, T., & Harabasz, J. (1974). *"A Dendrite Method for Cluster Analysis."* Communications in Statistics, 3(1), 1-27: [DOI (Taylor & Francis)](https://doi.org/10.1080/03610927408827101).
* **Spectral Clustering & ASE**: Adjacency Spectral Embedding for latent-space node clustering: [graspologic Documentation](https://graspologic.readthedocs.io/).

### Multi-Layer / Multiplex Networks

* **Multilayer Networks (survey)**: The formal framework for networks composed of multiple interacting layers, including multiplex networks where the same nodes recur across layers: [Wikipedia](https://en.wikipedia.org/wiki/Multidimensional_network).
  - **Foundational Survey**: Kivelä, M., Arenas, A., Barthelemy, M., Gleeson, J. P., Moreno, Y., & Porter, M. A. (2014). *"Multilayer networks."* Journal of Complex Networks, 2(3), 203-271: [DOI (Oxford)](https://doi.org/10.1093/comnet/cnu016) | [arXiv](https://arxiv.org/abs/1309.7233).
  - **Structure and Dynamics**: Boccaletti, S., et al. (2014). *"The structure and dynamics of multilayer networks."* Physics Reports, 544(1), 1-122: [DOI (Elsevier)](https://doi.org/10.1016/j.physrep.2014.07.001).

### Community Alignment Across Layers

* **Jaccard Index**: Set-overlap similarity used to decide when two per-layer communities represent the same group: [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index).
* **Hungarian (Kuhn-Munkres) Algorithm**: Optimal one-to-one assignment, used here to match each layer's communities against those already seen: [Wikipedia](https://en.wikipedia.org/wiki/Hungarian_algorithm).
  - **Reference**: Kuhn, H. W. (1955). *"The Hungarian method for the assignment problem."* Naval Research Logistics Quarterly, 2(1-2), 83-97: [DOI (Wiley)](https://doi.org/10.1002/nav.3800020109).
  - **Implementation**: [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

### Planned: Temporal Evolution

* **Omnibus Embedding**: Simultaneously embedding multiple matched-vertex graphs into a common canonical coordinate system: [graspologic Tutorial](https://graspologic-org.github.io/graspologic/tutorials/embedding/Omnibus.html).
  - **Foundational Paper**: Levin, K., Athreya, A., Tang, M., Lyzinski, V., & Priebe, C. E. (2017). *"A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference."* [arXiv:1705.09355](https://arxiv.org/abs/1705.09355).
* **Holm-Bonferroni Method**: Sequentially rejective procedure controlling family-wise error rates, for change-point detection across consecutive windows: [Wikipedia](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method).
  - **Foundational Paper**: Holm, S. (1979). *"A simple sequentially rejective multiple test procedure."* Scandinavian Journal of Statistics, 6(2), 65-70: [JSTOR](https://www.jstor.org/stable/4615733).

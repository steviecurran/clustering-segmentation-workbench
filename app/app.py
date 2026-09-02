#!/usr/bin/env python3

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from itertools import combinations

# ---------------------------------------------------------------------
# App / repository setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Clustering & Segmentation Workbench",
    page_icon="◉",
    layout="wide",
)


st.markdown( # TO STOP STREAMLIT BEING SO SHOUTY
    """
    <style>
    /* Metric headings */
    [data-testid="stMetricLabel"] {
        font-size: 17px;
        font-weight: 600;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


BUILT_IN_DATASETS = {
    "Customer segmentation": DATA_DIR / "customer-segmentation-synthetic.csv",
}

PALETTE = ["lime", "red", "blue", "gold", "orange", "green", "grey",
    "blueviolet", "fuchsia", "turquoise", "yellow"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h"]

def cluster_colour(cluster):
    return PALETTE[int(cluster) % len(PALETTE)]


def cluster_marker(cluster):
    return MARKERS[int(cluster) % len(MARKERS)]

RANDOM_STATE = 42
plot_font = 12


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_builtin(path_string):
    return pd.read_csv(path_string)


@st.cache_data(show_spinner=False)
def read_remote_csv(url):
    return pd.read_csv(url)


def style_axes(ax,plot_font, ax_width, grid=False,minor=True):
    for spine in ax.spines.values():
        spine.set_linewidth(ax_width)

    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True,
        pad=7, length=6, width=1.5, labelsize=plot_font,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True,
        length=3, width=1.2,
    )
    
    if minor:
        ax.minorticks_on()
        ax.tick_params(axis="both",which="minor",direction="in",top=True,
            right=True,length=3,width=1.2)
    
    if grid:
        ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.8)

    return plot_font 
 

def sensible_default_features(df):
    numeric = list(df.select_dtypes(include=np.number).columns)
    return [
        c for c in numeric
        if c.strip().lower() not in {"id", "index", "identifier"}
        and not c.strip().lower().endswith("_id")
    ] or numeric


def prepare_numeric_data(raw_df, features, missing_strategy):
    data = raw_df.loc[:, features].copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    initial_rows = len(data)

    if missing_strategy == "Drop rows with missing values":
        data = data.dropna()
    else:
        medians = data.median(numeric_only=True)
        data = data.fillna(medians)
        # Columns containing only NaNs cannot be median-imputed.
        data = data.dropna(axis=1, how="all")
        data = data.dropna()

    dropped_rows = initial_rows - len(data)
    return data, dropped_rows


def scale_data(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    return X, scaler


@st.cache_data(show_spinner=False)
def kmeans_diagnostics(X, max_k):
    max_k = int(max_k)
    ks = list(range(1, max_k + 1))
    inertia = []

    for k in ks:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE,
        )
        model.fit(X)
        inertia.append(model.inertia_)

    return ks, inertia


def fit_kmeans(X, n_clusters):
    model = KMeans(
        n_clusters=int(n_clusters),
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE,
    )
    labels = model.fit_predict(X)
    return model, labels


def cluster_colour(cluster_number):
    return PALETTE[int(cluster_number) % len(PALETTE)]


def scatter_clusters(data, x_col, y_col, labels, label_names=None, title=None):

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    plot_font = style_axes(ax, 10, 1.5)

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    for cluster in unique_labels:
        mask = labels == cluster
        display_label = (
            label_names.get(int(cluster), f"Cluster {cluster}")
            if label_names
            else f"Cluster {cluster}"
        )
   
        ax.scatter(data.loc[mask, x_col], data.loc[mask, y_col], s=25, alpha=0.75,
                   color=cluster_colour(cluster),
                   marker=cluster_marker(cluster),
                   edgecolors="dimgrey",
                   linewidths=0.6,label=display_label)

    ax.set_xlabel(x_col, fontsize=plot_font)
    ax.set_ylabel(y_col, fontsize=plot_font)
    if title:
        ax.set_title(title, fontsize=plot_font)

    ax.legend(fontsize=0.8*plot_font, loc="best", frameon=False)
    fig.tight_layout()
    return fig


def pca_group_scatter(data, x_col, y_col, labels, label_names):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    plot_font = style_axes(ax, 10, 1.5)

    labels = np.asarray(labels)

    for cluster in np.unique(labels):

        mask = labels == cluster

        display_label = label_names.get(
            int(cluster),
            f"Cluster {cluster}",
        )

        ax.scatter(
            data.loc[mask, x_col],
            data.loc[mask, y_col],
            s=25,
            alpha=0.75,
            color=cluster_colour(cluster),
            marker=cluster_marker(cluster),
            edgecolors="dimgrey",
            linewidths=0.6,
            label=display_label,
        )

    ax.set_xlabel(x_col, fontsize=plot_font)
    ax.set_ylabel(y_col, fontsize=plot_font)

    ax.legend(
        fontsize=0.8 * plot_font,
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()
    return fig

def correlation_figure(corr):
    n = len(corr.columns)
    # Give the correlation matrix enough physical space to remain legible
    # when Streamlit renders it across the main page.
    fig_size = min(11, max(9, 0.45 * n))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    label_font = 12

    image = ax.imshow(corr.values,cmap=plt.cm.rainbow,vmin=-1,
        vmax=1,interpolation="nearest",aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=label_font)
    ax.set_yticklabels(corr.index, fontsize=label_font)
    ax.tick_params(direction="out", top=False, right=False)

    if n <= 12:
        for i in range(n):
            for j in range(n):
                value = corr.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=0.8*label_font,
                    color="black" if abs(value) < 0.65 else "white",
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=label_font)
    ax.set_title("Pearson correlations between selected features", fontsize=label_font)
    fig.tight_layout()
    return fig


def strong_correlations(corr, threshold):
    rows = []
    cols = list(corr.columns)

    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            r = corr.loc[left, right]
            if abs(r) >= threshold:
                rows.append(
                    {
                        "Feature 1": left,
                        "Feature 2": right,
                        "Pearson r": r,
                        "|r|": abs(r),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Pearson r", "|r|"])

    return pd.DataFrame(rows).sort_values("|r|", ascending=False).reset_index(drop=True)


def hierarchical_figure(X, branches, sample_limit, data_for_elbow):
    max_k = min(10, len(data_for_elbow))
    ks, inertia = kmeans_diagnostics(data_for_elbow, max_k=max_k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    plot_font = style_axes(ax1, 15,1.5,grid=True)
    style_axes(ax2, 15,1.5)

    # Elbow plot
    ax1.plot(ks, inertia, marker="o", ls="-", color="blue", lw=2)
    if int(branches) in ks:
        y_branch = inertia[int(branches) - 1]
        ax1.plot(
            [branches, branches],
            [0, y_branch],
            color="red",
            ls="dotted",
            lw=2,
            label="Selected branch count",
        )
        ax1.annotate(
            f"WCSS: {y_branch:.3e}",
            xy=(branches, y_branch),
            xytext=(branches + 0.2, y_branch),
            color="red",
            fontweight="bold",
            fontsize= 0.8*plot_font,
        )
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Number of clusters", fontsize=plot_font)
    ax1.set_ylabel("WCSS", fontsize=plot_font)
    ax1.set_title("K-means clustering elbow plot", fontsize=plot_font)
    if int(branches) in ks:
        ax1.legend(fontsize=0.8*plot_font, loc="upper right", frameon=False)

    # Dendrogram
    if len(X) > sample_limit:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(len(X), sample_limit, replace=False)
        linkage_data = X[indices]
    else:
        linkage_data = X

    hierarchy = linkage(linkage_data, method="ward")
    dendrogram(
        hierarchy,
        truncate_mode="lastp",
        p=int(branches),
        show_leaf_counts=True,
        no_labels=False,
        ax=ax2,
    )

    ax2.set_title(
        f"Hierarchical clustering dendrogram — {branches} final branches",
        fontsize=plot_font + 1,
    )
    ax2.set_xlabel("Observations", fontsize=plot_font)
    ax2.set_ylabel("Distance", fontsize=plot_font)

    fig.tight_layout()
    return fig, min(len(X), sample_limit)


def pca_diagnostics(X, n_clusters, target_variance, manual_components=None):
    full_pca = PCA()
    full_pca.fit(X)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)

    if target_variance == "Manual":
        n_components = int(manual_components)
    else:
        target = float(target_variance)
        n_components = int(np.argmax(cumulative >= target) + 1)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    max_k = min(10, len(scores))
    ks, inertia = kmeans_diagnostics(scores, max_k=max_k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    plot_font = style_axes(ax1,15,1.5)
    style_axes(ax2,15,1.5)

    x = np.arange(1, len(cumulative) + 1)
    ax1.plot(x, cumulative, marker="o", ls="-", color="red", lw=2)

    if target_variance == "Manual":
        y_line = cumulative[n_components - 1]
        ax1.axhline(
            y=y_line,
            color="orange",
            linestyle=":",
            label="Manual selection",
        )
    else:
        ax1.axhline(
            y=float(target_variance),
            color="green",
            linestyle=":",
            label="Variance target",
        )

    ax1.axvline(
        x=n_components,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Cutoff ({n_components})",
    )
    ax1.set_xlabel("Number of components", fontsize=plot_font)
    ax1.set_ylabel("Cumulative explained variance", fontsize=plot_font)
    ax1.set_title("Explained variance by components", fontsize=plot_font)
    if len(cumulative) <= 20:
        ax1.set_xticks(x)
    ax1.legend(loc="lower right", fontsize=0.8*plot_font, frameon=False)

    ax2.plot(ks, inertia, marker="o", ls="-", color="blue", lw=2)
    if int(n_clusters) in ks:
        ax2.plot(n_clusters,inertia[int(n_clusters) - 1],marker="*",
            color="gold",markersize=14,markeredgecolor="black",
            label=f"Selected K={n_clusters}")
        ax2.legend(fontsize=0.8*plot_font, frameon=False)

    ax2.set_xlabel("Number of clusters", fontsize=plot_font)
    ax2.set_ylabel("Within-cluster sum of squares", fontsize=plot_font)
    ax2.set_title("K-means clustering in PCA space", fontsize=plot_font)
    ax2.set_xticks(ks)

    fig.tight_layout()
    return fig, pca, scores, cumulative, n_components


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Clustering & Segmentation Workbench")
st.caption(
    "Explore numeric datasets using correlations, hierarchical clustering, "
    "K-means segmentation and PCA."
)

with st.expander("What this app does", expanded=False):
    st.markdown(
        """
This workbench converts the original clustering notebook into an interactive
analysis pipeline. It is intended for exploratory segmentation rather than
supervised prediction.

**Workflow**

1. Load a built-in dataset, upload a CSV, or provide a remote CSV URL.
2. Select the numeric features used for clustering.
3. Explore correlations and hierarchical structure.
4. Fit K-means clusters and inspect their profiles.
5. Use PCA to reduce dimensionality and re-run K-means in the retained PCA space.

All clustering is performed on **standardised numeric features**.
"""
    )


# ---------------------------------------------------------------------
# Data source / sidebar
# ---------------------------------------------------------------------
st.sidebar.header("Data source")

source = st.sidebar.radio(
    "Choose source",
    ["Built-in example", "Upload CSV", "Remote CSV URL"],
)

raw_df = None
source_name = None

if source == "Built-in example":
    example_name = st.sidebar.selectbox(
        "Example dataset",
        list(BUILT_IN_DATASETS.keys()),
        index=0,
    )
    source_name = example_name
    raw_df = read_builtin(str(BUILT_IN_DATASETS[example_name]))

elif source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            source_name = uploaded.name
        except Exception as exc:
            st.sidebar.error(f"Could not read CSV: {exc}")

else:
    url = st.sidebar.text_input(
        "CSV URL",
        placeholder="https://example.com/data.csv",
    )
    if url:
        try:
            with st.spinner("Loading remote CSV..."):
                raw_df = read_remote_csv(url)
            source_name = url
        except Exception as exc:
            st.sidebar.error(f"Could not load remote CSV: {exc}")

if raw_df is None:
    st.info("Choose a dataset in the sidebar to begin.")
    st.stop()

if raw_df.empty:
    st.error("The selected dataset contains no rows.")
    st.stop()

numeric_columns = list(raw_df.select_dtypes(include=np.number).columns)

if len(numeric_columns) < 2:
    st.error(
        "This workbench needs at least two numeric columns. "
        "Encode or add numeric features before clustering."
    )
    st.stop()

st.sidebar.header("Analysis settings")


features = st.sidebar.multiselect(
    "Numeric features",
    numeric_columns,
    default=default_features,
    help="ID-like columns are excluded from the default selection, but can be added manually.",
)

if len(features) < 2:
    st.warning("Select at least two numeric features.")
    st.stop()

missing_strategy = st.sidebar.selectbox(
    "Missing values",
    ["Drop rows with missing values", "Median imputation"],
)

data, dropped_rows = prepare_numeric_data(
    raw_df,
    features,
    missing_strategy,
)

if len(data) < 3:
    st.error("Too few complete rows remain for clustering.")
    st.stop()

# If median imputation removes all-NaN columns, reflect that.
features = list(data.columns)

if len(features) < 2:
    st.error("Too few usable numeric features remain after preprocessing.")
    st.stop()

X, scaler = scale_data(data)

max_clusters = max(2, min(10, len(data) - 1))
default_k = min(4, max_clusters)

n_clusters = st.sidebar.slider(
    "Number of clusters",
    min_value=2,
    max_value=max_clusters,
    value=default_k,
)

st.sidebar.caption(f"Random seed: {RANDOM_STATE}")


# ---------------------------------------------------------------------
# Shared K-means fit
# ---------------------------------------------------------------------
kmeans_model, cluster_ids = fit_kmeans(X, n_clusters)

clustered_numeric = data.copy()
clustered_numeric["Cluster"] = cluster_ids


default_cluster_names = {cluster: f"Cluster {cluster}" for cluster in range(n_clusters)
}

# Read current labels from session state before drawing the K-means plot.
# The text widgets themselves are placed directly below that plot.
cluster_name_keys = {
    cluster: f"cluster-name-{source_name}-{n_clusters}-{cluster}"
    for cluster in range(n_clusters)
}

cluster_names = {
    cluster: st.session_state.get(
        cluster_name_keys[cluster],
        default_cluster_names[cluster],
    )
    for cluster in range(n_clusters)
}

clustered_numeric["Cluster label"] = clustered_numeric["Cluster"].map(cluster_names)

clustered_export = raw_df.loc[data.index].copy()
clustered_export["Cluster"] = cluster_ids
clustered_export["Cluster label"] = clustered_export["Cluster"].map(cluster_names)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tabs = st.tabs(
    [
        "Dataset",
        "Correlations",
        "Hierarchy",
        "K-means",
        "Cluster profiles",
        "PCA",
    ]
)

# ------------------------------- Dataset -----------------------------
with tabs[0]:
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows loaded", f"{len(raw_df):,}")
    c2.metric("Rows analysed", f"{len(data):,}")
    c3.metric("Numeric features", len(features))
    c4.metric("Rows removed", f"{dropped_rows:,}")

    non_numeric = [
        c for c in raw_df.columns
        if c not in raw_df.select_dtypes(include=np.number).columns
    ]

    if non_numeric:
        st.info(
            "Non-numeric columns are retained for export but not used directly in clustering: "
            + ", ".join(map(str, non_numeric))
        )

    if dropped_rows:
        st.warning(
            f"{dropped_rows:,} rows were removed because of missing/non-finite values "
            f"under the current preprocessing choice."
        )

    st.markdown("**Selected clustering features**")
    st.write(", ".join(map(str, features)))

    with st.expander("Preview data"):
        total_rows = len(raw_df)
        preview_cap = 20_000

        st.markdown("**Choose number of rows to display**")

        if total_rows <= 10:
            preview_rows = total_rows
            st.caption(f"Showing all {total_rows:,} rows.")

        else:
            slider_max = min(total_rows, preview_cap)

            row_options = np.geomspace(
                10,
                slider_max,
                num=50,
            )

            row_options = sorted(
                set(int(round(value)) for value in row_options)
            )

            if slider_max not in row_options:
                row_options.append(slider_max)

            default_rows = min(100, slider_max)

            default_value = min(
                row_options,
                key=lambda value: abs(value - default_rows),
            )

            preview_rows = st.select_slider(
                "Rows to preview",
                options=row_options,
                value=default_value,
                format_func=lambda value: f"{value:,}",
                label_visibility="collapsed",
            )

            if total_rows > preview_cap:
                show_all = st.checkbox(
                    f"Show all {total_rows:,} rows"
                )

                if show_all:
                    preview_rows = total_rows

            st.caption(
                f"Showing {preview_rows:,} of {total_rows:,} rows."
            )

        st.dataframe(
            raw_df.head(preview_rows),
            use_container_width=True,
            hide_index=True,
        )

    #st.dataframe(raw_df.head(100), use_container_width=True)

    with st.expander("Numeric summary"):
        st.dataframe(data.describe().T, use_container_width=True)


# ----------------------------- Correlations --------------------------
with tabs[1]:
    st.subheader("Correlation structure")

    corr_text_col, corr_slider_col = st.columns([2.2, 1])

    with corr_text_col:
        st.markdown(
            "The heatmap shows all pairwise Pearson correlations. "
            "Use the threshold to control which stronger relationships "
            "are listed in the table underneath."
        )

    with corr_slider_col:
        corr_threshold = st.slider(
            "Correlation threshold |r|",
            min_value=0.50,
            max_value=1.00,
            value=0.50,
            step=0.05,
            help=(
                "Only feature pairs with an absolute Pearson correlation at or above "
                "this value are shown in the table. The heatmap itself is unchanged."
            ),
            key="correlation-threshold",
        )

    corr = data.corr(numeric_only=True)
    strong = strong_correlations(corr, corr_threshold)

    fig = correlation_figure(corr)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        f"**Feature pairs with |Pearson r| ≥ {corr_threshold:.2f}**"
    )
    if strong.empty:
        st.info("No feature pairs exceed the selected threshold.")
    else:
        display_corr = strong.copy()
        display_corr["Pearson r"] = display_corr["Pearson r"].round(3)
        display_corr["|r|"] = display_corr["|r|"].round(3)
        st.dataframe(display_corr, use_container_width=True, hide_index=True)


# ------------------------------- Hierarchy ---------------------------
with tabs[2]:
    st.subheader("Hierarchical clustering exploration")

    st.markdown(
        "The dendrogram provides an exploratory view of how observations merge into "
        "larger groups. The elbow curve provides a complementary K-means compactness diagnostic."
    )

    hierarchy_left, hierarchy_right = st.columns(2)

    with hierarchy_left:
        branches = st.slider(
            "Final dendrogram branches",
            min_value=2,
            max_value=min(15, max_clusters),
            value=min(n_clusters, min(15, max_clusters)),
            help=(
                "Controls how many final branches are displayed in the truncated "
                "dendrogram. It does not change the K-means cluster count."
            ),
            key="dendrogram-branches",
        )

    sample_cap = min(5000, len(data))
    sample_default = min(2500, sample_cap)

    with hierarchy_right:
        dendrogram_sample = st.slider(
            "Dendrogram sample size",
            min_value=min(100, sample_cap),
            max_value=sample_cap,
            value=max(min(100, sample_cap), sample_default),
            step=max(1, min(100, sample_cap)),
            help=(
                "Number of rows used to calculate the hierarchical dendrogram. "
                "Ward linkage can be expensive for large datasets, so only the "
                "dendrogram is sampled. K-means diagnostics still use all analysed rows."
            ),
            key="dendrogram-sample",
        )

    st.caption(
        "The sample-size control affects only the dendrogram calculation. "
        "The elbow curve and K-means clustering continue to use the full analysed dataset."
    )

    with st.spinner("Calculating hierarchy..."):
        fig, used_rows = hierarchical_figure(X,branches=branches,sample_limit=dendrogram_sample,
            data_for_elbow=X)

    st.pyplot(fig, clear_figure=True)

    if used_rows < len(data):
        st.caption(
            f"Dendrogram calculated from a reproducible sample of {used_rows:,} "
            f"of {len(data):,} rows. K-means diagnostics use all analysed rows."
        )


# -------------------------------- K-means ----------------------------
with tabs[3]:
    st.subheader("K-means segmentation")

    default_x = features.index("Age") if "Age" in features else 0
    default_y = features.index("Income") if "Income" in features else (1 if len(features) > 1 else 0)

    x_col, y_col = st.columns(2)
    with x_col:
        x_feature = st.selectbox(
            "X-axis feature",
            features,
            index=default_x,
            key="kmeans-x",
        )
    with y_col:
        y_feature = st.selectbox(
            "Y-axis feature",
            features,
            index=default_y,
            key="kmeans-y",
        )

    if x_feature == y_feature:
        st.warning("Choose different X and Y features for a more useful projection.")


    fig = scatter_clusters(
        data,
        x_feature,
        y_feature,
        cluster_ids,
        label_names=cluster_names,
        title=f"K-means segmentation — K={n_clusters}",
    )
    st.pyplot(fig, clear_figure=True)

    with st.expander("Relabel groups", expanded=True):
        st.caption(
            "Inspect the segmentation above, then replace the numerical cluster "
            "names with meaningful group labels. The plot updates automatically, "
            "and the same labels are used throughout the rest of the app."
        )

        per_row = min(4, n_clusters)

        for row_start in range(0, n_clusters, per_row):
            name_columns = st.columns(per_row)

            for offset, cluster in enumerate(
                range(row_start, min(row_start + per_row, n_clusters))
            ):
                with name_columns[offset]:
                    colour = cluster_colour(cluster)
                    marker = cluster_marker(cluster)

                    st.markdown(
                            f"""
                            <div style="
                            display:flex;
                            align-items:center;
                            gap:8px;
                            margin-bottom:2px;
                            font-weight:600;
                            ">
                            <span style="
                            display:inline-block;
                            width:14px;
                            height:14px;
                            border-radius:50%;
                            background:{colour};
                            border:1px solid black;
                            "></span>
                            Cluster {cluster}
                            </div>
                            """,
                        unsafe_allow_html=True,
                    )

                    st.text_input(
                        f"Cluster {cluster}",
                        value=default_cluster_names[cluster],
                        key=cluster_name_keys[cluster],
                        label_visibility="collapsed",
                    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Clusters", n_clusters)
    metric_cols[1].metric("WCSS", f"{kmeans_model.inertia_:,.1f}")

    if n_clusters < len(data):
        try:
            sil = silhouette_score(X, cluster_ids)
            metric_cols[2].metric("Silhouette score", f"{sil:.3f}")
        except Exception:
            metric_cols[2].metric("Silhouette score", "—")

    with st.expander("What to look for"):
        st.markdown(
            """
- **Compactness:** lower WCSS generally indicates tighter clusters, but WCSS always falls as K increases.
- **Separation:** silhouette values nearer 1 indicate better-separated clusters; values around 0 indicate overlap.
- **Interpretability:** a statistically distinct cluster is only useful if its profile makes practical sense.
"""
        )


# ---------------------------- Cluster profiles -----------------------
with tabs[4]:
    st.subheader("Cluster interpretation")

    profile = clustered_numeric.groupby("Cluster")[features].mean()
    profile["N observations"] = clustered_numeric.groupby("Cluster").size()
    profile["Proportion"] = profile["N observations"] / len(clustered_numeric)
    profile.index = [cluster_names[int(i)] for i in profile.index]

    st.dataframe(profile.round(3), use_container_width=True)

    st.markdown("**Cluster sizes**")
    size_table = (
        clustered_numeric.groupby(["Cluster", "Cluster label"])
        .size()
        .rename("N observations")
        .reset_index()
    )
    size_table["Proportion"] = size_table["N observations"] / len(clustered_numeric)
    st.dataframe(size_table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download clustered CSV",
        clustered_export.to_csv(index=False).encode("utf-8"),
        file_name="clustered-data.csv",
        mime="text/csv",
    )


# ---------------------------------- PCA ------------------------------
with tabs[5]:
    st.subheader("Principal component analysis")

    left_control, right_control = st.columns(2)

    with left_control:
        variance_choice = st.selectbox(
            "Target explained variance",
            ["70%", "75%", "80%", "85%", "90%", "95%", "99%", "Manual"],
            index=2,
        )

    max_manual = min(len(features), len(data))
    with right_control:
        if variance_choice == "Manual":
            manual_components = st.slider(
                "Number of components",
                min_value=1,
                max_value=max_manual,
                value=min(3, max_manual),
            )
        else:
            manual_components = None

    if variance_choice == "Manual":
        variance_target = "Manual"
    else:
        variance_target = float(variance_choice.strip("%")) / 100

    fig, pca, scores, cumulative, n_components = pca_diagnostics(
        X,
        n_clusters=n_clusters,
        target_variance=variance_target,
        manual_components=manual_components,
    )
    st.pyplot(fig, clear_figure=True)

    captured = cumulative[n_components - 1]
    m1, m2 = st.columns(2)
    m1.metric("Components retained", n_components)
    m2.metric("Variance captured", f"{captured * 100:.1f}%")

    component_names = [f"Component {i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=component_names, index=data.index)

    # Match the original notebook: after reducing the feature space with PCA,
    # fit K-means again to the retained PCA scores.
    pca_kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=RANDOM_STATE,
        n_init="auto",
    )
    pca_cluster_ids = pca_kmeans.fit_predict(scores)

    # Quantify how well the PCA clusters separate in every 2D component pairing
pair_results = []

if n_components >= 2:
    for comp_x, comp_y in combinations(component_names, 2):

        pair_data = scores_df[
            [comp_x, comp_y]
        ].to_numpy()

        pair_score = silhouette_score(
            pair_data,
            pca_cluster_ids,
        )

        pair_results.append(
            {
                "Component X": comp_x,
                "Component Y": comp_y,
                "Silhouette score": pair_score,
            }
        )

    pair_results = (
        pd.DataFrame(pair_results)
        .sort_values(
            "Silhouette score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if n_components >= 2:
        p1, p2 = st.columns(2)
        with p1:
            pca_x = st.selectbox(
                "PCA X-axis",
                component_names,
                index=1,  # Notebook default: Component 2 on x-axis
            )
        with p2:
            pca_y = st.selectbox(
                "PCA Y-axis",
                component_names,
                index=0,  # Notebook default: Component 1 on y-axis
            )

        pca_plot_data = scores_df.copy()
        pca_plot_data["Cluster"] = pca_cluster_ids

        fig2 = pca_group_scatter(
            pca_plot_data,
            pca_x,
            pca_y,
            pca_cluster_ids,
            label_names=cluster_names,
        )
        st.pyplot(fig2, clear_figure=True)

        current_pair = scores_df[
            [pca_x, pca_y]
        ].to_numpy()

        current_pair_score = silhouette_score(
            current_pair,
            pca_cluster_ids,
        )

        best_pair = pair_results.iloc[0]

        score_col1, score_col2 = st.columns(2)

        score_col1.metric(
            "Current 2D silhouette",
            f"{current_pair_score:.3f}",
        )

        score_col2.metric(
            "Best 2D silhouette",
            f"{best_pair['Silhouette score']:.3f}",
        )

        st.caption(
            f"Best component pairing for 2D cluster separation: "
            f"{best_pair['Component X']} vs {best_pair['Component Y']}."
        )

        with st.expander("Compare PCA component pairings"):
            st.dataframe(
                pair_results.round(3),
                use_container_width=True,
                hide_index=True,
            )
            
        try:
            pca_sil = silhouette_score(scores, pca_cluster_ids)
            st.caption(
                f"K-means re-fitted in the retained PCA space, matching the notebook. "
                f"PCA-space silhouette score: {pca_sil:.3f}."
            )
        except Exception:
            st.caption(
                "K-means re-fitted in the retained PCA space, matching the notebook."
            )
    else:
        st.info("Retain at least two components to display a 2D PCA projection.")

    loadings = pd.DataFrame(
        pca.components_,
        columns=features,
        index=component_names,
    )

    with st.expander("PCA component loadings"):
        st.dataframe(loadings.round(4), use_container_width=True)

    st.download_button(
        "Download PCA loadings",
        loadings.to_csv().encode("utf-8"),
        file_name="pca-loadings.csv",
        mime="text/csv",
    )

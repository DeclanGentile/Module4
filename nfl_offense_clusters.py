import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.spatial import ConvexHull

import nfl_data_py as nfl

# --------- CONFIG ---------
YEARS = [2024]                 # e.g., [2022, 2023] if you want multiple seasons
K_RANGE = range(2, 9)          # elbow candidates
K_FINAL = 4                    # set after eyeballing the elbow
RANDOM_STATE = 0
LOGOS_DIR = "logos"            # folder with PNGs named like KC.png, PHI.png, DAL.png, ...
TARGET_LOGO_HEIGHT_PX = 16     # 40–64 are typical; all logos will be this height
TRIM_TRANSPARENT_BORDERS = True
HULL_ALPHA = 0.15              # transparency of the cluster hull fill
HULL_EDGE_ALPHA = 0.7          # edge visibility for the hull outline
# --------------------------

def load_pbp(years):
    pbp = nfl.import_pbp_data(years=years)
    pbp = pbp[pbp["play_type"].isin(["run", "pass"])].copy()
    return pbp

def team_offense_features(pbp):
    df = pbp.copy()
    df["off_epa_pp"] = df["epa"]
    df["success"] = (df["epa"] > 0).astype(int)
    if "pass" in df.columns:
        df["is_pass"] = df["pass"].astype(float).fillna(0.0)
    else:
        df["is_pass"] = (df["play_type"] == "pass").astype(int)
    df["yards_per_play"] = df["yards_gained"]

    g = (
        df.groupby(["season", "posteam"], dropna=True)
          .agg(
              off_epa_pp=("off_epa_pp", "mean"),
              off_success_rate=("success", "mean"),
              pass_rate=("is_pass", "mean"),
              yards_per_play=("yards_per_play", "mean"),
              plays=("play_id", "count")
          )
          .reset_index()
          .rename(columns={"posteam": "team"})
    )
    g = g[g["plays"] > 100].reset_index(drop=True)

    id_cols = ["season", "team"]
    feat_cols = ["off_epa_pp", "off_success_rate", "pass_rate", "yards_per_play"]
    return g[id_cols + feat_cols + ["plays"]].copy(), id_cols, feat_cols

def standardize_features(df, feature_cols):
    X = StandardScaler().fit_transform(df[feature_cols])
    return X

def elbow_plot(X, k_range, save_path="elbow_nfl_offense.png"):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xticks(list(k_range))
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.title("Elbow Method: NFL Offensive Clustering")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print("\nElbow inertias:")
    for k, inertia in zip(k_range, inertias):
        print(f"k={k}, inertia={inertia:.2f}")

def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    return km, labels

def summarize_clusters(df, id_cols, feature_cols):
    print("\nCluster means (raw scale):")
    print(df.groupby("cluster")[feature_cols].mean().round(3))
    print("\nExample teams per cluster:")
    for clus, g in df.groupby("cluster"):
        examples = g.sort_values("plays", ascending=False).head(2)[id_cols]
        print(f"  Cluster {clus}:\n{examples.to_string(index=False)}")

# --- Uniform logo helper: trim + resize each logo to a fixed height ---
def _add_logo(ax, xy, logo_path):
    try:
        img = Image.open(logo_path).convert("RGBA")
        if TRIM_TRANSPARENT_BORDERS:
            alpha = img.split()[-1]
            bbox = alpha.getbbox()
            if bbox:
                img = img.crop(bbox)
        w, h = img.size
        if h != TARGET_LOGO_HEIGHT_PX:
            new_w = int(round(w * (TARGET_LOGO_HEIGHT_PX / h)))
            img = img.resize((new_w, TARGET_LOGO_HEIGHT_PX), Image.LANCZOS)
        arr = np.asarray(img)
        oi = OffsetImage(arr, zoom=1.0)  # already normalized size
        ab = AnnotationBbox(oi, xy, frameon=False, zorder=3)
        ax.add_artist(ab)
        return True
    except Exception:
        return False

def _draw_cluster_hull(ax, x_vals, y_vals, color):
    """Draw a translucent convex hull around the (x,y) points."""
    pts = np.column_stack([x_vals, y_vals])
    # Need at least 3 points and non-collinear
    if pts.shape[0] < 3:
        return
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                facecolor=color, alpha=HULL_ALPHA, edgecolor=color,
                linewidth=1.5, zorder=0)
        # optional: slightly darker outline
        ax.plot(np.append(hull_pts[:, 0], hull_pts[0, 0]),
                np.append(hull_pts[:, 1], hull_pts[0, 1]),
                color=color, alpha=HULL_EDGE_ALPHA, linewidth=1.5, zorder=1)
    except Exception:
        # Degenerate cases (all points same line etc.) — skip hull
        pass

def scatter_2d_with_logos(df, X, feature_cols, save_path="offense_clusters_scatter.png", logos_dir=LOGOS_DIR):
    # Prefer interpretable axes; fallback to PCA only if needed
    candidates = [("off_epa_pp", "pass_rate"), ("yards_per_play", "off_success_rate")]
    for a, b in candidates:
        if a in feature_cols and b in feature_cols:
            x_col, y_col = a, b
            pca_mode = False
            break
    else:
        XY = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
        df = df.copy()
        df["_PC1"], df["_PC2"] = XY[:, 0], XY[:, 1]
        x_col, y_col = "_PC1", "_PC2"
        pca_mode = True

    x = df[x_col].values
    y = df[y_col].values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(True, linestyle="--", alpha=0.4)

# Add quadrant axis lines

    ax.set_xlabel("PC1" if pca_mode else x_col)
    ax.set_ylabel("PC2" if pca_mode else y_col)
    ax.set_title("NFL Offense Clusters (Logos + Hulls)")

    # consistent colors for clusters
    cmap = plt.get_cmap("tab10")
    cluster_ids = sorted(df["cluster"].unique())
    color_map = {cl: cmap(i % 10) for i, cl in enumerate(cluster_ids)}

    # 1) draw translucent hulls per cluster (behind everything)
    for cl in cluster_ids:
        m = (df["cluster"].values == cl)
        _draw_cluster_hull(ax, x[m], y[m], color_map[cl])

    # 2) faint backdrop points per cluster
    for cl in cluster_ids:
        m = (df["cluster"].values == cl)
        ax.scatter(x[m], y[m], color=color_map[cl], alpha=0.18, s=60, zorder=2, label=f"Cluster {cl}")

    # 3) overlay team logos
    for _, row in df.iterrows():
        team = row.get("team", None)
        xv, yv = row[x_col], row[y_col]
        logo_file = os.path.join(logos_dir, f"{team}.png") if team else None
        ok = bool(team) and os.path.isfile(logo_file) and _add_logo(ax, (xv, yv), logo_file)
        if not ok:
            # visible fallback dot with cluster color
            ax.scatter([xv], [yv], s=80, edgecolor="k", linewidth=0.5, alpha=0.9,
                       color=color_map[row["cluster"]], zorder=3)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    pbp = load_pbp(YEARS)
    team_df, id_cols, feat_cols = team_offense_features(pbp)

    X = standardize_features(team_df, feat_cols)
    elbow_plot(X, K_RANGE, save_path="elbow_nfl_offense.png")

    km, labels = fit_kmeans(X, K_FINAL)
    team_df = team_df.copy()
    team_df["cluster"] = labels

    summarize_clusters(team_df, id_cols, feat_cols)

    out_cols = id_cols + feat_cols + ["plays", "cluster"]
    team_df[out_cols].to_csv("nfl_offense_clusters.csv", index=False)
    print("\nSaved results to nfl_offense_clusters.csv")

    scatter_2d_with_logos(team_df, X, feat_cols, save_path="offense_clusters_scatter.png", logos_dir=LOGOS_DIR)

if __name__ == "__main__":
    main()

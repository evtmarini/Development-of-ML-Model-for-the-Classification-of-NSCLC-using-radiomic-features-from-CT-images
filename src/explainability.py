"""
Full Explainability module for radiomics ML projects with SHAP + full LIME support.

Main entrypoint: Explainability.run() or run_explainability()

Features:
- SHAP explainers with safe fallback (Tree/Linear/Kernel)
- LIME full support (raw values, top-k, plots, HTML per sample)
- Plots: summary, bar, waterfall (class-wise), dependence, clustering, 3D PCA
- CSV/HTML exports: global importance, top-k per class, common top-k, correlations, stability,
  SHAP/LIME vs model importance
- Raw artifacts: SHAP values, LIME explanations
- Fully modular, robust logging (stdout + file)

Usage:
    results = run_explainability(
        model=best_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=X_train.columns,
        output_dir=Path("results/explainability"),
        fold_name="Fold0"
    )
"""
from pathlib import Path
import logging
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Dict, Any

# optional LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception:
    _HAS_LIME = False

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

plt.switch_backend("Agg")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# file logging
file_handler = logging.FileHandler("explainability.log")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(file_handler)

warnings.filterwarnings("ignore", message="X has feature names")


class Explainability:
    """
    Explainability utilities for tabular radiomics ML models.

    Assumptions:
    - Classification by default (predict_proba available)
    - Supports binary and multiclass
    - Falls back safely for non-tree models
    - Full LIME integration (mirrors SHAP outputs)
    """

    def __init__(self, model, output_dir: Path = Path("results/explainability"), fold_name: str = "fold",
                 max_samples_shap: int = 100, random_state: int = 42, top_k: int = 10):
        self.model = model
        self.output_dir = Path(output_dir) / fold_name
        self.ext_dir = self.output_dir / "extended"
        self.max_samples_shap = max_samples_shap
        self.random_state = random_state
        self.top_k = top_k
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ext_dir.mkdir(parents=True, exist_ok=True)

    def _to_df(self, X, feature_names):
        return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

    def _choose_shap_explainer(self, X_bg: pd.DataFrame):
        try:
            expl = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
            return expl
        except Exception:
            pass
        try:
            expl = shap.LinearExplainer(self.model, X_bg, feature_perturbation="interventional")
            logger.info("Using LinearExplainer")
            return expl
        except Exception:
            pass
        bg = X_bg.sample(min(50, len(X_bg)), random_state=self.random_state)
        pred_fn = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
        expl = shap.KernelExplainer(pred_fn, bg)
        logger.info("Using KernelExplainer (fallback)")
        return expl

    def _save_fig(self, fig, path: Path, dpi: int = 300):
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to save figure {path}: {e}")

    #### SHAP METHODS ####
    def compute_shap_values(self, explainer, X_sample: pd.DataFrame):
        sv = explainer(X_sample)
        return sv, np.array(sv.values)

    def save_raw_shap(self, vals: np.ndarray, X_sample: pd.DataFrame):
        try:
            np.save(self.ext_dir / "shap_values.npy", vals)
            X_sample.to_csv(self.ext_dir / "shap_samples.csv", index=False)
            logger.info("Raw SHAP values saved")
            return str(self.ext_dir / "shap_values.npy")
        except Exception as e:
            logger.warning(f"Failed to save raw SHAP values: {e}")
            return None

    def shap_interactions(self, explainer, X_sample: pd.DataFrame):
        try:
            if hasattr(explainer, "shap_interaction_values"):
                iv = explainer.shap_interaction_values(X_sample)
                np.save(self.ext_dir / "shap_interactions.npy", iv)
                logger.info("SHAP interaction values saved")
                return str(self.ext_dir / "shap_interactions.npy")
        except Exception as e:
            logger.warning(f"SHAP interactions failed: {e}")
        return None

    def _save_shap_summary_plots(self, shap_values, X_plot: pd.DataFrame,
                                 class_names: Optional[Sequence[str]] = None):
        results = {}
        vals = np.array(shap_values.values)
        try:
            if vals.ndim == 3:
                for ci in range(vals.shape[2]):
                    cls_name = class_names[ci] if class_names and ci < len(class_names) else f"class_{ci}"
                    sv_cls = shap_values[..., ci]
                    for ptype in ["dot", "bar"]:
                        fig = plt.figure()
                        shap.summary_plot(sv_cls, X_plot, show=False, plot_type=ptype)
                        p = self.output_dir / f"shap_{ptype}_{cls_name}.png"
                        self._save_fig(fig, p)
                        results[f"shap_{ptype}_{cls_name}"] = str(p)
            else:
                for ptype in ["dot", "bar"]:
                    fig = plt.figure()
                    shap.summary_plot(shap_values, X_plot, show=False, plot_type=ptype)
                    p = self.output_dir / f"shap_{ptype}.png"
                    self._save_fig(fig, p)
                    results[f"shap_{ptype}"] = str(p)
        except Exception as e:
            logger.warning(f"SHAP summary plots failed: {e}")
        return results

    def compute_global_importance(self, vals: np.ndarray, feature_names: Sequence[str]):
        global_imp = np.abs(vals).mean(axis=(0, 2)) if vals.ndim == 3 else np.abs(vals).mean(axis=0)
        df = (pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": global_imp})
              .sort_values("Mean |SHAP|", ascending=False)
              .reset_index(drop=True))
        p_csv = self.ext_dir / "global_shap_importance.csv"
        df.to_csv(p_csv, index=False)
        try:
            fig = plt.figure(figsize=(8, 6))
            topn = df.head(15)
            plt.barh(topn["Feature"][::-1], topn["Mean |SHAP|"][::-1])
            plt.xlabel("Mean |SHAP|")
            plt.title("Global Feature Importance (Top 15)")
            p_img = self.ext_dir / "global_shap_importance.png"
            self._save_fig(fig, p_img)
        except Exception:
            pass
        return str(p_csv), df

    def top10_per_class(self, vals: np.ndarray, X_sample: pd.DataFrame, feature_names: Sequence[str]):
        results = {}
        if vals.ndim == 3:
            for ci in range(vals.shape[2]):
                vals_ci = vals[..., ci]
                mean_abs = np.abs(vals_ci).mean(axis=0)
                top_idx = np.argsort(mean_abs)[::-1][:self.top_k]
                df_top = pd.DataFrame({
                    "Feature": np.array(feature_names)[top_idx],
                    "Mean |SHAP|": mean_abs[top_idx]
                })
                p = self.ext_dir / f"top{self.top_k}_class{ci}.csv"
                df_top.to_csv(p, index=False)
                results[f"class_{ci}"] = (str(p), df_top)
        else:
            mean_abs = np.abs(vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:self.top_k]
            df_top = pd.DataFrame({
                "Feature": np.array(feature_names)[top_idx],
                "Mean |SHAP|": mean_abs[top_idx]
            })
            p = self.ext_dir / f"top{self.top_k}_all.csv"
            df_top.to_csv(p, index=False)
            results["all"] = (str(p), df_top)
        return results

    def top10_common_features(self, per_class_results: Dict[str, Any]):
        from collections import defaultdict
        lists = []
        for v in per_class_results.values():
            if v:
                lists.append(list(v[1]["Feature"]))
        counts, ranks = defaultdict(int), defaultdict(list)
        for lst in lists:
            for i, f in enumerate(lst, 1):
                counts[f] += 1
                ranks[f].append(i)
        rows = [{
            "Feature": f,
            "Count": cnt,
            "AvgRank": float(np.mean(ranks[f]))
        } for f, cnt in counts.items()]
        df_common = (pd.DataFrame(rows)
                     .sort_values(["Count", "AvgRank"], ascending=[False, True])
                     .reset_index(drop=True)) if rows else pd.DataFrame(
            columns=["Feature", "Count", "AvgRank"])
        p = self.ext_dir / f"top{self.top_k}_common.csv"
        df_common.to_csv(p, index=False)
        logger.info(f"Top-{self.top_k} common features saved to {p}")
        return str(p), df_common

    def save_waterfall(self, shap_values, which_sample=0):
        try:
            sv = (shap_values[which_sample]
                  if np.array(shap_values.values).ndim == 2
                  else shap_values[which_sample, :, 0])
            shap.plots.waterfall(sv, show=False)
            p = self.ext_dir / "shap_waterfall_sample0.png"
            plt.savefig(p, dpi=300)
            plt.close()
            return str(p)
        except Exception:
            return None

    def save_waterfall_per_class(self, shap_values, which_sample=0):
        try:
            vals = np.array(shap_values.values)
            if vals.ndim == 3:
                for ci in range(vals.shape[2]):
                    sv = shap_values[which_sample, :, ci]
                    shap.plots.waterfall(sv, show=False)
                    p = self.ext_dir / f"shap_waterfall_sample{which_sample}_class{ci}.png"
                    plt.savefig(p, dpi=300)
                    plt.close()
        except Exception as e:
            logger.warning(f"Per-class waterfall failed: {e}")

    def dependence_plot(self, shap_values, vals, X_sample, global_df):
        try:
            top_feat = global_df["Feature"].iloc[0]
            shap.dependence_plot(
                top_feat,
                vals[..., 0] if vals.ndim == 3 else vals,
                X_sample,
                show=False
            )
            p = self.ext_dir / f"shap_dependence_{top_feat}.png"
            plt.savefig(p, dpi=300)
            plt.close()
            return str(p)
        except Exception:
            return None

    def feature_correlations(self, vals, X_sample):
        try:
            corrs = []
            for i, col in enumerate(X_sample.columns):
                vcol = vals[..., i].mean(axis=1) if vals.ndim == 3 else vals[:, i]
                corrs.append(
                    np.corrcoef(X_sample[col].values[:len(vcol)], np.abs(vcol))[0, 1]
                    if len(vcol) > 1 else np.nan
                )
            df = pd.DataFrame({
                "Feature": X_sample.columns,
                "SHAP-Value Correlation": corrs
            })
            p = self.ext_dir / "shap_feature_corr.csv"
            df.to_csv(p, index=False)
            return str(p), df
        except Exception:
            return None, None

    def shap_clustering(self, vals, X_sample, y_test):
        try:
            mat = np.abs(vals).mean(axis=2) if vals.ndim == 3 else np.abs(vals)
            if mat.shape[1] >= 2 and mat.shape[0] > 2:
                n = min(3, mat.shape[0])
                km = KMeans(n_clusters=n, random_state=self.random_state)
                cl = km.fit_predict(mat)
                score = silhouette_score(mat, cl) if mat.shape[0] > 2 else np.nan
                pd.DataFrame({
                    "SampleIndex": X_sample.index[:mat.shape[0]],
                    "Cluster": cl
                }).to_csv(self.ext_dir / "shap_clusters.csv", index=False)
                fig = plt.figure()
                plt.scatter(mat[:, 0], mat[:, 1], c=cl, alpha=0.7)
                plt.xlabel("SHAP dim 0")
                plt.ylabel("SHAP dim 1")
                plt.title(f"SHAP clustering (Silhouette={score:.3f})")
                p_png = self.ext_dir / "shap_clusters.png"
                self._save_fig(fig, p_png)
                return str(self.ext_dir / "shap_clusters.csv"), str(p_png)
        except Exception:
            return None, None

    def shap_3d_pca(self, vals, X_sample, y_test):
        try:
            emb = np.abs(vals).mean(axis=1)
            if emb.shape[1] >= 3 and emb.shape[0] >= 3:
                pca = PCA(n_components=3, random_state=self.random_state)
                emb3d = pca.fit_transform(emb)
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(
                    emb3d[:, 0], emb3d[:, 1], emb3d[:, 2],
                    c=pd.Series(y_test).iloc[:len(emb3d)], alpha=0.8
                )
                ax.set_title("3D PCA of SHAP embeddings")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                plt.colorbar(sc, ax=ax, label="Class")
                p = self.ext_dir / "shap_3d_pca.png"
                self._save_fig(fig, p)
                return str(p)
        except Exception:
            return None

    def shap_stability(self, X_train, y_train, runs=3):
        counts = {}
        for seed in range(runs):
            Xr, yr = resample(X_train, y_train, random_state=seed)
            bg = Xr.sample(min(50, len(Xr)), random_state=seed)
            try:
                pred_fn = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
                expl = shap.Explainer(pred_fn, bg)
                sv = expl(Xr.sample(min(30, len(Xr)), random_state=seed))
                v = np.array(sv.values)
                mean_abs = (np.abs(v).mean(axis=(0, 2))
                            if v.ndim == 3 else np.abs(v).mean(axis=0))
                top_feats = np.array(X_train.columns)[np.argsort(mean_abs)[::-1][:self.top_k]]
                for f in top_feats:
                    counts[f] = counts.get(f, 0) + 1
            except Exception:
                continue
        df = pd.DataFrame(list(counts.items()), columns=["Feature", "Count"])
        if not df.empty:
            df["Stability (%)"] = (df["Count"] / runs) * 100
            p = self.ext_dir / "shap_feature_stability.csv"
            df.to_csv(p, index=False)
            return str(p), df
        return None, None

    def shap_vs_model_importance(self, global_df: pd.DataFrame):
        try:
            if hasattr(self.model, "feature_importances_"):
                model_imp = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                model_imp = np.abs(self.model.coef_).mean(axis=0)
            else:
                return None
            df = global_df.copy()
            df["ModelImportance"] = model_imp[:len(df)]
            df["Rank_SHAP"] = df["Mean |SHAP|"].rank(ascending=False)
            df["Rank_Model"] = df["ModelImportance"].rank(ascending=False)
            df["RankDiff"] = df["Rank_SHAP"] - df["Rank_Model"]
            p = self.ext_dir / "shap_vs_model_importance.csv"
            df.to_csv(p, index=False)
            return str(p)
        except Exception as e:
            logger.warning(f"SHAP vs model importance failed: {e}")
            return None

    #### LIME METHODS ####
    def compute_lime_values(self, X_train, X_test_sample, y_train):
        """
        Compute LIME explanations for all test samples.
        Returns: dict with per-sample HTMLs, top-k CSVs, global importance CSV.
        """
        if not _HAS_LIME:
            return None
        results = {}
        try:
            expl = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=list(X_train.columns),
                class_names=[str(c) for c in np.unique(y_train)],
                mode="classification"
            )
            lime_vals = []
            for i in range(len(X_test_sample)):
                exp = expl.explain_instance(
                    X_test_sample.iloc[i].values,
                    self.model.predict_proba
                )
                lime_vals.append(exp)
                html_path = self.ext_dir / f"lime_sample{i}.html"
                exp.save_to_file(str(html_path))
                results[f"lime_sample{i}"] = str(html_path)
            # aggregate global importance from mean absolute weights
            global_dict = {}
            for exp in lime_vals:
                for f, w in exp.as_list():
                    global_dict[f] = global_dict.get(f, []) + [abs(w)]
            rows = [{"Feature": f, "Mean |LIME|": np.mean(v)} for f, v in global_dict.items()]
            df_global = pd.DataFrame(rows).sort_values("Mean |LIME|", ascending=False)
            p_global = self.ext_dir / "global_lime_importance.csv"
            df_global.to_csv(p_global, index=False)
            results["global_lime_csv"] = str(p_global)
            results["_lime_vals_list"] = lime_vals  # store Lime objects for waterfall
            return results
        except Exception as e:
            logger.warning(f"LIME failed: {e}")
            return None

    def lime_waterfall(self, lime_vals):
        """
        Create waterfall-like bar plots for LIME explanations.
        lime_vals: list of LimeTabularExplainer objects (from compute_lime_values)
        """
        results = {}
        try:
            for i, exp in enumerate(lime_vals):
                feat, val = zip(*exp.as_list())
                val = np.array(val)
                feat = np.array(feat)
                # Sort by absolute contribution
                idx = np.argsort(np.abs(val))[::-1]
                feat, val = feat[idx], val[idx]
                fig, ax = plt.subplots(figsize=(8, max(4, len(feat)*0.3)))
                ax.barh(feat, val, color=["green" if v>0 else "red" for v in val])
                ax.set_xlabel("LIME Weight")
                ax.set_title(f"LIME Waterfall Sample {i}")
                plt.tight_layout()
                p = self.ext_dir / f"lime_waterfall_sample{i}.png"
                fig.savefig(p, dpi=300)
                plt.close(fig)
                results[f"lime_waterfall_sample{i}"] = str(p)
            return results
        except Exception as e:
            logger.warning(f"LIME waterfall failed: {e}")
            return None

    #### RUN ####
    def run(self, X_train, X_test, y_train, y_test,
            feature_names: Sequence[str], max_test_sample: int = 50):
        X_train = self._to_df(X_train, feature_names)
        X_test = self._to_df(X_test, feature_names)
        X_bg = X_train.sample(min(self.max_samples_shap, len(X_train)), random_state=self.random_state)
        X_test_sample = X_test.sample(min(max_test_sample, len(X_test)), random_state=self.random_state)
        results = {}
        try:
            expl = self._choose_shap_explainer(X_bg)
            sv, vals = self.compute_shap_values(expl, X_test_sample)
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            return {"error": str(e)}
        self.save_raw_shap(vals, X_test_sample)
        self.shap_interactions(expl, X_test_sample)
        class_names = list(getattr(self.model, "classes_", [])) if hasattr(self.model, "classes_") else None
        results.update(self._save_shap_summary_plots(sv, X_test_sample, class_names))
        p_csv, global_df = self.compute_global_importance(vals, feature_names)
        results["global_shap_csv"] = p_csv
        per_class = self.top10_per_class(vals, X_test_sample, feature_names)
        for k, v in per_class.items():
            results[f"top10_{k}"] = v[0]
        p_common, _ = self.top10_common_features(per_class)
        results["top10_common_features"] = p_common
        wf = self.save_waterfall(sv)
        if wf:
            results["waterfall_sample0"] = wf
        self.save_waterfall_per_class(sv)
        dep = self.dependence_plot(sv, vals, X_test_sample, global_df)
        if dep:
            results["dependence_top_feature"] = dep
        p_corr, _ = self.feature_correlations(vals, X_test_sample)
        if p_corr:
            results["feature_correlation_csv"] = p_corr
        p_clust, p_clust_png = self.shap_clustering(vals, X_test_sample, y_test)
        if p_clust:
            results["shap_clusters_csv"] = p_clust
        if p_clust_png:
            results["shap_clusters_png"] = p_clust_png
        p_3d = self.shap_3d_pca(vals, X_test_sample, y_test)
        if p_3d:
            results["shap_3d_pca"] = p_3d
        p_stab, _ = self.shap_stability(X_train, y_train)
        if p_stab:
            results["shap_feature_stability"] = p_stab
        p_cmp = self.shap_vs_model_importance(global_df)
        if p_cmp:
            results["shap_vs_model_importance"] = p_cmp

        # LIME run
        lime_results = self.compute_lime_values(X_train, X_test_sample, y_train)
        if lime_results:
            results.update(lime_results)
            # LIME waterfall plots
            lime_vals_list = lime_results.get("_lime_vals_list", None)
            if lime_vals_list:
                wf_lime = self.lime_waterfall(lime_vals_list)
                if wf_lime:
                    results.update(wf_lime)

        results["output_dir"] = str(self.output_dir)
        logger.info(f"Explainability complete. Outputs in {self.output_dir}")
        return results


def run_explainability(*, model, X_train, X_test, y_train, y_test,
                       feature_names, output_dir: Path = Path("results/explainability"),
                       fold_name: str = "fold"):
    exp = Explainability(model=model, output_dir=output_dir, fold_name=fold_name)
    return exp.run(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names
    )


if __name__ == "__main__":
    print("Explainability module ready — call run_explainability() with your data.")

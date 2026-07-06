import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from scipy import stats
import os, csv
from itertools import product
from functools import reduce
import tifffile
from pathlib import Path
from collections import OrderedDict
import json

import tifffile
from scipy.ndimage import center_of_mass, binary_erosion

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


class PlotHelper:
    """
    Helper for plotting MultiIndex dataframes with index:
        (condition, folder, repeat_id, frame)

    Provides:
      - tidy reset_index handling
      - frame filtering or frame binning
      - per-repeat averaging over frames
      - folder marker mapping
      - subplot scaffolding
      - bar+dot summary plots
      - time-trace plots

    NOTE: All time intervals are inclusive of lower and upper bounds
    """

    def __init__(self, 
                 analysis,
                 palette=None,
                 plot_t0 = 0.0, # in mins
                 time_per_frame = 1.0, # in mins/frame
                 save_outputs=False,
                 out_dir="plot_exports",
                 fig_format="svg",
                 dpi=300,
                 fig_width=2.0,
                 fig_height=1.3,
                 use_mpl_defaults=True
                ):
        
        self.analysis = analysis
        self.palette = palette

        # Master toggle
        self.save_outputs = save_outputs

        # Output folder
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

        # Figure export options
        self.fig_format = fig_format
        self.dpi = dpi
        self.fig_width = fig_width
        self.fig_height = fig_height

        # Apply default matplotlib style
        if use_mpl_defaults:
            self.set_mpl_defaults()

        # Store frame-to-time conversion params
        self.dt = 1.0              # time per frame
        self.time_unit = "frames"  # label
        self.frame_zero = 0        # reference frame
        self.t_zero = 0.0          # displayed time at frame_zero
        self.time_col = "time"     # derived column name

    def set_time_axis(
        self,
        dt=1.0,
        unit="frames",
        frame_zero=0,
        t_zero=0.0,
    ):
        """
        Define conversion from frame → displayed time.
    
        time = (frame - frame_zero)*dt + t_zero
        """
    
        self.dt = dt
        self.time_unit = unit
        self.frame_zero = frame_zero
        self.t_zero = t_zero

    def print_time(self):
        unrolled_df = self.analysis.df.reset_index()
        df_with_time = self.add_time_column(unrolled_df)
        return df_with_time
    
    def add_time_column(self, df, frame_col="frame"):
        """
        Add a derived time column for plotting only.
        Filtering/aggregation still uses integer frames.
        """
    
        df = df.copy()
    
        df[self.time_col] = (
            (df[frame_col] - self.frame_zero) * self.dt + self.t_zero
        )
    
        return df
        
    
    def set_mpl_defaults(self):
        """
        Apply consistent matplotlib rcParams for publication-style figures.
        """

        # Resolution + axis/tick colors
        mpl.rcParams["figure.dpi"] = 300
        mpl.rcParams["axes.edgecolor"] = "#424242"
        mpl.rcParams["xtick.color"] = "#424242"
        mpl.rcParams["ytick.color"] = "#424242"

        mpl.rcParams["svg.fonttype"] = "none"
        # mpl.rcParams["font.family"] = ["sans-serif"]
        # plt.rcParams["font.sans-serif"] = ["Arial"]

        # Font sizes
        SMALL_SIZE = 5
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        mpl.rc("font", size=SMALL_SIZE)
        mpl.rc("axes", titlesize=SMALL_SIZE)
        mpl.rc("axes", labelsize=SMALL_SIZE)
        mpl.rc("xtick", labelsize=SMALL_SIZE)
        mpl.rc("ytick", labelsize=SMALL_SIZE)
        mpl.rc("legend", fontsize=SMALL_SIZE)
        mpl.rc("figure", titlesize=SMALL_SIZE)

        plt.rcParams["xtick.major.size"] =  2.5
        plt.rcParams["xtick.minor.size"] =  1.5
        plt.rcParams["ytick.major.size"] =  2.5
        plt.rcParams["ytick.minor.size"] =  1.5
        plt.rcParams["xtick.major.pad"] = 0.5
        plt.rcParams["ytick.major.pad"] = 0.5

    def condition_palette(self):
        """Main color per condition (traces + errorbars)."""
        return {
            cond: colors["main"]
            for cond, colors in self.palette.items()
        }

    def bin_palette(self, frame_bins=("early", "late")):
        """
        Return a palette for frame_bin hue plots.
    
        Example output:
          {"ctrl_early": ..., "ctrl_late": ..., ...}
        """
        out = {}
    
        for cond, colors in self.palette.items():
            for b in frame_bins:
                out[f"{cond}_{b}"] = colors[b]
    
        return out
        
    
    def _make_stem(self, plot_name, var_tag, file_stem=None):
        """
        Create consistent base filename for all exported files.
    
        If file_stem is provided, use it exactly.
        Otherwise generate a timestamped name.
        """
    
        if file_stem is not None:
            return file_stem
    
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return f"{plot_name}_{var_tag}_{stamp}"

    def save_plot_bundle(
        self,
        fig,
        plot_name,
        var_tag,
        points_df,
        summary_df=None,
        params=None,
        file_stem=None,
    ):
        """
        Save everything needed to reproduce a plot:
    
          - SVG figure
          - raw datapoints CSV
          - summary statistics CSV
          - JSON metadata (plot parameters)
    
        Parameters
        ----------
        fig : matplotlib Figure
        plot_name : str
        var_tag : str
            Variable name or joined variable list
        points_df : DataFrame
            Raw points shown in plot
        summary_df : DataFrame or None
            Bar means, SD, SEM, etc.
        params : dict or None
            Plot settings (conditions, frame_range, bins...)
        file_stem : str or None
            Custom filename stem (no extension)
        """
    
        if not self.save_outputs:
            return
    
        stem = self._make_stem(plot_name, var_tag, file_stem=file_stem)
    
        # ----------------------------
        # Save figure
        # ----------------------------
        fig_path = self.out_dir / f"{stem}.{self.fig_format}"
        fig.savefig(fig_path, bbox_inches="tight", dpi=self.dpi)
    
        # ----------------------------
        # Save raw datapoints
        # ----------------------------
        points_path = self.out_dir / f"{stem}_points.csv"
        points_df.to_csv(points_path, index=False)
    
        # ----------------------------
        # Save summary statistics
        # ----------------------------
        if summary_df is not None:
            summary_path = self.out_dir / f"{stem}_summary.csv"
            summary_df.to_csv(summary_path, index=False)
    
        # ----------------------------
        # Save plot metadata
        # ----------------------------
        if params is not None:
            params_path = self.out_dir / f"{stem}_params.json"
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2)
    
        print(f"[saved figure] {fig_path}")
        print(f"[saved points] {points_path}")
        if summary_df is not None:
            print(f"[saved summary] {summary_path}")
        if params is not None:
            print(f"[saved params] {params_path}")

    def compute_bar_summary(self, points, group_cols):
        """
        Compute mean ± SD ± SEM across plotted points.
        """
    
        summary = (
            points.groupby(group_cols)["value_mean"]
                  .agg(mean="mean", sd="std", n="count")
                  .reset_index()
        )
    
        summary["sem"] = summary["sd"] / (summary["n"] ** 0.5)
    
        return summary
    

    # -----------------------------
    # Core data prep
    # -----------------------------
    def get_df_for_var(self, var_name):
        """
        Return the appropriate dataframe.
    
        If var_name starts with 'norm_', use analysis.norm_df.
        Otherwise use raw df.
        """
    
        if var_name.startswith("norm_"):
    
            if self.analysis.norm_df is None:
                raise ValueError("norm_df not built. Run analysis.build_norm_df().")
    
            return self.analysis.norm_df.copy(), var_name
    
        else:
            return self.analysis.df.reset_index().copy(), var_name

    def tidy(self, use_norm=False):
        """
        Return tidy dataframe for plotting.
    
        If use_norm=True:
          use analysis.norm_df if available.
        """
    
        if use_norm:
    
            if self.analysis.norm_df is None:
                raise ValueError(
                    "No norm_df found. Run analysis.build_norm_df() first."
                )
    
            return self.analysis.norm_df.copy()
    
        return self.analysis.df.reset_index().copy()


    def events_dict_to_df(self, events_dict):
    
        rows = []
    
        for condition, cond_data in events_dict.items():
            for folder, folder_data in cond_data.items():
                for repeat_id, repeat_data in folder_data.items():
                    for track_id, event_info in repeat_data.items():
    
                        rows.append({
                            "condition": condition,
                            "folder": folder,
                            "repeat_id": int(repeat_id),
                            "track_id": int(track_id),
                            "t0": int(event_info["t0"]),
                            "t_end": int(event_info["t_end"]),
                        })
    
        events_df = pd.DataFrame(rows)
    
        return events_df

    def unit_cols(self, df=None):
        """
        Columns defining one independent trace unit.
        Robust to norm_df (no object_id).
        """
    
        if df is None:
            df = self.analysis.df.reset_index()
    
        if self.analysis.mode == "trackmate":
            return ["condition", "folder", "repeat_id", "object_id"]
    
        if self.analysis.mode == "cellpose":
    
            # Raw object-level
            if "object_id" in df.columns:
                return ["condition", "folder", "repeat_id", "object_id"]
    
            # Norm/FOV-level
            return ["condition", "folder", "repeat_id"]
    
        if self.analysis.mode == "whole-fov":
            return ["condition", "folder", "repeat_id"]
    
        raise ValueError("Unknown mode")


    
    def build_points(
        self,
        var_name,
        conditions=None,
        frame_range=None,
        frame_bins=None,
        cellpose_unit="object",
    ):
        """
        Returns one row per experimental unit with value_mean.
    
        Supports:
          - raw vars: "FR_ratio"
          - normalized vars: "norm_FR_ratio"
    
        Cellpose:
          - raw + cellpose_unit="fov" collapses objects per frame
          - norm variables already live in collapsed norm_df
        """
    
        # ------------------------------------------------------------
        # Resolve correct dataframe + column name
        # ------------------------------------------------------------
        df, ycol = self.get_df_for_var(var_name)
    
        # ------------------------------------------------------------
        # Condition filtering
        # ------------------------------------------------------------
        if conditions is not None and len(conditions) > 0:
            df = df[df["condition"].isin(conditions)]
    
        # ------------------------------------------------------------
        # Frame filtering
        # ------------------------------------------------------------
        if frame_range is not None:
            t0, t1 = frame_range
            df = df[df["frame"].between(t0, t1)]
    
        # ------------------------------------------------------------
        # Frame binning (early/late windows)
        # ------------------------------------------------------------
        if frame_bins is not None:
            df["frame_bin"] = None
            for name, (lo, hi) in frame_bins.items():
                df.loc[df["frame"].between(lo, hi), "frame_bin"] = name
            df = df.dropna(subset=["frame_bin"])
            df["cond_bin"] = df["condition"] + "_" + df["frame_bin"]
            group_cols = ["condition", "frame_bin", "folder", "repeat_id"]
        else:
            group_cols = ["condition", "folder", "repeat_id"]
    
        # ------------------------------------------------------------
        # Special case: Cellpose + FOV aggregation
        # ------------------------------------------------------------
        if self.analysis.mode == "cellpose" and cellpose_unit == "fov":
    
            # Case A: raw df still has object_id → collapse objects
            if "object_id" in df.columns:
    
                # Collapse objects → per-frame mean + count cells
                per_frame = (
                    df.groupby(group_cols + ["frame"], as_index=False)
                      .agg(
                          value_mean=(ycol, "mean"),
                          n_cells=("object_id", "nunique"),
                      )
                )
    
                # Collapse frames → replicate-level point
                points = (
                    per_frame.groupby(group_cols, as_index=False)
                             .agg(
                                 value_mean=("value_mean", "mean"),
                                 total_cells=("n_cells", "sum"),
                                 n_frames=("frame", "nunique"),
                             )
                )
    
                return points
    
            # Case B: norm_df already collapsed → no object_id exists
            else:
                points = (
                    df.groupby(group_cols, as_index=False)[ycol]
                      .mean()
                      .rename(columns={ycol: "value_mean"})
                )
                return points

        # ------------------------------------------------------------
        # Default case: object-level (trackmate/cellpose)
        # ------------------------------------------------------------
        if self.analysis.mode in ["trackmate", "cellpose"]:
            if "object_id" in df.columns:
                group_cols = group_cols + ["object_id"]
        
                # In cellpose, object_id is NOT persistent across frames, i.e. cells are not tracked.
                # Each (frame, object_id) is an independent cell — don't average across frames.
                if self.analysis.mode == "cellpose":
                    group_cols = group_cols + ["frame"]
        # ------------------------------------------------------------
        # Per-unit mean across frames
        # ------------------------------------------------------------
        points = (
            df.groupby(group_cols, as_index=False)[ycol]
              .mean()
              .rename(columns={ycol: "value_mean"})
        )
    
        return points

    
    # -----------------------------
    # Plotting utilities
    # -----------------------------
    def make_axes(self, var_names, figsize=None):
        """Create consistent subplot layout."""
    
        var_names = [var_names] if isinstance(var_names, str) else var_names
    
        if figsize is None:
            figsize = (5 * len(var_names), 4)
        else:
            figsize = (figsize[0] * len(var_names), figsize[1])
    
        fig, axes = plt.subplots(
            ncols=len(var_names),
            figsize=figsize,
            sharex=True
        )
    
        axes = np.atleast_1d(axes)
        return fig, axes, var_names

    def folder_markers(self, folders):
        """Assign markers to folders."""
        marker_list = ["o", "s", "D", "h", "v", "^", "<", ">", "p"]
        return {
            f: marker_list[i % len(marker_list)]
            for i, f in enumerate(folders)
        }

    def get_style_for_var(self, style, var_name):
        if style is None:
            return None
        if var_name in style:
            return style[var_name]
        return style

    def apply_axes_style(self, ax, style=None):
        """
        Apply axis limits, ticks, labels, etc. in a unified way.
    
        style dict may contain:
          xlim, ylim
          xticks, yticks
          xminorticks, yminorticks
          xlabel, ylabel
          tick_params
        """
    
        if style is None:
            return
    
        # Limits
        if "xlim" in style:
            ax.set_xlim(style["xlim"])
    
        if "ylim" in style:
            ax.set_ylim(style["ylim"])
    
        # Tick locations
        if "xticks" in style:
            ax.xaxis.set_major_locator(ticker.FixedLocator(style['xticks']))
            # ax.set_xticks(style["xticks"])
    
        if "yticks" in style:
            ax.yaxis.set_major_locator(ticker.FixedLocator(style['yticks']))
            # ax.set_yticks(style["yticks"])
    
        # Minor ticks
        if "xminorticks" in style:
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(style['xminorticks']))
            # ax.set_xticklabels(style["xticklabels"])
    
        if "yminorticks" in style:
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(style['yminorticks']))
            # ax.set_yticklabels(style["yticklabels"])
    
        # Axis labels override
        if "xlabel" in style:
            ax.set_xlabel(style["xlabel"])
    
        if "ylabel" in style:
            ax.set_ylabel(style["ylabel"])
    
        # Tick appearance
        if "tick_params" in style:
            ax.tick_params(**style["tick_params"])


    # -----------------------------
    # Main plot types
    # -----------------------------
    def barplot(
        self,
        var_names,
        conditions=None,
        frame_range=(0, 60),
        show_folder_legend=False,
        cellpose_unit="fov",
        file_stem=None,
        orientation="vertical",
        style=None, 
        figsize=None
    ):
        """
        Bars = mean ± SD across repeats
        Dots = per-repeat values
        Marker shape = folder
        Color = condition
        """
        allpoints = {}
        fig, axes, var_names = self.make_axes(var_names, figsize=figsize)

        for ax, var_name in zip(axes, var_names):

            ax_style = self.get_style_for_var(style, var_name)
            self.apply_axes_style(ax, ax_style)

            points = self.build_points(
                var_name,
                conditions=conditions,
                frame_range=frame_range,
                cellpose_unit=cellpose_unit
            )
            allpoints[var_name]=points
            
            summary = self.compute_bar_summary(
                points,
                group_cols=["condition"]
            )

            markers = self.folder_markers(points["folder"].unique())

            cat_axis = "condition"
            val_axis = "value_mean"

            if orientation == "vertical":
                x = cat_axis
                y = val_axis
            else:
                x = val_axis
                y = cat_axis

            order = conditions if conditions is not None else None
            
            # Bars
            sns.barplot(
                ax=ax,
                data=points,
                x=x,
                y=y,
                hue="condition",
                order=order,
                palette=self.condition_palette(),
                errorbar="sd",
                alpha=0.4,
                err_kws={'linewidth': 1.0},
                legend=False,
                capsize=0.6
            )

            # Folder-coded dots
            for folder, m in markers.items():
                sns.stripplot(
                    ax=ax,
                    data=points[points["folder"] == folder],
                    x=x,
                    y=y,
                    hue="condition",
                    order=order,
                    palette=self.condition_palette(),
                    marker=m,
                    jitter=True,
                    size=3,
                    linewidth=0.5,
                    alpha=0.8,
                    legend=False
                )

            if orientation == "vertical":
                ax.set_ylabel(var_name)
                ax.set_xlabel("Condition")
            else:
                ax.set_ylabel("Condition")
                ax.set_xlabel(var_name)

            sns.despine(ax=ax)

            # --- EXPORT ---
            self.save_plot_bundle(
                fig=fig,
                plot_name="barplot",
                var_tag=var_name,
                points_df=points,
                summary_df=summary,
                params={
                    "conditions": conditions,
                    "frame_range": frame_range,
                    "cellpose_unit": cellpose_unit,
                },
                file_stem=file_stem,
            )

        # Folder legend
        if show_folder_legend:
            folder_list = sorted(points["folder"].unique())
            markers = self.folder_markers(folder_list)
            
            folder_handles = [
                mlines.Line2D(
                    [], [], color="black",
                    marker=markers[f],
                    linestyle="None",
                    label=str(f)
                )
                for f in folder_list
            ]

            axes[-1].legend(
                handles=folder_handles,
                title="Folder",
                bbox_to_anchor=(1.05, 1),
                loc="upper left"
            )

        plt.tight_layout()
        plt.show()
        return allpoints  

    def delta_barplot(
        self,
        var_names,
        conditions=None,
        frame_bins=None,
        show_folder_legend=False,
        cellpose_unit="fov",
        file_stem=None,
        orientation="vertical",
        style=None,
        figsize=None,
        normalize_to_pre=False
    ):
        """
        Bars = mean ± SD of Δ (late − early)
        Dots = per-repeat Δ values
        Marker shape = folder
        Color = condition
    
        Structurally identical to barplot(), except that
        value_mean is computed as (late − early).
    
        Parameters
        ----------
        normalize_to_pre : bool, optional
            If True, the delta is normalized to the pre (early) condition,
            i.e. value_mean = (late − early) / early.
            Default is False.
        """
    
        if frame_bins is None:
            frame_bins = {"early": (0, 5), "late": (50, 55)}
    
        assert "early" in frame_bins and "late" in frame_bins, \
            "delta_barplot requires 'early' and 'late' frame bins"
    
        allpoints = {}
        fig, axes, var_names = self.make_axes(var_names, figsize=figsize)
    
        for ax, var_name in zip(axes, var_names):
    
            # -----------------------------
            # Styling (identical)
            # -----------------------------
            ax_style = self.get_style_for_var(style, var_name)
            self.apply_axes_style(ax, ax_style)
    
            # -----------------------------
            # Build early/late points
            # -----------------------------
            raw = self.build_points(
                var_name,
                conditions=conditions,
                frame_bins=frame_bins,
                cellpose_unit=cellpose_unit
            )
    
            # -----------------------------
            # Construct Δ table
            # -----------------------------
            idx_cols = ["condition", "folder", "repeat_id"]
    
            early = (
                raw[raw["frame_bin"] == "early"]
                .set_index(idx_cols)["value_mean"]
                .rename("early")
            )
    
            late = (
                raw[raw["frame_bin"] == "late"]
                .set_index(idx_cols)["value_mean"]
                .rename("late")
            )
    
            points = (
                pd.concat([early, late], axis=1)
                .dropna()
                .reset_index()
            )
    
            # -----------------------------
            # Compute delta, with optional
            # normalization to pre (early)
            # -----------------------------
            if normalize_to_pre:
                points["value_mean"] = (points["late"] - points["early"]) / points["early"]
            else:
                points["value_mean"] = points["late"] - points["early"]
    
            allpoints[var_name] = points
    
            # -----------------------------
            # Summary (unchanged helper)
            # -----------------------------
            summary = self.compute_bar_summary(
                points,
                group_cols=["condition"]
            )
    
            markers = self.folder_markers(points["folder"].unique())
    
            cat_axis = "condition"
            val_axis = "value_mean"
    
            if orientation == "vertical":
                x, y = cat_axis, val_axis
            else:
                x, y = val_axis, cat_axis
    
            order = conditions if conditions is not None else None
    
            # -----------------------------
            # Bars (identical to barplot)
            # -----------------------------
            sns.barplot(
                ax=ax,
                data=points,
                x=x,
                y=y,
                hue="condition",
                order=order,
                palette=self.condition_palette(),
                errorbar="sd",
                alpha=0.5,
                err_kws={"linewidth": 1.0},
                legend=False,
                capsize=0.6
            )
    
            # -----------------------------
            # Folder-coded dots (identical)
            # -----------------------------
            for folder, m in markers.items():
                sns.stripplot(
                    ax=ax,
                    data=points[points["folder"] == folder],
                    x=x,
                    y=y,
                    hue="condition",
                    order=order,
                    palette=self.condition_palette(),
                    marker=m,
                    jitter=True,
                    size=3,
                    linewidth=0.5,
                    alpha=0.8,
                    legend=False
                )
    
            # -----------------------------
            # Labels + zero line
            # -----------------------------
            if normalize_to_pre:
                delta_label = f"Δ {var_name} / pre (late − early) / early"
            else:
                delta_label = f"Δ {var_name} (late − early)"
    
            if orientation == "vertical":
                ax.set_ylabel(delta_label)
                ax.set_xlabel("Condition")
                ax.axhline(0, color="#666666", linestyle="--", linewidth=1.0)
            else:
                ax.set_ylabel("Condition")
                ax.set_xlabel(delta_label)
                ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
    
            sns.despine(ax=ax)
    
            # -----------------------------
            # Export (same API)
            # -----------------------------
            self.save_plot_bundle(
                fig=fig,
                plot_name="delta_barplot",
                var_tag=var_name,
                points_df=points,
                summary_df=summary,
                params={
                    "conditions": conditions,
                    "frame_bins": frame_bins,
                    "cellpose_unit": cellpose_unit,
                    "normalize_to_pre": normalize_to_pre,
                },
                file_stem=file_stem,
            )
    
        # -----------------------------
        # Folder legend (unchanged)
        # -----------------------------
        if show_folder_legend:
            folder_list = sorted(points["folder"].unique())
            markers = self.folder_markers(folder_list)
    
            handles = [
                mlines.Line2D(
                    [], [], color="black",
                    marker=markers[f],
                    linestyle="None",
                    label=str(f)
                )
                for f in folder_list
            ]
    
            axes[-1].legend(
                handles=handles,
                title="Folder",
                bbox_to_anchor=(1.05, 1),
                loc="upper left"
            )
    
        plt.tight_layout()
        plt.show()
        return allpoints    

    def time_comp_barplot(
        self,
        var_names,
        conditions=None,
        frame_bins=None,
        cellpose_unit="fov",
        file_stem=None,
        style=None, 
        figsize=None
    ):

        if frame_bins is None:
            frame_bins = {"early": (0, 5), "late": (50, 55)}

        allpoints = {}
        fig, axes, var_names = self.make_axes(var_names, figsize=figsize)
    
        for ax, var_name in zip(axes, var_names):

            ax_style = self.get_style_for_var(style, var_name)
            self.apply_axes_style(ax, ax_style)
    
            points = self.build_points(
                var_name,
                conditions=conditions,
                frame_bins=frame_bins,
                cellpose_unit=cellpose_unit
            )
            points["cond_bin"] = points["condition"] + "_" + points["frame_bin"]
            
            summary = self.compute_bar_summary(
                points,
                group_cols=["condition", "frame_bin"]
            )
            
            allpoints[var_name]=points
    
            markers = self.folder_markers(points["folder"].unique())
            
            conditions_order = conditions if conditions is not None else list(points["condition"].unique())
            bins_order = list(frame_bins.keys())   # ["early","late"]
    
            # Bars
            sns.barplot(
                ax=ax,
                data=points,
                x="condition",
                y="value_mean",
                hue="frame_bin",
                order=conditions_order,
                hue_order=bins_order,
                # palette=self.bin_palette(frame_bins.keys()),
                palette={"early": "lightgray", "late": "darkgray"},  # temporary
                errorbar="sd",
                alpha=0.5,
                capsize=0.6,
                err_kws={'linewidth': 1.0},
                legend=False
            )
            
            # seaborn draws patches in nested order:
            # for each bin:
            #   for each condition:
            #       one bar
            i = 0
            for b in bins_order:                  # hue first
                for cond in conditions_order:     # then condition
                    patch = ax.patches[i]
                    patch.set_facecolor(self.palette[cond][b])
                    i += 1
    
            # --- Manual scatter dots (robust, correct colors) ---
            cond_to_x = {c: i for i, c in enumerate(conditions_order)}
            offsets = {"early": -0.2, "late": +0.2}
            
            # base x positions + dodge + jitter
            points = points.copy()
            points["xpos"] = points["condition"].map(cond_to_x)
            points["xpos"] += points["frame_bin"].map(offsets)
            points["xpos"] += np.random.uniform(-0.05, 0.05, size=len(points))
            
            for folder, m in markers.items():
                sub = points[points["folder"] == folder]
            
                for b in bins_order:
                    sub2 = sub[sub["frame_bin"] == b]
            
                    ax.scatter(
                        sub2["xpos"],
                        sub2["value_mean"],
                        marker=m,
                        s=15,
                        alpha=0.8,
                        color=[self.palette[c][b] for c in sub2["condition"]],
                        edgecolors="#424242",
                        linewidth=0.5,
                    )

            # ax.set_title(var_name)
            sns.despine(ax=ax)
            
            #removes extra padding from the comparative barplots
            ax.set_xlim(-0.5, len(conditions_order) - 0.5)

            self.save_plot_bundle(
                fig=fig,
                plot_name="timecomp",
                var_tag=var_name,
                points_df=points,
                summary_df=summary,
                params={
                    "conditions": conditions,
                    "frame_bins": frame_bins,
                    "cellpose_unit": cellpose_unit,
                },
                file_stem=file_stem,
            )
    
        plt.tight_layout()
        plt.show()
        return allpoints

    def histogram(
        self,
        var_names,
        conditions=None,
        frame_range=(0, 60),
        bins=30,
        density=False,
        element="step",
        fill=True,
        common_norm=False,
        alpha=0.5,
        cellpose_unit="fov",
        file_stem=None,
        kde=False,
        ccdf=False,            
        style=None,
        figsize=None,
        legend=False,
        log_axis=False,
        show_mean_median=False,
        show_mean_mode=False,
        normalize_mean=False,
        normalize_median=False,
        normalize_mode=False,
    ):
        allpoints = {}
        fig, axes, var_names = self.make_axes(var_names, figsize=figsize)
    
        for ax, var_name in zip(axes, var_names):
    
            ax_style = self.get_style_for_var(style, var_name)
            self.apply_axes_style(ax, ax_style)
    
            points = self.build_points(
                var_name,
                conditions=conditions,
                frame_range=frame_range,
                cellpose_unit=cellpose_unit
            )
    
            summary = (
                points
                .groupby("condition")["value_mean"]
                .agg(
                    n_cellpose_units="count",
                    mean="mean",
                    median="median",
                )
                .reset_index()
            )
    
            allpoints[var_name] = points
    
            order = (
                conditions
                if conditions is not None
                else list(points["condition"].unique())
            )
    
            # -----------------------------------
            # Plot: CCDF / KDE / Histogram
            # -----------------------------------
            if ccdf:
                sns.ecdfplot(
                    ax=ax,
                    data=points,
                    x="value_mean",
                    hue="condition",
                    hue_order=order,
                    palette=self.condition_palette(),
                    complementary=True,
                    legend=False,
                )
                ax.set_ylabel("P(X > x)")
                ax.set_xlabel(var_name)
    
                if log_axis:
                    ax.set_xscale('log')          # x for CCDF
                    ax.set_yscale('log')          # log y also useful for tail
                    ax.set_ylim(bottom=1e-3)      # avoid log(0)
    
            elif kde:
                plot_points = points.copy()
                if normalize_mean or normalize_median or normalize_mode:
                    if normalize_mode:
                        from scipy.stats import gaussian_kde
                        mode_map = {}
                        for cond, grp in plot_points.groupby("condition"):
                            vals = grp["value_mean"].dropna().values
                            kde  = gaussian_kde(vals)
                            xs   = np.linspace(vals.min(), vals.max(), 1000)
                            mode_map[cond] = xs[np.argmax(kde(xs))]
                        shift = plot_points["condition"].map(mode_map)
                    else:
                        shift = (
                            plot_points.groupby("condition")["value_mean"]
                            .transform("median" if normalize_median else "mean")
                        )
                    plot_points["value_mean"] = plot_points["value_mean"] - shift
                    summary = (
                        plot_points
                        .groupby("condition")["value_mean"]
                        .agg(
                            n_cellpose_units="count",
                            mean="mean",
                            median="median",
                        )
                        .reset_index()
                    )
                    
                sns.kdeplot(
                    ax=ax,
                    data=plot_points,
                    x="value_mean",
                    hue="condition",
                    hue_order=order,
                    palette=self.condition_palette(),
                    common_norm=common_norm,
                    fill=True,
                    alpha=0.3,
                    legend=False,
                )
                centre_label = " (mode-centred)" if normalize_mode else " (median-centred)" if normalize_median else " (mean-centred)" if normalize_mean else ""
                ax.set_xlabel(f"{var_name}{centre_label}")
                ax.set_ylabel("Density")
    
                if log_axis:
                    ax.set_xscale('log')          # <-- fixed: was yscale before
    
            else:
                sns.histplot(
                    ax=ax,
                    data=points,
                    x="value_mean",
                    hue="condition",
                    hue_order=order,
                    bins=bins,
                    palette=self.condition_palette(),
                    element=element,
                    fill=fill,
                    stat="density" if density else "count",
                    common_norm=common_norm,
                    alpha=alpha,
                    edgecolor=None,
                    legend=False,
                )
                ax.set_xlabel(var_name)
                ax.set_ylabel("Density" if density else "Count")
    
                if log_axis:
                    ax.set_xscale('log')          # <-- fixed: was yscale before
    
            # -----------------------------------
            # Mean / Median / Mode lines (all plot types)
            # -----------------------------------
            if show_mean_median or show_mean_mode:
                from scipy.stats import gaussian_kde as _gkde

                # Arrow length = 1/8 of the current y-axis range
                def _arrow(ax, x, color, linestyle):
                    ylo, yhi = ax.get_ylim()
                    arrow_len = (yhi - ylo) / 4
                    # Evaluate KDE height at x to start arrow from the curve
                    src = plot_points if kde and (normalize_mean or normalize_median or normalize_mode) else points
                    cond_vals = src[src["condition"] == cond]["value_mean"].dropna().values
                    if len(cond_vals) > 1:
                        y_start = float(_gkde(cond_vals)(x))
                    else:
                        y_start = arrow_len
                    y_start = min(y_start, arrow_len)
                    y_start = max(y_start, arrow_len * 0.5)  # ensure arrow has room

                    ax.annotate(
                        "",
                        xy=(x, 0),
                        xytext=(x, y_start),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=0.8,
                            linestyle=linestyle,
                        ),
                        zorder=6,
                    )

                for _, row in summary.iterrows():
                    cond  = row["condition"]
                    color = self.condition_palette()[cond]

                    # Mean (solid)
                    _arrow(ax, row["mean"], color, "solid")

                    if show_mean_mode:
                        src   = plot_points if kde and (normalize_mean or normalize_median or normalize_mode) else points
                        vals  = src[src["condition"] == cond]["value_mean"].dropna().values
                        if len(vals) > 1:
                            _kde_fit = _gkde(vals)
                            xs       = np.linspace(vals.min(), vals.max(), 1000)
                            mode     = xs[np.argmax(_kde_fit(xs))]
                            _arrow(ax, mode, color, "dashed")
                    else:
                        _arrow(ax, row["median"], color, "dashed")
    
            sns.despine(ax=ax)
    
            self.save_plot_bundle(
                fig=fig,
                plot_name="histogram",
                var_tag=var_name,
                points_df=points,
                summary_df=summary,
                params={
                    "conditions": conditions,
                    "frame_range": frame_range,
                    "cellpose_unit": cellpose_unit,
                    "bins": bins,
                    "density": density,
                    "ccdf": ccdf,
                },
                file_stem=file_stem,
            )
    
        plt.tight_layout()
        plt.show()

        return allpoints    
        
    def resolve_fit_cfg(self, fit_cfg, conditions_order):
        """
        Normalize per-condition fit configuration.
        """
    
        if fit_cfg is None:
            return {}
    
        out = {}
    
        for cond in conditions_order:
            cfg = fit_cfg.get(cond)
            if cfg is None:
                continue
    
            out[cond] = {
                "enabled": cfg.get("enabled", True),
                "model": cfg.get("model"),
                "p0": cfg.get("p0", None),
                "fit_range": cfg.get("fit_range", None),
                "color": cfg.get("color", self.palette[cond]["main"]),
                "linestyle": cfg.get("linestyle", "--"),
                "linewidth": cfg.get("linewidth", 1.5),
            }
    
        return out


    def plot_traces(
        self,
        var_names,
        conditions=None,
        frame_range=(0, 60),
        plot_mean=True,
        errorbar=None,
        cellpose_frame_stat="mean",  # "mean"/"median"
        file_stem=None,
        style=None, 
        figsize=None,
        fit_cfg=None
    ):
        """
        Plot full time traces.
    
        Trackmate / fov:
          - faint unit traces (object tracks or movie)
          - bold condition mean
    
        Cellpose:
          - collapse objects within each frame
          - unit = movie (folder × repeat)
          - each frame point = mean/median across objects
        """
    
        # ----------------------------
        # Subplots
        # ----------------------------
        fig, axes, var_names = self.make_axes(var_names, figsize=figsize)
    
        for ax, var_name in zip(axes, var_names):

            # ----------------------------------------
            # Resolve raw vs norm dataframe automatically
            # ----------------------------------------
            df_plot, ycol = self.get_df_for_var(var_name)

            # ----------------------------------------
            # Conditionn Filtering
            # ----------------------------------------
            if conditions is not None:
                df_plot = df_plot[df_plot["condition"].isin(conditions)]

            # ----------------------------
            # Frame filtering
            # ----------------------------
            t0, t1 = frame_range
            df_plot = df_plot[df_plot["frame"].between(t0, t1)]

            # ----------------------------
            # Add time columns
            # ----------------------------
            df_plot = self.add_time_column(df_plot)
            
            ax_style = self.get_style_for_var(style, var_name)
            self.apply_axes_style(ax, ax_style)

            # ----------------------------------------
            # Cellpose raw variables: collapse objects per frame
            # ----------------------------------------
            if self.analysis.mode == "cellpose" and "object_id" in df_plot.columns:
        
                aggfunc = cellpose_frame_stat
        
                df_plot = (
                    df_plot.groupby(
                        ["condition", "folder", "repeat_id", "frame"],
                        as_index=False
                    )
                    .agg(
                        value_mean=(ycol, aggfunc),
                        n_cells=("object_id", "nunique"),
                    )
                )
        
                # Movie replicate unit
                df_plot["unit"] = (
                    df_plot["folder"].astype(str)
                    + "_rep" + df_plot["repeat_id"].astype(str)
                )
        
                ycol = "value_mean"

                # re-add time column
                df_plot = self.add_time_column(df_plot)
        
            else:
                # ----------------------------------------
                # Trackmate / whole_fov / norm_df
                # ----------------------------------------
                # unit_cols = self.unit_cols()
                unit_cols = self.unit_cols(df_plot)
                df_plot["unit"] = df_plot[unit_cols].astype(str).agg("_".join, axis=1)

            mean_trace = (
                df_plot.groupby(["condition", "frame"])[ycol]
                  .agg(mean="mean", sd="std", n="count")
                  .reset_index()
            )
            mean_trace["sem"] = mean_trace["sd"] / np.sqrt(mean_trace["n"])
            mean_trace = self.add_time_column(mean_trace)
    
            # Individual traces (no averaging)
            sns.lineplot(
                ax=ax,
                data=df_plot,
                # x="frame",
                x=self.time_col,
                y=ycol,
                hue="condition",
                units="unit",
                estimator=None,
                alpha=0.4,
                linewidth=0.5,
                palette=self.condition_palette(),
                legend=False
            )
    
            # Mean trace per condition
            if plot_mean:
                sns.lineplot(
                    ax=ax,
                    data=df_plot,
                    # x="frame",
                    x=self.time_col,
                    y=ycol,
                    hue="condition",
                    estimator="mean",
                    errorbar=errorbar, # None for no
                    err_kws={"edgecolor":"none",
                             "linewidth": 0, 
                            },
                    linewidth=1.5,
                    palette=self.condition_palette(),
                    legend=False
                )

            if fit_cfg is not None:
                
                conditions_order = list(mean_trace["condition"].unique())
                fit_map = self.resolve_fit_cfg(fit_cfg, conditions_order)
                
                for cond, cfg in fit_map.items():
                
                    if not cfg["enabled"]:
                        continue
                
                    cond_trace = mean_trace[mean_trace["condition"] == cond]

                    # Frame domain (for masking)
                    frame_vals = cond_trace["frame"].values
                    
                    # Time domain (for fitting)
                    time_vals = cond_trace[self.time_col].values
                    ydata = cond_trace["mean"].values
                    
                    # Apply fit range IN FRAMES
                    if cfg["fit_range"] is not None:
                        lo_f, hi_f = cfg["fit_range"]
                        mask = (frame_vals >= lo_f) & (frame_vals <= hi_f)
                        time_vals = time_vals[mask]
                        ydata = ydata[mask]
                    else:
                        time_vals = time_vals
                    
                    xdata = time_vals   # fitting uses time

                    # xdata = cond_trace[self.time_col].values
                    # ydata = cond_trace["mean"].values
                
                    # # Apply fit range
                    # if cfg["fit_range"] is not None:
                    #     lo, hi = cfg["fit_range"]
                    #     mask = (xdata >= lo) & (xdata <= hi)
                    #     xdata = xdata[mask]
                    #     ydata = ydata[mask]

                    fit_res = self.fit_trace(
                        xdata,
                        ydata,
                        model=cfg["model"],
                        p0=cfg.get("p0", None),
                        # bounds=cfg.get("bounds", None),
                    )
                    
                    popt = fit_res["params"]
                    perr = fit_res["errors"]   
                
                    model_func = self.get_fit_model(cfg["model"])
                    
                    t_fit = np.linspace(xdata.min(), xdata.max(), 300)
                    y_fit = model_func(t_fit, *popt)
                
                    ax.plot(
                        t_fit,
                        y_fit,
                        color=cfg["color"],
                        linestyle=cfg["linestyle"],
                        linewidth=cfg["linewidth"],
                        label=f"{cond} fit"
                    )

                    print(f"\nFit results for {cond}")
                    print("model:", fit_res["model"])
                    print("params:", fit_res["params"])
                    print("errors:", fit_res["errors"])

                    if fit_res["half_life"] is not None:
                        print(
                            f"t½ = {fit_res['half_life']:.4g} "
                            f"± {fit_res['half_life_error']:.4g} {self.time_unit}"
                        )


            sns.despine(fig=fig)
            
            self.save_plot_bundle(
                fig=fig,
                plot_name="traces",
                var_tag=var_name,
                points_df=df_plot,
                summary_df=mean_trace,
                params={
                    "conditions": conditions,
                    "frame_range": frame_range,
                },
                file_stem=file_stem,
            )

            # ax.set_title(var_name)
            # ax.set_xlabel("Frame")
            ax.set_xlabel(f"Time ({self.time_unit})")
            ax.set_ylabel(var_name)
            #sns.despine(ax=ax)
    
        plt.tight_layout()
        plt.show()

    
    def compute_band(self, df, xcol, ycol, groupcol,
                 mode="sd",
                 qlow=0.16,
                 qhigh=0.84):
        """
        Compute uncertainty bands for each group at each x.
    
        Parameters
        ----------
        mode : {"sd", "sem", "quantile"}
            sd       = mean ± 1 SD
            sem      = mean ± SEM
            quantile = [qlow, qhigh] band
    
        qlow/qhigh : float
            Quantiles used if mode="quantile"
            Common choices:
              0.16–0.84  (~68%)
              0.025–0.975 (~95%)
              0.25–0.75  (IQR)
        """
    
        if mode in ["sd", "sem"]:
            stats = (
                df.groupby([groupcol, xcol])[ycol]
                  .agg(["mean", "std", "count"])
                  .reset_index()
            )
    
            if mode == "sd":
                err = stats["std"]
            else:  # sem
                err = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    
            stats["low"] = stats["mean"] - err
            stats["high"] = stats["mean"] + err
            return stats[[groupcol, xcol, "mean", "low", "high"]]
    
        elif mode == "quantile":
            stats = (
                df.groupby([groupcol, xcol])[ycol]
                  .agg(
                      mean="mean",
                      low=lambda x: x.quantile(qlow),
                      high=lambda x: x.quantile(qhigh),
                  )
                  .reset_index()
            )
            return stats
    
        else:
            raise ValueError("mode must be 'sd', 'sem', or 'quantile'")

    def draw_bands(self, ax, df_pre, df_post, var_name,
               palette, bands):

        for band_cfg in bands:
    
            mode = band_cfg.get("mode", "quantile")
            alpha = band_cfg.get("alpha", 0.15)
    
            qlow = band_cfg.get("qlow", 0.16)
            qhigh = band_cfg.get("qhigh", 0.84)
    
            # Compute stats
            band_pre = self.compute_band(
                df_pre, self.time_col, var_name, "condition",
                mode=mode, qlow=qlow, qhigh=qhigh
            )
            band_post = self.compute_band(
                df_post, self.time_col, var_name, "condition",
                mode=mode, qlow=qlow, qhigh=qhigh
            )
    
            # Draw for each condition
            for cond in band_pre["condition"].unique():
    
                color = palette[cond]
    
                cpre = band_pre[band_pre["condition"] == cond]
                cpost = band_post[band_post["condition"] == cond]
    
                ax.fill_between(
                    cpre[self.time_col], cpre["low"], cpre["high"],
                    alpha=alpha, color=color,linewidth=0, edgecolor="none", zorder=3
                )
    
                ax.fill_between(
                    cpost[self.time_col], cpost["low"], cpost["high"],
                    alpha=alpha, color=color,linewidth=0, edgecolor="none", zorder=3
                )
    
    
    def plot_aligned_traces_with_events(
        self,
        var_names,
        conditions=None,
        pre_window=20,
        gap_left=6,
        gap_right=5,
        post_window=21,
        min_margin=20,
        min_duration=20,
        plot_mean=True,
        shade_gap=True,
        show_tend=False,
        bands=None,
        file_stem=None,
        style=None, 
        figsize=None,
        ttest_lengths=None,
        baseline_width=5,
        return_baselines=False,
        require_full_window=False,
    ):
        """
        Align each tracked cell trajectory by its own t0.
    
        Requires tracking mode (trackmate).
        Uses canonical schema:
    
          df index:
            condition, folder, repeat_id, frame, object_id
    
          events_df index:
            condition, folder, repeat_id, object_id
        """
    
        # ----------------------------
        # Require tracking mode
        # ----------------------------
        if self.analysis.events_df is None:
            raise ValueError("No events_df found. Run trackmate mode first.")
    
        # ----------------------------
        # Plot
        # ----------------------------
        fig, axes, var_names = self.make_axes(var_names, figsize=figsize)
    
        for ax, var_name in zip(axes, var_names):

            # ----------------------------
            # Tidy data
            # ----------------------------
            # df = self.tidy()
            df, var_name = self.get_df_for_var(var_name)
            events = self.analysis.events_df.reset_index()
        
            # Optional condition filtering
            if conditions is not None and len(conditions) > 0:
                df = df[df["condition"].isin(conditions)]
                events = events[events["condition"].isin(conditions)]
        
            # ----------------------------
            # Merge t0/t_end into traces
            # ----------------------------
            df = df.merge(
                events,
                on=["condition", "folder", "repeat_id", "object_id"],
                how="inner"
            )
        
            # ----------------------------
            # Compute max frame per object
            # ----------------------------
            max_frames = (
                df.groupby(["condition", "folder", "repeat_id", "object_id"])["frame"]
                  .max()
                  .reset_index(name="max_frame")
            )
        
            df = df.merge(max_frames, on=["condition","folder","repeat_id","object_id"])
        
            # ----------------------------
            # Quality cuts
            # ----------------------------

            if require_full_window:
                # STRICT: must cover entire plotting window
                df = df[
                    (df["t0"] >= pre_window) &
                    ((df["max_frame"] - df["t0"]) >= post_window)
                ]
            else:
                # LOOSE: original behavior
                df = df[
                    (df["t0"] > min_margin) &
                    ((df["max_frame"] - df["t0"]) > min_margin) &
                    ((df["t_end"] - df["t0"]) > min_duration)
                ]
            # ----------------------------
            # Relative time axis
            # ----------------------------
            df["t_rel"] = df["frame"] - df["t0"]

            # Add column to convert from frames to time
            df = self.add_time_column(df, frame_col="t_rel")
        
            # Restrict plotting window
            df = df[
                (df["t_rel"] >= -pre_window) &
                (df["t_rel"] <= post_window)
            ]
        
            # Remove excluded gap
            df = df[
                (df["t_rel"] <= -gap_left) |
                (df["t_rel"] >= gap_right)
            ]
        
            # Unique unit per object
            df["unit"] = (
                df["folder"].astype(str)
                + "_rep" + df["repeat_id"].astype(str)
                + "_obj" + df["object_id"].astype(str)
            )

            # ----------------------------
            # Baseline windows
            # ----------------------------
            pre_base = df[
                (df["t_rel"] >= -gap_left - baseline_width) &
                (df["t_rel"] <= -gap_left)
            ]
            
            post_base = df[
                (df["t_rel"] >= gap_right) &
                (df["t_rel"] <= gap_right + baseline_width)
            ]
            
            pre_base = pre_base[pre_base[var_name] != 0]
            post_base = post_base[post_base[var_name] != 0]

            obj_pre_base = (
                pre_base
                .groupby(["condition", "unit"])[var_name]
                .mean()
                .reset_index(name="baseline_pre")
            )
            
            obj_post_base = (
                post_base
                .groupby(["condition", "unit"])[var_name]
                .mean()
                .reset_index(name="baseline_post")
            )
            
            df_baseline = obj_pre_base.merge(
                obj_post_base,
                on=["condition", "unit"],
                how="inner"
            )

            ax_style = self.get_style_for_var(style, var_name)
            self.apply_axes_style(ax, ax_style)
    
            df_pre  = df[df["t_rel"] <= -gap_left]
            df_post = df[df["t_rel"] >= gap_right]

            df_pre = df_pre[df_pre[var_name] != 0]
            df_post = df_post[df_post[var_name] != 0]
    
            # Individual traces
            for seg in [df_pre, df_post]:
                sns.lineplot(
                    ax=ax,
                    data=seg,
                    # x="t_rel",
                    x=self.time_col,
                    y=var_name,
                    #hue="condition",
                    units="unit",
                    estimator=None,
                    alpha=0.25,
                    linewidth=0.75,
                    color= '#BBBBBB' ,#self.condition_palette(),
                    legend=False,
                    zorder=1, #to set the individual lines in the background
                )
    
            # Mean traces
            if plot_mean:
                for seg in [df_pre, df_post]:
                    sns.lineplot(
                        ax=ax,
                        data=seg,
                        # x="t_rel",
                        x=self.time_col,
                        y=var_name,
                        hue="condition",
                        estimator="mean",
                        errorbar=None,
                        linewidth=1.5,
                        palette=self.condition_palette(),
                        legend=False,
                        zorder=4
                    )
    
            # Bands
            if bands is not None:
                self.draw_bands(
                    ax=ax,
                    df_pre=df_pre,
                    df_post=df_post,
                    var_name=var_name,
                    palette=self.condition_palette(),
                    bands=bands,
                )
    
            # Shade gap
            if shade_gap:
                ax.axvspan(-gap_left, gap_right, alpha=0.15)
    
                ax.axvline(0, linestyle="--", linewidth=1)
    
            if show_tend:
                tend_rel = (df["t_end"] - df["t0"]).median()
                ax.axvline(tend_rel, linestyle=":", linewidth=2)
    
            # ax.set_title(f"{var_name} aligned to division")
            ax.set_xlabel("Frames relative to t0")
            ax.set_ylabel(var_name)
    
            sns.despine(ax=ax)

                        # --- SAVE ---
            if file_stem is not None:

                # Per-frame condition stats
                summary = (
                    df.groupby(["condition",self.time_col])[var_name]
                    .agg(
                        n_traces="count",
                        mean="mean",
                        sd="std",
                    )
                    .reset_index()
                )
                summary["sem"] = summary["sd"] / np.sqrt(summary["n_traces"])

                # Trace count per condition
                trace_counts = (
                    df.groupby("condition")["unit"]
                    .nunique()
                    .reset_index(name="n_unique_traces")
                )
                summary = summary.merge(trace_counts, on="condition")

                # Per-unit trace values (wide format: one column per unit)
                traces_wide = (
                    df.pivot_table(
                        index=["condition", self.time_col],
                        columns="unit",
                        values=var_name,
                    )
                    .reset_index()
                )

                # Merge stats + individual traces
                summary = summary.merge(traces_wide, on=["condition", self.time_col])

                self.save_plot_bundle(
                    fig=fig,
                    plot_name="aligned_traces",
                    var_tag=var_name,
                    points_df=df,
                    summary_df=summary,       
                    params={
                        "conditions": conditions,
                        "pre_window": pre_window,
                        "post_window": post_window,
                        "gap_left": gap_left,
                        "gap_right": gap_right,
                    },
                    file_stem=file_stem,
                )
    
        plt.tight_layout()
        plt.show()

        if return_baselines:
            return df_pre, df_post, df_baseline
        else:
            return df_pre, df_post

    # ----------------------------
    # Built-in fit models
    # ----------------------------
    def linear(self, t, m, b):
        return m * t + b
    
    def exp_decay(self, t, A, tau, C):
        return A * np.exp(-t / tau) + C
    
    def exp_rise(self, t, A, tau, C):
        return A * (1 - np.exp(-t / tau)) + C
    
    def logistic(self, t, L, k, t0, C):
        return L / (1 + np.exp(-k * (t - t0))) + C

    def exp_decay_delayed(self, t, A, tau, t0, C):
        """
        Exponential decay with onset delay t0.
    
        y(t) = A + C                              for t < t0
               A * exp(-(t - t0)/tau) + C         for t >= t0
        """
        t = np.asarray(t)
    
        y = np.full_like(t, A + C, dtype=float)
        mask = t >= t0
    
        y[mask] = A * np.exp(-(t[mask] - t0) / tau) + C
        return y

    def ratio_exp_delayed_over_exp_decay(self, t, C, D, tau, t2):
        return C*np.exp((t-t2)/tau) - D

    def exp_rise_delayed(self, t, A, tau, t0, C):
        """
        Exponential rise with onset delay t0.
    
        y(t) = C                          for t < t0
               A * (1 - exp(-(t - t0)/tau)) + C   for t >= t0
        """
        t = np.asarray(t)
    
        y = np.full_like(t, C, dtype=float)
        mask = t >= t0
    
        y[mask] = A * (1 - np.exp(-(t[mask] - t0) / tau)) + C
        return y

    def get_fit_model(self, model):
        """
        Return callable model function by name.
        If name is not a string, assume it's a custom model.
        """

        models = {
            "linear": self.linear, 
            "exp_decay": self.exp_decay,
            "exp_rise": self.exp_rise,
            "logistic": self.logistic,
            "exp_decay_delayed": self.exp_decay_delayed,
            "exp_rise_delayed": self.exp_rise_delayed,
            "ratio_exp_delayed_over_exp_decay": self.ratio_exp_delayed_over_exp_decay,
        }

        if isinstance(model, str):
            if model not in models:
                raise ValueError(
                    f"Unknown model '{model}'. Available: {list(models.keys())}"
                )
            return models[model]

        return model

    def get_default_p0(self, t, y, model):
        """
        Generate automatic initial guesses for curve fitting.
        """
    
        model_name = model if isinstance(model, str) else model.__name__
    
        t = np.asarray(t)
        y = np.asarray(y)
    
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        tmin, tmax = np.nanmin(t), np.nanmax(t)
    
        A0 = ymax - ymin
        C0 = ymin
        tau0 = 0.5 * (tmax - tmin)
    
        # Safety
        if tau0 <= 0:
            tau0 = 1.0
    
        if model_name in ("exp_decay", "exp_rise"):
            return (A0, tau0, C0)
    
        if model_name in ("exp_rise_delayed", "exp_decay_delayed"):
            t00 = tmin
            return (A0, tau0, t00, C0)

        if model_name == "linear":
            # slope estimate from endpoints
            m0 = (ymax - ymin) / (tmax - tmin) if tmax > tmin else 0.0
            b0 = ymin - m0 * tmin
            return (m0, b0)
            
        # Fallback
        return None
   
    def fit_trace(
        self,
        t,
        y,
        model,
        p0=None,
        # bounds=None,
    ):
        """
        Fit a single (t, y) trace and compute derived quantities.
        """
    
        # Auto-generate p0 if missing
        if p0 is None:
            p0 = self.get_default_p0(t, y, model)
    
        # Auto-generate bounds if missing
        # if bounds is None:
        #     bounds = self.get_default_bounds(model, t)
    
        # Resolve model
        f_model = self.get_fit_model(model)
        model_name = model if isinstance(model, str) else model.__name__
    
        popt, pcov = curve_fit(
            f_model,
            t,
            y,
            p0=p0,
            # bounds=bounds,
            maxfev=10000,
        )
    
        perr = np.sqrt(np.diag(pcov))
    
        # --- Half-life logic (unchanged) ---
        half_life = None
        half_life_err = None
    
        if f_model in (
            self.exp_decay,
            self.exp_rise,
            self.exp_decay_delayed,
            self.exp_rise_delayed,
        ):
            tau = popt[1]
            tau_err = perr[1]
            half_life = np.log(2) * tau
            half_life_err = np.log(2) * tau_err
    
        return {
            "params": popt,
            "errors": perr,
            "cov": pcov,
            "model": model_name,
            "half_life": half_life,
            "half_life_error": half_life_err,
        }

    def fit_average_trace(
        self,
        var_name,
        model="exp_decay",
        conditions=None,
        fit_range=None,
        p0=None,
        # bounds=(-np.inf, np.inf),
        print_results=True,
    ):
        """
        Fit average condition traces using scipy.optimize.curve_fit.
    
        Parameters
        ----------
        var_name : str
            Variable to fit (raw or norm_)
    
        model : str or callable
            Built-in model name or custom function f(t, ...)
    
        fit_range : tuple (tmin, tmax) given in frames
            Restrict fit region
    
        p0 : list
            Initial guess for parameters
    
        bounds : tuple
            Parameter bounds for curve_fit
    
        Returns
        -------
        results : dict
            Fit params + errors per condition
        """
    
        # Resolve correct dataframe
        df, ycol = self.get_df_for_var(var_name)
    
        # Filter conditions
        if conditions is not None:
            df = df[df["condition"].isin(conditions)]
    
        # Compute mean trace per condition
        mean_trace = (
            df.groupby(["condition", "frame"])[ycol]
              .mean()
              .reset_index()
        )
    
        # Apply fit window
        if fit_range is not None:
            lo, hi = fit_range
            mean_trace = mean_trace[mean_trace["frame"].between(lo, hi)]
    
        # Resolve model function
        if isinstance(model, str):
            f_model = self.get_fit_model(model)
            model_name = model
        else:
            f_model = model
            model_name = model.__name__
    
        results = {}

        for cond in mean_trace["condition"].unique():

            sub = mean_trace[mean_trace["condition"] == cond]
            t = sub["frame"].values
            y = sub[ycol].values
        
            res = self.fit_trace(
                t,
                y,
                model=model,
                p0=p0,
                bounds=bounds,
            )
        
            results[cond] = res
        
            if print_results:
                print(f"\n=== Fit results: {cond} ({res['model']}) ===")
                for i, (val, err) in enumerate(zip(res["params"], res["errors"])):
                    print(f"  p{i}: {val:.4g} ± {err:.4g}")
        
                if res["half_life"] is not None:
                    print(
                        f"  t½: {res['half_life']:.4g} "
                        f"± {res['half_life_error']:.4g}"
                    )

    def aligned_timecomp_barplot(
        self,
        var_names,
        conditions=None,
        pre_window=20,
        gap_left=6,
        gap_right=5,
        post_window=21,
        min_margin=20,
        min_duration=20,
        collapse_func="mean",
        file_stem=None,
        style=None,
        figsize=None,
    ):
    
        if self.analysis.events_df is None:
            raise ValueError("No events_df found. Run trackmate mode first.")
    
        allpoints = {}
        fig, axes, var_names = self.make_axes(var_names, figsize=figsize)
    
        for ax, var_name in zip(axes, var_names):
    
            ax_style = self.get_style_for_var(style, var_name)
            self.apply_axes_style(ax, ax_style)
    
            df, var_name = self.get_df_for_var(var_name)
            events = self.analysis.events_df.reset_index()
    
            if conditions is not None:
                df = df[df["condition"].isin(conditions)]
                events = events[events["condition"].isin(conditions)]
    
            df = df.merge(
                events,
                on=["condition", "folder", "repeat_id", "object_id"],
                how="inner"
            )
    
            max_frames = (
                df.groupby(["condition", "folder", "repeat_id", "object_id"])["frame"]
                  .max()
                  .reset_index(name="max_frame")
            )
    
            df = df.merge(max_frames,
                          on=["condition","folder","repeat_id","object_id"])
    
            df = df[
                (df["t0"] > min_margin) &
                ((df["max_frame"] - df["t0"]) > min_margin) &
                ((df["t_end"] - df["t0"]) > min_duration)
            ]
    
            df["t_rel"] = df["frame"] - df["t0"]
    
            df = df[
                (df["t_rel"] >= -pre_window) &
                (df["t_rel"] <= post_window)
            ]
    
            df = df[
                (df["t_rel"] <= -gap_left) |
                (df["t_rel"] >= gap_right)
            ]
    
            df["unit"] = (
                df["folder"].astype(str)
                + "_rep" + df["repeat_id"].astype(str)
                + "_obj" + df["object_id"].astype(str)
            )
    
            # -----------------------------------
            # Split aligned epochs
            # -----------------------------------
            df_pre  = df[df["t_rel"] <= -gap_left]
            df_post = df[df["t_rel"] >= gap_right]
    
            df_pre  = df_pre[df_pre[var_name] != 0]
            df_post = df_post[df_post[var_name] != 0]
    
            # -----------------------------------
            # Collapse to unit level
            # -----------------------------------
            obj_pre = (
                df_pre
                .groupby(["condition", "folder", "unit"])[var_name]
                .agg(collapse_func)
                .reset_index(name="value_mean")
            )
            obj_pre["frame_bin"] = "early"
    
            obj_post = (
                df_post
                .groupby(["condition", "folder", "unit"])[var_name]
                .agg(collapse_func)
                .reset_index(name="value_mean")
            )
            obj_post["frame_bin"] = "late"
    
            points = pd.concat([obj_pre, obj_post], ignore_index=True)
    
            allpoints[var_name] = points
    
            conditions_order = (
                conditions if conditions is not None
                else list(points["condition"].unique())
            )
    
            bins_order = ["early", "late"]
    
            # -----------------------------------
            # Bars
            # -----------------------------------
            sns.barplot(
                ax=ax,
                data=points,
                x="condition",
                y="value_mean",
                hue="frame_bin",
                order=conditions_order,
                hue_order=bins_order,
                errorbar="sd",
                alpha=0.5,
                capsize=0.6,
                err_kws={'linewidth': 1.0},
                legend=False
            )
    
            # recolor bars using your palette system
            i = 0
            for b in bins_order:
                for cond in conditions_order:
                    patch = ax.patches[i]
                    patch.set_facecolor(self.palette[cond][b])
                    i += 1
    
            # -----------------------------------
            # Manual scatter overlay (same logic)
            # -----------------------------------
            markers = self.folder_markers(points["folder"].unique())
            cond_to_x = {c: i for i, c in enumerate(conditions_order)}
            offsets = {"early": -0.2, "late": +0.2}
    
            points = points.copy()
            points["xpos"] = points["condition"].map(cond_to_x)
            points["xpos"] += points["frame_bin"].map(offsets)
            points["xpos"] += np.random.uniform(-0.05, 0.05, size=len(points))
    
            for folder, m in markers.items():
                sub = points[points["folder"] == folder]
    
                for b in bins_order:
                    sub2 = sub[sub["frame_bin"] == b]
    
                    ax.scatter(
                        sub2["xpos"],
                        sub2["value_mean"],
                        marker=m,
                        s=15,
                        alpha=0.8,
                        color=[self.palette[c][b] for c in sub2["condition"]],
                        edgecolors="#424242",
                        linewidth=0.5,
                    )
    
            sns.despine(ax=ax)
            ax.set_xlim(-0.5, len(conditions_order) - 0.5)
    
            self.save_plot_bundle(
                fig=fig,
                plot_name="aligned_timecomp",
                var_tag=var_name,
                points_df=points,
                summary_df=None,
                params={
                    "pre_window": pre_window,
                    "post_window": post_window,
                    "gap_left": gap_left,
                    "gap_right": gap_right,
                    "collapse_func": collapse_func,
                },
                file_stem=file_stem,
            )
    
        plt.tight_layout()
        plt.show()
    
        return allpoints

    def detrend_and_smooth(
        self,
        series,
        smooth_sigma=1.0,
        baseline_window=50,
        baseline_pct=10,
        detrend_mode="percentile",  
        poly_degree=2,               
        smooth_only=True,
    ):    
        """
        Denoise and detrend a 1D array of fluorescence values.
    
        Steps:
          1. Estimate a rolling baseline as the low percentile
             over a sliding window
          2. Divide by the baseline (ΔF/F style)
          3. Gaussian smooth to reduce high-frequency noise
    
        NaN values are ignored throughout and preserved in the output.
    
        Parameters
        ----------
        series : np.ndarray
            1D array of values, may contain NaN.
        smooth_sigma : float
            Sigma (in frames) for Gaussian smoothing. 0 = skip smoothing.
        baseline_window : int
            Rolling window size (in frames) for baseline estimation.
        baseline_pct : float
            Percentile used to define the baseline (e.g. 10 = 10th percentile).
    
        Returns
        -------
        np.ndarray
            Detrended, smoothed array of the same length.
        """
        from scipy.ndimage import gaussian_filter1d
    
        out = series.copy().astype(float)

        if smooth_only:
            if smooth_sigma > 0:
                nan_mask = np.isnan(out)
                if not nan_mask.all():
                    x = np.arange(len(out))
                    interp = np.interp(x, x[~nan_mask], out[~nan_mask])
                    smoothed = gaussian_filter1d(interp, sigma=smooth_sigma)
                    smoothed[nan_mask] = np.nan
                    out = smoothed
            return out

        # ------------------------------------------------------------------
        # 1. Detrend
        # ------------------------------------------------------------------
        if detrend_mode == "polynomial":
            x = np.arange(len(out))
            valid = ~np.isnan(out)
            if valid.sum() < poly_degree + 1:
                return out
            coeffs   = np.polyfit(x[valid], out[valid], poly_degree)
            baseline = np.polyval(coeffs, x)
            out = out - baseline

        else:
            # Rolling baseline: low percentile over a sliding window
            half = baseline_window // 2
            baseline = np.full_like(out, np.nan)

            for i in range(len(out)):
                lo = max(0, i - half)
                hi = min(len(out), i + half + 1)
                window_vals = out[lo:hi]
                valid = window_vals[~np.isnan(window_vals)]
                if len(valid) > 0:
                    baseline[i] = np.nanpercentile(valid, baseline_pct)

            # ΔF/F: divide by baseline
            out = (out - baseline) / (baseline + 1e-10)

        # ------------------------------------------------------------------
        # 2. Gaussian smooth (NaN-aware: interpolate → smooth → restore NaNs)
        # ------------------------------------------------------------------
        if smooth_sigma > 0:
            nan_mask = np.isnan(out)
            if not nan_mask.all():
                x = np.arange(len(out))
                interp = np.interp(x, x[~nan_mask], out[~nan_mask])
                smoothed = gaussian_filter1d(interp, sigma=smooth_sigma)
                smoothed[nan_mask] = np.nan
                out = smoothed

        return out
        
    # def _get_valid_cell_keys(
    #     self,
    #     conditions=None,
    #     frame_range=None,
    #     exclude_dividing=True,
    #     exclude_late_start=True,
    #     min_track_start=0,
    #     exclude_nuc_zero=True,
    #     exclude_short_tracks=True,
    #     min_track_length=10,
    # ):
    #     """
    #     Return a DataFrame of (condition, folder, repeat_id, object_id) tuples
    #     that pass all requested quality filters.
    
    #     Parameters
    #     ----------
    #     conditions : list[str] or None
    #         Restrict to these conditions. None = all.
    #     frame_range : tuple(int, int) or None
    #         (t0_frame, t1_frame) inclusive. Used to define expected window.
    #     exclude_dividing : bool
    #         If True, exclude cells where t0 is not NaN (detected division).
    #     exclude_late_start : bool
    #         If True, exclude cells whose track starts after min_track_start.
    #         Catches post-division daughters that were never assigned a t0.
    #     min_track_start : int
    #         Maximum allowed first frame for a cell track.
    #         Only used when exclude_late_start=True.
    #         Defaults to 0 (track must start at frame 0).
    #         If frame_range is provided and min_track_start=0, 
    #         automatically uses frame_range[0].
    #     exclude_nuc_zero : bool
    #         If True, exclude cells that have any frame with nuc_area == 0
    #         within the requested frame_range. These are likely segmentation
    #         failures at division.
    #     exclude_short_tracks : bool
    #         If True, exclude cells whose track (within frame_range) is shorter
    #         than min_track_length frames.
    #     min_track_length : int
    #         Minimum number of valid (non-NaN nuc_area) frames required.
    #         Only used when exclude_short_tracks=True.
    
    #     Returns
    #     -------
    #     valid_keys : pd.DataFrame
    #         Columns: condition, folder, repeat_id, object_id
    #     """
    
    #     cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    
    #     raw_df = self.analysis.df.reset_index()
    
    #     # ------------------------------------------------------------------
    #     # Condition filtering
    #     # ------------------------------------------------------------------
    #     if conditions is not None and len(conditions) > 0:
    #         raw_df = raw_df[raw_df["condition"].isin(conditions)]
    
    #     # ------------------------------------------------------------------
    #     # Frame range filtering
    #     # ------------------------------------------------------------------
    #     if frame_range is not None:
    #         t0_f, t1_f = frame_range
    #         raw_df = raw_df[raw_df["frame"].between(t0_f, t1_f)]
    
    #     # ------------------------------------------------------------------
    #     # Start with all unique cell keys present in the (filtered) raw_df
    #     # ------------------------------------------------------------------
    #     valid_keys = raw_df[cell_keys].drop_duplicates().copy()
    
    #     # ------------------------------------------------------------------
    #     # Filter 1: exclude dividing cells (t0 is not NaN)
    #     # ------------------------------------------------------------------
    #     if exclude_dividing and self.analysis.events_df is not None:
    
    #         events = self.analysis.events_df.reset_index()
    
    #         if conditions is not None and len(conditions) > 0:
    #             events = events[events["condition"].isin(conditions)]
    
    #         non_dividing = events[events["t0"].isna()][cell_keys]
    
    #         valid_keys = valid_keys.merge(non_dividing, on=cell_keys, how="inner")
    
    #         print(f"[filter] After exclude_dividing: {len(valid_keys)} cells")
    
    #     # ------------------------------------------------------------------
    #     # Filter 2: exclude late-starting tracks
    #     # ------------------------------------------------------------------
    #     if exclude_late_start:
    
    #         # If frame_range given and min_track_start is at default 0,
    #         # snap to frame_range[0] so we don't penalise clipped data
    #         effective_min_start = min_track_start
    #         if frame_range is not None and min_track_start == 0:
    #             effective_min_start = frame_range[0]
    
    #         first_frames = (
    #             raw_df.groupby(cell_keys)["frame"]
    #             .min()
    #             .reset_index(name="first_frame")
    #         )
    
    #         early_enough = first_frames[
    #             first_frames["first_frame"] <= effective_min_start
    #         ][cell_keys]
    
    #         valid_keys = valid_keys.merge(early_enough, on=cell_keys, how="inner")
    
    #         print(f"[filter] After exclude_late_start "
    #               f"(min_track_start={effective_min_start}): {len(valid_keys)} cells")
    
    #     # ------------------------------------------------------------------
    #     # Filter 3: exclude cells with any nuc_area == 0 in window
    #     # ------------------------------------------------------------------
    #     if exclude_nuc_zero:
    
    #         cells_with_zero_nuc = (
    #             raw_df[raw_df["nuc_area"] == 0][cell_keys]
    #             .drop_duplicates()
    #         )
    
    #         # Anti-join: keep cells NOT in cells_with_zero_nuc
    #         merged = valid_keys.merge(
    #             cells_with_zero_nuc.assign(_bad=True),
    #             on=cell_keys,
    #             how="left",
    #         )
    #         valid_keys = merged[merged["_bad"].isna()][cell_keys].copy()
    
    #         print(f"[filter] After exclude_nuc_zero: {len(valid_keys)} cells")
    
    #     # ------------------------------------------------------------------
    #     # Filter 4: exclude short tracks
    #     # ------------------------------------------------------------------
    #     if exclude_short_tracks:
    
    #         # Count valid (non-NaN nuc_area, i.e. nuc_area > 0) frames per cell
    #         track_lengths = (
    #             raw_df[raw_df["nuc_area"] > 0]
    #             .groupby(cell_keys)["frame"]
    #             .nunique()
    #             .reset_index(name="track_length")
    #         )
    
    #         long_enough = track_lengths[
    #             track_lengths["track_length"] >= min_track_length
    #         ][cell_keys]
    
    #         valid_keys = valid_keys.merge(long_enough, on=cell_keys, how="inner")
    
    #         print(f"[filter] After exclude_short_tracks "
    #               f"(min_track_length={min_track_length}): {len(valid_keys)} cells")
    
    #     print(f"[filter] Final valid cells: {len(valid_keys)}")
    
    #     return valid_keys.reset_index(drop=True)
    def _get_valid_cell_keys(
        self,
        conditions=None,
        frame_range=None,
        exclude_dividing=True,
        exclude_late_start=True,
        min_track_start=0,
        exclude_nuc_zero=True,
        exclude_cell_zero=False,
        exclude_short_tracks=True,
        min_track_length=10,
    ):
        """
        Return a DataFrame of (condition, folder, repeat_id, object_id) tuples
        that pass all requested quality filters.
    
        Parameters
        ----------
        conditions : list[str] or None
            Restrict to these conditions. None = all.
        frame_range : tuple(int, int) or None
            (t0_frame, t1_frame) inclusive. Used to define expected window.
        exclude_dividing : bool
            If True, exclude cells where t0 is not NaN (detected division).
        exclude_late_start : bool
            If True, exclude cells whose track starts after min_track_start.
            Catches post-division daughters that were never assigned a t0.
        min_track_start : int
            Maximum allowed first frame for a cell track.
            Only used when exclude_late_start=True.
            Defaults to 0 (track must start at frame 0).
            If frame_range is provided and min_track_start=0, 
            automatically uses frame_range[0].
        exclude_nuc_zero : bool
            If True, exclude cells that have any frame with nuc_area == 0
            within the requested frame_range. These are likely segmentation
            failures at division.
        exclude_cell_zero : bool
            If True, exclude cells that have any frame with cell_area == 0
            within the requested frame_range. Catches tracks where the cell
            body association failed (e.g. nucleus tracked but no overlapping
            cell body mask found). Default False for backward compatibility;
            set True when using the new nucleus-based tracker.
        exclude_short_tracks : bool
            If True, exclude cells whose track (within frame_range) is shorter
            than min_track_length frames.
        min_track_length : int
            Minimum number of valid (non-NaN nuc_area) frames required.
            Only used when exclude_short_tracks=True.
    
        Returns
        -------
        valid_keys : pd.DataFrame
            Columns: condition, folder, repeat_id, object_id
        """
    
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    
        raw_df = self.analysis.df.reset_index()
    
        # ------------------------------------------------------------------
        # Condition filtering
        # ------------------------------------------------------------------
        if conditions is not None and len(conditions) > 0:
            raw_df = raw_df[raw_df["condition"].isin(conditions)]
    
        # ------------------------------------------------------------------
        # Frame range filtering
        # ------------------------------------------------------------------
        if frame_range is not None:
            t0_f, t1_f = frame_range
            raw_df = raw_df[raw_df["frame"].between(t0_f, t1_f)]
    
        # ------------------------------------------------------------------
        # Start with all unique cell keys present in the (filtered) raw_df
        # ------------------------------------------------------------------
        valid_keys = raw_df[cell_keys].drop_duplicates().copy()
    
        # ------------------------------------------------------------------
        # Filter 1: exclude dividing cells (t0 is not NaN)
        # ------------------------------------------------------------------
        if exclude_dividing and self.analysis.events_df is not None:
    
            events = self.analysis.events_df.reset_index()
    
            if conditions is not None and len(conditions) > 0:
                events = events[events["condition"].isin(conditions)]
    
            non_dividing = events[events["t0"].isna()][cell_keys]
    
            valid_keys = valid_keys.merge(non_dividing, on=cell_keys, how="inner")
    
            print(f"[filter] After exclude_dividing: {len(valid_keys)} cells")
    
        # ------------------------------------------------------------------
        # Filter 2: exclude late-starting tracks
        # ------------------------------------------------------------------
        if exclude_late_start:
    
            effective_min_start = min_track_start
            if frame_range is not None and min_track_start == 0:
                effective_min_start = frame_range[0]
    
            first_frames = (
                raw_df.groupby(cell_keys)["frame"]
                .min()
                .reset_index(name="first_frame")
            )
    
            early_enough = first_frames[
                first_frames["first_frame"] <= effective_min_start
            ][cell_keys]
    
            valid_keys = valid_keys.merge(early_enough, on=cell_keys, how="inner")
    
            print(f"[filter] After exclude_late_start "
                  f"(min_track_start={effective_min_start}): {len(valid_keys)} cells")
    
        # ------------------------------------------------------------------
        # Filter 3: exclude cells with any nuc_area == 0 in window
        # ------------------------------------------------------------------
        if exclude_nuc_zero:
    
            cells_with_zero_nuc = (
                raw_df[raw_df["nuc_area"] == 0][cell_keys]
                .drop_duplicates()
            )
    
            merged = valid_keys.merge(
                cells_with_zero_nuc.assign(_bad=True),
                on=cell_keys,
                how="left",
            )
            valid_keys = merged[merged["_bad"].isna()][cell_keys].copy()
    
            print(f"[filter] After exclude_nuc_zero: {len(valid_keys)} cells")
    
        # ------------------------------------------------------------------
        # Filter 3b: exclude cells with any cell_area == 0 in window
        # ------------------------------------------------------------------
        if exclude_cell_zero:
    
            cells_with_zero_cell = (
                raw_df[raw_df["cell_area"] == 0][cell_keys]
                .drop_duplicates()
            )
    
            merged = valid_keys.merge(
                cells_with_zero_cell.assign(_bad=True),
                on=cell_keys,
                how="left",
            )
            valid_keys = merged[merged["_bad"].isna()][cell_keys].copy()
    
            print(f"[filter] After exclude_cell_zero: {len(valid_keys)} cells")
    
        # ------------------------------------------------------------------
        # Filter 4: exclude short tracks
        # ------------------------------------------------------------------
        if exclude_short_tracks:
    
            track_lengths = (
                raw_df[raw_df["nuc_area"] > 0]
                .groupby(cell_keys)["frame"]
                .nunique()
                .reset_index(name="track_length")
            )
    
            long_enough = track_lengths[
                track_lengths["track_length"] >= min_track_length
            ][cell_keys]
    
            valid_keys = valid_keys.merge(long_enough, on=cell_keys, how="inner")
    
            print(f"[filter] After exclude_short_tracks "
                  f"(min_track_length={min_track_length}): {len(valid_keys)} cells")
    
        print(f"[filter] Final valid cells: {len(valid_keys)}")
    
        return valid_keys.reset_index(drop=True)
    
    def plot_heatmap_zscore(
        self,
        var_name,
        conditions=None,
        frame_range=None,
        dividing_only=False,
        non_dividing_only=False,
        # --- new quality filter toggles ---
        exclude_dividing=False,
        exclude_late_start=False,
        min_track_start=0,
        exclude_nuc_zero=False,
        exclude_short_tracks=False,
        min_track_length=10,
        # ----------------------------------
        figsize=None,
        cmap="cividis",
        vmin=-2,
        vmax=2,
        style=None,
        cell_order=None,
        detrend=False,
        smooth_sigma=1.0,
        baseline_window=41,
        baseline_pct=10,
        peak_prominence=0.5,
        min_coverage=1.0,
        file_stem=None,
    ):
        """
        2D heatmap of per-cell z-scored traces.
    
        X-axis : frame (or time, using the configured time axis)
        Y-axis : cells, ordered sequentially by (condition, folder, repeat_id, object_id)
        Color  : z-score relative to each cell's own mean and std
    
        Only valid in 'trackmate' mode.
    
        Parameters
        ----------
        var_name : str
            Variable to plot (supports 'norm_*' prefix).
        conditions : list[str] or None
            Restrict to these conditions. None = all.
        frame_range : tuple(int, int) or None
            (t0_frame, t1_frame) inclusive. None = full movie.
        dividing_only : bool
            If True, include only cells where t0 is not NaN.
        non_dividing_only : bool
            If True, include only cells where t0 is NaN.
        exclude_dividing : bool
            Use _get_valid_cell_keys() to exclude dividing cells.
        exclude_late_start : bool
            Use _get_valid_cell_keys() to exclude late-starting tracks.
        min_track_start : int
            Maximum allowed first frame for a track (used with exclude_late_start).
        exclude_nuc_zero : bool
            Use _get_valid_cell_keys() to exclude cells with any nuc_area == 0.
        exclude_short_tracks : bool
            Use _get_valid_cell_keys() to exclude short tracks.
        min_track_length : int
            Minimum track length in frames (used with exclude_short_tracks).
        figsize : tuple or None
            Figure size. Defaults to (fig_width*4, fig_height*0.2*n_cells).
        cmap : str
            Colormap for the heatmap. Default 'RdBu_r'.
        vmin, vmax : float
            Color scale limits in z-score units.
        style : dict or None
            Axis style dict passed to apply_axes_style().
        """
    
        if self.analysis.mode != "trackmate":
            raise ValueError("plot_heatmap_zscore is only available in 'trackmate' mode.")
    
        if dividing_only and non_dividing_only:
            raise ValueError("dividing_only and non_dividing_only cannot both be True.")
    
        # ------------------------------------------------------------------
        # 1. Get data
        # ------------------------------------------------------------------
        df, ycol = self.get_df_for_var(var_name)
    
        if conditions is not None and len(conditions) > 0:
            df = df[df["condition"].isin(conditions)]
    
        if frame_range is not None:
            t0_f, t1_f = frame_range
            df = df[df["frame"].between(t0_f, t1_f)]
    
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
        supplied_order = cell_order is not None
    
        # ------------------------------------------------------------------
        # 2. Filter by division status if requested (legacy flags)
        # ------------------------------------------------------------------
        if not supplied_order:
            if (dividing_only or non_dividing_only) and self.analysis.events_df is not None:
                events = self.analysis.events_df.reset_index()
                if conditions is not None:
                    events = events[events["condition"].isin(conditions)]
    
                if non_dividing_only:
                    keep = events[events["t0"].isna()]
                else:
                    keep = events[events["t0"].notna()]
    
                keep_keys = keep.set_index(cell_keys).index
                df_keys   = df.set_index(cell_keys).index
                df = df[df_keys.isin(keep_keys)]
    
        # ------------------------------------------------------------------
        # 2b. Apply new quality filters via shared helper
        # ------------------------------------------------------------------
        if not supplied_order and any([
            exclude_dividing,
            exclude_late_start,
            exclude_nuc_zero,
            exclude_short_tracks,
        ]):
            valid_keys = self._get_valid_cell_keys(
                conditions=conditions,
                frame_range=frame_range,
                exclude_dividing=exclude_dividing,
                exclude_late_start=exclude_late_start,
                min_track_start=min_track_start,
                exclude_nuc_zero=exclude_nuc_zero,
                exclude_short_tracks=exclude_short_tracks,
                min_track_length=min_track_length,
            )
    
            df = df.merge(valid_keys, on=cell_keys, how="inner")
    
        # ------------------------------------------------------------------
        # 3. Mask frames where nuc_area == 0
        # ------------------------------------------------------------------
        raw_df = self.analysis.df.reset_index()
        zero_nuc = raw_df[raw_df["nuc_area"] == 0][
            ["condition", "folder", "repeat_id", "object_id", "frame"]
        ]
        df = df.merge(
            zero_nuc.assign(_bad=True),
            on=["condition", "folder", "repeat_id", "object_id", "frame"],
            how="left",
        )
        df.loc[df["_bad"] == True, ycol] = np.nan
        df = df.drop(columns="_bad")
    
        # ------------------------------------------------------------------
        # 3b. Filter cells by track coverage
        # ------------------------------------------------------------------
        if not supplied_order and min_coverage > 0.0:
            all_frames = self.analysis.df.reset_index()["frame"]
            if frame_range is not None:
                expected_frames = frame_range[1] - frame_range[0] + 1
            else:
                expected_frames = all_frames.max() - all_frames.min() + 1
    
            coverage = (
                df.groupby(cell_keys)[ycol]
                  .apply(lambda x: x.notna().sum())
                  .reset_index()
                  .rename(columns={ycol: "n_valid"})
            )
            coverage["coverage"] = coverage["n_valid"] / expected_frames
            sufficient = coverage[coverage["coverage"] >= min_coverage][cell_keys]
    
            df = df.merge(sufficient, on=cell_keys, how="inner")
    
        # ------------------------------------------------------------------
        # 4. Z-score per cell
        # ------------------------------------------------------------------
        if detrend:
            chunks = []
            for keys, group in df.groupby(cell_keys):
                group = group.copy()
                group[ycol] = self.detrend_and_smooth(
                    group[ycol].values, smooth_sigma, baseline_window, baseline_pct)
                chunks.append(group)
            df = pd.concat(chunks, ignore_index=True)
    
        stats = (
            df.groupby(cell_keys)[ycol]
              .agg(cell_mean="mean", cell_std="std")
              .reset_index()
        )
        df = df.merge(stats, on=cell_keys)
        df["zscore"] = (df[ycol] - df["cell_mean"]) / (df["cell_std"] + 1e-10)
    
        # ------------------------------------------------------------------
        # 5. Build 2D matrix
        # ------------------------------------------------------------------
        all_frames = self.analysis.df.reset_index()["frame"]
        if frame_range is not None:
            frame_vals = np.arange(frame_range[0], frame_range[1] + 1)
        else:
            frame_vals = np.arange(all_frames.min(), all_frames.max() + 1)
    
        frame_to_col = {f: i for i, f in enumerate(frame_vals)}
    
        if supplied_order:
            cell_order = (
                cell_order
                .merge(df[cell_keys].drop_duplicates(), on=cell_keys, how="inner")
                .reset_index(drop=True)
            )
        else:
            cell_order = (
                df[cell_keys]
                .drop_duplicates()
                .sort_values(cell_keys)
                .reset_index(drop=True)
            )
    
        cell_order["cell_idx"] = cell_order.index
        df = df.merge(cell_order, on=cell_keys)
    
        n_cells  = len(cell_order)
        n_frames = len(frame_vals)
        matrix   = np.full((n_cells, n_frames), np.nan)
    
        for _, row in df.iterrows():
            col = frame_to_col.get(row["frame"])
            if col is not None:
                matrix[int(row["cell_idx"]), col] = row["zscore"]
    
        # Remove entirely empty rows
        non_empty  = ~np.all(np.isnan(matrix), axis=1)
        matrix     = matrix[non_empty]
        cell_order = cell_order[non_empty].reset_index(drop=True)
    
        # Sort by first peak location
        if not supplied_order:
            from scipy.signal import find_peaks
    
            def _first_peak_idx(row, prominence=0.5):
                nan_mask = np.isnan(row)
                if nan_mask.all():
                    return n_frames
                x      = np.arange(len(row))
                interp = np.interp(x, x[~nan_mask], row[~nan_mask])
                peaks, _ = find_peaks(interp, prominence=prominence)
                if len(peaks) == 0:
                    return n_frames
                return peaks[0]
    
            first_peak_indices = np.array([
                _first_peak_idx(matrix[i], prominence=peak_prominence)
                for i in range(len(matrix))
            ])
            sort_order = np.argsort(first_peak_indices, kind="stable")
            matrix     = matrix[sort_order]
            cell_order = cell_order.iloc[sort_order].reset_index(drop=True)
    
        n_cells = len(cell_order)
    
        # ------------------------------------------------------------------
        # 6. Plot
        # ------------------------------------------------------------------
        if figsize is None:
            row_height = 0.01
            figsize = (self.fig_width * 4, max(1.5, row_height * n_cells))
    
        fig, ax = plt.subplots(figsize=figsize)
    
        time_vals = (frame_vals - self.frame_zero) * self.dt + self.t_zero
    
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            extent=[time_vals[0], time_vals[-1], n_cells - 0.5, -0.5],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
        )
    
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f"z-score  ({var_name})")
    
        ax.set_xlabel(self.time_col)
        ax.set_ylabel("cell index")
        ax.set_title(var_name)
    
        self.apply_axes_style(ax, style)
    
        sns.despine(ax=ax)
        plt.tight_layout()

        # ------------------------------------------------------------------
        # 7. Save bundle
        # ------------------------------------------------------------------
        if self.save_outputs:
            rows = []

            for cell_idx, cell_row in cell_order.iterrows():
                for frame_idx, frame in enumerate(frame_vals):
                    rows.append({
                        "condition":    cell_row["condition"],
                        "folder":       cell_row["folder"],
                        "repeat_id":    cell_row["repeat_id"],
                        "object_id":    cell_row["object_id"],
                        "heatmap_row":  cell_row["cell_idx"],
                        "frame":        frame,
                        "time":         time_vals[frame_idx],
                        "zscore":       matrix[cell_row["cell_idx"], frame_idx],
                    })
            points_df = pd.DataFrame(rows)

            self.save_plot_bundle(
                fig=fig,
                plot_name="heatmap_zscore",
                var_tag=var_name,
                points_df=points_df,
                file_stem=file_stem,
            )

        plt.show()

        return {
            "matrix":     matrix,
            "cell_order": cell_order,
            "frame_vals": frame_vals,
            "time_vals":  time_vals,
        }
        plt.show()
    
        # return {
        #     "matrix":     matrix,
        #     "cell_order": cell_order,
        #     "frame_vals": frame_vals,
        #     "time_vals":  time_vals,
        # }

    def plot_heatmap(
        self,
        var,
        df_indexed,
        figsize=None,
        cmap="cividis",
        vmin=-2,
        vmax=2,
        peak_prominence=0.5,
        cell_order=None,
        style=None,
        file_stem=None,
    ):
        """
        2D heatmap of per-cell z-scored traces, built from a pre-processed
        df_indexed produced by build_signal_df().

        Unlike plot_heatmap_zscore(), this method does NO data loading,
        filtering, detrending, or interpolation — all of that is expected
        to have been done already by build_signal_df(). This guarantees
        the heatmap always shows exactly the same cells and signal as the
        rest of your analysis.

        Parameters
        ----------
        var        : str   — column name in df_indexed to plot
        df_indexed : DataFrame — MultiIndex (condition, folder, repeat_id,
                     object_id, frame) as produced by build_signal_df()
        figsize    : tuple or None
        cmap       : colormap string, default 'cividis'
        vmin, vmax : float — z-score color scale limits
        peak_prominence : float — prominence for first-peak sort order
        cell_order : DataFrame or None — pre-computed cell ordering with
                     columns (condition, folder, repeat_id, object_id);
                     if None, cells are sorted by first peak location
        style      : dict or None — passed to apply_axes_style()
        file_stem  : str or None — passed to save_plot_bundle()

        Returns
        -------
        dict with keys: matrix, cell_order, frame_vals, time_vals
        """

        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
        df = df_indexed.reset_index()

        if var not in df.columns:
            raise ValueError(f"Column '{var}' not found in df_indexed. "
                             f"Available: {df.columns.tolist()}")

        # ── Frame axis ───────────────────────────────────────────────────
        frame_vals = np.sort(df["frame"].unique())

        # ── Z-score per cell ─────────────────────────────────────────────
        stats = (
            df.groupby(cell_keys)[var]
              .agg(cell_mean="mean", cell_std="std")
              .reset_index()
        )
        df = df.merge(stats, on=cell_keys)
        df["zscore"] = (df[var] - df["cell_mean"]) / (df["cell_std"] + 1e-10)

        # ── Cell ordering ─────────────────────────────────────────────────
        supplied_order = cell_order is not None

        if supplied_order:
            cell_order = (
                cell_order
                .merge(df[cell_keys].drop_duplicates(), on=cell_keys, how="inner")
                .reset_index(drop=True)
            )
        else:
            cell_order = (
                df[cell_keys]
                .drop_duplicates()
                .sort_values(cell_keys)
                .reset_index(drop=True)
            )

        cell_order = cell_order.copy()
        cell_order["cell_idx"] = cell_order.index
        df = df.merge(cell_order, on=cell_keys)

        # ── Build matrix ──────────────────────────────────────────────────
        n_cells  = len(cell_order)
        n_frames = len(frame_vals)
        frame_to_col = {f: i for i, f in enumerate(frame_vals)}

        matrix = np.full((n_cells, n_frames), np.nan)
        for _, row in df.iterrows():
            col = frame_to_col.get(row["frame"])
            if col is not None:
                matrix[int(row["cell_idx"]), col] = row["zscore"]

        # Remove entirely empty rows
        non_empty  = ~np.all(np.isnan(matrix), axis=1)
        matrix     = matrix[non_empty]
        cell_order = cell_order[non_empty].reset_index(drop=True)

        # ── Sort by first peak ────────────────────────────────────────────
        if not supplied_order:
            from scipy.signal import find_peaks

            def _first_peak_idx(row):
                nan_mask = np.isnan(row)
                if nan_mask.all():
                    return n_frames
                x      = np.arange(len(row))
                interp = np.interp(x, x[~nan_mask], row[~nan_mask])
                peaks, _ = find_peaks(interp, prominence=peak_prominence)
                return peaks[0] if len(peaks) > 0 else n_frames

            sort_order = np.argsort(
                [_first_peak_idx(matrix[i]) for i in range(len(matrix))],
                kind="stable"
            )
            matrix     = matrix[sort_order]
            cell_order = cell_order.iloc[sort_order].reset_index(drop=True)

        n_cells = len(cell_order)

        # ── Plot ──────────────────────────────────────────────────────────
        if figsize is None:
            figsize = (self.fig_width * 4, max(1.5, 0.01 * n_cells))

        fig, ax = plt.subplots(figsize=figsize)
        time_vals = (frame_vals - self.frame_zero) * self.dt + self.t_zero

        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            extent=[time_vals[0], time_vals[-1], n_cells - 0.5, -0.5],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
        )

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f"z-score  ({var})")
        ax.set_xlabel(self.time_col)
        ax.set_ylabel("cell index")
        ax.set_title(var)

        self.apply_axes_style(ax, style)
        sns.despine(ax=ax)
        plt.tight_layout()

        # ── Save ──────────────────────────────────────────────────────────
        if self.save_outputs:
            rows = []
            for cell_idx, cell_row in cell_order.iterrows():
                for frame_idx, frame in enumerate(frame_vals):
                    rows.append({
                        "condition":   cell_row["condition"],
                        "folder":      cell_row["folder"],
                        "repeat_id":   cell_row["repeat_id"],
                        "object_id":   cell_row["object_id"],
                        "heatmap_row": cell_row["cell_idx"],
                        "frame":       frame,
                        "time":        time_vals[frame_idx],
                        "zscore":      matrix[cell_row["cell_idx"], frame_idx],
                    })
            self.save_plot_bundle(
                fig=fig,
                plot_name="heatmap",
                var_tag=var,
                points_df=pd.DataFrame(rows),
                file_stem=file_stem,
            )

        plt.show()

        return {
            "matrix":     matrix,
            "cell_order": cell_order,
            "frame_vals": frame_vals,
            "time_vals":  time_vals,
        }
    
    def plot_single_cell_dual(
        self,
        var_name_left,
        var_name_right,
        conditions=None,
        frame_range=None,
        style_left=None,
        style_right=None,
        seed=None,
        detrend=False,
        smooth_sigma=1.0,
        baseline_window=41,
        baseline_pct=10,
        # --- new quality filter toggles ---
        exclude_dividing=True,
        exclude_late_start=True,
        min_track_start=0,
        exclude_nuc_zero=True,
        exclude_short_tracks=True,
        min_track_length=10,
    ):
        """
        Exploratory plot: pick one random cell (filtered by quality criteria)
        and plot two variables against each other on a shared time axis
        using twin y-axes.
    
        Only valid in 'trackmate' mode (requires self.analysis.events_df).
    
        Parameters
        ----------
        var_name_left : str
            Variable to plot on the left y-axis (supports 'norm_*' prefix).
        var_name_right : str
            Variable to plot on the right y-axis (supports 'norm_*' prefix).
        conditions : list[str] or None
            Restrict cell selection to these conditions. None = all.
        frame_range : tuple(int, int) or None
            (t0_frame, t1_frame) inclusive range to display. None = all frames.
        style_left : dict or None
            Axis-style dict for the left axis (ylim, yticks, ylabel, …).
        style_right : dict or None
            Axis-style dict for the right axis.
        seed : int or None
            Random seed for reproducible cell selection.
        exclude_dividing : bool
            Exclude cells with a detected division event (t0 not NaN).
        exclude_late_start : bool
            Exclude tracks that start after min_track_start.
        min_track_start : int
            Maximum allowed first frame for a track.
        exclude_nuc_zero : bool
            Exclude cells with any nuc_area == 0 in the window.
        exclude_short_tracks : bool
            Exclude tracks shorter than min_track_length.
        min_track_length : int
            Minimum number of valid frames required.
        """
    
        if self.analysis.mode != "trackmate":
            raise ValueError("plot_single_cell_dual is only available in 'trackmate' mode.")
    
        if self.analysis.events_df is None:
            raise ValueError("No events_df found. Run process_images() first.")
    
        # ------------------------------------------------------------------
        # 1. Get valid (non-dividing / quality-filtered) cells
        # ------------------------------------------------------------------
        valid_keys = self._get_valid_cell_keys(
            conditions=conditions,
            frame_range=frame_range,
            exclude_dividing=exclude_dividing,
            exclude_late_start=exclude_late_start,
            min_track_start=min_track_start,
            exclude_nuc_zero=exclude_nuc_zero,
            exclude_short_tracks=exclude_short_tracks,
            min_track_length=min_track_length,
        )
    
        if valid_keys.empty:
            raise ValueError(
                "No cells passed all quality filters for the given conditions."
            )
    
        # ------------------------------------------------------------------
        # 2. Pick a random cell from the valid set
        # ------------------------------------------------------------------
        rng  = np.random.default_rng(seed)
        cell = valid_keys.sample(n=1, random_state=rng.integers(0, 2**31)).iloc[0]
    
        condition = cell["condition"]
        folder    = cell["folder"]
        repeat_id = cell["repeat_id"]
        object_id = cell["object_id"]
    
        print(
            f"Selected cell — condition: '{condition}', folder: {folder}, "
            f"repeat_id: {repeat_id}, object_id: {object_id}"
        )
    
        # ------------------------------------------------------------------
        # 3. Extract the time-series for both variables
        # ------------------------------------------------------------------
        def _get_cell_series(var_name):
            df, ycol = self.get_df_for_var(var_name)
    
            mask = (
                (df["condition"]  == condition)  &
                (df["folder"]     == folder)     &
                (df["repeat_id"]  == repeat_id)  &
                (df["object_id"]  == object_id)
            )
            series = df.loc[mask, ["frame", ycol]].copy()
    
            # Mask frames where nuc_area is zero
            raw_df = self.analysis.df.reset_index()
            nuc_mask = (
                (raw_df["condition"]  == condition)  &
                (raw_df["folder"]     == folder)     &
                (raw_df["repeat_id"]  == repeat_id)  &
                (raw_df["object_id"]  == object_id)
            )
            invalid_frames = raw_df.loc[
                nuc_mask & (raw_df["nuc_area"] == 0), "frame"
            ]
            series.loc[series["frame"].isin(invalid_frames), ycol] = np.nan
    
            if frame_range is not None:
                t0_f, t1_f = frame_range
                series = series[series["frame"].between(t0_f, t1_f)]
    
            series = self.add_time_column(series)
            return series, ycol
    
        series_left,  ycol_left  = _get_cell_series(var_name_left)
        series_right, ycol_right = _get_cell_series(var_name_right)
    
        if detrend:
            series_left[ycol_left]   = self.detrend_and_smooth(
                series_left[ycol_left].values,
                smooth_sigma, baseline_window, baseline_pct,
            )
            series_right[ycol_right] = self.detrend_and_smooth(
                series_right[ycol_right].values,
                smooth_sigma, baseline_window, baseline_pct,
            )
    
        if series_left.empty or series_right.empty:
            raise ValueError(
                "No data found for the selected cell in the given frame_range."
            )
    
        # ------------------------------------------------------------------
        # 4. Plot
        # ------------------------------------------------------------------
        fig, ax_left = plt.subplots(
            figsize=(self.fig_width * 3, self.fig_height * 2)
        )
       # ax_left.set_ylim([0.2, 1.2])
        ax_right = ax_left.twinx()
      #  ax_right.set_ylim([0.5, 1.5])
    
        color_left  = "#1f77b4"
        color_right = "#d62728"
    
        ax_left.plot(
            series_left[self.time_col],
            series_left[ycol_left],
            color=color_left,
            linewidth=1.2,
            label=var_name_left,
        )
    
        ax_right.plot(
            series_right[self.time_col],
            series_right[ycol_right],
            color=color_right,
            linewidth=1.2,
            label=var_name_right,
        )
    
        # X-axis limits
        all_frames = self.analysis.df.reset_index()["frame"]
        if frame_range is not None:
            ax_left.set_xlim(
                (frame_range[0] - self.frame_zero) * self.dt + self.t_zero,
                (frame_range[1] - self.frame_zero) * self.dt + self.t_zero,
            )
        else:
            ax_left.set_xlim(
                (all_frames.min() - self.frame_zero) * self.dt + self.t_zero,
                (all_frames.max() - self.frame_zero) * self.dt + self.t_zero,
            )
    
        ax_left.set_xlabel(self.time_col)
        ax_left.set_ylabel(
            style_left.get("ylabel", var_name_left) if style_left else var_name_left,
            color=color_left,
        )
        ax_right.set_ylabel(
            style_right.get("ylabel", var_name_right) if style_right else var_name_right,
            color=color_right,
        )
    
        ax_left.tick_params(axis="y",  colors=color_left)
        ax_right.tick_params(axis="y", colors=color_right)
    
        self.apply_axes_style(ax_left,  style_left)
        self.apply_axes_style(ax_right, style_right)
    
        handles = [
            mlines.Line2D([], [], color=color_left,  linewidth=1.2,
                          linestyle="-",  label=var_name_left),
            mlines.Line2D([], [], color=color_right, linewidth=1.2,
                          linestyle="--", label=var_name_right),
        ]
        ax_left.legend(handles=handles, loc="upper left", frameon=False)
    
        title = (
            f"cell {object_id}  |  {condition}  "
            f"|  folder {folder}  |  rep {repeat_id}"
        )
        ax_left.set_title(title)
    
        sns.despine(ax=ax_left,  right=False)
        sns.despine(ax=ax_right, left=False)
    
        plt.tight_layout()
        plt.show()
    
        return {
            "condition": condition,
            "folder":    folder,
            "repeat_id": repeat_id,
            "object_id": object_id,
            "series_left":  series_left,
            "series_right": series_right,
        }

  
 
    def find_peaks_per_cell(
        self,
        var_name,
        df_indexed=None,   # <-- if provided, use this instead of self.analysis.df
        conditions=None,
        frame_range=None,
        prominence=0.1,
        relative_prominence=False,
        distance=10,
        height=None,
        width=2,
        detrend=False,
        smooth_sigma=1.0,
        baseline_window=50,
        baseline_pct=10,
        exclude_first_n_frames=0,
    ):
        """
        Detect peaks in a variable for every tracked cell, returning a
        DataFrame of peak events in the same style as the rest of the library.
    
        Only valid in 'trackmate' mode.
    
        Parameters
        ----------
        var_name : str
            Variable to search for peaks (supports 'norm_*' prefix).
        conditions : list[str] or None
            Restrict to these conditions. None = all.
        frame_range : tuple(int, int) or None
            (t0_frame, t1_frame) inclusive. None = full movie.
        prominence : float
            Minimum peak prominence passed to scipy.signal.find_peaks.
        distance : int
            Minimum number of frames between successive peaks.
        height : float or None
            Minimum peak height. None = no height threshold.
        detrend : bool
            Apply detrend_and_smooth() before peak detection.
        smooth_sigma : float
            Gaussian smooth sigma in frames (used when detrend=True).
        baseline_window : int
            Rolling baseline window in frames (used when detrend=True).
        baseline_pct : float
            Percentile for baseline estimation (used when detrend=True).
        exclude_first_n_frames : int
            Discard peaks whose peak_frame is earlier than this value.
    
        Returns
        -------
        peaks_df : DataFrame
            One row per detected peak with columns:
              condition, folder, repeat_id, object_id,
              peak_frame, peak_value, prominence
        """
        from scipy.signal import find_peaks as _find_peaks
    
        if self.analysis.mode != "trackmate":
            raise ValueError("find_peaks_per_cell is only available in 'trackmate' mode.")
    
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    
        if df_indexed is not None:
            # pull the variable column out of the pre-sanitised df
            df = df_indexed[[var_name]].reset_index()
            ycol = var_name
        else:
            df, ycol = self.get_df_for_var(var_name)
    
        if conditions is not None and len(conditions) > 0:
            df = df[df["condition"].isin(conditions)]
    
        if frame_range is not None:
            t0_f, t1_f = frame_range
            df = df[df["frame"].between(t0_f, t1_f)]
    
        rows = []
    
        for keys, group in df.groupby(cell_keys):
            condition, folder, repeat_id, object_id = keys
            group = group.sort_values("frame")
    
            frames = group["frame"].values
            values = group[ycol].values.astype(float)

            # NaN-aware: interpolate gaps so find_peaks sees a clean signal
            nan_mask = np.isnan(values)
            if nan_mask.all():
                continue
            if nan_mask.any():
                x = np.arange(len(values))
                values = np.interp(x, x[~nan_mask], values[~nan_mask])

            # Scale prominence to cell's own signal range
            if relative_prominence:
                signal_range = np.nanmax(values) - np.nanmin(values)
                if signal_range == 0:
                    continue
                _prominence = prominence * signal_range
            else:
                _prominence = prominence

            find_kwargs = dict(prominence=_prominence, distance=distance, width=width)
    
            # Optionally detrend/smooth before peak detection
            if detrend:
                values = self.detrend_and_smooth(
                    values, smooth_sigma, baseline_window, baseline_pct,
                    smooth_only=False,
                )
    

    
            # find_kwargs = dict(prominence=prominence, distance=distance)
            if height is not None:
                find_kwargs["height"] = height
    
            peak_indices, props = _find_peaks(values, **find_kwargs)
    
            for i, pidx in enumerate(peak_indices):
                pframe = int(frames[pidx])
    
                if pframe < exclude_first_n_frames:
                    continue
    
                rows.append({
                    "condition":  condition,
                    "folder":     folder,
                    "repeat_id":  repeat_id,
                    "object_id":  object_id,
                    "peak_frame": pframe,
                    "peak_value": float(values[pidx]),
                    "prominence": float(props["prominences"][i]),
                })
    
        peaks_df = pd.DataFrame(rows)
        print(f"[find_peaks_per_cell] {len(peaks_df)} peaks found across "
              f"{peaks_df[cell_keys].drop_duplicates().shape[0] if len(peaks_df) > 0 else 0} cells.")

        if len(peaks_df) > 0:
            peaks_df = peaks_df.sort_values(cell_keys + ["peak_frame"])
            peaks_df["pulse_number"] = peaks_df.groupby(cell_keys).cumcount() + 1
            
        return peaks_df
        

    def _interpolate_gaps(
        self,
        t: np.ndarray,
        y: np.ndarray,
        threshold: float = 7.0,
        n_fill: int = 20,
        n_context: int = 10,
        verbose: bool = True,
    ):
        """
        Detect and fill gaps in a time series using PCHIP interpolation.
     
        Parameters
        ----------
        t : array-like
            Time points (must be sorted ascending, no duplicates).
        y : array-like
            Signal values corresponding to t.
        threshold : float
            Minimum gap size (in t units) to trigger interpolation.
            Default 7.0 — tune to your sampling interval.
        n_fill : int
            Number of interpolated points inserted inside each gap.
        n_context : int
            Number of real points on each side of a gap used to fit
            the PCHIP spline.
        verbose : bool
            Print a summary of detected gaps.
     
        Returns
        -------
        t_out : np.ndarray
            Time points of the filled series (sorted).
        y_out : np.ndarray
            Signal values of the filled series.
        is_interp : np.ndarray of bool
            True where a value was interpolated, False where it is original.
     
        Example
        -------
        >>> t_out, y_out, mask = interpolate_gaps(t, y, threshold=7, n_fill=20)
        >>> # plot original
        >>> plt.plot(t_out[~mask], y_out[~mask], 'b-', label='measured')
        >>> # plot interpolated sections
        >>> plt.plot(t_out[mask], y_out[mask], 'b--', alpha=0.5, label='PCHIP')
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
     
        gaps = []
        diffs = np.diff(t)
        for i, d in enumerate(diffs):
            if d > threshold:
                gaps.append((i, t[i], t[i + 1], d))
     
        if verbose:
            if gaps:
                print(f"interpolate_gaps: {len(gaps)} gap(s) found (threshold={threshold}):")
                for _, t0, t1, w in gaps:
                    print(f"  t={t0:.2f} → {t1:.2f}  width={w:.2f}")
            else:
                print(f"interpolate_gaps: no gaps found (threshold={threshold}).")
     
        if not gaps:
            return t.copy(), y.copy(), np.zeros(len(t), dtype=bool)
     
        t_extra, y_extra = [], []
     
        for idx, t_start, t_end, _ in gaps:
            left  = max(0, idx - n_context + 1)
            right = min(len(t) - 1, idx + n_context)
            t_ctx = np.concatenate([t[left : idx + 1], t[idx + 1 : right + 1]])
            y_ctx = np.concatenate([y[left : idx + 1], y[idx + 1 : right + 1]])
     
            pchip  = PchipInterpolator(t_ctx, y_ctx)
            t_fill = np.linspace(t_start, t_end, n_fill + 2)[1:-1]
            y_fill = pchip(t_fill)
     
            t_extra.append(t_fill)
            y_extra.append(y_fill)
     
        t_fill_all = np.concatenate(t_extra)
        y_fill_all = np.concatenate(y_extra)
     
        t_out     = np.concatenate([t, t_fill_all])
        y_out     = np.concatenate([y, y_fill_all])
        is_interp = np.concatenate([np.zeros(len(t), bool),
                                    np.ones(len(t_fill_all), bool)])
     
        order     = np.argsort(t_out)
        return t_out[order], y_out[order], is_interp[order]
        

    def build_signal_df(
        self,
        vars, # e.g. ["FR_nuc_mem_conc_ratio", "iRFP_cr_nuc_conc_ratio"]
        var_params=None,
        invert_vars=None, # e.g. ["iRFP_cr_nuc_conc_ratio"] — replaces invert_b
        conditions=None,
        exclude_dividing=False,
        frame_range=None,
        detrend=False,
        detrend_mode="percentile",
        poly_degree=2,
        smooth_only=False,
        smooth_sigma=1.0,
        baseline_window=50,
        baseline_pct=10,
        outlier_clip=None,
        min_coverage=0.5,
        interpolate_gaps=False, 
        gap_threshold=7.0,
        gap_n_fill=20,
        gap_n_context=10,
        min_track_start=None,

    ):
        """
        Load, optionally detrend, merge, and sanitise two variables into a
        single MultiIndex dataframe ready for window extraction.
    
        Returns
        -------
        df_indexed : DataFrame
            Indexed by (condition, folder, repeat_id, object_id, frame).
        """
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]

        if invert_vars is None:
            invert_vars = []
    
        signal_dfs = []
        for var in vars:
            df_var, ycol = self.get_df_for_var(var)

            # Resolve per-variable params, falling back to global defaults
            vp = var_params.get(var, {}) if var_params is not None else {}
            _smooth_sigma    = vp.get("smooth_sigma",    smooth_sigma)
            _smooth_only     = vp.get("smooth_only",     smooth_only)
            _baseline_window = vp.get("baseline_window", baseline_window)
            _baseline_pct    = vp.get("baseline_pct",    baseline_pct)
            _detrend_mode    = vp.get("detrend_mode",    detrend_mode)
            _poly_degree     = vp.get("poly_degree",     poly_degree)
            _detrend         = vp.get("detrend",         detrend)
    
            if conditions is not None:
                df_var = df_var[df_var["condition"].isin(conditions)]
            if frame_range is not None:
                t0_f, t1_f = frame_range
                df_var = df_var[df_var["frame"].between(t0_f, t1_f)]

            # Exclude dividing cells
            if exclude_dividing and self.analysis.events_df is not None:
                events = self.analysis.events_df.reset_index()
                non_dividing = events[events["t0"].isna()][cell_keys]
                df_var = df_var.merge(non_dividing, on=cell_keys, how="inner")

            # Exclude late-starting tracks
            if min_track_start is not None:
                first_frames = (
                    df_var.groupby(cell_keys)["frame"]
                    .min()
                    .reset_index(name="first_frame")
                )
                early_enough = first_frames[
                    first_frames["first_frame"] <= min_track_start
                ][cell_keys]
                df_var = df_var.merge(early_enough, on=cell_keys, how="inner")

            # Clip outliers from bad ratios
            if outlier_clip is not None:
                df_var.loc[df_var[ycol].abs() > outlier_clip, ycol] = np.nan
            # Masks zeros as nan
            df_var.loc[df_var[ycol] == 0.0, ycol] = np.nan

            if "cell_area" in df_var.columns:
                df_var.loc[df_var["cell_area"] == 0, ycol] = np.nan

            if min_coverage is not None:
                coverage = df_var.groupby(cell_keys)[ycol].transform(
                    lambda x: x.notna().mean()
                )
                n_before = df_var[cell_keys].drop_duplicates().shape[0]
                df_var = df_var[coverage >= min_coverage]
                n_after = df_var[cell_keys].drop_duplicates().shape[0]
                print(f"[build_signal_df] {var}: {n_after}/{n_before} cells pass coverage>={min_coverage}")

            if interpolate_gaps:
                chunks = []
                for keys, group in df_var.groupby(cell_keys):
                    group = group.copy().sort_values("frame")
                    t = group["frame"].values.astype(float)
                    y = group[ycol].values.astype(float)
            
                    valid = ~np.isnan(y)
                    if valid.sum() < 2:
                        chunks.append(group)
                        continue
            
                    # Only interpolate within the active window
                    first_valid = np.where(valid)[0][0]
                    last_valid  = np.where(valid)[0][-1]
                    active_mask = np.zeros(len(t), dtype=bool)
                    active_mask[first_valid:last_valid + 1] = True
            
                    t_active = t[active_mask]
                    y_active = y[active_mask]
            
                    # Among active frames, pass real values only to interpolate_gaps
                    valid_active = ~np.isnan(y_active)

                    t_valid = t_active[valid_active]
                    y_valid = y_active[valid_active]
                    
                    # Deduplicate — keep first occurrence of any duplicate frame
                    _, unique_idx = np.unique(t_valid, return_index=True)
                    t_valid = t_valid[unique_idx]
                    y_valid = y_valid[unique_idx]

                    if len(t_valid) < 2:
                        chunks.append(group)
                        continue
                    
                    t_out, y_out, _ = self._interpolate_gaps(
                        t_valid, y_valid,
                        threshold=gap_threshold,
                        n_fill=gap_n_fill,
                        n_context=gap_n_context,
                        verbose=False,
                    )
                    
                    # if valid_active.sum() < 2:
                    #     chunks.append(group)
                    #     continue
            
                    # t_out, y_out, _ = self._interpolate_gaps(
                    #     t_active[valid_active], y_active[valid_active],
                    #     threshold=gap_threshold,
                    #     n_fill=gap_n_fill,
                    #     n_context=gap_n_context,
                    #     verbose=False,
                    # )
            
                    # Rebuild df: interpolated active window + NaN for inactive frames
                    interp_df = pd.DataFrame({
                        "frame": t_out.astype(int),
                        ycol:    y_out,
                    })
                    for k, v in zip(cell_keys, keys):
                        interp_df[k] = v
            
                    # Add back the inactive frames as NaN
                    inactive_frames = t[~active_mask].astype(int)
                    if len(inactive_frames) > 0:
                        nan_df = pd.DataFrame({
                            "frame": inactive_frames,
                            ycol:    np.nan,
                        })
                        for k, v in zip(cell_keys, keys):
                            nan_df[k] = v
                        interp_df = pd.concat([interp_df, nan_df], ignore_index=True)
            
                    chunks.append(interp_df.sort_values("frame"))
            
                df_var = pd.concat(chunks, ignore_index=True)
        
            if _detrend:
                chunks = []
                for keys, group in df_var.groupby(cell_keys):
                    group = group.copy()
                    group[ycol] = self.detrend_and_smooth(
                        group[ycol].values, _smooth_sigma, _baseline_window,
                        _baseline_pct, smooth_only=_smooth_only, detrend_mode=_detrend_mode, 
                        poly_degree=_poly_degree,
                    )
                    chunks.append(group)
                df_var = pd.concat(chunks, ignore_index=True)

            if var in invert_vars:
                df_var[ycol] = -df_var[ycol]
            
            signal_dfs.append(df_var[cell_keys + ["frame", ycol]])
    
        df_merged = reduce(
            lambda left, right: left.merge(right, on=cell_keys + ["frame"], how="outer"),
            signal_dfs
        )
        df_merged = df_merged.groupby(cell_keys + ["frame"], as_index=False).mean()
        df_indexed = df_merged.set_index(cell_keys + ["frame"]).sort_index()
    
        print(f"[build_signal_df] shape={df_indexed.shape}")
        print(df_indexed[vars].describe().round(4))
    
        return df_indexed
    
    
    def extract_eta_windows(
        self,
        df_indexed,
        peaks_df,
        extract_vars,
        pre=20,
        post=40,
        exclude_first_n_frames=0,
        skip_first_peak=False,
        require_full_window=True,
        pre_std_threshold=None,
    ):
        """
        Extract aligned windows around each peak event.
    
        Parameters
        ----------
        df_indexed : DataFrame
            Output of build_signal_df() — indexed by cell_keys + frame.
        peaks_df : DataFrame
            Output of find_peaks_per_cell().
        extract_vars : list[str]
            Variables to extract (must be columns in df_indexed).
        pre, post : int
            Frames before/after each trigger peak.
        exclude_first_n_frames : int
            Skip peaks before this frame.
        require_full_window : bool
            Discard events with missing frames or NaNs in the post window.
        pre_std_threshold : float or None
            If set, discard events where any variable's pre-window std
            exceeds this value. Useful for keeping only isolated pulses.
    
        Returns
        -------
        cond_results : dict
            Keyed by condition. Each value:
              "windows"  : {var_name: (n_events, win_len) array}
              "n_events" : int
              "rel_time" : 1D array of frame offsets
        """
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
        rel_time = np.arange(-pre, post + 1)
        win_len  = len(rel_time)
    
        conditions = peaks_df["condition"].unique().tolist()
        cond_results = {}
    
        for cond in conditions:
            cond_peaks = peaks_df[peaks_df["condition"] == cond]

            # Build first-peak lookup per cell
            if skip_first_peak:
                first_peak_lookup = (
                    cond_peaks.groupby(cell_keys)["peak_frame"]
                    .min()
                )
            
            aligned = {v: [] for v in extract_vars}
            n_rejected_window  = 0
            n_rejected_std     = 0
    
            for _, ev in cond_peaks.iterrows():
                if int(ev["peak_frame"]) < exclude_first_n_frames:
                    continue
    
                cell_key = (
                    ev["condition"],
                    ev["folder"],
                    ev["repeat_id"],
                    np.int64(ev["object_id"]),
                )
                t0     = int(ev["peak_frame"])

                                # skip the first peak per cell if requested
                if skip_first_peak:
                    key = (ev["condition"], ev["folder"], ev["repeat_id"], np.int64(ev["object_id"]))
                    if t0 == first_peak_lookup.get(key, None):
                        continue

                
                frames = t0 + rel_time
    
                idx = pd.MultiIndex.from_tuples(
                    [cell_key + (f,) for f in frames],
                    names=cell_keys + ["frame"],
                )
    
                try:
                    win = df_indexed.reindex(idx)
                except Exception:
                    continue
    
                if require_full_window:
                    if win.shape[0] != win_len:
                        n_rejected_window += 1
                        continue
                    if win.iloc[pre:][list(extract_vars)].isna().any().any():
                        n_rejected_window += 1
                        continue
    
                if pre_std_threshold is not None:
                    pre_vals = win.iloc[:pre][list(extract_vars)]
                    if (pre_vals.std() > pre_std_threshold).any():
                        n_rejected_std += 1
                        continue
    
                for v in extract_vars:
                    aligned[v].append(win[v].to_numpy().astype(float))
    
            # Stack into arrays
            windows = {}
            for v in extract_vars:
                if len(aligned[v]) == 0:
                    windows[v] = np.empty((0, win_len))
                else:
                    windows[v] = np.vstack(aligned[v])
    
            n_events = len(aligned[extract_vars[0]])
            cond_results[cond] = {
                "windows":  windows,
                "n_events": n_events,
                "rel_time": rel_time,
            }
    
            print(
                f"[extract_eta_windows] {cond}: {n_events} events kept  |  "
                f"rejected: {n_rejected_window} (incomplete window)  "
                f"{n_rejected_std} (noisy pre-window)"
            )
    
        return cond_results
    
    
    def normalise_eta_windows(
        self,
        cond_results,
        pre,
        mode="baseline_subtract",
    ):
        """
        Normalise and/or invert extracted windows in-place (returns a copy).
    
        Parameters
        ----------
        cond_results : dict
            Output of extract_eta_windows().
        pre : int
            Number of pre-trigger frames (used as baseline segment).
        mode : str or None
            "baseline_subtract", "zscore", "fold_change", or None.
        
        Returns
        -------
        out : dict
            Same structure as cond_results with normalised windows.
        """
    
        out = {}
    
        for cond, cond_data in cond_results.items():
            new_windows = {}
    
            for var, X in cond_data["windows"].items():
                X = X.copy()
    
                if mode is not None and X.shape[0] > 0:
                    pre_seg = X[:, :pre]
                    mu = np.nanmean(pre_seg, axis=1, keepdims=True)
                    sd = np.nanstd( pre_seg, axis=1, ddof=1, keepdims=True)
    
                    if mode == "baseline_subtract":
                        X = X - mu
                    elif mode == "zscore":
                        X = (X - mu) / (sd + 1e-10)
                    elif mode == "fold_change":
                        X = X / (mu + 1e-10)
                    else:
                        raise ValueError(
                            "mode must be None, 'baseline_subtract', "
                            "'zscore', or 'fold_change'."
                        )
    
                new_windows[var] = X
    
            out[cond] = {
                "windows":  new_windows,
                "n_events": cond_data["n_events"],
                "rel_time": cond_data["rel_time"],
            }
    
        return out
        
#without any significance bars
    
    # def plot_eta_windows(
    #     self,
    #     windows_list,
    #     vars,
    #     trigger_vars,
    #     conditions=None,
    #     normalize_mode=None,
    #     colors=None,
    #     figsize=None,
    #     style=None,
    #     legend=True,
    #     file_stem=None,
    # ):
    #     if conditions is None:
    #         conditions = list(windows_list[0].keys())
    
    #     n_conds  = len(conditions)
    #     n_panels = len(trigger_vars)
    
    #     # ------------------------------------------------------------------
    #     # 1. Build color map
    #     # ------------------------------------------------------------------
    #     if colors is None:
    #         if self.palette is not None:
    #             _cycle = [v["main"] for v in self.palette.values()]
    #         else:
    #             _cycle = [f"C{i}" for i in range(10)]
    #         colors = {var: _cycle[i % len(_cycle)] for i, var in enumerate(vars)}
    
    #     # ------------------------------------------------------------------
    #     # 2. Y-axis label
    #     # ------------------------------------------------------------------
    #     if normalize_mode == "zscore":
    #         ylabel = "signal (z-score)"
    #     elif normalize_mode == "baseline_subtract":
    #         ylabel = "Δ signal (baseline-subtracted)"
    #     elif normalize_mode == "fold_change":
    #         ylabel = "signal (fold change)"
    #     else:
    #         ylabel = "signal (a.u.)"
    
    #     # ------------------------------------------------------------------
    #     # 3. Figure layout
    #     # ------------------------------------------------------------------
    #     if figsize is None:
    #         panel_w = self.fig_width * 4
    #         panel_h = self.fig_height * 2
    #     else:
    #         panel_w, panel_h = figsize
    
    #     fig, axes = plt.subplots(
    #         nrows=n_conds,
    #         ncols=n_panels,
    #         figsize=(panel_w * n_panels, panel_h * n_conds),
    #         squeeze=False,
    #     )
    
    #     _fs = plt.rcParams["font.size"]
    
    #     # ------------------------------------------------------------------
    #     # 4. Plot
    #     # ------------------------------------------------------------------
    #     for col_idx, (cond_data, trigger_var) in enumerate(
    #         zip(windows_list, trigger_vars)
    #     ):
    #         for row_idx, cond in enumerate(conditions):
    #             ax = axes[row_idx][col_idx]
    
    #             data = cond_data.get(
    #                 cond,
    #                 {"windows": {}, "n_events": 0, "rel_time": np.array([])}
    #             )
    
    #             rel_time         = data["rel_time"]
    #             rel_time_display = rel_time * self.dt
    
    #             for var in vars:
    #                 X = data["windows"].get(var, np.empty((0, len(rel_time))))
    #                 if X.shape[0] == 0:
    #                     continue
    
    #                 mean  = np.nanmean(X, axis=0)
    #                 sem   = np.nanstd(X, axis=0, ddof=1) / np.sqrt(X.shape[0])
    #                 color = colors[var]
    
    #                 ax.plot(rel_time_display, mean, color=color,
    #                         linewidth=1.2, label=f"{var}  (n={X.shape[0]})")
    #                 ax.fill_between(rel_time_display, mean - sem, mean + sem,
    #                                 color=color, alpha=0.25, linewidth=0)
    
    #             # --------------------------------------------------------------
    #             # 5. Axes styling
    #             # --------------------------------------------------------------
    #             ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    #             ax.axhline(0, color="black", linewidth=0.5, linestyle=":",  zorder=1)
    
    #             title = f"trigger: {trigger_var}\n{cond}  (n={data['n_events']} events)"
    #             ax.set_title(title, fontsize=_fs, pad=3)
    #             ax.set_xlabel(f"time relative to trigger ({self.time_unit})", fontsize=_fs)
    #             if col_idx == 0:
    #                 ax.set_ylabel(ylabel, fontsize=_fs)
    
    #             if style is not None:
    #                 for var_key, s in style.items():
    #                     if "xlim"   in s: ax.set_xlim(s["xlim"])
    #                     if "ylim"   in s: ax.set_ylim(s["ylim"])
    #                     if "xticks" in s: ax.set_xticks(s["xticks"])
    #                     if "yticks" in s: ax.set_yticks(s["yticks"])
    
    #             ax.tick_params(labelsize=_fs)
    #             sns.despine(ax=ax)
    
    #             if legend:
    #                 ax.legend(frameon=False, fontsize=_fs)
    
    #     plt.tight_layout()
    
    #     # ------------------------------------------------------------------
    #     # 6. Save bundle
    #     # ------------------------------------------------------------------
    #     if self.save_outputs:
    #         rows = []
    #         for col_idx, (cond_data, trigger_var) in enumerate(
    #             zip(windows_list, trigger_vars)
    #         ):
    #             for cond in conditions:
    #                 data = cond_data.get(
    #                     cond,
    #                     {"windows": {}, "n_events": 0, "rel_time": np.array([])}
    #                 )
    #                 for var, X in data["windows"].items():
    #                     if X.shape[0] == 0:
    #                         continue
    #                     for ev_idx, trace in enumerate(X):
    #                         for t_idx, val in enumerate(trace):
    #                             rows.append({
    #                                 "condition":       cond,
    #                                 "trigger_var":     trigger_var,
    #                                 "signal":          var,
    #                                 "event_idx":       ev_idx,
    #                                 "rel_time_frames": data["rel_time"][t_idx],
    #                                 "rel_time":        data["rel_time"][t_idx] * self.dt,
    #                                 "time_unit":       self.time_unit,
    #                                 "value":           val,
    #                             })
    
    #         points_df = pd.DataFrame(rows)
    
    #         self.save_plot_bundle(
    #             fig=fig,
    #             plot_name="eta_windows",
    #             var_tag="_".join(trigger_vars),
    #             points_df=points_df,
    #             file_stem=file_stem,
    #         )
    
    #     plt.show()
    
    #     return fig, axes
    
    

    def plot_eta_windows(
        self,
        windows_list,
        vars,
        trigger_vars,
        conditions=None,
        normalize_mode=None,
        colors=None,
        figsize=None,
        style=None,
        legend=True,
        file_stem=None,
        show_significance=False,
        effect_size_threshold=0.2,
    ):
        if conditions is None:
            conditions = list(windows_list[0].keys())
    
        n_conds  = len(conditions)
        n_panels = len(trigger_vars)
    
        # ------------------------------------------------------------------
        # 1. Build color map
        # ------------------------------------------------------------------
        if colors is None:
            if self.palette is not None:
                _cycle = [v["main"] for v in self.palette.values()]
            else:
                _cycle = [f"C{i}" for i in range(10)]
            colors = {var: _cycle[i % len(_cycle)] for i, var in enumerate(vars)}
    
        # ------------------------------------------------------------------
        # 2. Y-axis label
        # ------------------------------------------------------------------
        if normalize_mode == "zscore":
            ylabel = "signal (z-score)"
        elif normalize_mode == "baseline_subtract":
            ylabel = "Δ signal (baseline-subtracted)"
        elif normalize_mode == "fold_change":
            ylabel = "signal (fold change)"
        else:
            ylabel = "signal (a.u.)"
    
        # ------------------------------------------------------------------
        # 3. Figure layout
        # ------------------------------------------------------------------
        if figsize is None:
            panel_w = self.fig_width * 4
            panel_h = self.fig_height * 2
        else:
            panel_w, panel_h = figsize
    
        # If showing significance, carve each panel into a main axes (85%)
        # and a small bar axes below (15%)
        if show_significance:
            from matplotlib.gridspec import GridSpecFromSubplotSpec
            fig, outer = plt.subplots(
                nrows=n_conds,
                ncols=n_panels,
                figsize=(panel_w * n_panels, panel_h * n_conds),
                squeeze=False,
            )
            # Replace each outer cell with a gridspec split
            axes     = np.empty((n_conds, n_panels), dtype=object)
            bar_axes = np.empty((n_conds, n_panels), dtype=object)
            for r in range(n_conds):
                for c in range(n_panels):
                    outer[r][c].remove()
                    gs = GridSpecFromSubplotSpec(
                        2, 1,
                        subplot_spec=outer[r][c].get_subplotspec(),
                        height_ratios=[1, 10],
                        hspace=0.1,
                    )
                    axes[r][c]     = fig.add_subplot(gs[1])
                    bar_axes[r][c] = fig.add_subplot(gs[0], sharex=axes[r][c])
        else:
            fig, axes = plt.subplots(
                nrows=n_conds,
                ncols=n_panels,
                figsize=(panel_w * n_panels, panel_h * n_conds),
                squeeze=False,
            )
            bar_axes = None
    
        _fs = plt.rcParams["font.size"]
    
        # ------------------------------------------------------------------
        # 4. Plot
        # ------------------------------------------------------------------
        for col_idx, (cond_data, trigger_var) in enumerate(
            zip(windows_list, trigger_vars)
        ):
            for row_idx, cond in enumerate(conditions):
                ax = axes[row_idx][col_idx]
    
                data = cond_data.get(
                    cond,
                    {"windows": {}, "n_events": 0, "rel_time": np.array([])}
                )
    
                rel_time         = data["rel_time"]
                rel_time_display = rel_time * self.dt
    
                means = {}
                for var in vars:
                    X = data["windows"].get(var, np.empty((0, len(rel_time))))
                    if X.shape[0] == 0:
                        continue
    
                    mean  = np.nanmean(X, axis=0)
                    sem   = np.nanstd(X, axis=0, ddof=1) / np.sqrt(X.shape[0])
                    color = colors[var]
                    means[var] = mean
    
                    ax.plot(rel_time_display, mean, color=color,
                            linewidth=1.2, label=f"{var}  (n={X.shape[0]})")
                    ax.fill_between(rel_time_display, mean - sem, mean + sem,
                                    color=color, alpha=0.25, linewidth=0)
    
                # --------------------------------------------------------------
                # 5. Axes styling
                # --------------------------------------------------------------
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
                ax.axhline(0, color="black", linewidth=0.5, linestyle=":",  zorder=1)
    
                title = f"trigger: {trigger_var}\n{cond}  (n={data['n_events']} events)"
                #ax.set_title(title, fontsize=_fs, pad=3)
                if row_idx == n_conds - 1:
                    ax.set_xlabel(f"time relative to trigger ({self.time_unit})", fontsize=_fs)
                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=_fs)
    
                if style is not None:
                    for var_key, s in style.items():
                        if "xlim"   in s: ax.set_xlim(s["xlim"])
                        if "ylim"   in s: ax.set_ylim(s["ylim"])
                        if "xticks" in s: ax.set_xticks(s["xticks"])
                        if "yticks" in s: ax.set_yticks(s["yticks"])
    
                ax.tick_params(labelsize=_fs)
                sns.despine(ax=ax)
    
                if legend:
                    ax.legend(frameon=False, fontsize=_fs)
    
                # --------------------------------------------------------------
                # 6. Significance bar axes
                # --------------------------------------------------------------
                if show_significance and means:
                    bax = bar_axes[row_idx][col_idx]
    
                    for var_idx, var in enumerate(vars):
                        if var not in means:
                            continue
                        if var == trigger_var:         
                            continue
                        mean      = means[var]
                        color     = colors[var]
                        peak      = np.nanmax(np.abs(mean))
                        threshold = peak * effect_size_threshold
                        sig       = np.abs(mean) > threshold
                        sig_times = rel_time_display[sig]
    
                        if len(sig_times) > 0:
                            bax.scatter(sig_times,
                                        np.full_like(sig_times, var_idx),
                                        color=color, s=5, marker="|",
                                        linewidths=0.8, zorder=3)
    
                    bax.set_ylim(-0.5, len(vars) - 0.5)
                    bax.axis("off")
                    bax.set_yticks([])
                    bax.set_xlabel("")
                    bax.tick_params(labelbottom=False)
                    bax.set_title(title, fontsize=_fs, pad=3)
                    sns.despine(ax=bax, left=True)
    
        plt.tight_layout()
    
        # ------------------------------------------------------------------
        # 7. Save bundle
        # ------------------------------------------------------------------
        if self.save_outputs:
            rows = []
            for col_idx, (cond_data, trigger_var) in enumerate(
                zip(windows_list, trigger_vars)
            ):
                for cond in conditions:
                    data = cond_data.get(
                        cond,
                        {"windows": {}, "n_events": 0, "rel_time": np.array([])}
                    )
                    for var, X in data["windows"].items():
                        if X.shape[0] == 0:
                            continue
                        for ev_idx, trace in enumerate(X):
                            for t_idx, val in enumerate(trace):
                                rows.append({
                                    "condition":       cond,
                                    "trigger_var":     trigger_var,
                                    "signal":          var,
                                    "event_idx":       ev_idx,
                                    "rel_time_frames": data["rel_time"][t_idx],
                                    "rel_time":        data["rel_time"][t_idx] * self.dt,
                                    "time_unit":       self.time_unit,
                                    "value":           val,
                                })
    
            points_df = pd.DataFrame(rows)
    
            self.save_plot_bundle(
                fig=fig,
                plot_name="eta_windows",
                var_tag="_".join(trigger_vars),
                points_df=points_df,
                file_stem=file_stem,
            )
    
        plt.show()
    
        return fig, axes


                

    def plot_eta_cross_trigger(
        self,
        var_a,
        var_b,
        invert_b=False,
        peaks_a=None,
        peaks_b=None,
        conditions=None,
        frame_range=None,
        pre=20,
        post=40,
        exclude_first_n_frames=0,
        require_full_window=True,
        normalize_mode=None,
        detrend=False,
        smooth_sigma=1.0,
        baseline_window=50,
        baseline_pct=10,
        prominence=0.1,
        distance=10,
        height=None,
        color_a=None,
        color_b=None,
        figsize=None,
        style=None,
        file_stem=None,
        legend=True,
    ):
        """
        Cross-trigger analysis: reveal the temporal relationship between two
        oscillating variables by running ETA in both directions and plotting
        them side by side.
    
        For each condition, two panels are produced:
          Left  — trigger on var_a peaks, trace of var_b (and var_a) aligned
          Right — trigger on var_b peaks, trace of var_a (and var_b) aligned
    
        If var_b systematically leads var_a, the left panel will show var_b
        peaking before t=0, while the right panel will show var_a peaking
        after t=0 (and vice versa).  Symmetric panels indicate no consistent
        lag.
    
        Only valid in 'trackmate' mode.
    
        Parameters
        ----------
        var_a, var_b : str
            The two variables to cross-trigger on (support 'norm_*' prefix).
        peaks_a, peaks_b : DataFrame or None
            Pre-computed peaks tables from find_peaks_per_cell().
            If None, peaks are detected automatically using the prominence /
            distance / height / detrend parameters below.
        conditions : list[str] or None
            Restrict to these conditions. None = all conditions in data.
        frame_range : tuple(int, int) or None
            (t0_frame, t1_frame) inclusive. Filters data before peak
            detection and window extraction.
        pre : int
            Frames before each trigger peak (window = [-pre, post]).
        post : int
            Frames after each trigger peak.
        exclude_first_n_frames : int
            Skip peaks whose peak_frame < this value to avoid edge artefacts.
        require_full_window : bool
            If True, discard events where any frame in [-pre, post] is missing
            or NaN in either variable.
        normalize_mode : str or None
            Per-event normalisation before averaging:
              None                – raw values
              "baseline_subtract" – subtract per-event pre-trigger mean
              "zscore"            – z-score to per-event pre-trigger mean/SD
              "fold_change"       – divide by per-event pre-trigger mean
        detrend : bool
            Apply detrend_and_smooth() to each cell trace before peak
            detection and window extraction (only used when peaks are auto-
            detected, i.e. peaks_a / peaks_b is None).
        smooth_sigma : float
            Gaussian smooth sigma in frames (used when detrend=True).
        baseline_window : int
            Rolling baseline window in frames (used when detrend=True).
        baseline_pct : float
            Percentile for baseline estimation (used when detrend=True).
        prominence : float
            Minimum peak prominence for auto peak detection.
        distance : int
            Minimum frames between peaks for auto peak detection.
        height : float or None
            Minimum peak height for auto peak detection.
        color_a, color_b : color spec or None
            Colours for var_a and var_b traces.  Defaults to the first two
            entries of the condition palette (or 'C0'/'C1').
        figsize : tuple or None
            (width_per_panel, height).  Defaults to
            (self.fig_width * 4, self.fig_height * 2).
        style : dict or None
            Axis style passed to apply_axes_style().  A nested dict keyed
            by condition name applies per-condition overrides.
        file_stem : str or None
            Base filename for save_plot_bundle(). Auto-generated if None.
    
        Returns
        -------
        fig : matplotlib Figure
        axes : ndarray of Axes, shape (n_conditions, 2)
            axes[i, 0] = trigger-on-A panel for condition i
            axes[i, 1] = trigger-on-B panel for condition i
        results : dict
            Keyed by condition, each value is a dict with:
              "trigger_a" – ETA result when triggering on var_a
              "trigger_b" – ETA result when triggering on var_b
            Each ETA result contains:
              "rel_time"  – 1D array of frame offsets
              "windows"   – dict var_name -> (n_events, win_len) array
              "n_events"  – number of valid events
        """
    
        if self.analysis.mode != "trackmate":
            raise ValueError(
                "plot_eta_cross_trigger is only available in 'trackmate' mode."
            )
    
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    
        # ------------------------------------------------------------------
        # 1. Resolve conditions
        # ------------------------------------------------------------------
        df_raw = self.analysis.df.reset_index()
    
        if conditions is None:
            conditions = sorted(df_raw["condition"].unique().tolist())
    
        # ------------------------------------------------------------------
        # 2. Auto-detect peaks where not supplied
        # ------------------------------------------------------------------
        _peak_kwargs = dict(
            conditions=conditions,
            frame_range=frame_range,
            prominence=prominence,
            distance=distance,
            height=height,
            detrend=detrend,
            smooth_sigma=smooth_sigma,
            baseline_window=baseline_window,
            baseline_pct=baseline_pct,
            exclude_first_n_frames=exclude_first_n_frames,
        )
    
        if peaks_a is None:
            peaks_a = self.find_peaks_per_cell(var_name=var_a, **_peak_kwargs)
        if peaks_b is None:
            peaks_b = self.find_peaks_per_cell(var_name=var_b, **_peak_kwargs)
    
        if len(peaks_a) == 0 and len(peaks_b) == 0:
            raise ValueError(
                "No peaks found for either variable. "
                "Adjust prominence/distance/height thresholds."
            )
    
        # ------------------------------------------------------------------
        # 3. Build merged signal dataframe for window extraction
        #    (same pattern as plot_eta_comparison)
        # ------------------------------------------------------------------
        signal_dfs = []
        for var in (var_a, var_b):
            df_var, ycol = self.get_df_for_var(var)
    
            if conditions is not None:
                df_var = df_var[df_var["condition"].isin(conditions)]
            if frame_range is not None:
                t0_f, t1_f = frame_range
                df_var = df_var[df_var["frame"].between(t0_f, t1_f)]
    
            if detrend:
                chunks = []
                for keys, group in df_var.groupby(cell_keys):
                    group = group.copy()
                    group[ycol] = self.detrend_and_smooth(
                        group[ycol].values, smooth_sigma, baseline_window,
                        baseline_pct, smooth_only=False,
                    )
                    chunks.append(group)
                df_var = pd.concat(chunks, ignore_index=True)
    
            signal_dfs.append(df_var[cell_keys + ["frame", ycol]])
    
        df_merged = signal_dfs[0].merge(signal_dfs[1], on=cell_keys + ["frame"], how="outer")
        # Tighten the outlier clip (add this after df_merged is built)
        for var in (var_a, var_b):
            df_merged.loc[df_merged[var].abs() > 10, var] = np.nan  # tighter: signals are 0-2
        df_merged = df_merged.groupby(cell_keys + ["frame"], as_index=False).mean()
        # for var in (var_a, var_b):
        #     # Mask physically impossible values — adjust the threshold to your signal range
        #     df_merged.loc[df_merged[var].abs() > 100, var] = np.nan
        df_indexed = df_merged.set_index(cell_keys + ["frame"]).sort_index()
    
        rel_time = np.arange(-pre, post + 1)
        win_len  = len(rel_time)
    
        # ------------------------------------------------------------------
        # 4. Helper: extract aligned windows for one set of peaks
        # ------------------------------------------------------------------
        def _extract_windows(peaks_df, extract_vars):
            """
            Returns a dict keyed by condition, each value a dict:
              "windows"  : {var_name: (n_events, win_len) array}
              "n_events" : int
            """
            cond_results = {}
    
            for cond in conditions:
                cond_peaks = peaks_df[peaks_df["condition"] == cond]
                aligned = {v: [] for v in extract_vars}
    
                for _, ev in cond_peaks.iterrows():
                    if int(ev["peak_frame"]) < exclude_first_n_frames:
                        continue
    
                    cell_key = (
                        ev["condition"],
                        ev["folder"],
                        ev["repeat_id"],
                        np.int64(ev["object_id"]),
                    )
                    t0     = int(ev["peak_frame"])
                    frames = t0 + rel_time
    
                    idx = pd.MultiIndex.from_tuples(
                        [cell_key + (f,) for f in frames],
                        names=cell_keys + ["frame"],
                    )
    
                    try:
                        win = df_indexed.reindex(idx)
                    except Exception:
                        continue
    
                    if require_full_window:
                        if win.shape[0] != win_len:
                            continue
                        post_win = win.iloc[pre:]
                        if post_win[list(extract_vars)].isna().any().any():
                            continue
    
                    for v in extract_vars:
                        aligned[v].append(win[v].to_numpy().astype(float))
    
                # Stack raw windows
                raw_windows = {}
                for v in extract_vars:
                    if len(aligned[v]) == 0:
                        raw_windows[v] = np.empty((0, win_len))
                    else:
                        raw_windows[v] = np.vstack(aligned[v])
    
                n_events = len(aligned[extract_vars[0]])
    
                # Joint valid mask (same logic as plot_eta_comparison)
                if normalize_mode is not None and n_events > 0:
                    all_sds = []
                    for v in extract_vars:
                        X = raw_windows[v]
                        if X.shape[0] == 0:
                            continue
                        sd = np.nanstd(X[:, :pre], axis=1, ddof=1)
                        all_sds.append(sd)
                    if all_sds:
                        min_sd = np.nanpercentile(
                            np.stack(all_sds, axis=1).min(axis=1), 10
                        )
                        valid = np.all(np.stack(all_sds, axis=1) > min_sd, axis=1)
                    else:
                        valid = np.ones(n_events, dtype=bool)
                else:
                    valid = np.ones(n_events, dtype=bool)
    
                # Normalise
                windows = {}
                for v in extract_vars:
                    X = raw_windows[v]
                    if X.shape[0] == 0:
                        windows[v] = X
                        continue
                    X = X[valid]
                    if normalize_mode is not None and X.shape[0] > 0:
                        mu = np.nanmean(X[:, :pre], axis=1, keepdims=True)
                        sd = np.nanstd( X[:, :pre], axis=1, ddof=1, keepdims=True)
                        if normalize_mode == "baseline_subtract":
                            X = X - mu
                        elif normalize_mode == "zscore":
                            X = (X - mu) / (sd + 1e-10)
                        elif normalize_mode == "fold_change":
                            X = X / (mu + 1e-10)
                        else:
                            raise ValueError(
                                "normalize_mode must be None, 'baseline_subtract', "
                                "'zscore', or 'fold_change'."
                            )
                    if invert_b and v == var_b:
                        X = -X
                    windows[v] = X
    
                n_valid = int(valid.sum())
                cond_results[cond] = {"windows": windows, "n_events": n_valid}
                print(
                    f"[plot_eta_cross_trigger] {cond}: "
                    f"trigger={extract_vars[0] if len(cond_peaks)==0 else peaks_df.name if hasattr(peaks_df,'name') else '?'}  "
                    f"{n_valid} events"
                )
    
            return cond_results

        # Temporarily patch to debug
        for _, ev in peaks_a[peaks_a["condition"] == conditions[0]].head(3).iterrows():
            cell_key = (ev["condition"], ev["folder"], ev["repeat_id"], np.int64(ev["object_id"]))
            t0 = int(ev["peak_frame"])
            frames = t0 + rel_time
            idx = pd.MultiIndex.from_tuples(
                [cell_key + (f,) for f in frames],
                names=cell_keys + ["frame"],
            )
            win = df_indexed.reindex(idx)
            print("win shape:", win.shape)
            print("expected win_len:", win_len)
            print("pre:", pre)
            print("pre-window slice shape:", win.iloc[:pre].shape)
            print("pre-window values (first row):", win[var_a].values[:pre])
            print("any NaN in pre-window:", np.isnan(win[var_a].values[:pre]).any())
            print("raw values range:", win[var_a].min(), "to", win[var_a].max())
            print()
    
        # ------------------------------------------------------------------
        # 5. Run extraction for both trigger directions
        # ------------------------------------------------------------------
        print(f"--- Triggering on {var_a} ---")
        cond_results_a = _extract_windows(peaks_a, [var_a, var_b])
    
        print(f"--- Triggering on {var_b} ---")
        cond_results_b = _extract_windows(peaks_b, [var_b, var_a])
    
        # ------------------------------------------------------------------
        # 6. Resolve colours
        # ------------------------------------------------------------------
        if self.palette is not None:
            _cycle = [v["main"] for v in self.palette.values()]
        else:
            _cycle = [f"C{i}" for i in range(10)]
    
        if color_a is None:
            color_a = _cycle[0] if len(_cycle) > 0 else "C0"
        if color_b is None:
            color_b = _cycle[1] if len(_cycle) > 1 else "C1"
    
        signal_colors = {var_a: color_a, var_b: color_b}
    
        # ------------------------------------------------------------------
        # 7. Plot — rows = conditions, columns = [trigger-A, trigger-B]
        # ------------------------------------------------------------------
        n_conds = len(conditions)
    
        if figsize is None:
            panel_w = self.fig_width * 4
            panel_h = self.fig_height * 2
        else:
            panel_w, panel_h = figsize
    
        fig, axes = plt.subplots(
            nrows=n_conds,
            ncols=2,
            figsize=(panel_w * 2, panel_h * n_conds),
            squeeze=False,
        )
    
        rel_time_display = rel_time * self.dt
    
        if normalize_mode == "zscore":
            ylabel = "signal (z-score)"
        elif normalize_mode == "baseline_subtract":
            ylabel = "Δ signal (baseline-subtracted)"
        elif normalize_mode == "fold_change":
            ylabel = "signal (fold change)"
        else:
            ylabel = "signal (a.u.)"
    
        results = {}
    
        for row_idx, cond in enumerate(conditions):
    
            eta_a = cond_results_a.get(cond, {"windows": {}, "n_events": 0})
            eta_b = cond_results_b.get(cond, {"windows": {}, "n_events": 0})
    
            results[cond] = {
                "trigger_a": {"rel_time": rel_time, **eta_a},
                "trigger_b": {"rel_time": rel_time, **eta_b},
            }
    
            for col_idx, (eta, trigger_name, plot_vars) in enumerate([
                (eta_a, var_a, [var_a, var_b]),
                (eta_b, var_b, [var_b, var_a]),
            ]):
                ax = axes[row_idx, col_idx]
    
                for v in plot_vars:
                    X = eta["windows"].get(v, np.empty((0, win_len)))
                    if X.shape[0] == 0:
                        continue
    
                    color = signal_colors[v]
                    mean  = np.nanmean(X, axis=0)
                    sem   = np.nanstd(X, axis=0, ddof=1) / np.sqrt(X.shape[0])
    
                    ax.plot(
                        rel_time_display,
                        mean,
                        color=color,
                        linewidth=1.2,
                        label=f"{v}  (n={X.shape[0]})",
                    )
                    ax.fill_between(
                        rel_time_display,
                        mean - sem,
                        mean + sem,
                        color=color,
                        alpha=0.25,
                        linewidth=0,
                    )

                    ax.set_title(title, pad=3)
                    ax.set_xlabel(f"time relative to trigger ({self.time_unit})")
                    if col_idx == 0:
                        ax.set_ylabel(ylabel)
                
                    ax.tick_params(which="both")
                
                    if legend:
                        ax.legend(frameon=False)
                    
                # ax.axvline(0, color="#424242", linewidth=0.8, linestyle="--")
                # ax.axhline(0, color="#424242", linewidth=0.5, linestyle=":", alpha=0.6)
    
                # ax.set_xlabel(f"time relative to trigger ({self.time_unit})")
                # ax.set_ylabel(ylabel)
                # ax.set_title(
                #     f"{cond}  —  trigger: {trigger_name}"
                #     f"  (n={eta['n_events']})"
                # )
                # if legend:
                #     ax.legend(frameon=False)
    
                ax_style = self.get_style_for_var(style, cond) if style is not None else None
                self.apply_axes_style(ax, ax_style)
                sns.despine(ax=ax)
    
        plt.tight_layout()
    
        # ------------------------------------------------------------------
        # 8. Save bundle
        # ------------------------------------------------------------------
        if self.save_outputs:
            rows = []
            for cond, cond_result in results.items():
                for direction, eta in [("trigger_a", cond_result["trigger_a"]),
                                       ("trigger_b", cond_result["trigger_b"])]:
                    for var, X in eta["windows"].items():
                        for ev_idx, trace in enumerate(X):
                            for t_idx, val in enumerate(trace):
                                rows.append({
                                    "condition":  cond,
                                    "direction":  direction,
                                    "signal":     var,
                                    "event_idx":  ev_idx,
                                    "rel_time":   rel_time[t_idx],
                                    "value":      val,
                                })
    
            points_df = pd.DataFrame(rows)
    
            self.save_plot_bundle(
                fig=fig,
                plot_name="eta_cross_trigger",
                var_tag=f"{var_a}_x_{var_b}",
                points_df=points_df,
                summary_df=None,
                params={
                    "var_a":                  var_a,
                    "var_b":                  var_b,
                    "conditions":             conditions,
                    "pre":                    pre,
                    "post":                   post,
                    "normalize_mode":         normalize_mode,
                    "exclude_first_n_frames": exclude_first_n_frames,
                },
                file_stem=file_stem,
            )
    
        plt.show()
    
        return fig, axes, results


    def plot_peak_count_hist2d(
        self,
        peaks_list,
        var_names,
        conditions=None,
        density=False,
        normalize=False,
        pairs=None,
        bins=None,
        cmap="cividis",
        figsize=None,
        style=None,
        file_stem=None,
    ):
        """
        2D histograms of per-cell peak counts, pairing variables against each other.
    
        For each pair of variables, plots a 2D histogram where each cell
        contributes one point: (n_peaks_var_a, n_peaks_var_b).
    
        Parameters
        ----------
        peaks_list : list[DataFrame]
            One peaks DataFrame per variable, from find_peaks_per_cell().
            Must be same length and order as var_names.
        var_names : list[str]
            Names of the variables, used for axis labels.
        conditions : list[str] or None
            Restrict to these conditions. None = all.
        pairs : list[tuple(int,int)] or None
            Which pairs to plot, as indices into var_names.
            None = all pairwise combinations.
        bins : array or int or None
            Bins for hist2d. Default: integer bins 0.5 to max_peaks+0.5.
        cmap : str
            Colormap. Default 'YlOrRd'.
        figsize : tuple or None
            (width_per_panel, height). Default (fig_width*2, fig_height*2).
        style : dict or None
        file_stem : str or None
    
        Returns
        -------
        fig, axes
        counts_df : DataFrame
            Per-cell peak counts for all variables, one row per cell.
        """
        from itertools import combinations
    
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    
        # ------------------------------------------------------------------
        # 1. Build per-cell peak count table for each variable
        # ------------------------------------------------------------------
        count_dfs = []
        for peaks_df, var in zip(peaks_list, var_names):
            if conditions is not None:
                peaks_df = peaks_df[peaks_df["condition"].isin(conditions)]
    
            counts = (
                peaks_df.groupby(cell_keys)
                .size()
                .reset_index(name=var)
            )
            count_dfs.append(counts)
    
        # Outer join so cells with 0 peaks in one var still appear
        counts_df = count_dfs[0]
        for cdf in count_dfs[1:]:
            counts_df = counts_df.merge(cdf, on=cell_keys, how="outer")
        counts_df = counts_df.fillna(0).astype(
            {var: int for var in var_names}
        )
    
        if conditions is not None:
            counts_df = counts_df[counts_df["condition"].isin(conditions)]
    
        # ------------------------------------------------------------------
        # 2. Resolve pairs
        # ------------------------------------------------------------------
        if pairs is None:
            pairs = list(combinations(range(len(var_names)), 2))
    
        n_pairs = len(pairs)
    
        # ------------------------------------------------------------------
        # 3. Figure layout
        # ------------------------------------------------------------------
        if figsize is None:
            panel_w = self.fig_width * 2
            panel_h = self.fig_height * 2
        else:
            panel_w, panel_h = figsize
    
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_pairs,
            figsize=(panel_w * n_pairs, panel_h),
            squeeze=False,
        )
        axes = axes[0]
    
        # ------------------------------------------------------------------
        # 4. Plot each pair
        # ------------------------------------------------------------------
        for ax, (i, j) in zip(axes, pairs):
            var_x = var_names[i]
            var_y = var_names[j]
    
            x = counts_df[var_x].values
            y = counts_df[var_y].values
    
            if bins is None:
                max_count = max(x.max(), y.max(), 1)
                _bins = np.arange(0.5, max_count + 1.5, 1)
            else:
                _bins = bins
    
            counts_2d, xedges, yedges = np.histogram2d(x, y, bins=[_bins, _bins])

            if density:
                weights = np.ones(len(x)) / len(x)
                _, _, _, im = ax.hist2d(x, y, bins=[_bins, _bins], 
                                        cmap=cmap, weights=weights,
                                        density=True)
                fig.colorbar(im, ax=ax, label="density")
            elif normalize:
                weights = np.ones(len(x)) / len(x) * 100
                _, _, _, im = ax.hist2d(x, y, bins=[_bins, _bins],
                                        cmap=cmap, weights=weights)
                fig.colorbar(im, ax=ax, label="% cells")
            else:
                _, _, _, im = ax.hist2d(x, y, bins=[_bins, _bins], cmap=cmap)
                fig.colorbar(im, ax=ax, label="n cells")

            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))    

    
            ax.set_xlabel(f"n peaks  ({var_x})")
            ax.set_ylabel(f"n peaks  ({var_y})")
            ax.set_title(f"{var_x}\nvs\n{var_y}")
    
            # Diagonal reference line
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, lim], [0, lim], color="#888888",
                    linewidth=0.8, linestyle="--", zorder=0)

            r, p = stats.spearmanr(x, y)
            pearson_r, pearson_p = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f"r={r:.2f}\np={p:.2e}",
                    transform=ax.transAxes, va="top", fontsize=8, color='w')
    
            self.apply_axes_style(ax, style)
            sns.despine(ax=ax)
    
        plt.tight_layout()
    
        # ------------------------------------------------------------------
        # 5. Save bundle
        # ------------------------------------------------------------------
        if self.save_outputs:
            # Points df: one row per cell with all peak counts
            points_df = counts_df.copy()
            points_df["time_unit"] = self.time_unit

            # Correlation df — one row per pair
            corr_rows = []
            for i, j in pairs:
                var_x = var_names[i]
                var_y = var_names[j]
                x = counts_df[var_x].values
                y = counts_df[var_y].values
                r, p = stats.spearmanr(x, y)
                pearson_r, pearson_p = stats.pearsonr(x, y)
                corr_rows.append({
                    "var_x":      var_x,
                    "var_y":      var_y,
                    "spearman_r": r,
                    "spearman_p": p,
                    "pearson_r":  pearson_r,
                    "pearson_p":  pearson_p,
                    "n_cells":    len(counts_df),
                })
            corr_df = pd.DataFrame(corr_rows)

            # Merge both into one save call via summary_df
            self.save_plot_bundle(
                fig=fig,
                plot_name="peak_count_hist2d",
                var_tag="_".join(var_names),
                points_df=points_df,
                summary_df=corr_df,
                file_stem=file_stem,
            )

        plt.show()
    
        return fig, axes, counts_df



    def plot_pulse_prominence_by_number(
        self,
        peaks_list,
        var_names,
        conditions=None,
        max_pulses=5,
        figsize=None,
        palette=None,
        style=None,
        file_stem=None,
    ):
        #cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    
        # ------------------------------------------------------------------
        # 1. Filter and tag each peaks df
        # ------------------------------------------------------------------
        tagged = []
        for peaks_df, var in zip(peaks_list, var_names):
            df = peaks_df.copy()
            if conditions is not None:
                df = df[df["condition"].isin(conditions)]
            df = df[df["pulse_number"] <= max_pulses]
            df["variable"] = var
            tagged.append(df)
    
        combined_df = pd.concat(tagged, ignore_index=True)
    
        # ------------------------------------------------------------------
        # 2. Figure layout
        # ------------------------------------------------------------------
        n_vars = len(var_names)
    
        if figsize is None:
            panel_w = self.fig_width * 2
            panel_h = self.fig_height * 2
        else:
            panel_w, panel_h = figsize
    
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_vars,
            figsize=(panel_w * n_vars, panel_h),
            squeeze=False,
        )
        axes = axes[0]
    
        # ------------------------------------------------------------------
        # 3. Plot
        # ------------------------------------------------------------------
        _palette = palette or [f"C{i}" for i in range(max_pulses)]
    
        for ax, (df, var) in zip(axes, zip(tagged, var_names)):
            sns.violinplot(
                data=df,
                x="pulse_number",
                y="prominence",
                inner="box",
                cut=0,
                palette=_palette[:max_pulses],
                ax=ax,
            )
            ax.set_xlabel("pulse number")
            ax.set_ylabel("peak prominence (a.u.)")
            ax.set_title(var)
            self.apply_axes_style(ax, style)
            sns.despine(ax=ax)
    
        plt.tight_layout()
    
        # ------------------------------------------------------------------
        # 4. Save bundle
        # ------------------------------------------------------------------
        if self.save_outputs:
            self.save_plot_bundle(
                fig=fig,
                plot_name="pulse_prominence_by_number",
                var_tag="_".join(var_names),
                points_df=combined_df,
                file_stem=file_stem,
            )
    
        plt.show()
    
        return fig, axes, combined_df

    def plot_representative_traces_with_peaks(
        self,
        df_raw_indexed,
        df_proc_indexed,
        peaks_list,
        vars,
        pairs=None,
        colors=None,
        conditions=None,
        seed=None,
        object_id=None,
        counts_df=None,
        min_peaks=1,
        figsize=(14, 6),
        linewidth_raw=0.8,
        linewidth_proc=1.2,
        file_stem=None,
        invert_vars=None,
    ):
        cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    
        # ------------------------------------------------------------------
        # 1. Colors
        # ------------------------------------------------------------------
        if colors is None:
            _cycle = ["#aa3377", "#292562", "#888888", "#1D9E75"]
            colors = {var: _cycle[i % len(_cycle)] for i, var in enumerate(vars)}
    
        # ------------------------------------------------------------------
        # 2. Candidate cells
        # ------------------------------------------------------------------
        df_cells = df_proc_indexed.reset_index()[cell_keys].drop_duplicates()
        if conditions is not None:
            df_cells = df_cells[df_cells["condition"].isin(conditions)]
        if counts_df is not None:
            df_cells = df_cells.merge(counts_df[cell_keys], on=cell_keys, how="inner")
    
        for peaks_df, var in zip(peaks_list, vars):
            peak_counts = (
                peaks_df.groupby(cell_keys)
                .size()
                .reset_index(name=f"n_{var}")
            )
            df_cells = df_cells.merge(peak_counts, on=cell_keys, how="inner")
            df_cells = df_cells[df_cells[f"n_{var}"] >= min_peaks]
    
        if len(df_cells) == 0:
            raise ValueError("No candidate cells found. Try lowering min_peaks or relaxing filters.")
    
        # ------------------------------------------------------------------
        # 3. Pick cell
        # ------------------------------------------------------------------
        if object_id is not None:
            row = df_cells[df_cells["object_id"] == object_id].iloc[0]
        else:
            rng = np.random.default_rng(seed)
            row = df_cells.iloc[rng.integers(len(df_cells))]
    
        cell = row[cell_keys].to_dict()
        print(f"Plotting cell: {cell}  ({len(df_cells)} candidates)")
    
        # ------------------------------------------------------------------
        # 4. Pull traces
        # ------------------------------------------------------------------
        def get_trace(df_idx):
            try:
                return df_idx.loc[
                    (cell["condition"], cell["folder"],
                     cell["repeat_id"], cell["object_id"])
                ].sort_index()
            except KeyError:
                raise ValueError(f"Cell {cell} not found in df_indexed.")
    
        raw_trace  = get_trace(df_raw_indexed)
        proc_trace = get_trace(df_proc_indexed)
    
        times_raw  = (raw_trace.index.values  - self.frame_zero) * self.dt + self.t_zero
        times_proc = (proc_trace.index.values - self.frame_zero) * self.dt + self.t_zero
    
        # ------------------------------------------------------------------
        # 5. Pull peaks
        # ------------------------------------------------------------------
        def get_cell_peaks(peaks_df):
            m = np.ones(len(peaks_df), dtype=bool)
            for k, v in cell.items():
                m &= peaks_df[k] == v
            return peaks_df[m].sort_values("peak_frame")
    
        cell_peaks = {var: get_cell_peaks(pdf) for var, pdf in zip(vars, peaks_list)}
    
       ## ------------------------------------------------------------------
        # 6. Plot
        # ------------------------------------------------------------------
        if pairs is None:
            from itertools import combinations
            var_pairs = list(combinations(vars, 2))
        else:
            # Accept either index pairs or var name pairs
            if len(pairs) > 0 and isinstance(pairs[0][0], int):
                var_pairs = [(vars[i], vars[j]) for i, j in pairs]
            else:
                var_pairs = pairs
    
        n_pairs = len(var_pairs)
    
        fig, axes = plt.subplots(2, n_pairs, figsize=figsize, sharex=True)
    
        # Handle single pair case where axes is 1D
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
    
        for col_idx, (var_a, var_b) in enumerate(var_pairs):
            ax_raw  = axes[0, col_idx]
            ax_raw2 = ax_raw.twinx()
            ax_proc = axes[1, col_idx]
    
            # --- Top: raw signal, twin y-axes ---
            for var, _ax in [(var_a, ax_raw), (var_b, ax_raw2)]:
                y = raw_trace[var].values.astype(float)
                if invert_vars is not None and var in invert_vars:
                    y = -y
                _ax.plot(times_raw, y, linewidth=linewidth_raw,
                         color=colors[var], alpha=0.9)
    
            ax_raw.set_ylabel(var_a.split("_")[0], color=colors[var_a])
            ax_raw2.set_ylabel(var_b.split("_")[0], color=colors[var_b])
            ax_raw.tick_params(axis="y", labelcolor=colors[var_a])
            ax_raw2.tick_params(axis="y", labelcolor=colors[var_b])
            sns.despine(ax=ax_raw, right=False)
            sns.despine(ax=ax_raw2, left=False)
    
            # --- Bottom: processed signal, single axis ---
            for var in [var_a, var_b]:
                y = proc_trace[var].values.astype(float)
                if invert_vars is not None and var in invert_vars:
                    y = -y
                ax_proc.plot(times_proc, y, linewidth=linewidth_proc,
                             color=colors[var], alpha=0.9)
    
                for _, pk in cell_peaks[var].iterrows():
                    pf = int(pk["peak_frame"])
                    pt = (pf - self.frame_zero) * self.dt + self.t_zero
                    if pf in proc_trace.index:
                        pval = proc_trace.loc[pf, var]
                        ax_proc.scatter(pt, pval, color=colors[var], zorder=5,
                                        s=20, edgecolors="white", linewidths=0.8)
    
            ax_proc.axhline(0, linewidth=0.5, linestyle=":", color="#aaaaaa", alpha=0.6)
            ax_proc.set_xlabel(f"time ({self.time_unit})")
            ymin, ymax = ax_proc.get_ylim()
            margin = (ymax - ymin) * 0.1
            ax_proc.set_ylim(ymin - margin, ymax + margin)
            sns.despine(ax=ax_proc)
    
            # y label only on leftmost panel
            if col_idx == 0:
                ax_proc.set_ylabel("signal (a.u.)")
    
            # Panel title
            ax_raw.set_title(
                f"{var_a.split('_')[0]} vs {var_b.split('_')[0]}",
                color="#555555"
            )
    
        title_str = (f"{cell['condition']}  |  folder {cell['folder']}  |  "
                     f"rep {cell['repeat_id']}  |  cell {cell['object_id']}")
        fig.suptitle(title_str, color="#555555", y=1.01)
        plt.subplots_adjust(hspace=0.25, wspace=0.5, top=0.93)
    
        # ------------------------------------------------------------------
        # 7. Save bundle
        # ------------------------------------------------------------------
        if self.save_outputs:
            rows = []
            for var in vars:
                # Raw trace
                for frame, val in zip(raw_trace.index.values, raw_trace[var].values):
                    rows.append({
                        "condition":  cell["condition"],
                        "folder":     cell["folder"],
                        "repeat_id":  cell["repeat_id"],
                        "object_id":  cell["object_id"],
                        "frame":      frame,
                        "time":       (frame - self.frame_zero) * self.dt + self.t_zero,
                        "signal":     var,
                        "panel":      "raw",
                        "value":      val,
                        "is_peak":    False,
                        "pulse_number": np.nan,
                    })
                # Processed trace + peak annotations
                for frame, val in zip(proc_trace.index.values, proc_trace[var].values):
                    pf_row = cell_peaks[var][cell_peaks[var]["peak_frame"] == frame]
                    is_peak = len(pf_row) > 0
                    pulse_num = int(pf_row["pulse_number"].values[0]) if (
                        is_peak and "pulse_number" in pf_row.columns
                    ) else np.nan
                    rows.append({
                        "condition":    cell["condition"],
                        "folder":       cell["folder"],
                        "repeat_id":    cell["repeat_id"],
                        "object_id":    cell["object_id"],
                        "frame":        frame,
                        "time":         (frame - self.frame_zero) * self.dt + self.t_zero,
                        "signal":       var,
                        "panel":        "processed",
                        "value":        val,
                        "is_peak":      is_peak,
                        "pulse_number": pulse_num,
                    })
    
            points_df = pd.DataFrame(rows)
            if file_stem is None:
                seed_str    = f"_s{seed}"    if seed    is not None else ""
                file_stem   = (f"c{cell['condition']}-f{cell['folder']}"
                               f"-r{cell['repeat_id']}-o{cell['object_id']}"
                               f"{seed_str}_p{min_peaks}")
            
            self.save_plot_bundle(
                fig=fig,
                plot_name="representative_traces",
                var_tag="_".join(v.split("_")[0] for v in vars),
                points_df=points_df,
                file_stem=file_stem,
            )
    
        plt.show()
        return fig, cell


    
    def compute_eta_statistics(cond_results, conditions=None, alpha=0.05):
        """
        Compute per-timepoint statistics on ETA windows.
        Tests whether the mean is significantly different from zero
        at each timepoint using a one-sample Wilcoxon signed-rank test.
    
        Parameters
        ----------
        cond_results : dict
            Output of normalise_eta_windows().
        conditions : list or None
        alpha : float
            Significance threshold.
    
        Returns
        -------
        stats_df : DataFrame
            One row per (condition, trigger, signal, timepoint) with columns:
            mean, sem, p_value, significant
        """
        from scipy.stats import wilcoxon, ttest_1samp
    
        rows = []
        for cond, data in cond_results.items():
            if conditions is not None and cond not in conditions:
                continue
            rel_time = data["rel_time"]
            for var, X in data["windows"].items():
                if X.shape[0] < 5:
                    continue
                for t_idx, t in enumerate(rel_time):
                    vals = X[:, t_idx]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) < 5:
                        continue
                    mean = np.mean(vals)
                    sem  = np.std(vals, ddof=1) / np.sqrt(len(vals))
                    # Wilcoxon tests if median is different from zero
                    try:
                        _, p = wilcoxon(vals)
                    except ValueError:
                        p = 1.0
                    rows.append({
                        "condition":   cond,
                        "signal":      var,
                        "rel_time":    t,
                        "mean":        mean,
                        "sem":         sem,
                        "p_value":     p,
                        "significant": p < alpha,
                        "n":           len(vals),
                    })
    
        return pd.DataFrame(rows)

def cell_name(cell):
    """
    Generate a standardised name string for a cell dict.
    e.g. {'condition': '235D_EGF', 'folder': '3', 'repeat_id': '8', 'object_id': 39}
         -> 'c235D_EGF-f3-r8-o39'
    """
    return (f"c{cell['condition']}-f{cell['folder']}"
            f"-r{cell['repeat_id']}-o{cell['object_id']}")
    
    
def save_cell_tiffs(
    cell,
    analysis,
    plotter,
    pad=100,
    save_crop=True,
    save_outline=True,
):
    out_dir = Path(plotter.out_dir) / "celltrack_img_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = cell_name(cell)

    # ── Find the right file ───────────────────────────────────────────────
    cell_keys = ["condition", "folder", "repeat_id", "object_id"]
    folder_idx   = int(cell["folder"]) - 1
    folder_files = analysis.all_filenames[folder_idx]

    fluor_path = None
    lbm_path   = None

    for filepath in folder_files:
        file_index = f"folder{cell['folder']}-file_{filepath.stem}"
        try:
            condition, folder, repeat_id = analysis.get_file_index_info(file_index)
        except ValueError:
            continue
        if condition == cell["condition"] and repeat_id == cell["repeat_id"]:
            fluor_path = filepath
            sub_path   = filepath.parent / filepath.stem
            candidates = [
                sub_path / f"LblImg_{filepath.stem}_trackfile.tif",
                sub_path / f"LblImg_{filepath.stem}-trackfile.tif",
            ]
            lbm_path = next((p for p in candidates if p.exists()), None)
            break

    if fluor_path is None or lbm_path is None:
        raise ValueError(f"Could not find files for cell {cell}")

    # ── Load stacks ───────────────────────────────────────────────────────
    fluor = tifffile.imread(fluor_path)   # (T, C, H, W)
    lbm   = tifffile.imread(lbm_path)    # (T, H, W)

    oid      = int(cell["object_id"])
    T, C, H, W = fluor.shape
    mask     = (lbm == oid)              # (T, H, W) bool

    # ── Mean centroid ─────────────────────────────────────────────────────
    centroids = []
    for t in range(T):
        if mask[t].any():
            cy, cx = center_of_mass(mask[t])
            centroids.append((cy, cx))

    if len(centroids) == 0:
        raise ValueError(f"No mask pixels found for object_id={oid}")

    mean_cy = int(round(np.mean([c[0] for c in centroids])))
    mean_cx = int(round(np.mean([c[1] for c in centroids])))

    # ── Crop bounds — shared by both crop outputs ─────────────────────────
    r0 = max(0, mean_cy - pad)
    r1 = min(H, mean_cy + pad)
    c0 = max(0, mean_cx - pad)
    c1 = min(W, mean_cx + pad)

    # ── Cropped fluorescence TIFF ─────────────────────────────────────────
    if save_crop:
        crop = fluor[:, :, r0:r1, c0:c1]
        crop_path = out_dir / f"{name}_raw_img.tif"
        tifffile.imwrite(str(crop_path), crop, imagej=True,
                         metadata={"axes": "TCYX"})
        print(f"[save_cell_tiffs] Saved crop        → {crop_path}")

    # ── Outline masks ─────────────────────────────────────────────────────
    if save_outline:
        outline_full = np.zeros((T, H, W), dtype=np.uint8)
        outline_crop = np.zeros((T, r1 - r0, c1 - c0), dtype=np.uint8)

        for t in range(T):
            if mask[t].any():
                eroded = binary_erosion(mask[t], iterations=1)
                outline = (mask[t] & ~eroded).astype(np.uint8)
                outline_full[t]  = outline
                outline_crop[t]  = outline[r0:r1, c0:c1]

        full_path = out_dir / f"{name}_cell_outline_full.tif"
        tifffile.imwrite(str(full_path), outline_full, imagej=True,
                         metadata={"axes": "TYX"})
        print(f"[save_cell_tiffs] Saved outline full → {full_path}")

        crop_path = out_dir / f"{name}_cell_outline_crop.tif"
        tifffile.imwrite(str(crop_path), outline_crop, imagej=True,
                         metadata={"axes": "TYX"})
        print(f"[save_cell_tiffs] Saved outline crop → {crop_path}")

    return out_dir

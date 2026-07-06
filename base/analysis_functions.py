
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.optimize
from scipy import ndimage
import os, csv
from itertools import product
import tifffile
from pathlib import Path
import re

class ImageProcessor():
    def __init__(self, image_paths, mode, **kwargs):

        # ----------------------------
        # Input paths / files
        # ----------------------------
        self.image_paths = image_paths
        self.all_filenames = [
            list(path.glob("*.tif"))
            for path in self.image_paths
        ]

        self.mode = mode  # "whole-fov" / "cellpose" / "trackmate"
        
        self.nuc_bin_masks = None
        self.cyto_outlines = None
        self.cyto_ring_masks = None
    
        # ----------------------------
        # Image metadata (filled later)
        # ----------------------------
        self.H = None   # height (pixels)
        self.W = None   # width  (pixels)
        self.F = None   # number of frames (set per movie)
    
        # ----------------------------
        # Segmentation / filtering params
        # ----------------------------
        self.nuc_area_threshold = 500   # area integrated across fov, in pixels
        self.cell_area_threshold = 2000 # area integrated across fov, in pixels
        self.margin = 150       # fiducial margin in pixels
    
        # ----------------------------
        # Channel ordering
        # ----------------------------
        self.channels = ["BFP", "GFP", "FR", "iRFP"]

        # ----------------------------
        # Canonincal df schema for all three modes
        # ----------------------------
        self.index_names = ["condition", "folder", "repeat_id", "frame", "object_id"]
    
        # ----------------------------
        # Main outputs
        # ----------------------------
        self.df = None          # built after process_images()
        self.norm_df = None     # time-normalized df
        self.events_df = None   # built only for tracking mode

        for k,v in kwargs.items():
            setattr(self, k, v)


    def get_file_index_info(self, file_index):
        name = Path(file_index).stem

        pattern = re.compile(
        r"""
        folder(?P<folder>\d+)-       # folder prefix (required)
        file_
        (?P<repeat_id>\d+)           # repeat id
        [_-]
        (?P<condition>.+?)           # condition
        (?:[_-](?P<replicate>\d+))?  # replicate (parsed but NOT used as folder)
        $
        """,
        re.VERBOSE
        )

        m = pattern.match(name)
        if not m:
            raise ValueError(f"Could not parse filename: {name}")

        return (m['condition'], m['folder'], m['repeat_id'])
 

    def compute_vars_cellpose_frame_fast(
        self,
        frame,
        labels,
        obj_ids=None,
    ):
        """
        Vectorized per-object quantification for ONE frame.
    
        Parameters
        ----------
        frame : int
            Frame index
    
        labels : 2D array
            Segmentation label image (0 = background)
    
        obj_ids : array-like or None
            Object IDs to measure. If None, measure all nonzero IDs.
    
        Returns
        -------
        rows : list[dict]
            One row per object with measurements.
        """

        # Ensure integer labels
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int32)
    
        # ----------------------------
        # Object IDs in this frame
        # ----------------------------
        if obj_ids is None:
            obj_ids = np.unique(labels)
            obj_ids = obj_ids[obj_ids != 0]
    
        if len(obj_ids) == 0:
            return []
    
        # ----------------------------
        # Extract intensity channels (2D)
        # ----------------------------
        FR  = self.tiff_data[frame, self.channels.index("FR")]
        GFP = self.tiff_data[frame, self.channels.index("GFP")]
        BFP = self.tiff_data[frame, self.channels.index("BFP")]
        iRFP= self.tiff_data[frame, self.channels.index("iRFP")]
    
        # Masks (2D)
        nuc = self.nuc_bin_masks[frame]
        mem = self.cyto_outlines[frame]
        cell = self.cell_bin_masks[frame]

        # ----------------------------
        # Vectorized sums over objects
        # ----------------------------
        FR_nuc   = ndimage.sum(FR  * nuc, labels, index=obj_ids)
        FR_mem   = ndimage.sum(FR  * mem, labels, index=obj_ids)
    
        GFP_nuc  = ndimage.sum(GFP * nuc, labels, index=obj_ids)
        GFP_mem  = ndimage.sum(GFP * mem, labels, index=obj_ids)
    
        BFP_nuc  = ndimage.sum(BFP * nuc, labels, index=obj_ids)
        BFP_mem  = ndimage.sum(BFP * mem, labels, index=obj_ids)

        iRFP_nuc = ndimage.sum(iRFP * nuc, labels, index=obj_ids)
        iRFP_mem = ndimage.sum(iRFP * mem, labels, index=obj_ids)
    
        nuc_area = ndimage.sum(nuc, labels, index=obj_ids)
        mem_area = ndimage.sum(mem, labels, index=obj_ids)
        cell_area = ndimage.sum(cell, labels, index=obj_ids)

        centroids = ndimage.center_of_mass(
            np.ones_like(labels),
            labels,
            index=obj_ids
        )
    
        # ----------------------------
        # Build rows
        # ----------------------------
        rows = []
    
        for oid, (y,x), frn, frm, gfn, gfm, bfn, bfm, irn, irm, na, ma, ca in zip(
            obj_ids,
            centroids,
            FR_nuc, FR_mem,
            GFP_nuc, GFP_mem,
            BFP_nuc, BFP_mem,
            iRFP_nuc, iRFP_mem,
            nuc_area, mem_area,
            cell_area
        ):
            oid = int(oid)

            frnc = frn / (na + 1e-3)
            frmc = frm / (ma + 1e-3)
            frnmr = frn / (frm + 1e-3)
            frnmcr = frnc / (frmc + 1e-3)
            gfnc = gfn / (na + 1e-3)
            gfmc = gfm / (ma + 1e-3)
            gfnmcr = gfnc / (gfmc + 1e-3)
            bfnc = bfn / (na + 1e-3)
            bfmc = bfm / (ma + 1e-3)
            bfnmcr = bfnc / (bfmc + 1e-3)
            irnc = irn / (na + 1e-3)
            irmc = irm / (ma + 1e-3)
            irnmcr = irnc / (irmc + 1e-3)
            
            gfirmr = gfm / (irm + 1e-3)
            frbfnr = frn / (bfn + 1e-3)
            frirmr = frm / (irm + 1e-3)
            gffrmr = gfm / (frm + 1e-3)
            frgfmr = frm / (gfm + 1e-3)
    
            row = {
                "frame": frame,
                "object_id": oid,
    
                "FR_nuc": frn,
                "FR_mem": frm,
                "GFP_nuc": gfn,
                "GFP_mem": gfm,
                "BFP_nuc": bfn,
                "BFP_mem": bfm,
                "iRFP_nuc": irn,
                "iRFP_mem": irm,
    
                "nuc_area": na,
                "mem_area": ma,
                "cell_area": ca,

                # Centroid coordinates
                "x": float(x),
                "y": float(y),
    
                # Derived quantities
                "FR_nuc_conc": frnc,
                "FR_mem_conc": frmc,
                "GFP_nuc_conc": gfnc,
                "GFP_mem_conc": gfmc,
                "BFP_nuc_conc": bfnc,
                "BFP_mem_conc": bfmc,
                "FR_nuc_mem_ratio": frnmr,
                "FR_nuc_mem_conc_ratio": frnmcr,
                "BFP_nuc_mem_conc_ratio": bfnmcr,
                "iRFP_nuc_mem_conc_ratio": irnmcr,
                "GFP_iRFP_mem_ratio": gfirmr,
                "FR_BFP_nuc_ratio": frbfnr,
                "FR_iRFP_mem_ratio": frirmr,
                "GFP_FR_mem_ratio": gffrmr,
                "FR_GFP_mem_ratio": frgfmr
            }

            rows.append(row)

        # Add cyto ring information if available
        if self.cyto_ring_masks is not None:
            cr =  self.cyto_ring_masks[frame] # Mask
            iRFP_cr = ndimage.sum(iRFP * cr, labels, index=obj_ids) # Intensity
            cr_area = ndimage.sum(cr, labels, index=obj_ids) # Area
            for i, (ircr, cra, irn) in enumerate(zip(iRFP_cr, cr_area, iRFP_nuc)):
                ircrc = ircr / (cra + 1e-3)
                irnc = irn / (cra + 1e-3)
                ircrnr = ircr / (irn + 1e-3)
                ircrncr = ircrc / (irnc + 1e-3)
                
                rows[i]['iRFP_cr'] = ircr
                rows[i]['cr_area'] = cra
                rows[i]['iRFP_cr_conc'] = ircrc
                rows[i]['iRFP_cr_nuc_ratio'] = ircrnr
                rows[i]['iRFP_cr_nuc_conc_ratio'] = ircrncr
                    
        return rows


    def compute_vars(self, tiff_data, nuc_mask, cyto_mask, cell_mask, cyto_ring_mask, axis):
        """
        Compute fluorescence variables given *binary* masks.
    
        Returns dict of arrays (one value per frame).
        """
    
        GFP  = tiff_data[:, self.channels.index("GFP")]
        FR   = tiff_data[:, self.channels.index("FR")]
        BFP  = tiff_data[:, self.channels.index("BFP")]
        iRFP = tiff_data[:, self.channels.index("iRFP")]
    
        vs = {}
    
        # Intensities
        vs["FR_nuc"]   = np.sum(FR   * nuc_mask, axis=axis)
        vs["FR_mem"]   = np.sum(FR   * cyto_mask, axis=axis)
        vs["GFP_nuc"]  = np.sum(GFP  * nuc_mask, axis=axis)
        vs["GFP_mem"]  = np.sum(GFP  * cyto_mask, axis=axis)
        vs["BFP_nuc"]  = np.sum(BFP  * nuc_mask, axis=axis)
        vs["BFP_mem"]  = np.sum(BFP  * cyto_mask, axis=axis)
        vs["iRFP_nuc"] = np.sum(iRFP * nuc_mask, axis=axis)
        vs["iRFP_mem"] = np.sum(iRFP * cyto_mask, axis=axis)
    
        # Areas
        vs["nuc_area"] = np.sum(nuc_mask, axis=axis)
        vs["mem_area"] = np.sum(cyto_mask, axis=axis)
        vs["cell_area"] = np.sum(cell_mask, axis=axis)
    
        # Concentrations
        vs["FR_nuc_conc"] = vs["FR_nuc"] / (vs["nuc_area"] + 1e-3)
        vs["FR_mem_conc"] = vs["FR_mem"] / (vs["mem_area"] + 1e-3)
        vs["GFP_nuc_conc"] = vs["GFP_nuc"] / (vs["nuc_area"] + 1e-3)
        vs["GFP_mem_conc"] = vs["GFP_mem"] / (vs["mem_area"] + 1e-3)
        vs["BFP_nuc_conc"] = vs["BFP_nuc"] / (vs["nuc_area"] + 1e-3)
        vs["BFP_mem_conc"] = vs["BFP_mem"] / (vs["mem_area"] + 1e-3)
        vs["iRFP_nuc_conc"] = vs["iRFP_nuc"] / (vs["nuc_area"] + 1e-3)
        vs["iRFP_mem_conc"] = vs["iRFP_mem"] / (vs["mem_area"] + 1e-3)
    
        # Ratios
        vs["FR_nuc_mem_ratio"] = vs["FR_nuc"] / (vs["FR_mem"] + 1e-3)
        vs["FR_nuc_mem_conc_ratio"] = vs["FR_nuc_conc"] / (vs["FR_mem_conc"] + 1e-3)
        vs["GFP_nuc_mem_conc_ratio"] = vs["GFP_nuc_conc"] / (vs["GFP_mem_conc"] + 1e-3)
        vs["BFP_nuc_mem_conc_ratio"] = vs["BFP_nuc_conc"] / (vs["BFP_mem_conc"] + 1e-3)
        vs["iRFP_nuc_mem_conc_ratio"] = vs["iRFP_nuc_conc"] / (vs["iRFP_mem_conc"] + 1e-3)
        
        vs["GFP_iRFP_mem_ratio"] = vs["GFP_mem"] / (vs["iRFP_mem"] + 1e-3)
        vs["FR_BFP_nuc_ratio"] = vs["FR_nuc"] / (vs["BFP_nuc"] + 1e-3)
        vs["FR_iRFP_mem_ratio"] = vs["FR_mem"] / (vs["iRFP_mem"] + 1e-3)
        vs["GFP_FR_mem_ratio"] = vs["GFP_mem"] / (vs["FR_mem"] + 1e-3)
        vs["FR_GFP_mem_ratio"] = vs["FR_mem"] / (vs["GFP_mem"] + 1e-3)
        

        if cyto_ring_mask is not None:
            vs["iRFP_cr"] = np.sum(iRFP * cyto_ring_mask, axis=axis)
            vs["BFP_cr"] = np.sum(BFP * cyto_ring_mask, axis=axis)
            vs["FR_cr"] = np.sum(FR * cyto_ring_mask, axis=axis)
            vs["cr_area"] = np.sum(cyto_ring_mask, axis=axis)
            vs["iRFP_cr_conc"] = vs["iRFP_cr"] / (vs["cr_area"] + 1e-3)
            vs["iRFP_cr_nuc_ratio"] = vs["iRFP_cr"] / (vs["iRFP_nuc"] + 1e-3)
            vs["iRFP_cr_nuc_conc_ratio"] = vs["iRFP_cr_conc"] / (vs["iRFP_nuc_conc"] + 1e-3)
            vs["BFP_cr_conc"] = vs["BFP_cr"] / (vs["cr_area"] + 1e-3)
            vs["BFP_cr_nuc_ratio"] = vs["BFP_cr"] / (vs["BFP_nuc"] + 1e-3)
            vs["BFP_cr_nuc_conc_ratio"] = vs["BFP_cr_conc"] / (vs["BFP_nuc_conc"] + 1e-3)
            vs["FR_cr_conc"] = vs["FR_cr"] / (vs["cr_area"] + 1e-3)
            vs["FR_cr_nuc_ratio"] = vs["FR_cr"] / (vs["FR_nuc"] + 1e-3)
            vs["FR_cr_nuc_conc_ratio"] = vs["FR_cr_conc"] / (vs["FR_nuc_conc"] + 1e-3)
    
        return vs


    def append_timeseries_rows(
        self,
        rows,
        condition, folder, repeat_id,
        object_id,
        values_dict
    ):
        """
        Append one row per frame for this object_id.
        """
    
        F = len(next(iter(values_dict.values())))
    
        for frame in range(F):
            row = {
                "condition": condition,
                "folder": folder,
                "repeat_id": repeat_id,
                "frame": frame,
                "object_id": object_id,
            }
    
            for k, arr in values_dict.items():
                row[k] = float(arr[frame])
    
            rows.append(row)

    def process_images(self):

        # ------------------------------------------------------------
        # Accumulate all measurement rows here
        # ------------------------------------------------------------
        all_rows = []
        
        # Track-level division events (trackmate only)
        event_rows = []
        
        # ------------------------------------------------------------
        # Loop over all movies
        # ------------------------------------------------------------
        for path_num, path_filenames in enumerate(self.all_filenames):
        
            for filename in path_filenames:
        
                file_index = f"folder{path_num+1}-file_{filename.stem}"
                print(f"\nProcessing file: {filename}")
        
                condition, folder, repeat_id = self.get_file_index_info(file_index)
        
                sub_path = Path(filename.parent, filename.stem)
        
                # ----------------------------
                # Load image stacks
                # ----------------------------
                tiff_data = tifffile.imread(filename)
                self.tiff_data = tiff_data
                F = tiff_data.shape[0]
                self.F = F
                if self.H is None or self.W is None:
                    self.H, self.W = tiff_data.shape[-2:]

                cell_bin_masks = tifffile.imread(
                    Path(sub_path, "all_cyto_masks.tif")
                )
                # Make binary
                cell_bin_masks = (cell_bin_masks > 0).astype(np.uint8)
                self.cell_bin_masks = cell_bin_masks
        
                nuc_bin_masks = tifffile.imread(
                    Path(sub_path, "all_nuc_bin_masks.tif")
                )
                self.nuc_bin_masks = nuc_bin_masks
                
                cyto_outlines = tifffile.imread(
                    Path(sub_path, "all_cyto_outlines.tif")
                )
                self.cyto_outlines = cyto_outlines

                if Path(sub_path, 'all_cyto_ring_masks.tif').exists():
                    cr_masks = tifffile.imread(
                        Path(sub_path, 'all_cyto_ring_masks.tif')
                    )
                else:
                    cr_masks = None
                self.cyto_ring_masks = cr_masks
        
                print(f"Frames: {F}")
        
                # ============================================================
                # MODE 1: Whole field of view
                # ============================================================
                if self.mode == 'whole-fov':
        
                    object_id = 0
        
                    values = self.compute_vars(
                        tiff_data,
                        nuc_bin_masks,
                        cyto_outlines,
                        cell_bin_masks,
                        cr_masks,
                        axis=(1, 2)
                    )
        
                    self.append_timeseries_rows(
                        all_rows,
                        condition, folder, repeat_id,
                        object_id,
                        values
                    )
        
                    continue
        

                # ============================================================
                # MODE 2: Cellpose segmentation (objects per frame)
                # ============================================================
                if self.mode == "cellpose":

                    # roi_masks is non-binary cell_masks (numbered)
                    roi_masks = tifffile.imread(
                        Path(sub_path, "all_cyto_masks.tif")
                    ).astype(np.int32)
                    self.roi_masks = roi_masks
        
                    # Loop frame-by-frame (no object loop)
                    for frame in range(F):

                        labels = roi_masks[frame]

                        # Compute ALL object measurements in this frame
                        frame_rows = self.compute_vars_cellpose_frame_fast(frame, labels)

                        # Add experiment metadata
                        for row in frame_rows:
                            row.update({
                                "condition": condition,
                                "folder": folder,
                                "repeat_id": repeat_id,
                            })
                            
                        all_rows.extend(frame_rows)
        
                # ============================================================
                # MODE 3: TrackMate tracking (persistent objects)
                # ============================================================
                elif self.mode == "trackmate":

                    paths = [
                        Path(sub_path, f"LblImg_{filename.stem}_trackfile.tif"),
                        Path(sub_path, f"LblImg_{filename.stem}-trackfile.tif"),
                    ]
                    
                    roi_masks = tifffile.imread(
                        next(p for p in paths if p.exists())
                    )

                    
                    track_info = pd.read_csv(
                        Path(sub_path, f"{filename.stem}_spots.csv"),
                        # low_memory=False
                    )
        
                    track_ids = (
                        track_info["TRACK_ID"]
                        .dropna()
                        .unique()
                    )
        
                    track_ids = track_ids[2:].astype(int) + 1
        
                    # Loop tracks
                    for track_id in track_ids:
        
                        roi = (roi_masks == track_id).astype(np.uint8)

                        nuc_masked = roi * nuc_bin_masks #changed from nuc_masks to nuc_bin_masks
                        cyto_masked = roi * cyto_outlines
                        cr_masked = roi * cr_masks if cr_masks is not None else None

                        # # Check that there are no zeros in this nuc_masked
                        # if np.any(nuc_masked):
                        #     continue
        
                        # Compute full time series for this track
                        values = self.compute_vars(
                            tiff_data,
                            nuc_masked,
                            cyto_masked,
                            roi,
                            cr_masked,
                            axis=(1,2)
                        )
        
                        self.append_timeseries_rows(
                            all_rows,
                            condition, folder, repeat_id,
                            object_id=int(track_id),
                            values_dict=values
                        )
        
                        # -------- Division t0/t_end detection --------
                        track_id_str = str(track_id - 1)
        
                        counts = (
                            track_info[track_info["TRACK_ID"] == track_id_str]
                            .groupby("FRAME")
                            .size()
                        )
        
                        div_frames = counts[counts > 1].index
        
                        if len(div_frames) > 0:
                            t0 = int(div_frames.min())
                            t_end = int(div_frames.max())
                        else:
                            t0 = np.nan
                            t_end = np.nan
        
                        event_rows.append({
                            "condition": condition,
                            "folder": folder,
                            "repeat_id": repeat_id,
                            "object_id": int(track_id),
                            "t0": t0,
                            "t_end": t_end,
                        })
        
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
        
        # ============================================================
        # FINAL: Build dataframe once
        # ============================================================
        print("\nBuilding dataframe...")
        
        self.df = pd.DataFrame(all_rows)

        if self.mode == "cellpose":
        
            # ----------------------------
            # Area thresholds
            # ----------------------------
            self.df = self.df[self.df["nuc_area"] >= self.nuc_area_threshold]
            self.df = self.df[self.df["cell_area"] >= self.cell_area_threshold]
        
            # ----------------------------
            # Margin exclusion
            # ----------------------------
            self.df = self.df[
                (self.df["x"] > self.margin) &
                (self.df["x"] < self.W - self.margin) &
                (self.df["y"] > self.margin) &
                (self.df["y"] < self.H - self.margin)
            ]
        
        self.df = self.df.set_index(
            ["condition", "folder", "repeat_id", "frame", "object_id"]
        ).sort_index()
        
        print("Final df shape:", self.df.shape)
        
        # ============================================================
        # FINAL: Build events dataframe once
        # ============================================================
        if len(event_rows) > 0:
        
            self.events_df = pd.DataFrame(event_rows)
        
            self.events_df = self.events_df.set_index(
                ["condition", "folder", "repeat_id", "object_id"]
            ).sort_index()
        
            print("Division events shape:", self.events_df.shape)
        
        else:
            self.events_df = None


    def normalize_variables(self, avg_over_n=5, cellpose_stat="mean"):
        """
        Build a normalized dataframe.
    
        Trackmate / whole-fov:
          norm_df is just df with norm_* columns added
    
        Cellpose:
          norm_df is a collapsed FOV-per-frame dataframe
          normalized per replicate movie.
        """
    
        df = self.df.reset_index()
    
        # Numeric variables only
        var_cols = df.select_dtypes(include="number").columns
        var_cols = [c for c in var_cols if c not in ["frame", "track_id", "repeat_id", "object_id"]]
    
        # ============================================================
        # Trackmate: normalize per track
        # ============================================================
        if self.mode == "trackmate":
    
            group_cols = ["condition", "folder", "repeat_id", "object_id"]
    
            for var in var_cols:
                baseline = (
                    df.groupby(group_cols)[var]
                      .transform(lambda x: x.iloc[:avg_over_n].mean())
                )
                df[f"norm_{var}"] = df[var] / baseline
    
            self.norm_df = df
            # return self.norm_df
    
        # ============================================================
        # Whole-FOV: normalize per movie replicate
        # ============================================================
        elif self.mode == "whole-fov":
    
            group_cols = ["condition", "folder", "repeat_id"]
    
            for var in var_cols:
                baseline = (
                    df.groupby(group_cols)[var]
                      .transform(lambda x: x.iloc[:avg_over_n].mean())
                )
                df[f"norm_{var}"] = df[var] / baseline
    
            self.norm_df = df
            # return self.norm_df
    
        # ============================================================
        # Cellpose: collapse → normalize per movie replicate
        # ============================================================
        elif self.mode == "cellpose":
    
            # Collapse objects → one mean value per frame per replicate
            df_fov = (
                df.groupby(["condition", "folder", "repeat_id", "frame"],
                           as_index=False)[var_cols]
                  .agg(cellpose_stat)
            )
    
            group_cols = ["condition", "folder", "repeat_id"]
    
            for var in var_cols:
                baseline = (
                    df_fov.groupby(group_cols)[var]
                          .transform(lambda x: x.iloc[:avg_over_n].mean())
                )
                df_fov[f"norm_{var}"] = df_fov[var] / baseline
    
            self.norm_df = df_fov
            # return self.norm_df

        else:
            raise ValueError(f"Unknown mode: {self.mode}")



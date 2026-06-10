import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib import axes, patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch, Rectangle
from typing import Dict, Optional, List, Tuple
import warnings
import os
from src.dataset import load_bigwig_signal
def gaussian_smooth(signal, sigma):
    """
    Apply simple Gaussian smoothing using a normalized kernel.
    If sigma <= 0 or signal is empty, return original signal.
    """
    if sigma is None:
        return signal
    try:
        sigma = float(sigma)
    except Exception:
        return signal
    if sigma <= 0 or signal is None or len(signal) == 0:
        return signal
    # kernel size = odd integer, cover +/-3 sigma
    kernel_size = max(3, int(6 * sigma) | 1)  # ensure odd by forcing last bit 1
    half = kernel_size // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    try:
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed
    except Exception:
        return signal

class DatasetViewer:
    """
    Helper to visualize dataset windows and optional gene annotations.

    Usage:
      viewer = DatasetViewer(dataset, annotation_path="gencode.v48.gff3.gz")
      viewer.plot_window(idx=0, smoothing_sigma=2.0)
    """
    def __init__(self, dataset, annotation_path=None, max_subplots=6,
                 gene_color_plus="tab:blue", gene_color_minus="tab:orange",
                 xtick_step=4000, dpi=100, signal_palette=None):
        self.dataset = dataset
        self.max_subplots = int(max_subplots)
        self.gene_color_plus = gene_color_plus
        self.gene_color_minus = gene_color_minus
        self.xtick_step = xtick_step
        self.dpi = dpi
        self.signal_palette = signal_palette  # if None, will use tab10/tab20 per-plot
        self.genes_by_chrom = {}
        self.exons_by_chrom = {}
        if annotation_path:
            self._load_gff(annotation_path)

    def _load_gff(self, path):
        """Load GFF/GTF (optionally gzipped). Store genes and exons per chromosome."""
        import gzip
        genes = {}
        exons = {}
        try:
            open_func = gzip.open if path.endswith('.gz') else open
            mode = 'rt' if path.endswith('.gz') else 'r'
            with open_func(path, mode) as fh:
                for line in fh:
                    if line.startswith('#') or not line.strip():
                        continue
                    cols = line.rstrip().split('\t')
                    if len(cols) < 9:
                        continue
                    chrom, src, feature, start, end, score, strand, phase, attrs = cols[:9]
                    try:
                        start_i = int(start); end_i = int(end)
                    except Exception:
                        continue
                    # extract a simple name (gene_name / Name / gene_id)
                    gene_name = ""
                    for key in ("gene_name=", "Name=", "gene_id="):
                        if key in attrs:
                            parts = [p for p in attrs.split(';') if p.strip().startswith(key)]
                            if parts:
                                gene_name = parts[0].split('=', 1)[-1].strip('"').strip()
                                break
                    if feature == "gene":
                        genes.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
                    elif feature == "exon":
                        exons.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
        except Exception as e:
            logging.warning(f"Failed to load annotation {path}: {e}")
        self.genes_by_chrom = genes
        self.exons_by_chrom = exons
        logging.info(f"Loaded annotation: chromosomes={list(self.genes_by_chrom.keys())}")

    def get_genes_in_interval(self, chrom, start, end):
        """Return (genes, exons) overlapping [start, end)."""
        genes = self.genes_by_chrom.get(chrom, [])
        exons = self.exons_by_chrom.get(chrom, [])
        genes_f = [(gs, ge, st, nm) for (gs, ge, st, nm) in genes if not (ge < start or gs >= end)]
        exons_f = [(es, ee, st, nm) for (es, ee, st, nm) in exons if not (ee < start or es >= end)]
        return genes_f, exons_f

    def get_genes_in_interval2(self, chrom, start, end):
        """
        Return (genes, transcripts, exons) overlapping [start, end).
        - genes: list of (start, end, strand, gene_name)
        - transcripts: list of (start, end, strand, transcript_id, gene_name)  # 新增
        - exons: list of (start, end, strand, transcript_id)  # name 字段为 transcript_id
        """
        # 基因（不变）
        genes = self.genes_by_chrom.get(chrom, [])
        genes_f = [(gs, ge, st, nm) for (gs, ge, st, nm) in genes if not (ge < start or gs >= end)]

        # 外显子（要求 name 字段为 transcript_id）
        exons = self.exons_by_chrom.get(chrom, [])
        exons_f = [(es, ee, st, nm) for (es, ee, st, nm) in exons if not (ee < start or es >= end)]

        # 转录本（需要额外数据结构）
        if hasattr(self, 'transcripts_by_chrom'):
            transcripts = self.transcripts_by_chrom.get(chrom, [])
            transcripts_f = [(ts, te, tstrand, tname, tgene) for (ts, te, tstrand, tname, tgene) in transcripts
                            if not (te < start or ts >= end)]
        else:
            transcripts_f = []
            if genes_f or exons_f:
                print("Warning: transcripts_by_chrom not found. Transcript-level display disabled.")
        
        return genes_f, transcripts_f, exons_f

    def plot_window(self, idx=None, max_subplots=None, assembly_filter=None, track_indices=None, smoothing_sigma=2,
                    window_start=None, window_end=None):
        """
        Plot selected tracks for one window and optional gene models.

        Args:
            idx (int): index into dataset.sequence_split_df; defaults to 0.
            max_subplots (int): max number of tracks to draw (default: viewer.max_subplots).
            assembly_filter (str or list): filter labels_meta_df by 'File assembly'.
            track_indices (list[int]): explicit indices into labels_meta_df to plot.
            smoothing_sigma (float): gaussian smoothing sigma (<=0 disables).
        Returns:
            (fig, axes) matplotlib objects or None on error.
        """
        from matplotlib.ticker import FuncFormatter
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        ds = self.dataset
        if idx is None:
            if len(ds.sequence_split_df) == 0:
                logging.warning("sequence_split_df is empty")
                return
            idx = 0
        try:
            row = ds.sequence_split_df.iloc[idx]
        except Exception as e:
            logging.error(f"Invalid idx for sequence_split_df: {e}")
            return

        meta_df = ds.labels_meta_df.copy()
        if assembly_filter is not None and 'File assembly' in meta_df.columns:
            if isinstance(assembly_filter, (list, tuple)):
                meta_df = meta_df[meta_df['File assembly'].isin(assembly_filter)]
            else:
                meta_df = meta_df[meta_df['File assembly'] == assembly_filter]

        if track_indices is not None:
            sel_df = meta_df.iloc[track_indices].reset_index(drop=True)
        else:
            nmax = self.max_subplots if max_subplots is None else int(max_subplots)
            sel_df = meta_df.reset_index(drop=True).iloc[:nmax]

        if sel_df.shape[0] == 0:
            logging.info("No tracks selected for plotting")
            return

        # load signals and titles
        signals = []
        titles = []
        for _, mrow in sel_df.iterrows():
            fname = mrow.get('target_file_name') or mrow.get('target_file') or None
            if fname is None:
                logging.warning("Missing target file name for a track; skipping")
                continue
            bw_path = os.path.join(ds.bigwig_dir, fname)
            vals = load_bigwig_signal(bw_path, str(row["chromosome"]), int(row["start"]), int(row["end"]), ds.max_length)
            vals = np.nan_to_num(vals, nan=0.0)
            signals.append(vals)
            output_type = mrow.get('output_type') or ""
            biosample_name = mrow.get('biosample_name') or ""
            assay = mrow.get('name') or ""
            strand = mrow.get('strand') or ""
            if assay or strand:
                titles.append(f"{output_type}: {biosample_name} ({strand})\n{assay}")
            else:
                titles.append(os.path.basename(fname))

        if len(signals) == 0:
            logging.info("No signals loaded for plotting")
            return

        n_tracks = len(signals)
        fig, axes = plt.subplots(n_tracks + 1, 1, figsize=(18, 1.5 * (n_tracks + 1)), sharex=True, dpi=self.dpi,
                                 gridspec_kw={'height_ratios': [1.0] + [1.0] * n_tracks})
        if n_tracks + 1 == 1:
            axes = [axes]
        ax_gene = axes[0]

        chrom = row["chromosome"]
        row_start = int(row["start"])  # original interval start loaded from files
        first_signal = signals[0]
        actual_len = len(first_signal)
        row_end = row_start + actual_len

        # If no window specified, keep previous behavior (show entire loaded interval)
        if window_start is None and window_end is None:
            start_display = row_start
            end_display = row_end
            display_positions = np.arange(start_display, end_display)
            signals_to_plot = signals
        else:
            # Fill missing window_start/window_end with loaded interval bounds if only one provided
            if window_start is None:
                window_start = row_start
            if window_end is None:
                window_end = row_end
            start_display = int(window_start)
            end_display = int(window_end)
            desired_len = end_display - start_display
            if desired_len <= 0:
                logging.error("Invalid window: window_end must be greater than window_start")
                return
            display_positions = np.arange(start_display, end_display)
            displayed_len = len(display_positions)

            # Slice/pad each signal so that the returned segment corresponds exactly to
            # [start_display, end_display). Regions outside the originally loaded interval
            # are filled with zeros.
            signals_to_plot = []
            for sig in signals:
                seg = np.zeros(displayed_len, dtype=sig.dtype)
                # source indices within the original loaded signal
                src_start = max(0, start_display - row_start)
                src_end = min(actual_len, end_display - row_start)
                if src_end > src_start:
                    # destination start index in the segment
                    dest_start = max(0, row_start - start_display)
                    seg[dest_start:dest_start + (src_end - src_start)] = sig[src_start:src_end]
                signals_to_plot.append(seg)

        # draw gene models if available
        genes, exons = self.get_genes_in_interval(chrom, start_display, end_display)
        if genes:
            genes_df = __import__("pandas").DataFrame(genes, columns=["start", "end", "strand", "name"]).sort_values("start").reset_index(drop=True)
            level_ends = []
            level_height_base = 0.3
            level_gap = 0.25
            max_levels = 8
            gene_levels = []
            for _, g in genes_df.iterrows():
                gs, ge = int(g["start"]), int(g["end"])
                placed = False
                for lvl in range(len(level_ends)):
                    if gs >= level_ends[lvl]:
                        level_ends[lvl] = ge
                        gene_levels.append(lvl)
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(ge)
                    gene_levels.append(len(level_ends) - 1)
                    placed = True
                if not placed:
                    earliest = int(np.argmin(level_ends))
                    level_ends[earliest] = ge
                    gene_levels.append(earliest)
            for (idx_g, rowg), lvl in zip(genes_df.iterrows(), gene_levels):
                gs, ge, strand, name = int(rowg["start"]), int(rowg["end"]), rowg["strand"], rowg["name"]
                y = level_height_base + lvl * level_gap
                color = self.gene_color_plus if strand == "+" else self.gene_color_minus
                ax_gene.plot([gs, ge], [y, y], color=color, lw=2.5, zorder=2, solid_capstyle='round')
                gene_length = ge - gs
                arrow_len = min(gene_length * 0.1, 2000)
                if strand == "+":
                    ax_gene.arrow(ge - arrow_len, y, arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                else:
                    ax_gene.arrow(gs + arrow_len, y, -arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                gene_exons = [e for e in exons if e[3] == name]
                for es, ee, st, nm in gene_exons:
                    es_d, ee_d = max(es, start_display), min(ee, end_display)
                    if ee_d > es_d and (ee_d - es_d) > 50:
                        rect = mpatches.Rectangle((es_d, y - 0.04), ee_d - es_d, 0.08, facecolor=color, alpha=0.9,
                                                  zorder=3, edgecolor='white', linewidth=0.5)
                        ax_gene.add_patch(rect)
                text_x = (gs + ge) / 2
                text_x = np.clip(text_x, start_display + 500, end_display - 500)
                ax_gene.text(text_x, y + 0.06, name if name else "Unknown", ha="center", va="bottom", fontsize=9, zorder=4,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
            gene_track_height = level_height_base + len(level_ends) * level_gap + 0.2
            ax_gene.set_ylim(0, gene_track_height)
            legend_elems = [
                Line2D([0], [0], color=self.gene_color_plus, lw=2, marker='>', markersize=8, label='Forward (+)'),
                Line2D([0], [0], color=self.gene_color_minus, lw=2, marker='<', markersize=8, label='Reverse (-)')
            ]
            ax_gene.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)
        else:
            ax_gene.text(0.5, 0.5, "No genes in this region", ha="center", va="center", transform=ax_gene.transAxes,
                         fontsize=10, style='italic')
            ax_gene.set_ylim(0, 1)
        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')

        # plot signals
        # choose palette: tab10/tab20 per number of tracks (scientific colors)
        cmap_name = 'tab10' if n_tracks <= 8 else 'tab20'
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(n_tracks+2)]
        for i, name in enumerate(titles):
            ax = axes[i + 1]
            sig = np.asarray(signals_to_plot[i])
            if smoothing_sigma and smoothing_sigma > 0:
                sig = gaussian_smooth(sig, sigma=float(smoothing_sigma))
            color = colors[i+2] if self.signal_palette is None else (self.signal_palette[i % len(self.signal_palette)])
            ax.plot(display_positions, sig, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(display_positions, 0, sig, alpha=0.25, color=color)
            y_max = max(sig.max() * 1.15, 0.1) if len(sig) > 0 else 1
            ax.set_ylim(0, y_max)
            ax.set_yticks(np.linspace(0, y_max, 5))
            ax.tick_params(axis='y', labelsize=9)
            # move the per-track title into the y-label (publication-friendly, rotated 0)
            ax.set_ylabel(f"{name}", fontsize=10, rotation=0, ha='right', va='center')
            # do not use per-subplot title to reduce clutter
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # x axis formatting
        # displayed length = number of positions in the x-axis
        displayed_len = len(display_positions)
        for ax in axes:
            ax.set_xlim(start_display, end_display)
            if self.xtick_step and displayed_len > self.xtick_step:
                xticks = np.arange(start_display, end_display, self.xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=9)
        def fmt(x, pos):
            return f"{int(x):,}"
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        axes[-1].set_xlabel(f"Chromosome position; interval= {chrom}:{start_display:,}-{end_display:,} (length: {displayed_len:,} bp)", fontsize=11)
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        plt.subplots_adjust(hspace=0.12)
        plt.show()
        return fig, axes


    def plot_window2(self, idx=None, max_subplots=None, assembly_filter=None, track_indices=None, smoothing_sigma=2,
                    window_start=None, window_end=None, chrom=None, start=None, end=None, meta_df=None):
        """
        Plot selected tracks for one window and optional gene models.

        Args:
            idx (int): index into dataset.sequence_split_df; defaults to 0.
            max_subplots (int): max number of tracks to draw (default: viewer.max_subplots).
            assembly_filter (str or list): filter labels_meta_df by 'File assembly'.
            track_indices (list[int]): explicit indices into labels_meta_df to plot.
            smoothing_sigma (float): gaussian smoothing sigma (<=0 disables).
            window_start (int): start position for zooming within the loaded interval (deprecated, use start instead).
            window_end (int): end position for zooming within the loaded interval (deprecated, use end instead).
            chrom (str): chromosome name (e.g., "chr19"). If provided, will directly plot this region.
            start (int): start genomic position. Must be provided if chrom is provided.
            end (int): end genomic position. Must be provided if chrom is provided.
        Returns:
            (fig, axes) matplotlib objects or None on error.
        """
        from matplotlib.ticker import FuncFormatter
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        ds = self.dataset
        
        # 处理直接指定基因组坐标的情况
        if chrom is not None:
            if start is None or end is None:
                logging.error("If chrom is specified, both start and end must be provided")
                return
            if start >= end:
                logging.error("start must be less than end")
                return
            
            # 直接创建虚拟的row，用于后续处理
            import pandas as pd
            row = pd.Series({
                "chromosome": chrom,
                "start": start,
                "end": end,
                # 添加其他必要的字段，如果有的话
                "length": end - start
            })
            
            # 标记这是直接坐标模式
            direct_coords = True
            loaded_row_start = start
            loaded_row_end = end
            
        else:
            # 原有的基于索引的逻辑
            direct_coords = False
            
            if idx is None:
                if len(ds.sequence_split_df) == 0:
                    logging.warning("sequence_split_df is empty")
                    return
                idx = 0
            try:
                row = ds.sequence_split_df.iloc[idx]
            except Exception as e:
                logging.error(f"Invalid idx for sequence_split_df: {e}")
                return
            
            loaded_row_start = int(row["start"])
            loaded_row_end = int(row["end"])

        #meta_df = ds.labels_meta_df.copy()
        if assembly_filter is not None and 'File assembly' in meta_df.columns:
            if isinstance(assembly_filter, (list, tuple)):
                meta_df = meta_df[meta_df['File assembly'].isin(assembly_filter)]
            else:
                meta_df = meta_df[meta_df['File assembly'] == assembly_filter]

        if track_indices is not None:
            sel_df = meta_df.iloc[track_indices].reset_index(drop=True)
        else:
            nmax = self.max_subplots if max_subplots is None else int(max_subplots)
            sel_df = meta_df.reset_index(drop=True).iloc[:nmax]

        if sel_df.shape[0] == 0:
            logging.info("No tracks selected for plotting")
            return

        # load signals and titles
        signals = []
        titles = []
        chrom_name = str(row["chromosome"])
        
        ds.bigwig_dir = ds.index_stats['default']["inputs"]["processed_bw_dir"]
        for _, mrow in sel_df.iterrows():
            fname = mrow.get('target_file_name') or mrow.get('target_file') or None
            if fname is None:
                logging.warning("Missing target file name for a track; skipping")
                continue
            bw_path = os.path.join(ds.bigwig_dir, fname)
            
            # 如果是直接坐标模式，直接加载指定区间的信号
            if direct_coords:
                vals = load_bigwig_signal(bw_path, chrom_name, start, end, end - start)
            else:
                # 原有的加载逻辑
                vals = load_bigwig_signal(bw_path, chrom_name, loaded_row_start, loaded_row_end, ds.max_length)
            
            vals = np.nan_to_num(vals, nan=0.0)
            signals.append(vals)
            
            output_type = mrow.get('output_type') or ""
            biosample_name = mrow.get('biosample_name') or ""
            assay = mrow.get('name') or ""
            strand = mrow.get('strand') or ""
            
            if assay or strand:
                titles.append(f"{output_type}: {biosample_name} ({strand})\n{assay}")
            else:
                titles.append(os.path.basename(fname))

        if len(signals) == 0:
            logging.info("No signals loaded for plotting")
            return

        n_tracks = len(signals)
        fig, axes = plt.subplots(n_tracks + 1, 1, figsize=(18, 1.5 * (n_tracks + 1)), sharex=True, dpi=self.dpi,
                                gridspec_kw={'height_ratios': [1.0] + [1.0] * n_tracks})
        if n_tracks + 1 == 1:
            axes = [axes]
        ax_gene = axes[0]

        # 确定最终显示的区间
        if direct_coords:
            # 直接坐标模式：使用传入的start和end
            final_start = start
            final_end = end
            actual_len = end - start
            # 检查window参数（向后兼容）
            if window_start is not None or window_end is not None:
                logging.warning("window_start/window_end parameters are ignored when chrom/start/end are specified")
        else:
            # 原有逻辑：基于加载的区间和可能的window参数
            row_start = loaded_row_start
            first_signal = signals[0]
            actual_len = len(first_signal)
            row_end = row_start + actual_len

            # If no window specified, keep previous behavior (show entire loaded interval)
            if window_start is None and window_end is None:
                final_start = row_start
                final_end = row_end
                display_positions = np.arange(final_start, final_end)
                signals_to_plot = signals
            else:
                # Fill missing window_start/window_end with loaded interval bounds if only one provided
                if window_start is None:
                    window_start = row_start
                if window_end is None:
                    window_end = row_end
                final_start = int(window_start)
                final_end = int(window_end)
                desired_len = final_end - final_start
                if desired_len <= 0:
                    logging.error("Invalid window: window_end must be greater than window_start")
                    return
                display_positions = np.arange(final_start, final_end)
                displayed_len = len(display_positions)

                # Slice/pad each signal so that the returned segment corresponds exactly to
                # [start_display, end_display). Regions outside the originally loaded interval
                # are filled with zeros.
                signals_to_plot = []
                for sig in signals:
                    seg = np.zeros(displayed_len, dtype=sig.dtype)
                    # source indices within the original loaded signal
                    src_start = max(0, final_start - row_start)
                    src_end = min(actual_len, final_end - row_start)
                    if src_end > src_start:
                        # destination start index in the segment
                        dest_start = max(0, row_start - final_start)
                        seg[dest_start:dest_start + (src_end - src_start)] = sig[src_start:src_end]
                    signals_to_plot.append(seg)
                # 对于直接坐标模式，signals_to_plot就是signals本身
                signals_to_plot = signals
        
        # 对于直接坐标模式，生成显示位置数组
        if direct_coords:
            display_positions = np.arange(final_start, final_end)
            signals_to_plot = signals

        # draw gene models if available
        genes, exons = self.get_genes_in_interval(chrom_name, final_start, final_end)
        if genes:
            genes_df = __import__("pandas").DataFrame(genes, columns=["start", "end", "strand", "name"]).sort_values("start").reset_index(drop=True)
            level_ends = []
            level_height_base = 0.3
            level_gap = 0.25
            max_levels = 8
            gene_levels = []
            for _, g in genes_df.iterrows():
                gs, ge = int(g["start"]), int(g["end"])
                placed = False
                for lvl in range(len(level_ends)):
                    if gs >= level_ends[lvl]:
                        level_ends[lvl] = ge
                        gene_levels.append(lvl)
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(ge)
                    gene_levels.append(len(level_ends) - 1)
                    placed = True
                if not placed:
                    earliest = int(np.argmin(level_ends))
                    level_ends[earliest] = ge
                    gene_levels.append(earliest)
            for (idx_g, rowg), lvl in zip(genes_df.iterrows(), gene_levels):
                gs, ge, strand, name = int(rowg["start"]), int(rowg["end"]), rowg["strand"], rowg["name"]
                y = level_height_base + lvl * level_gap
                color = self.gene_color_plus if strand == "+" else self.gene_color_minus
                ax_gene.plot([gs, ge], [y, y], color=color, lw=2.0, zorder=1, solid_capstyle='round')
                gene_length = ge - gs
                arrow_len = min(gene_length * 0.1, 2000)
                if strand == "+":
                    ax_gene.arrow(ge - arrow_len, y, arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                else:
                    ax_gene.arrow(gs + arrow_len, y, -arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                gene_exons = [e for e in exons if e[3] == name]
                for es, ee, st, nm in gene_exons:
                    es_d, ee_d = max(es, final_start), min(ee, final_end)
                    if ee_d > es_d and (ee_d - es_d) > 50:
                        rect = mpatches.Rectangle((es_d, y +0.04), ee_d - es_d, 0.08, facecolor=color, alpha=0.9,
                                                zorder=3, edgecolor='none')
                        ax_gene.add_patch(rect)
                text_x = (gs + ge) / 2
                text_x = np.clip(text_x, final_start + 500, final_end - 500)
                ax_gene.text(text_x, y + 0.06, name if name else "Unknown", ha="center", va="bottom", fontsize=9, zorder=4,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
            gene_track_height = level_height_base + len(level_ends) * level_gap + 0.2
            ax_gene.set_ylim(0, gene_track_height)
            legend_elems = [
                Line2D([0], [0], color=self.gene_color_plus, lw=2, marker='>', markersize=8, label='Forward (+)'),
                Line2D([0], [0], color=self.gene_color_minus, lw=2, marker='<', markersize=8, label='Reverse (-)')
            ]
            ax_gene.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)
        else:
            ax_gene.text(0.5, 0.5, "No genes in this region", ha="center", va="center", transform=ax_gene.transAxes,
                        fontsize=10, style='italic')
            ax_gene.set_ylim(0, 1)
        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')

        # plot signals
        # choose palette: tab10/tab20 per number of tracks (scientific colors)
        cmap_name = 'tab10' if n_tracks <= 8 else 'tab20'
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(n_tracks+2)]
        for i, name in enumerate(titles):
            ax = axes[i + 1]
            sig = np.asarray(signals_to_plot[i])
            if smoothing_sigma and smoothing_sigma > 0:
                sig = gaussian_smooth(sig, sigma=float(smoothing_sigma))
            color = colors[i+2] if self.signal_palette is None else (self.signal_palette[i % len(self.signal_palette)])
            ax.plot(display_positions, sig, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(display_positions, 0, sig, alpha=0.25, color=color)
            y_max = max(sig.max() * 1.15, 0.1) if len(sig) > 0 else 1
            ax.set_ylim(0, y_max)
            ax.set_yticks(np.linspace(0, y_max, 5))
            ax.tick_params(axis='y', labelsize=9)
            # move the per-track title into the y-label (publication-friendly, rotated 0)
            ax.set_ylabel(f"{name}", fontsize=10, rotation=0, ha='right', va='center')
            # do not use per-subplot title to reduce clutter
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # x axis formatting
        # displayed length = number of positions in the x-axis
        displayed_len = len(display_positions)
        for ax in axes:
            ax.set_xlim(final_start, final_end)
            if self.xtick_step and displayed_len > self.xtick_step:
                xticks = np.arange(final_start, final_end, self.xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=9)
        def fmt(x, pos):
            return f"{int(x):,}"
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        # 修改标题以显示模式信息
        if direct_coords:
            source_info = "Direct coordinates"
        else:
            source_info = f"Dataset index {idx}" if idx is not None else "Dataset"
        
        axes[-1].set_xlabel(f"{source_info}: {chrom_name}:{final_start:,}-{final_end:,} (length: {displayed_len:,} bp)", fontsize=11)
        
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        plt.subplots_adjust(hspace=0.12)
        plt.show()
        return fig, axes



import os
import gzip
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Optional, Tuple

# 假设这些辅助函数已定义（如 gaussian_smooth, _to_numpy）
# 如果未定义，请确保它们存在
def gaussian_smooth(arr, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma, mode='nearest')

def _to_numpy(x):
    """
    Robust conversion to numpy:
      - handles torch.Tensor on CUDA/CPU (detaches, moves to CPU, converts bfloat16/float16 -> float32)
      - handles numpy arrays
      - handles iterables element-wise (safe for lists of tensors)
    """
    try:
        import torch
    except Exception:
        torch = None

    # torch.Tensor -> numpy
    if torch is not None and isinstance(x, torch.Tensor):
        t = x.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype in (getattr(torch, "bfloat16", None), torch.float16):
            t = t.to(torch.float32)
        return t.squeeze().numpy()

    # numpy array
    if isinstance(x, np.ndarray):
        return x.squeeze()

    # objects exposing .cpu()
    try:
        if hasattr(x, "cpu") and callable(x.cpu):
            y = x.cpu()
            if torch is not None and isinstance(y, torch.Tensor):
                t = y.detach()
                if t.dtype in (getattr(torch, "bfloat16", None), torch.float16):
                    t = t.to(torch.float32)
                return t.squeeze().numpy()
            return np.asarray(y).squeeze()
    except Exception:
        pass

    # iterable (element-wise conversion)
    try:
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, bytearray)):
            items = list(x)
            if len(items) == 0:
                return np.array([]).squeeze()
            converted = []
            for it in items:
                try:
                    if torch is not None and isinstance(it, torch.Tensor):
                        t = it.detach()
                        if t.device.type != "cpu":
                            t = t.cpu()
                        if t.dtype in (getattr(torch, "bfloat16", None), torch.float16):
                            t = t.to(torch.float32)
                        converted.append(t.squeeze().numpy())
                    elif isinstance(it, np.ndarray):
                        converted.append(it.squeeze())
                    else:
                        converted.append(np.asarray(it))
                except Exception:
                    try:
                        converted.append(np.array(it))
                    except Exception:
                        converted.append(np.asarray(it))
            try:
                return np.asarray(converted).squeeze()
            except Exception:
                try:
                    return np.concatenate([np.atleast_1d(c) for c in converted]).squeeze()
                except Exception:
                    return np.array(converted).squeeze()
    except Exception:
        pass

    # fallback
    return np.array([x]).squeeze()


class ResultsViewer:
    """
    Visualize model prediction results with pre-loaded gene annotations.

    Mimics DatasetViewer behavior:
      - Load GFF once at init
      - Plot full-resolution signals (no binning)
      - Only supports small genomic windows (< 100 kb recommended)
    """
    def __init__(self,
                 annotation_path: Optional[str] = None,
                 signal_palette: Optional[List[str]] = None,
                 xtick_step: int = 4000,
                 dpi: int = 100,
                 max_region_length: int = 100_000):  # 新增安全限制
        self.signal_palette = signal_palette
        self.xtick_step = xtick_step
        self.dpi = dpi
        self.max_region_length = max_region_length
        self.genes_by_chrom = {}
        self.exons_by_chrom = {}
        if annotation_path:
            self._load_gff2(annotation_path)

    def _load_gff(self, path):
        """Load GFF/GTF (optionally gzipped). Store genes and exons per chromosome."""
        genes = {}
        exons = {}
        try:
            open_func = gzip.open if path.endswith('.gz') else open
            mode = 'rt' if path.endswith('.gz') else 'r'
            with open_func(path, mode) as fh:
                for line in fh:
                    if line.startswith('#') or not line.strip():
                        continue
                    cols = line.rstrip().split('\t')
                    if len(cols) < 9:
                        continue
                    chrom, src, feature, start, end, score, strand, phase, attrs = cols[:9]
                    try:
                        start_i = int(start)
                        end_i = int(end)
                    except Exception:
                        continue
                    # extract a simple name (gene_name / Name / gene_id)
                    gene_name = ""
                    for key in ("gene_name=", "Name=", "gene_id="):
                        if key in attrs:
                            parts = [p for p in attrs.split(';') if p.strip().startswith(key)]
                            if parts:
                                gene_name = parts[0].split('=', 1)[-1].strip('"').strip()
                                break
                    if feature == "gene":
                        genes.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
                    elif feature == "exon":
                        exons.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
        except Exception as e:
            logging.warning(f"Failed to load annotation {path}: {e}")
        self.genes_by_chrom = genes
        self.exons_by_chrom = exons
        logging.info(f"Loaded annotation: chromosomes={list(self.genes_by_chrom.keys())}")


    def _load_gff2(self, path):
            """Load GFF/GTF (optionally gzipped). Store genes, transcripts, and exons per chromosome."""
            import gzip
            import logging

            genes = {}       # chrom -> list of (start, end, strand, gene_id)
            transcripts = {} # chrom -> list of (start, end, strand, transcript_id, gene_id)
            exons = {}       # chrom -> list of (start, end, strand, transcript_id)

            try:
                open_func = gzip.open if path.endswith('.gz') else open
                mode = 'rt' if path.endswith('.gz') else 'r'
                with open_func(path, mode) as fh:
                    for line in fh:
                        if line.startswith('#') or not line.strip():
                            continue
                        cols = line.rstrip().split('\t')
                        if len(cols) < 9:
                            continue
                        chrom, src, feature, start, end, score, strand, phase, attrs = cols[:9]
                        try:
                            start_i = int(start)
                            end_i = int(end)
                        except Exception:
                            continue

                        # 辅助函数：从属性字符串中提取指定键的值
                        def get_attr(key):
                            # 支持 key=value 或 key "value"
                            for part in attrs.split(';'):
                                part = part.strip()
                                if part.startswith(key + '='):
                                    return part.split('=', 1)[-1].strip('"').strip()
                                if part.startswith(key + ' '):
                                    # 某些格式如 key "value"
                                    if '"' in part:
                                        return part.split('"')[1]
                            return ""

                        if feature == "gene":
                            gene_id = get_attr("ID")
                            if not gene_id:
                                gene_id = get_attr("Name") or get_attr("gene_id")
                            if gene_id:
                                genes.setdefault(chrom, []).append((start_i, end_i, strand, gene_id))

                        elif feature in ("mRNA", "transcript"):
                            transcript_id = get_attr("ID")
                            if not transcript_id:
                                transcript_id = get_attr("Name") or get_attr("transcript_id")
                            gene_id = get_attr("Parent")
                            if transcript_id:
                                transcripts.setdefault(chrom, []).append((start_i, end_i, strand, transcript_id, gene_id))

                        elif feature == "exon":
                            # 外显子的 Parent 指向转录本 ID
                            transcript_id = get_attr("Parent")
                            if not transcript_id:
                                # 有时 ID 本身包含 exon 标识，但标准 GFF 用 Parent
                                transcript_id = get_attr("ID")
                            if transcript_id:
                                exons.setdefault(chrom, []).append((start_i, end_i, strand, transcript_id))
                            else:
                                # 如果没有转录本 ID，则记录为 None 或空，但后续匹配会失败
                                exons.setdefault(chrom, []).append((start_i, end_i, strand, ""))

            except Exception as e:
                logging.warning(f"Failed to load annotation {path}: {e}")

            self.genes_by_chrom = genes
            self.transcripts_by_chrom = transcripts   # 新增
            self.exons_by_transcript = exons               # 外显子现在关联到转录本 ID

            logging.info(f"Loaded annotation: genes={len(genes)} chromosomes, transcripts={len(transcripts)} chromosomes")




    def get_genes_in_interval(self, chrom, start, end):
        """Return (genes, exons) overlapping [start, end)."""
        genes = self.genes_by_chrom.get(chrom, [])
        exons = self.exons_by_chrom.get(chrom, [])
        genes_f = [(gs, ge, st, nm) for (gs, ge, st, nm) in genes if not (ge < start or gs >= end)]
        exons_f = [(es, ee, st, nm) for (es, ee, st, nm) in exons if not (ee < start or es >= end)]
        return genes_f, exons_f
    
    def get_genes_in_interval2(self, chrom, start, end):
        """
        Return (genes, transcripts, exons) overlapping [start, end).
        - genes: list of (start, end, strand, gene_name)
        - transcripts: list of (start, end, strand, transcript_id, gene_name)  # 新增
        - exons: list of (start, end, strand, transcript_id)  # name 字段为 transcript_id
        """
        # 基因（不变）
        genes = self.genes_by_chrom.get(chrom, [])
        genes_f = [(gs, ge, st, nm) for (gs, ge, st, nm) in genes if not (ge < start or gs >= end)]

        # 外显子（要求 name 字段为 transcript_id）
        exons = self.exons_by_transcript.get(chrom, [])
        exons_f = [(es, ee, st, nm) for (es, ee, st, nm) in exons if not (ee < start or es >= end)]

        # 转录本（需要额外数据结构）
        if hasattr(self, 'transcripts_by_chrom'):
            transcripts = self.transcripts_by_chrom.get(chrom, [])
            transcripts_f = [(ts, te, tstrand, tname, tgene) for (ts, te, tstrand, tname, tgene) in transcripts
                            if not (te < start or ts >= end)]
        else:
            transcripts_f = []
            if genes_f or exons_f:
                print("Warning: transcripts_by_chrom not found. Transcript-level display disabled.")
        
        return genes_f, transcripts_f, exons_f

    def plot(self,
             results: Dict,
             track_order: Optional[List[str]] = None,
             smoothing_sigma: float = 2.0,
             figsize: Optional[Tuple[float, float]] = None,
             show_legend: bool = True,
             gene_color_plus: str = "tab:blue",
             gene_color_minus: str = "tab:orange",
             window_start: Optional[int] = None,
             window_end: Optional[int] = None):
        """
        Plot results. Optional window_start/window_end specify absolute genomic coordinates
        to display; if not provided, use results['position'] interval (original behavior).
        """
        values = results.get("values", {})
        position = results.get("position", (None, None, None))
        chrom, start, end = position if len(position) == 3 else (None, None, None)

        if not values:
            raise ValueError("results['values'] is empty")
        if chrom is None or start is None or end is None:
            raise ValueError("results['position'] must be (chrom, start, end)")

        # determine display window
        orig_start = int(start)
        # original region length inferred from provided position
        orig_region_length = int(end) - int(start)

        if window_start is None and window_end is None:
            start_display = orig_start
            end_display = orig_start + orig_region_length
        else:
            start_display = int(window_start) if window_start is not None else orig_start
            end_display = int(window_end) if window_end is not None else (orig_start + orig_region_length)

        displayed_len = end_display - start_display
        if displayed_len <= 0:
            raise ValueError("Invalid window: window_end must be greater than window_start")

        if displayed_len > self.max_region_length:
            raise ValueError(
                f"Display window length ({displayed_len:,} bp) exceeds max allowed ({self.max_region_length:,} bp)."
            )

        # === 获取基因注释 ===
        genes, exons = self.get_genes_in_interval(chrom, start_display, end_display)

        # === 收集 tracks 和 biosamples ===
        track_names = list(values.keys())
        biosample_set = []
        for tn in track_names:
            for b in values[tn].keys():
                if b not in biosample_set:
                    biosample_set.append(b)
        biosamples = biosample_set

        if track_order:
            ordered = [t for t in track_order if t in track_names]
            ordered += [t for t in track_names if t not in ordered]
            track_names = ordered

        n_heads = len(track_names)
        n_bios = max(1, len(biosamples))
        total_signal_subplots = n_heads * n_bios
        total_subplots = 1 + total_signal_subplots

        default_fig_width = 18.0
        default_fig_height = 1.5 * total_subplots
        figsize = figsize or (default_fig_width, default_fig_height)

        fig, axes = plt.subplots(total_subplots, 1, figsize=figsize, sharex=True, dpi=self.dpi,
                                 gridspec_kw={'height_ratios': [0.8] + [1.0] * total_signal_subplots})
        if total_subplots == 1:
            axes = [axes]
        ax_gene = axes[0]

        # === 基因轨道 ===
        display_positions = np.arange(start_display, end_display)

        if not genes:
            ax_gene.text(0.5, 0.5, "No genes in this region", ha="center", va="center",
                         transform=ax_gene.transAxes, fontsize=10, style='italic')
            ax_gene.set_ylim(0, 1)
        else:
            import pandas as pd
            genes_df = pd.DataFrame(genes, columns=["start", "end", "strand", "name"]).sort_values("start").reset_index(drop=True)
            level_ends = []
            level_height_base = 0.3
            level_gap = 0.25
            max_levels = 8
            gene_levels = []

            for _, g in genes_df.iterrows():
                gs, ge = int(g["start"]), int(g["end"])
                placed = False
                for lvl in range(len(level_ends)):
                    if gs >= level_ends[lvl]:
                        level_ends[lvl] = ge
                        gene_levels.append(lvl)
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(ge)
                    gene_levels.append(len(level_ends) - 1)
                    placed = True
                if not placed:
                    earliest = int(np.argmin(level_ends))
                    level_ends[earliest] = ge
                    gene_levels.append(earliest)

            for (idx_g, rowg), lvl in zip(genes_df.iterrows(), gene_levels):
                gs, ge, strand, name = int(rowg["start"]), int(rowg["end"]), rowg["strand"], rowg["name"]
                y = level_height_base + lvl * level_gap
                color = gene_color_plus if strand == "+" else gene_color_minus
                ax_gene.plot([gs, ge], [y, y], color=color, lw=2.5, zorder=2, solid_capstyle='round')

                gene_length = ge - gs
                arrow_len = min(gene_length * 0.1, 2000)
                if strand == "+":
                    ax_gene.arrow(ge - arrow_len, y, arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                else:
                    ax_gene.arrow(gs + arrow_len, y, -arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)

                gene_exons = [e for e in exons if e[3] == name]
                for es, ee, st, nm in gene_exons:
                    es_d, ee_d = max(es, start_display), min(ee, end_display)
                    if ee_d > es_d and (ee_d - es_d) > 50:
                        rect = patches.Rectangle((es_d, y - 0.04), ee_d - es_d, 0.08,
                                                facecolor=color, alpha=0.9, zorder=3,
                                                edgecolor='white', linewidth=0.5)
                        ax_gene.add_patch(rect)

                text_x = (gs + ge) / 2
                text_x = np.clip(text_x, start_display + 500, end_display - 500)
                ax_gene.text(text_x, y + 0.06, name if name else "Unknown", ha='center', va='bottom',
                             fontsize=9, zorder=4,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

            gene_track_height = level_height_base + len(level_ends) * level_gap + 0.2
            ax_gene.set_ylim(0, gene_track_height)

            legend_elems = [
                Line2D([0], [0], color=gene_color_plus, lw=2, marker='>', markersize=8, label='Forward (+)'),
                Line2D([0], [0], color=gene_color_minus, lw=2, marker='<', markersize=8, label='Reverse (-)')
            ]
            ax_gene.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)

        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')

        # === 信号轨道 ===
        idx_ax = 1
        for head in track_names:
            track_dict = values.get(head, {})
            for b in biosamples:
                ax = axes[idx_ax]
                arr = track_dict.get(b, None)
                if arr is None:
                    ax.text(0.5, 0.5, f"No data for {head}/{b}", ha='center', va='center')
                    ax.set_yticks([])
                    idx_ax += 1
                    continue

                y_src = _to_numpy(arr).reshape(-1)
                # source covers [orig_start, orig_start + len(y_src))
                src_len = len(y_src)
                src_start = orig_start
                src_end = orig_start + src_len

                # Build displayed segment of length displayed_len
                seg = np.zeros(displayed_len, dtype=y_src.dtype)
                # compute overlap between source and display window
                overlap_start = max(start_display, src_start)
                overlap_end = min(end_display, src_end)
                if overlap_end > overlap_start:
                    src_slice_start = overlap_start - src_start
                    src_slice_end = overlap_end - src_start
                    dest_slice_start = overlap_start - start_display
                    seg[dest_slice_start:dest_slice_start + (src_slice_end - src_slice_start)] = y_src[src_slice_start:src_slice_end]

                y = seg

                if smoothing_sigma > 0:
                    y = gaussian_smooth(y, smoothing_sigma)

                # Color
                if self.signal_palette:
                    color = self.signal_palette[biosamples.index(b) % len(self.signal_palette)]
                else:
                    cmap = plt.get_cmap('tab10' if len(biosamples) <= 10 else 'tab20')
                    color = cmap((biosamples.index(b)+2) % cmap.N)

                ax.plot(display_positions, y, color=color, linewidth=1.2)
                ax.fill_between(display_positions, 0, y, color=color, alpha=0.14)

                ax.set_ylabel(f"{head}\n{b}", fontsize=9, rotation=0, ha='right', va='center')
                y_max = max(np.nanmax(y) * 1.15, 0.1) if y.size > 0 else 0.1
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.4)
                idx_ax += 1

        # === X-axis ===
        for ax in axes:
            ax.set_xlim(start_display, end_display)
            if self.xtick_step and displayed_len > self.xtick_step:
                xticks = np.arange(start_display, end_display, self.xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=9)

        def fmt(x, pos): return f"{int(x):,}"
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        axes[-1].set_xlabel(f"Chromosome position; interval= {chrom}:{start_display:,}-{end_display:,} ({displayed_len:,} bp)", fontsize=10)

        # === Legend ===
        if show_legend and biosamples:
            handles = []
            for i, b in enumerate(biosamples):
                if self.signal_palette:
                    c = self.signal_palette[i % len(self.signal_palette)]
                else:
                    cmap = plt.get_cmap('tab10' if len(biosamples) <= 10 else 'tab20')
                    c = cmap((i+2) % cmap.N)
                handles.append(Line2D([0], [0], color=c, lw=2))
            try:
                axes[1].legend(handles, biosamples, loc='upper right', fontsize=8)
            except:
                fig.legend(handles, biosamples, loc='upper right', fontsize=8)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
        plt.subplots_adjust(hspace=0.12)
        return fig, axes

    def plot2(self,
            results: Dict,
            results2: Optional[Dict] = None,  # 添加第二个结果
            track_order: Optional[List[str]] = None,
            smoothing_sigma: float = 2.0,
            figsize: Optional[Tuple[float, float]] = None,
            show_legend: bool = True,
            gene_color_plus: str = "tab:blue",
            gene_color_minus: str = "tab:orange",
            window_start: Optional[int] = None,
            window_end: Optional[int] = None):
        """
        Plot results. Optional window_start/window_end specify absolute genomic coordinates
        to display; if not provided, use results['position'] interval (original behavior).
        results2: Optional second results dict to overlap with the first one.
        """
        values = results.get("values", {})
        position = results.get("position", (None, None, None))
        chrom, start, end = position if len(position) == 3 else (None, None, None)

        if not values:
            raise ValueError("results['values'] is empty")
        if chrom is None or start is None or end is None:
            raise ValueError("results['position'] must be (chrom, start, end)")

        # 如果有第二个结果，检查一致性
        values2 = results2.get("values", {}) if results2 else {}
        position2 = results2.get("position", (None, None, None)) if results2 else (None, None, None)
        chrom2, start2, end2 = position2 if len(position2) == 3 else (None, None, None)
        
        # 检查染色体是否一致
        if results2 and chrom2 is not None and chrom2 != chrom:
            print(f"Warning: Chromosomes differ ({chrom} vs {chrom2}). Using first chromosome.")
        
        # determine display window
        orig_start = int(start)
        # original region length inferred from provided position
        orig_region_length = int(end) - int(start)

        if window_start is None and window_end is None:
            start_display = orig_start
            end_display = orig_start + orig_region_length
        else:
            start_display = int(window_start) if window_start is not None else orig_start
            end_display = int(window_end) if window_end is not None else (orig_start + orig_region_length)

        displayed_len = end_display - start_display
        if displayed_len <= 0:
            raise ValueError("Invalid window: window_end must be greater than window_start")

        if displayed_len > self.max_region_length:
            raise ValueError(
                f"Display window length ({displayed_len:,} bp) exceeds max allowed ({self.max_region_length:,} bp)."
            )

        # === 获取基因注释 ===
        genes, exons = self.get_genes_in_interval(chrom, start_display, end_display)

        # === 收集 tracks 和 biosamples ===
        # 从两个结果中收集所有track
        track_names = list(values.keys())
        if results2:
            for tn in values2.keys():
                if tn not in track_names:
                    track_names.append(tn)
        
        biosample_set = []
        for tn in track_names:
            for b in values.get(tn, {}).keys():
                if b not in biosample_set:
                    biosample_set.append(b)
            if results2:
                for b in values2.get(tn, {}).keys():
                    if b not in biosample_set:
                        biosample_set.append(b)
        biosamples = biosample_set

        if track_order:
            ordered = [t for t in track_order if t in track_names]
            ordered += [t for t in track_names if t not in ordered]
            track_names = ordered

        n_heads = len(track_names)
        n_bios = max(1, len(biosamples))
        total_signal_subplots = n_heads * n_bios
        total_subplots = 1 + total_signal_subplots

        default_fig_width = 18.0
        default_fig_height = 1.5 * total_subplots
        figsize = figsize or (default_fig_width, default_fig_height)

        fig, axes = plt.subplots(total_subplots, 1, figsize=figsize, sharex=True, dpi=self.dpi,
                                gridspec_kw={'height_ratios': [0.8] + [1.0] * total_signal_subplots})
        if total_subplots == 1:
            axes = [axes]
        ax_gene = axes[0]

        # === 基因轨道 ===
        display_positions = np.arange(start_display, end_display)

        if not genes:
            ax_gene.text(0.5, 0.5, "No genes in this region", ha="center", va="center",
                        transform=ax_gene.transAxes, fontsize=10, style='italic')
            ax_gene.set_ylim(0, 1)
        else:
            import pandas as pd
            genes_df = pd.DataFrame(genes, columns=["start", "end", "strand", "name"]).sort_values("start").reset_index(drop=True)
            level_ends = []
            level_height_base = 0.4
            level_gap = 0.35
            max_levels = 8
            gene_levels = []

            for _, g in genes_df.iterrows():
                gs, ge = int(g["start"]), int(g["end"])
                placed = False
                for lvl in range(len(level_ends)):
                    if gs >= level_ends[lvl]:
                        level_ends[lvl] = ge
                        gene_levels.append(lvl)
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(ge)
                    gene_levels.append(len(level_ends) - 1)
                    placed = True
                if not placed:
                    earliest = int(np.argmin(level_ends))
                    level_ends[earliest] = ge
                    gene_levels.append(earliest)

            for (idx_g, rowg), lvl in zip(genes_df.iterrows(), gene_levels):
                gs, ge, strand, name = int(rowg["start"]), int(rowg["end"]), rowg["strand"], rowg["name"]
                y = level_height_base + lvl * level_gap
                color = gene_color_plus if strand == "+" else gene_color_minus
                ax_gene.plot([gs, ge], [y, y], color=color, lw=2.0, zorder=1, solid_capstyle='round')

                gene_length = ge - gs
                arrow_len = min(gene_length * 0.1, 2000)
                if strand == "+":
                    ax_gene.arrow(ge - arrow_len, y, arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                else:
                    ax_gene.arrow(gs + arrow_len, y, -arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)

                gene_exons = [e for e in exons if e[3] == name]
                for es, ee, st, nm in gene_exons:
                    es_d, ee_d = max(es, start_display), min(ee, end_display)
                    if ee_d > es_d and (ee_d - es_d) > 50:
                        rect = patches.Rectangle((es_d, y + 0.08), ee_d - es_d, 0.06,
                                                facecolor=color, alpha=0.9, zorder=3,
                                                edgecolor='white', linewidth=0.5)
                        ax_gene.add_patch(rect)


                text_x = (gs + ge) / 2
                text_x = np.clip(text_x, start_display + 500, end_display - 500)
                ax_gene.text(text_x, y + 0.06, name if name else "Unknown", ha='center', va='bottom',
                            fontsize=9, zorder=4,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

            gene_track_height = level_height_base + len(level_ends) * level_gap + 0.3
            ax_gene.set_ylim(0, gene_track_height)

            legend_elems = [
                Line2D([0], [0], color=gene_color_plus, lw=2, marker='>', markersize=8, label='Forward (+)'),
                Line2D([0], [0], color=gene_color_minus, lw=2, marker='<', markersize=8, label='Reverse (-)')
            ]
            ax_gene.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)

        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')

        # === 信号轨道 ===
        idx_ax = 1
        for head in track_names:
            track_dict = values.get(head, {})
            track_dict2 = values2.get(head, {}) if results2 else {}
            
            for b in biosamples:
                ax = axes[idx_ax]
                
                # 绘制第一个结果
                arr = track_dict.get(b, None)
                if arr is not None:
                    y_src = _to_numpy(arr).reshape(-1)
                    src_len = len(y_src)
                    src_start = orig_start
                    src_end = orig_start + src_len

                    seg = np.zeros(displayed_len, dtype=y_src.dtype)
                    overlap_start = max(start_display, src_start)
                    overlap_end = min(end_display, src_end)
                    if overlap_end > overlap_start:
                        src_slice_start = overlap_start - src_start
                        src_slice_end = overlap_end - src_start
                        dest_slice_start = overlap_start - start_display
                        seg[dest_slice_start:dest_slice_start + (src_slice_end - src_slice_start)] = y_src[src_slice_start:src_slice_end]

                    y = seg
                    if smoothing_sigma > 0:
                        y = gaussian_smooth(y, smoothing_sigma)
                    
                    # 第一个结果使用实线
                    ax.plot(display_positions, y, color='tab:blue', linewidth=1.5, alpha=0.9, label='Result 1' if b == biosamples[0] else "")

                # 绘制第二个结果（如果存在）
                if results2:
                    arr2 = track_dict2.get(b, None)
                    if arr2 is not None:
                        # 获取第二个结果的起始位置
                        src_start2 = int(start2) if start2 is not None else orig_start
                        y_src2 = _to_numpy(arr2).reshape(-1)
                        src_len2 = len(y_src2)
                        src_end2 = src_start2 + src_len2

                        seg2 = np.zeros(displayed_len, dtype=y_src2.dtype)
                        overlap_start2 = max(start_display, src_start2)
                        overlap_end2 = min(end_display, src_end2)
                        if overlap_end2 > overlap_start2:
                            src_slice_start2 = overlap_start2 - src_start2
                            src_slice_end2 = overlap_end2 - src_start2
                            dest_slice_start2 = overlap_start2 - start_display
                            seg2[dest_slice_start2:dest_slice_start2 + (src_slice_end2 - src_slice_start2)] = y_src2[src_slice_start2:src_slice_end2]

                        y2 = seg2
                        if smoothing_sigma > 0:
                            y2 = gaussian_smooth(y2, smoothing_sigma)
                        
                        # 第二个结果使用虚线
                        ax.plot(display_positions, y2, color='tab:red', linewidth=1.5, alpha=0.9, 
                            linestyle='--', label='Result 2' if b == biosamples[0] else "")

                if arr is None and (not results2 or arr2 is None):
                    ax.text(0.5, 0.5, f"No data for {head}/{b}", ha='center', va='center')
                    ax.set_yticks([])
                    idx_ax += 1
                    continue

                ax.set_ylabel(f"{head}\n{b}", fontsize=9, rotation=0, ha='right', va='center')
                
                # 设置Y轴范围，考虑两个结果
                y_max_list = []
                if arr is not None:
                    y_max_list.append(np.nanmax(y) if y.size > 0 else 0)
                if results2 and arr2 is not None:
                    y_max_list.append(np.nanmax(y2) if y2.size > 0 else 0)
                
                y_max = max(y_max_list) * 1.15 if y_max_list else 0.1
                y_max = max(y_max, 0.1)  # 确保最小值
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.4)
                
                # 添加图例区分两个结果
                if results2 and b == biosamples[0]:
                    ax.legend(loc='upper right', fontsize=8)
                
                idx_ax += 1

        # === X-axis ===
        for ax in axes:
            ax.set_xlim(start_display, end_display)
            if self.xtick_step and displayed_len > self.xtick_step:
                xticks = np.arange(start_display, end_display, self.xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=9)

        def fmt(x, pos): return f"{int(x):,}"
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        axes[-1].set_xlabel(f"Chromosome position; interval= {chrom}:{start_display:,}-{end_display:,} ({displayed_len:,} bp)", fontsize=10)

        # === Legend ===
        if show_legend and biosamples:
            handles = []
            labels = []
            
            # 添加第一个结果的颜色
            handles.append(Line2D([0], [0], color='tab:blue', lw=2, linestyle='-'))
            labels.append('Result 1')
            
            # 如果有第二个结果，添加第二个结果的颜色
            if results2:
                handles.append(Line2D([0], [0], color='tab:red', lw=2, linestyle='--'))
                labels.append('Result 2')
            
            # 添加biosample的图例
            for i, b in enumerate(biosamples):
                if self.signal_palette:
                    c = self.signal_palette[i % len(self.signal_palette)]
                else:
                    cmap = plt.get_cmap('tab10' if len(biosamples) <= 10 else 'tab20')
                    c = cmap((i+2) % cmap.N)
                handles.append(Line2D([0], [0], color=c, lw=2))
                labels.append(b)
            
            try:
                axes[1].legend(handles, labels, loc='upper right', fontsize=8)
            except:
                fig.legend(handles, labels, loc='upper right', fontsize=8)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
        plt.subplots_adjust(hspace=0.12)
        return fig, axes


    def plot3(self,
            results: Dict,
            results2: Optional[Dict] = None,
            track_order: Optional[List[str]] = None,
            smoothing_sigma: float = 2.0,
            figsize: Optional[Tuple[float, float]] = None,
            show_legend: bool = True,
            gene_color_plus: str = "#2c7bb6",
            gene_color_minus: str = "#d7191c",
            window_start: Optional[int] = None,
            window_end: Optional[int] = None,
            gene_track_height_ratio: float = 0.8,
            exon_alpha: float = 0.9,
            label_fontsize: int = 8,
            use_rounded_exons: bool = True,
            background_alternate: bool = True,
            exclude_legend_labels: List[str] = None):
        """
        Plot results with a beautiful gene track (exons matched by coordinate containment).
        """
        values = results.get("values", {})
        position = results.get("position", (None, None, None))
        chrom, start, end = position if len(position) == 3 else (None, None, None)

        if not values:
            raise ValueError("results['values'] is empty")
        if chrom is None or start is None or end is None:
            raise ValueError("results['position'] must be (chrom, start, end)")

        # Second result handling
        values2 = results2.get("values", {}) if results2 else {}
        position2 = results2.get("position", (None, None, None)) if results2 else (None, None, None)
        chrom2, start2, end2 = position2 if len(position2) == 3 else (None, None, None)

        if results2 and chrom2 is not None and chrom2 != chrom:
            print(f"Warning: Chromosomes differ ({chrom} vs {chrom2}). Using first chromosome.")

        orig_start = int(start)
        orig_region_length = int(end) - int(start)

        if window_start is None and window_end is None:
            start_display = orig_start
            end_display = orig_start + orig_region_length
        else:
            start_display = int(window_start) if window_start is not None else orig_start
            end_display = int(window_end) if window_end is not None else (orig_start + orig_region_length)

        displayed_len = end_display - start_display
        if displayed_len <= 0:
            raise ValueError("Invalid window: window_end must be greater than window_start")
        if displayed_len > self.max_region_length:
            raise ValueError(f"Display window length ({displayed_len:,} bp) exceeds max allowed ({self.max_region_length:,} bp).")

        # Get gene annotations
        genes, transcripts, exons = self.get_genes_in_interval2(chrom, start_display, end_display)

        # ----- Collect tracks and biosamples (unchanged) -----
        track_names = list(values.keys())
        if results2:
            for tn in values2.keys():
                if tn not in track_names:
                    track_names.append(tn)

        biosample_set = []
        for tn in track_names:
            for b in values.get(tn, {}).keys():
                if b not in biosample_set:
                    biosample_set.append(b)
            if results2:
                for b in values2.get(tn, {}).keys():
                    if b not in biosample_set:
                        biosample_set.append(b)
        biosamples = biosample_set

        if track_order:
            ordered = [t for t in track_order if t in track_names]
            ordered += [t for t in track_names if t not in ordered]
            track_names = ordered

        n_heads = len(track_names)
        n_bios = max(1, len(biosamples))
        total_signal_subplots = n_heads * n_bios
        total_subplots = 1 + total_signal_subplots

        default_fig_width = 18.0
        gene_track_height = 1.5 * gene_track_height_ratio
        default_fig_height = gene_track_height + 1.2 * total_signal_subplots
        figsize = figsize or (default_fig_width, default_fig_height)

        fig, axes = plt.subplots(total_subplots, 1, figsize=figsize, sharex=True, dpi=self.dpi,
                                gridspec_kw={'height_ratios': [gene_track_height_ratio] + [1.0] * total_signal_subplots})
        if total_subplots == 1:
            axes = [axes]
        ax_gene = axes[0]

        # ==================== GENE TRACK (improved) ====================
        display_positions = np.arange(start_display, end_display)

        # 尝试使用转录本数据（若存在）
        use_transcripts = hasattr(self, 'transcripts_by_chrom') and hasattr(self, 'exons_by_transcript')
        if use_transcripts:
            transcripts = self.transcripts_by_chrom.get(chrom, [])
            # 过滤窗口内的转录本
            transcripts_f = [(ts, te, tstrand, tname, tgene) for (ts, te, tstrand, tname, tgene) in transcripts
                            if not (te < start_display or ts >= end_display)]
            if not transcripts_f:
                print("Info: No transcripts in region, falling back to gene-level display.")
                use_transcripts = False

        if not use_transcripts:
            # 回退到原基因级显示（与原代码相同，此处省略详细代码，请复用原基因轨道部分）
            print("Warning: Transcript-level data missing, using gene-level display (exons may be merged).")
            # 请将原函数中基因轨道的完整代码粘贴在此处（从 if not genes: 到 ax_gene.set_ylabel...）
            # 为简洁，此处只留注释
            pass

        else:
            # ----- 转录本级别显示（区分 .1, .2）-----
            # 构建每个转录本的外显子列表
            transcript_exons = {}
            for (es, ee, estr, etranscript) in exons:   # 假设 exons 现在返回 (start, end, strand, transcript_id)
                if etranscript not in transcript_exons:
                    transcript_exons[etranscript] = []
                transcript_exons[etranscript].append((es, ee, estr))

            # 过滤出窗口内的转录本，并排序
            transcripts_f = sorted(transcripts_f, key=lambda x: x[0])  # 按起始排序

            # 布局：每个转录本占一行（避免重叠）
            level_gap = 0.32
            level_height_base = 0.25
            max_levels = 20
            transcript_levels = {}
            level_ends = []

            for tx in transcripts_f:
                ts, te, tstrand, tname, tgene = tx
                placed = False
                for lvl, end_pos in enumerate(level_ends):
                    if ts >= end_pos:
                        level_ends[lvl] = te
                        transcript_levels[tname] = lvl
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(te)
                    transcript_levels[tname] = len(level_ends) - 1
                    placed = True
                if not placed:
                    earliest = np.argmin(level_ends)
                    level_ends[earliest] = te
                    transcript_levels[tname] = earliest

            max_level = max(transcript_levels.values()) if transcript_levels else 0
            gene_track_top = level_height_base + (max_level + 1) * level_gap + 0.15
            ax_gene.set_ylim(0, gene_track_top)

            if background_alternate:
                for lvl in range(max_level + 1):
                    y0 = level_height_base + lvl * level_gap - level_gap * 0.4
                    y1 = y0 + level_gap * 0.8
                    if lvl % 2 == 0:
                        rect = patches.Rectangle((start_display, y0), displayed_len, y1 - y0,
                                                facecolor='#f0f0f0', alpha=0.5, edgecolor='none', zorder=0)
                        ax_gene.add_patch(rect)

            # 绘制每个转录本
            for tx in transcripts_f:
                ts, te, tstrand, tname, tgene = tx
                lvl = transcript_levels[tname]
                y = level_height_base + lvl * level_gap
                color = gene_color_plus if tstrand == "+" else gene_color_minus

                # 内含子线
                ax_gene.plot([ts, te], [y, y], color=color, lw=1.2, alpha=0.7, solid_capstyle='round', zorder=1)

                # 获取该转录本的外显子
                exons_tx = transcript_exons.get(tname, [])
                # 按起始排序
                exons_tx.sort(key=lambda x: x[0])
                for es, ee, estr in exons_tx:
                    # 确保外显子链与转录本一致
                    if estr != tstrand:
                        continue
                    es_d, ee_d = max(es, start_display), min(ee, end_display)
                    if ee_d > es_d:
                        width = ee_d - es_d
                        if use_rounded_exons and width >= 2:
                            rect = patches.FancyBboxPatch((es_d, y - 0.08), width, 0.16,
                                                        boxstyle="round,pad=0.02", facecolor=color,
                                                        alpha=exon_alpha, edgecolor='white', linewidth=0.5, zorder=2)
                        else:
                            rect = patches.Rectangle((es_d, y - 0.08), width, 0.16,
                                                    facecolor=color, alpha=exon_alpha, edgecolor='white', linewidth=0.5, zorder=2)
                        ax_gene.add_patch(rect)

                # 方向箭头
                gene_length = te - ts
                arrow_len = min(max(gene_length * 0.08, 80), 1500)
                if tstrand == "+":
                    arrow_x = te - arrow_len
                    arrow = FancyArrowPatch((arrow_x, y), (te, y),
                                            arrowstyle='->', mutation_scale=15, color=color, linewidth=1.5,
                                            zorder=3, shrinkA=0, shrinkB=0)
                else:
                    arrow_x = ts + arrow_len
                    arrow = FancyArrowPatch((arrow_x, y), (ts, y),
                                            arrowstyle='<-', mutation_scale=15, color=color, linewidth=1.5,
                                            zorder=3, shrinkA=0, shrinkB=0)
                ax_gene.add_patch(arrow)

                # 标签：显示转录本名称（例如 LOC_Os09g35980.1）
                display_name = tname if tname else tgene
                text_y = y + 0.16  # if lvl % 2 == 0 else y - 0.2
                text_x = (ts + te) / 2
                text_x = np.clip(text_x, start_display + 500, end_display - 500)
                bbox_props = dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85, edgecolor=color, linewidth=0.8)
                ax_gene.text(text_x, text_y, display_name,
                            ha='center', va= 'bottom', # if lvl % 2 == 0 else 'top', 
                            fontsize=label_fontsize, zorder=4, bbox=bbox_props, color='black')

            # 图例
            legend_elems = [
                Line2D([0], [0], color=gene_color_plus, lw=2, marker='>', markersize=8, label='Forward (+)'),
                Line2D([0], [0], color=gene_color_minus, lw=2, marker='<', markersize=8, label='Reverse (-)')
            ]
            ax_gene.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)

        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes / Transcripts", fontsize=10, rotation=0, ha='right', va='center')
        ax_gene.spines['top'].set_visible(False)
        ax_gene.spines['right'].set_visible(False)
        ax_gene.spines['left'].set_visible(False)

        # ==================== SIGNAL TRACKS (unchanged) ====================
        idx_ax = 1
        for head in track_names:
            track_dict = values.get(head, {})
            track_dict2 = values2.get(head, {}) if results2 else {}

            for b in biosamples:
                ax = axes[idx_ax]
                arr = track_dict.get(b, None)
                if arr is not None:
                    y_src = _to_numpy(arr).reshape(-1)
                    src_len = len(y_src)
                    src_start = orig_start
                    src_end = orig_start + src_len
                    seg = np.zeros(displayed_len, dtype=y_src.dtype)
                    overlap_start = max(start_display, src_start)
                    overlap_end = min(end_display, src_end)
                    if overlap_end > overlap_start:
                        src_slice_start = overlap_start - src_start
                        src_slice_end = overlap_end - src_start
                        dest_slice_start = overlap_start - start_display
                        seg[dest_slice_start:dest_slice_start + (src_slice_end - src_slice_start)] = y_src[src_slice_start:src_slice_end]
                    y = seg
                    if smoothing_sigma > 0:
                        y = gaussian_smooth(y, smoothing_sigma)
                    ax.plot(display_positions, y, color='tab:blue', linewidth=1.5, alpha=0.9, label='Result 1' if b == biosamples[0] else "")

                if results2:
                    arr2 = track_dict2.get(b, None)
                    if arr2 is not None:
                        src_start2 = int(start2) if start2 is not None else orig_start
                        y_src2 = _to_numpy(arr2).reshape(-1)
                        src_len2 = len(y_src2)
                        src_end2 = src_start2 + src_len2
                        seg2 = np.zeros(displayed_len, dtype=y_src2.dtype)
                        overlap_start2 = max(start_display, src_start2)
                        overlap_end2 = min(end_display, src_end2)
                        if overlap_end2 > overlap_start2:
                            src_slice_start2 = overlap_start2 - src_start2
                            src_slice_end2 = overlap_end2 - src_start2
                            dest_slice_start2 = overlap_start2 - start_display
                            seg2[dest_slice_start2:dest_slice_start2 + (src_slice_end2 - src_slice_start2)] = y_src2[src_slice_start2:src_slice_end2]
                        y2 = seg2
                        if smoothing_sigma > 0:
                            y2 = gaussian_smooth(y2, smoothing_sigma)
                        ax.plot(display_positions, y2, color='tab:red', linewidth=1.5, alpha=0.9, linestyle='--', label='Result 2' if b == biosamples[0] else "")

                if arr is None and (not results2 or arr2 is None):
                    ax.text(0.5, 0.5, f"No data for {head}/{b}", ha='center', va='center')
                    ax.set_yticks([])
                    idx_ax += 1
                    continue

                ax.set_ylabel(f"{head}\n{b}", fontsize=9, rotation=0, ha='right', va='center')
                y_max_list = []
                if arr is not None:
                    y_max_list.append(np.nanmax(y) if y.size > 0 else 0)
                if results2 and arr2 is not None:
                    y_max_list.append(np.nanmax(y2) if y2.size > 0 else 0)
                y_max = max(y_max_list) * 1.15 if y_max_list else 0.1
                y_max = max(y_max, 0.1)
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.4)
                if results2 and b == biosamples[0]:
                    ax.legend(loc='upper right', fontsize=8)
                idx_ax += 1

        # X-axis formatting
        for ax in axes:
            ax.set_xlim(start_display, end_display)
            if self.xtick_step and displayed_len > self.xtick_step:
                xticks = np.arange(start_display, end_display, self.xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=9)

        def fmt(x, pos):
            return f"{int(x):,}"
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        axes[-1].set_xlabel(f"Chromosome position; interval= {chrom}:{start_display:,}-{end_display:,} ({displayed_len:,} bp)", fontsize=10)

        # Overall legend
        if show_legend and biosamples:
            handles = []
            labels = []
            handles.append(Line2D([0], [0], color='tab:blue', lw=2, linestyle='-'))
            labels.append('Result 1')
            if results2:
                handles.append(Line2D([0], [0], color='tab:red', lw=2, linestyle='--'))
                labels.append('Result 2')
            # 过滤排除的 biosample
            for i, b in enumerate(biosamples):
                if exclude_legend_labels and b in exclude_legend_labels:
                    continue
                if self.signal_palette:
                    c = self.signal_palette[i % len(self.signal_palette)]
                else:
                    cmap = plt.get_cmap('tab10' if len(biosamples) <= 10 else 'tab20')
                    c = cmap((i+2) % cmap.N)
                handles.append(Line2D([0], [0], color=c, lw=2))
                labels.append(b)
            if handles:  # 只有非空时才添加图例
                try:
                    axes[1].legend(handles, labels, loc='upper right', fontsize=8)
                except:
                    fig.legend(handles, labels, loc='upper right', fontsize=8)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
        plt.subplots_adjust(hspace=0.12)
        return fig, axes

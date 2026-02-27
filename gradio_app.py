"""
Gradio web UI for the Communication and Memory Overhead Tool.

Launch:
    cd 291P
    pip install gradio
    python gradio_app.py

Then open the URL printed in the terminal (default http://127.0.0.1:7860).
"""

import sys
from io import StringIO
from pathlib import Path

import gradio as gr

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from comm_overhead import (
    TopologyKind,
    build_topology,
    CollectiveKind,
    collective_latency_and_volume,
    per_gpu_memory_bytes,
    analyze_config,
)
from comm_overhead.analyze import Config

TOPOLOGY_CHOICES = ["ring", "tree", "hierarchical", "switch", "mesh", "torus"]
TOPOLOGY_MAP = {
    "ring": TopologyKind.RING,
    "tree": TopologyKind.TREE,
    "hierarchical": TopologyKind.HIERARCHICAL,
    "switch": TopologyKind.SWITCH,
    "mesh": TopologyKind.MESH,
    "torus": TopologyKind.TORUS,
}
COLLECTIVE_CHOICES = ["ALL_REDUCE", "ALL_GATHER", "REDUCE_SCATTER", "BROADCAST", "ALL_TO_ALL"]
COLLECTIVE_MAP = {
    "ALL_REDUCE": CollectiveKind.ALL_REDUCE,
    "ALL_GATHER": CollectiveKind.ALL_GATHER,
    "REDUCE_SCATTER": CollectiveKind.REDUCE_SCATTER,
    "BROADCAST": CollectiveKind.BROADCAST,
    "ALL_TO_ALL": CollectiveKind.ALL_TO_ALL,
}

# ─── Metrics that the user can toggle ───────────────────────────────────────
COLLECTIVE_METRICS = [
    "Latency (ms)",
    "Volume per GPU (GB)",
    "Steps",
    "Intra-node latency (ms)",
    "Inter-node latency (ms)",
]

ANALYSIS_METRICS = [
    "Per-GPU Memory (GB)",
    "Memory Breakdown",
    "DP AllReduce Latency (ms)",
    "TP Per-layer Latency (ms)",
    "CP All2All Latency (ms)",
    "Total Comm/Step Upper Bound (ms)",
    "Collective Reports Table",
]


def _fmt(val, unit="", precision=4):
    if val is None:
        return "N/A"
    return f"{val:.{precision}f} {unit}".strip()


def _grid_dims_from_inputs(grid_nx, grid_ny, N, topology):
    """
    Parse n_x, n_y from UI; treat 0/None/empty as auto. Validate when both given.
    Returns (nx, ny) for build_topology/Config; for mesh/torus both can be None (= auto).
    """
    try:
        nx = int(float(grid_nx)) if grid_nx is not None and float(grid_nx) > 0 else None
    except (TypeError, ValueError):
        nx = None
    try:
        ny = int(float(grid_ny)) if grid_ny is not None and float(grid_ny) > 0 else None
    except (TypeError, ValueError):
        ny = None
    if topology not in ("mesh", "torus"):
        return None, None
    if nx is not None and ny is not None:
        if nx * ny != N:
            raise ValueError(f"n_x × n_y must equal N (GPU count). Got n_x={nx}, n_y={ny}, N={N}. Leave both empty to auto-infer.")
    return nx, ny


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1 – Single Collective Operation
# ═══════════════════════════════════════════════════════════════════════════════

def run_collective(
    topology, collective_op, N, M_str, gpus_per_node, grid_nx, grid_ny,
    tree_algo, verbose, selected_metrics,
):
    try:
        topo_kind = TOPOLOGY_MAP[topology]
        coll_kind = COLLECTIVE_MAP[collective_op]
        M = float(M_str)
        N = int(N)
        gpn = int(gpus_per_node) if gpus_per_node else None
        nx, ny = _grid_dims_from_inputs(grid_nx, grid_ny, N, topology)
        topo = build_topology(kind=topo_kind, N=N, gpus_per_node=gpn, n_x=nx, n_y=ny)

        verbose_output = ""
        if verbose:
            old_stdout = sys.stdout
            sys.stdout = buf = StringIO()
            res = collective_latency_and_volume(
                coll_kind, M, topo,
                algorithm="tree" if tree_algo else None,
                verbose=True,
            )
            verbose_output = buf.getvalue()
            sys.stdout = old_stdout
        else:
            res = collective_latency_and_volume(
                coll_kind, M, topo,
                algorithm="tree" if tree_algo else None,
                verbose=False,
            )

        lines = []
        lines.append(f"## {collective_op} on {topology.upper()} (N={N})\n")
        
        # Build table data
        table_rows = []
        if "Latency (ms)" in selected_metrics:
            table_rows.append(["Latency", _fmt(res.latency_s * 1000, 'ms')])
        if "Volume per GPU (GB)" in selected_metrics:
            table_rows.append(["Volume per GPU", _fmt(res.volume_bytes / 1e9, 'GB')])
        if "Steps" in selected_metrics:
            table_rows.append(["Steps", str(res.steps)])
        if "Intra-node latency (ms)" in selected_metrics and res.intra_latency_s is not None:
            table_rows.append(["Intra-node latency", _fmt(res.intra_latency_s * 1000, 'ms')])
        if "Inter-node latency (ms)" in selected_metrics and res.inter_latency_s is not None:
            table_rows.append(["Inter-node latency", _fmt(res.inter_latency_s * 1000, 'ms')])
        
        # Generate markdown table
        if table_rows:
            # Calculate column width for alignment (minimum 20 characters)
            max_label_width = max(max(len(row[0]) for row in table_rows), 20)
            max_value_width = max(max(len(row[1]) for row in table_rows), 15)
            
            # Table header (left-aligned)
            lines.append(f"| {'Metric':<{max_label_width}} | {'Value':<{max_value_width}} |")
            lines.append(f"|{'-' * (max_label_width + 2)}|{'-' * (max_value_width + 2)}|")
            
            # Table rows (left-aligned)
            for label, value in table_rows:
                lines.append(f"| {label:<{max_label_width}} | {value:<{max_value_width}} |")
            lines.append("")

        if verbose and verbose_output.strip():
            lines.append("---")
            lines.append("### Verbose formula breakdown:")
            lines.append("```")
            lines.append(verbose_output)
            lines.append("```")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2 – Full Analysis (DP / TP / CP + Memory)
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(
    topology, num_gpus, dp, tp, cp, params_str,
    batch_size, seq_length, hidden_size, num_layers,
    gpus_per_node, grid_nx, grid_ny,
    verbose, selected_metrics,
):
    try:
        num_gpus_int = int(num_gpus)
        nx, ny = _grid_dims_from_inputs(grid_nx, grid_ny, num_gpus_int, topology)
        config = Config(
            num_gpus=num_gpus_int,
            topology_kind=TOPOLOGY_MAP[topology],
            gpus_per_node=int(gpus_per_node) if gpus_per_node else None,
            dp_degree=int(dp),
            tp_degree=int(tp),
            cp_degree=int(cp),
            num_parameters=float(params_str),
            batch_size=int(batch_size) if batch_size else None,
            seq_length=int(seq_length) if seq_length else None,
            hidden_size=int(hidden_size) if hidden_size else None,
            num_layers=int(num_layers) if num_layers else None,
            nx=nx,
            ny=ny,
        )
        include_bd = "Memory Breakdown" in selected_metrics

        verbose_output = ""
        if verbose:
            old_stdout = sys.stdout
            sys.stdout = buf = StringIO()
            result = analyze_config(config, include_memory_breakdown=include_bd, verbose=True)
            verbose_output = buf.getvalue()
            sys.stdout = old_stdout
        else:
            result = analyze_config(config, include_memory_breakdown=include_bd, verbose=False)

        lines = []
        lines.append(f"## Analysis: {topology.upper()}, GPUs={num_gpus}, DP={dp}, TP={tp}, CP={cp}\n")
        
        # Build main metrics table
        main_table_rows = []
        if "Per-GPU Memory (GB)" in selected_metrics:
            main_table_rows.append(["Per-GPU Memory (no ZeRO)", _fmt(result.per_gpu_memory_bytes / 1e9, 'GB')])
        
        if "Memory Breakdown" in selected_metrics and result.memory_breakdown:
            bd = result.memory_breakdown
            main_table_rows.append(["  - Weights", _fmt(bd.weights_bytes / 1e9, 'GB')])
            main_table_rows.append(["  - Gradients", _fmt(bd.gradients_bytes / 1e9, 'GB')])
            main_table_rows.append(["  - Optimizer", _fmt(bd.optimizer_bytes / 1e9, 'GB')])
        
        if result.gpus_per_node is not None:
            main_table_rows.append(["GPUs per node", f"{result.gpus_per_node}, Nodes: {result.num_nodes}"])
        
        if "DP AllReduce Latency (ms)" in selected_metrics and result.dp_allreduce_latency_s is not None:
            main_table_rows.append(["DP AllReduce Latency", _fmt(result.dp_allreduce_latency_s * 1000, 'ms')])
        
        for r in result.collective_reports:
            if "TP" in r.name and "TP Per-layer Latency (ms)" in selected_metrics:
                main_table_rows.append(["TP Per-layer Latency", _fmt(r.latency_s * 1000, 'ms')])
            if "CP" in r.name and "CP All2All Latency (ms)" in selected_metrics:
                main_table_rows.append(["CP All2All Latency", _fmt(r.latency_s * 1000, 'ms')])
        
        if "Total Comm/Step Upper Bound (ms)" in selected_metrics and result.total_comm_time_per_step_upper_bound_s is not None:
            main_table_rows.append(["Total Comm/Step (upper)", _fmt(result.total_comm_time_per_step_upper_bound_s * 1000, 'ms')])
        
        # Generate main metrics table
        if main_table_rows:
            max_label_width = max(max(len(row[0]) for row in main_table_rows), 25)
            max_value_width = max(max(len(row[1]) for row in main_table_rows), 15)
            
            lines.append("| Metric | Value |")
            lines.append(f"|{'-' * (max_label_width + 2)}|{'-' * (max_value_width + 2)}|")
            
            for label, value in main_table_rows:
                lines.append(f"| {label:<{max_label_width}} | {value:<{max_value_width}} |")
            lines.append("")
        
        # Collective Reports Table
        if "Collective Reports Table" in selected_metrics and result.collective_reports:
            lines.append("### Collective Reports\n")
            
            # Calculate column widths
            max_name_width = max(max(len(r.name) for r in result.collective_reports), 35)
            latency_width = 12
            steps_width = 6
            vol_width = 10
            
            # Table header
            lines.append(f"| {'Collective':<{max_name_width}} | {'Latency(ms)':>{latency_width}} | {'Steps':>{steps_width}} | {'Vol(GB)':>{vol_width}} | Details |")
            lines.append(f"|{'-' * (max_name_width + 2)}|{'-' * (latency_width + 2)}|{'-' * (steps_width + 2)}|{'-' * (vol_width + 2)}|---------|")
            
            # Table rows
            for r in result.collective_reports:
                intra_str = ""
                if r.intra_latency_s is not None:
                    intra_str = f"intra={r.intra_latency_s*1000:.2f}ms, inter={r.inter_latency_s*1000:.2f}ms"
                lines.append(
                    f"| {r.name:<{max_name_width}} | {r.latency_s*1000:>{latency_width}.2f} | {r.steps:>{steps_width}} | {r.volume_bytes/1e9:>{vol_width}.2f} | {intra_str} |"
                )
            lines.append("")

        if verbose and verbose_output.strip():
            lines.append("---")
            lines.append("### Verbose formula breakdown:")
            lines.append("```")
            lines.append(verbose_output)
            lines.append("```")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 3 – Run All (comparison table)
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_comparison():
    try:
        M = 140e9
        rows = []

        for topo_name, topo_kind in [
            ("ring", TopologyKind.RING),
            ("tree", TopologyKind.TREE),
            ("hierarchical", TopologyKind.HIERARCHICAL),
            ("switch", TopologyKind.SWITCH),
            ("mesh", TopologyKind.MESH),
        ]:
            for N in [8, 16]:
                if topo_kind != TopologyKind.HIERARCHICAL:
                    gpn_list = [None]
                else:
                    gpn_list = [None, 4] if N == 8 else [None, 4, 8]
                for gpn in gpn_list:
                    topo = build_topology(kind=topo_kind, N=N, gpus_per_node=gpn)
                    res = collective_latency_and_volume(CollectiveKind.ALL_REDUCE, M, topo)
                    tag = f"collective | {topo_name} | N={N}"
                    if gpn is not None:
                        tag += f" | gpn={gpn}"
                    latency = f"{res.latency_s * 1000:.2f}"
                    steps = str(res.steps)
                    intra = f"{res.intra_latency_s * 1000:.2f}" if res.intra_latency_s else ""
                    inter = f"{res.inter_latency_s * 1000:.2f}" if res.inter_latency_s else ""
                    rows.append(["collective", topo_name, N, "", "" if gpn is None else gpn, latency, steps, intra, inter])

        # rows.append(["", "", "", "", ""])
        
        for topo_name, topo_kind in [
            ("ring", TopologyKind.RING),
            ("tree", TopologyKind.TREE),
            ("hierarchical", TopologyKind.HIERARCHICAL),
            ("switch", TopologyKind.SWITCH),
            ("mesh", TopologyKind.MESH),
        ]:
            gpn = 4 if topo_kind == TopologyKind.HIERARCHICAL else None
            config = Config(num_gpus=8, topology_kind=topo_kind, dp_degree=8, num_parameters=70e9, gpus_per_node=gpn)
            result = analyze_config(config, include_memory_breakdown=False)
            dp_lat = f"{result.dp_allreduce_latency_s * 1000:.2f}" if result.dp_allreduce_latency_s else "N/A"
            tag = f"analysis  | {topo_name} | 8 GPUs | DP=8"
            if gpn:
                tag += f" | gpn={gpn}"
            rows.append(["analysis", topo_name, "8", "8", "" if gpn is None else gpn, dp_lat, "", "", ""])

        return rows
    except Exception as e:
        return [[f"Error: {str(e)}", "", "", "", ""]]


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 4 – Memory Only
# ═══════════════════════════════════════════════════════════════════════════════

def run_memory(params_str, show_breakdown):
    try:
        params = float(params_str)
        if show_breakdown:
            total, bd = per_gpu_memory_bytes(params, return_breakdown=True)
            rows = [
                ["Model parameters", f"{params:.2e}", ""],
                ["Total memory", f"{total / 1e9:.2f} GB", f"{total / (1024**3):.2f} GiB"],
                ["", "", ""],
                ["Weights", f"{bd.weights_bytes / 1e9:.2f} GB", ""],
                ["Gradients", f"{bd.gradients_bytes / 1e9:.2f} GB", ""],
                ["Optimizer (AdamW: m_t + v_t + master)", f"{bd.optimizer_bytes / 1e9:.2f} GB", ""],
            ]
            return rows
        else:
            total = per_gpu_memory_bytes(params, return_breakdown=False)
            rows = [
                ["Total memory", f"{total / 1e9:.2f} GB", f"{total / (1024**3):.2f} GiB"],
            ]
            return rows
    except Exception as e:
        return [[f"Error: {str(e)}", "", ""]]


# ═══════════════════════════════════════════════════════════════════════════════
#  Build the Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

def update_topology_fields(topology):
    is_hierarchical = topology == "hierarchical"
    is_2d_grid = topology in ("mesh", "torus")
    return (
        gr.update(visible=is_hierarchical),  # gpn
        gr.update(visible=is_2d_grid),      # nx
        gr.update(visible=is_2d_grid),       # ny
    )

def update_operation_fields(operation):
    is_allreduce = operation == "ALL_REDUCE"
    return (
        gr.update(visible=is_allreduce)
    )

def build_app():
    with gr.Blocks(title="Comm & Memory Overhead Tool") as app:
        gr.Markdown(
            "# Communication & Memory Overhead Tool\n"
            "Calculate communication latency, volume, and memory overhead for distributed "
            "LLM training under various topologies and parallelism configurations.\n"
            "Select a tab, configure parameters, **choose which metrics to display**, then click **Run**."
        )

        # ─── Tab 1: Single Collective ────────────────────────────────────
        with gr.Tab("Single Collective"):
            with gr.Row():
                with gr.Column(scale=1):
                    coll_topology = gr.Dropdown(TOPOLOGY_CHOICES, value="ring", label="Topology")
                    coll_op = gr.Dropdown(COLLECTIVE_CHOICES, value="ALL_REDUCE", label="Collective Operation")
                    coll_N = gr.Number(value=8, label="N (GPUs)", precision=0)
                    coll_M = gr.Textbox(value="140e9", label="M (tensor size in bytes)")
                with gr.Column(scale=1):
                    coll_gpn = gr.Number(value=None, label="GPUs per node (hierarchical only)", precision=0, visible=False)
                    coll_nx = gr.Number(value=None, label="n_x (mesh & torus)", placeholder="auto if empty", precision=0, visible=False)
                    coll_ny = gr.Number(value=None, label="n_y (mesh & torus)", placeholder="auto if empty", precision=0, visible=False)
                    coll_tree = gr.Checkbox(value=False, label="Use tree algorithm (AllReduce only)", visible=True)
                    coll_verbose = gr.Checkbox(value=False, label="Verbose (show formulas)")

            coll_metrics = gr.CheckboxGroup(
                COLLECTIVE_METRICS,
                value=["Latency (ms)", "Volume per GPU (GB)", "Steps"],
                label="Select metrics to display",
            )
            coll_btn = gr.Button("Run", variant="primary", size="lg")
            coll_output = gr.Markdown(label="Results")

            coll_topology.change(
                fn=update_topology_fields,
                inputs=[coll_topology],
                outputs=[coll_gpn, coll_nx, coll_ny],
            )
            coll_op.change(
                fn=update_operation_fields,
                inputs=[coll_op],
                outputs=[coll_tree],
            )

            coll_btn.click(
                fn=run_collective,
                inputs=[
                    coll_topology, coll_op, coll_N, coll_M,
                    coll_gpn, coll_nx, coll_ny,
                    coll_tree, coll_verbose, coll_metrics,
                ],
                outputs=coll_output,
            )

        # ─── Tab 2: Full Analysis ────────────────────────────────────────
        with gr.Tab("Full Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    ana_topology = gr.Dropdown(TOPOLOGY_CHOICES, value="ring", label="Topology")
                    ana_gpus = gr.Number(value=8, label="Total GPUs", precision=0)
                    ana_dp = gr.Number(value=8, label="DP degree", precision=0)
                    ana_tp = gr.Number(value=1, label="TP degree", precision=0)
                    ana_cp = gr.Number(value=1, label="CP degree", precision=0)
                    ana_params = gr.Textbox(value="70e9", label="Model parameters (e.g. 70e9)")
                with gr.Column(scale=1):
                    ana_batch = gr.Number(value=None, label="Batch size (optional)", precision=0)
                    ana_seq = gr.Number(value=None, label="Seq length (optional, needed for TP/CP)", precision=0)
                    ana_hidden = gr.Number(value=None, label="Hidden size (optional, needed for TP/CP)", precision=0)
                    ana_layers = gr.Number(value=None, label="Num layers (optional, for total comm/step)", precision=0)
                    ana_gpn = gr.Number(value=None, label="GPUs per node (hierarchical only)", precision=0, visible=False)
                    ana_nx = gr.Number(value=None, label="n_x (mesh & torus)", placeholder="auto if empty", precision=0, visible=False)
                    ana_ny = gr.Number(value=None, label="n_y (mesh & torus)", placeholder="auto if empty", precision=0, visible=False)
                    ana_verbose = gr.Checkbox(value=False, label="Verbose (show formulas)")

            ana_metrics = gr.CheckboxGroup(
                ANALYSIS_METRICS,
                value=[
                    "Per-GPU Memory (GB)",
                    "Memory Breakdown",
                    "DP AllReduce Latency (ms)",
                    "Collective Reports Table",
                ],
                label="Select metrics to display",
            )
            ana_btn = gr.Button("Run", variant="primary", size="lg")
            ana_output = gr.Markdown(label="Results")

            ana_topology.change(
                fn=update_topology_fields,
                inputs=[ana_topology],
                outputs=[ana_gpn, ana_nx, ana_ny],
            )
            
            ana_btn.click(
                fn=run_analysis,
                inputs=[
                    ana_topology, ana_gpus, ana_dp, ana_tp, ana_cp, ana_params,
                    ana_batch, ana_seq, ana_hidden, ana_layers,
                    ana_gpn, ana_nx, ana_ny,
                    ana_verbose, ana_metrics,
                ],
                outputs=ana_output,
            )

        # ─── Tab 3: Memory Calculator ────────────────────────────────────
        with gr.Tab("Memory Calculator"):
            with gr.Row():
                mem_params = gr.Textbox(value="70e9", label="Model parameters (e.g. 70e9 for 70B)")
                mem_bd = gr.Checkbox(value=True, label="Show breakdown")
            mem_btn = gr.Button("Run", variant="primary", size="lg")
            mem_output = gr.Dataframe(
                label="Per-GPU Memory (no ZeRO)",
                headers=["Item", "Value (GB)", "Value (GiB)"],
                wrap=True,
            )

            mem_btn.click(
                fn=run_memory,
                inputs=[mem_params, mem_bd],
                outputs=mem_output,
            )

        # ─── Tab 4: Run All (Comparison Table) ───────────────────────────
        with gr.Tab("Run All (Comparison)"):
            gr.Markdown(
                "Run all predefined topology/config combinations and display a comparison table.\n"
                "Includes AllReduce on all topologies (N=8, 16) and full analysis (DP=8, 70B)."
            )
            runall_btn = gr.Button("Run All", variant="primary", size="lg")
            runall_output = gr.Dataframe(
                label="Comparison Table",
                # headers=["Tag", "Latency(ms)", "Steps", "Intra(ms)", "Inter(ms)"],
                headers=["Type", "Topology", "Number of GPUs", "DP Degree", "gpn", "Latency(ms)", "Steps", "Intra(ms)", "Inter(ms)"],
                wrap=True,
            )

            runall_btn.click(fn=run_all_comparison, inputs=[], outputs=runall_output)

    return app


if __name__ == "__main__":
    app = build_app()
    # app.launch(theme=gr.themes.Soft())
    app.launch()

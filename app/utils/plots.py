"""Plotly visualization helpers for the Streamlit dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Union, Set


def plot_sensor_network_topology(adj_matrix: np.ndarray, sensor_flows: Optional[np.ndarray] = None) -> go.Figure:
    """Creates a clearly labelled schematic network topology of the 36 traffic sensors (non-geographic)."""
    n_nodes = adj_matrix.shape[0]
    
    # Position nodes in a clean circular schematic layout with logical spacing
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x_nodes = np.cos(angles) * 10.0
    y_nodes = np.sin(angles) * 10.0

    degrees = np.sum(adj_matrix, axis=1)
    color_vals = sensor_flows if sensor_flows is not None and len(sensor_flows) == n_nodes else degrees
    color_title = "Flow [0,1]" if sensor_flows is not None else "Neighbors"

    # Edge traces
    edge_x = []
    edge_y = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] > 0:
                edge_x.extend([x_nodes[i], x_nodes[j], None])
                edge_y.extend([y_nodes[i], y_nodes[j], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color='rgba(71, 85, 105, 0.45)'),
        hoverinfo='none',
        mode='lines',
        name='Corridor Link (Adjacency)'
    )

    node_hover_text = [
        f"<b>Sensor S{i:02d}</b><br>Connectivity: {int(degrees[i])} adjacent sensors<br>" +
        (f"Observed Flow: {color_vals[i]:.3f} (Normalized)" if sensor_flows is not None else "Loop Station on Northern Virginia/D.C. Highway Network (UCI #734)")
        for i in range(n_nodes)
    ]

    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        text=[f"S{i:02d}" for i in range(n_nodes)],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_hover_text,
        marker=dict(
            showscale=True,
            colorscale='Blues' if sensor_flows is None else 'Viridis',
            color=color_vals,
            size=22,
            colorbar=dict(
                thickness=14,
                title=color_title,
                xanchor='left',
                len=0.75
            ),
            line=dict(width=2, color='#0F172A')
        ),
        name='Sensor Station'
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"<b>Schematic Highway Network Topology</b> ({n_nodes} Sensors, {int(np.sum(adj_matrix))} Connected Links — Northern Virginia / D.C.)",
                font=dict(size=16, color="#0F172A")
            ),
            annotations=[
                dict(
                    text="<b>NON-GEOGRAPHIC SCHEMATIC</b>: Actual GPS coordinates not provided in UCI #734 (Zhao et al., 2019).<br>Layout visualizes topological sensor adjacency and upstream/downstream flow dependencies.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.08,
                    xanchor="center", yanchor="top",
                    font=dict(size=11, color="#64748B")
                )
            ],
            showlegend=False,
            hovermode='closest',
            margin=dict(b=35, l=15, r=15, t=45),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=420,
            template="plotly_white"
        )
    )
    return fig


def plot_sensor_flow_strip_dataset_a(
    sensor_flows: Optional[np.ndarray] = None,
    selected_sensor: Union[int, List[int], Set[int]] = 0
) -> go.Figure:
    """Renders a clean 36-sensor flow strip for Dataset A, highlighting active/selected sensor(s).
    
    Replaces tangled circular graphs with a structured linear sensor array
    showing normalized flow [0, 1] across all 36 detector stations.
    """
    n_sensors = 36
    sensor_labels = [f"S{i:02d}" for i in range(n_sensors)]
    
    if isinstance(selected_sensor, (list, tuple, set)):
        selected_set = {int(s) for s in selected_sensor}
    else:
        selected_set = {int(selected_sensor)}
    
    if sensor_flows is None or len(sensor_flows) != n_sensors:
        flows = np.zeros(n_sensors)
    else:
        flows = np.clip(np.array(sensor_flows, dtype=float), 0.0, 1.0)
        
    bar_colors = []
    border_widths = []
    border_colors = []
    
    for i in range(n_sensors):
        if i in selected_set:
            bar_colors.append('#2563EB')  # Highlighted selected sensor: primary blue
            border_widths.append(2.5)
            border_colors.append('#0F172A')
        elif flows[i] >= 0.65:
            bar_colors.append('#F97316')  # High throughput (>0.65)
            border_widths.append(1)
            border_colors.append('#EA580C')
        elif flows[i] >= 0.40:
            bar_colors.append('#38BDF8')  # Moderate flow
            border_widths.append(1)
            border_colors.append('#0284C7')
        else:
            bar_colors.append('#94A3B8')  # Low flow
            border_widths.append(1)
            border_colors.append('#64748B')
            
    hover_texts = [
        f"<b>Sensor S{i:02d}</b>{' [SELECTED]' if i in selected_set else ''}<br>"
        f"Current Normalized Flow: {flows[i]:.3f}<br>"
        f"Regime: {'Near-Capacity Surge' if flows[i] >= 0.65 else ('Balanced Demand' if flows[i] >= 0.40 else 'Low Demand')}"
        for i in range(n_sensors)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sensor_labels,
        y=flows,
        marker=dict(
            color=bar_colors,
            line=dict(width=border_widths, color=border_colors)
        ),
        hoverinfo='text',
        hovertext=hover_texts,
        showlegend=False
    ))
    
    # Add horizontal threshold line at 0.65 (Near-Capacity Surge)
    fig.add_hline(
        y=0.65,
        line_dash="dot",
        line_color="#EA580C",
        line_width=1.5,
        annotation_text="High Flow Threshold (0.65)",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#EA580C")
    )
    
    fig.update_layout(
        title=dict(
            text="<b>36-Sensor Corridor Flow Distribution</b> (UCI #734 — Northern Virginia / D.C. Freeway Network)",
            font=dict(size=13, color="#0F172A"),
            x=0.01,
            y=0.96
        ),
        xaxis=dict(
            title="Loop Detector Stations (S00 – S35)",
            tickfont=dict(size=9.5),
            showgrid=False
        ),
        yaxis=dict(
            title="Normalized Flow [0, 1]",
            range=[0.0, 1.05],
            showgrid=True,
            gridcolor="#F1F5F9",
            dtick=0.2
        ),
        template="plotly_white",
        height=240,
        margin=dict(l=45, r=25, t=40, b=45)
    )
    return fig


def plot_sensor_flow_heatmap_dataset_a(
    df_val_a: pd.DataFrame,
    selected_sensors: Union[int, List[int], Set[int]] = 0,
    step_start: Optional[int] = None,
    n_steps: int = 48
) -> go.Figure:
    """Renders a spatiotemporal flow heatmap across all 36 sensors over consecutive 15-minute time steps."""
    if isinstance(selected_sensors, (list, tuple, set)):
        selected_set = {int(s) for s in selected_sensors}
    else:
        selected_set = {int(selected_sensors)}

    all_steps = sorted(df_val_a['step_idx'].unique())
    if step_start is None or step_start not in all_steps:
        step_start = all_steps[0]

    start_idx = all_steps.index(step_start)
    slice_steps = all_steps[start_idx : start_idx + n_steps]

    sub_df = df_val_a[df_val_a['step_idx'].isin(slice_steps)].copy()
    val_col = 'persistence_pred' if 'persistence_pred' in sub_df.columns else 'target'

    pivot = sub_df.pivot(index='sensor_idx', columns='step_idx', values=val_col)
    pivot = pivot.reindex(index=range(36), columns=slice_steps, fill_value=0.0)

    y_labels = [f"{'⭐ ' if i in selected_set else '   '}S{i:02d}" for i in range(36)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Step {s}" for s in slice_steps],
        y=y_labels,
        colorscale='Viridis',
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(
            title="Flow [0,1]",
            thickness=14,
            len=0.85
        ),
        hovertemplate="Station: %{y}<br>Timeline: %{x}<br>Observed Flow: %{z:.3f} (Normalized)<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>36-Sensor Spatiotemporal Flow Heatmap</b> (Normalized Flow [0, 1] over 15-Minute Intervals)",
            font=dict(size=13, color="#0F172A")
        ),
        xaxis=dict(
            title="15-Minute Observation Steps (Shared Timeline)",
            tickangle=-45,
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            title="Highway Stations (⭐ Highlighted)",
            dtick=1,
            tickfont=dict(size=9.5)
        ),
        template="plotly_white",
        height=520,
        margin=dict(l=75, r=30, t=45, b=60)
    )
    return fig




def plot_corridor_schematic_dataset_b(current_volume: Optional[float] = None, forecast_volume: Optional[float] = None) -> go.Figure:
    """Generates a schematic straight-line highway corridor overview for Interstate 94 (UCI 492)."""
    fig = go.Figure()

    # Highway mainline representation (Westbound/Eastbound highway corridor)
    x_line = [0, 10]
    y_line = [0, 0]

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        line=dict(color='#334155', width=12),
        hoverinfo='skip',
        showlegend=False
    ))

    # Center dash line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        line=dict(color='#F8FAFC', width=2, dash='dash'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Verified station and clearly labeled illustrative corridor landmarks
    landmarks = [
        {
            "display": "<b>Minneapolis W.</b><br><span style='font-size:10px;color:#64748B;'>(Illustrative)</span>",
            "name": "Minneapolis Terminal (Illustrative)",
            "mp": 0.8,
            "pos": "bottom center",
            "desc": "Western Metropolitan Terminal (Illustrative Boundary)"
        },
        {
            "display": "<b>Upstream Inflow</b><br><span style='font-size:10px;color:#64748B;'>(Illustrative)</span>",
            "name": "Upstream Inflow Segment (Illustrative)",
            "mp": 3.0,
            "pos": "top center",
            "desc": "Upstream Highway Merge Segment (Illustrative)"
        },
        {
            "display": "<b>Station ATR 301</b><br><span style='font-size:11px;color:#DC2626;'><b>[Official UCI 492]</b></span>",
            "name": "Station ATR 301 (Official UCI 492)",
            "mp": 5.2,
            "pos": "top center",
            "desc": "MN DoT Automatic Traffic Recorder (Station 301)<br>Sole Verified Loop Sensor in Dataset B"
        },
        {
            "display": "<b>Downstream Segment</b><br><span style='font-size:10px;color:#64748B;'>(Illustrative)</span>",
            "name": "Downstream Segment (Illustrative)",
            "mp": 7.4,
            "pos": "bottom center",
            "desc": "Downstream Highway Segment (Illustrative)"
        },
        {
            "display": "<b>St. Paul E.</b><br><span style='font-size:10px;color:#64748B;'>(Illustrative)</span>",
            "name": "St. Paul Terminal (Illustrative)",
            "mp": 9.2,
            "pos": "top center",
            "desc": "Eastern Metropolitan Terminal (Illustrative Boundary)"
        }
    ]

    for lm in landmarks:
        is_sensor = "ATR 301" in lm["name"]
        color = "#EF4444" if is_sensor else "#3B82F6"
        size = 22 if is_sensor else 14

        vol_text = ""
        if is_sensor:
            if current_volume is not None:
                vol_text += f"<br>Current Observed: {current_volume:,.0f} veh/h"
            if forecast_volume is not None:
                vol_text += f"<br>1h-Ahead Forecast: {forecast_volume:,.0f} veh/h"

        fig.add_trace(go.Scatter(
            x=[lm["mp"]], y=[0],
            mode='markers+text',
            text=[lm["display"]],
            textposition=lm["pos"],
            hoverinfo='text',
            hovertext=[f"<b>{lm['name']}</b><br>{lm['desc']}{vol_text}"],
            marker=dict(size=size, color=color, line=dict(color='#0F172A', width=2)),
            showlegend=False
        ))

    # Corridor orientation indicator
    fig.add_annotation(
        x=5.2, y=-0.65,
        text="<b>CORRIDOR ORIENTATION: INTERSTATE 94 (Westbound: St. Paul ➔ Minneapolis) ▶▶</b>",
        showarrow=False,
        font=dict(size=11, color="#1E293B", weight="bold")
    )

    fig.update_layout(
        title=dict(
            text="<b>Interstate 94 Corridor Schematic</b> (Station ATR 301 — Illustrative Landmark Layout)",
            font=dict(size=15, color="#0F172A")
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 10.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        height=280,
        margin=dict(l=20, r=20, t=40, b=25),
        template="plotly_white"
    )
    return fig


def plot_diurnal_traffic_profile(df_b: pd.DataFrame) -> go.Figure:
    """Generates 24-hour diurnal profile curves comparing weekday vs weekend traffic volumes."""
    df = df_b.copy()
    if 'date_time' in df.columns:
        df['dt'] = pd.to_datetime(df['date_time'])
        df['hour'] = df['dt'].dt.hour
        df['is_weekend'] = df['dt'].dt.dayofweek.isin([5, 6])
    else:
        # Fallback if raw columns missing
        df['hour'] = 0
        df['is_weekend'] = False

    weekday_stats = df[~df['is_weekend']].groupby('hour')['traffic_volume'].agg(['mean', 'std']).reset_index()
    weekend_stats = df[df['is_weekend']].groupby('hour')['traffic_volume'].agg(['mean', 'std']).reset_index()

    fig = go.Figure()

    # Weekday curve + ribbon
    fig.add_trace(go.Scatter(
        x=weekday_stats['hour'],
        y=weekday_stats['mean'],
        mode='lines+markers',
        name='Weekday Mean (Mon-Fri)',
        line=dict(color='#1f77b4', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=list(weekday_stats['hour']) + list(weekday_stats['hour'])[::-1],
        y=list(weekday_stats['mean'] + weekday_stats['std']) + list(weekday_stats['mean'] - weekday_stats['std'])[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Weekday ±1 Std Dev'
    ))

    # Weekend curve + ribbon
    fig.add_trace(go.Scatter(
        x=weekend_stats['hour'],
        y=weekend_stats['mean'],
        mode='lines+markers',
        name='Weekend Mean (Sat-Sun)',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=list(weekend_stats['hour']) + list(weekend_stats['hour'])[::-1],
        y=list(weekend_stats['mean'] + weekend_stats['std']) + list(weekend_stats['mean'] - weekend_stats['std'])[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Weekend ±1 Std Dev'
    ))

    fig.update_layout(
        title="Dataset B: Diurnal Traffic Profiles (Weekday vs. Weekend)",
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode='linear', dtick=2),
        yaxis=dict(title="Traffic Volume (Vehicles / Hour)"),
        hovermode="x unified",
        template="plotly_white",
        height=450
    )
    return fig


def plot_actual_vs_predicted_series(
    x_vals: Any,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
    title: str = "Forecast vs Actuals",
    y_label: str = "Traffic Measurement",
    x_label: str = "Time Index"
) -> go.Figure:
    """Creates a dual-panel interactive time series chart with forecasts and residual error bars."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Observed vs Forecast Series", "Prediction Residuals (Actual - Forecast)")
    )

    # Actual observations
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_actual,
        mode='lines',
        name='Actual Ground Truth',
        line=dict(color='#2ca02c', width=2.5)
    ), row=1, col=1)

    # Model prediction
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_pred,
        mode='lines',
        name='Model Forecast',
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)

    # Persistence baseline
    if y_baseline is not None:
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_baseline,
            mode='lines',
            name='Persistence Baseline',
            line=dict(color='#d62728', width=1.5, dash='dot')
        ), row=1, col=1)

    # Residuals
    residuals = y_actual - y_pred
    fig.add_trace(go.Bar(
        x=x_vals, y=residuals,
        name='Residual Error',
        marker=dict(color=np.where(residuals >= 0, '#1f77b4', '#d62728')),
        opacity=0.65
    ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#0F172A"),
            y=0.98,
            x=0.01
        ),
        xaxis2=dict(title=x_label),
        yaxis=dict(title=y_label),
        yaxis2=dict(title="Error"),
        hovermode="x unified",
        template="plotly_white",
        height=540,
        margin=dict(l=55, r=25, t=65, b=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        )
    )
    return fig


def plot_leaderboard_bar(df: pd.DataFrame, metric: str = "mae", y_label: str = "MAE") -> go.Figure:
    """Generates benchmark comparison bar chart from validation leaderboard."""
    df_sorted = df.sort_values(metric, ascending=True).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted['model_name'],
        y=df_sorted[metric],
        text=[f"{v:.4f}" if v < 1 else f"{v:.1f}" for v in df_sorted[metric]],
        textposition='outside',
        marker=dict(color=['#2ca02c' if m == df_sorted[metric].min() else '#1f77b4' for m in df_sorted[metric]]),
        name=y_label
    ))

    fig.update_layout(
        title=f"Validation {y_label} Comparison Across Models",
        xaxis=dict(title="Candidate Model"),
        yaxis=dict(title=y_label),
        template="plotly_white",
        height=400
    )
    return fig


def plot_fundamental_diagram_curve() -> go.Figure:
    """Visualizes the Fundamental Diagram of Traffic Flow (Greenshields Model: q = k * v)."""
    # Theoretical Greenshields parameters
    v_f = 100.0   # Free-flow speed (km/h)
    k_j = 120.0   # Jam density (veh/km)
    k_crit = k_j / 2.0  # Critical density (60 veh/km)
    q_max = (v_f * k_j) / 4.0  # Capacity flow (3,000 veh/h)

    # Densities
    k_free = np.linspace(0, k_crit, 100)
    q_free = v_f * k_free * (1.0 - k_free / k_j)

    k_cong = np.linspace(k_crit, k_j, 100)
    q_cong = v_f * k_cong * (1.0 - k_cong / k_j)

    fig = go.Figure()

    # Free flow branch
    fig.add_trace(go.Scatter(
        x=k_free, y=q_free,
        mode='lines',
        name='Free-Flow Regime (Stable, High Speed)',
        line=dict(color='#2ca02c', width=3)
    ))

    # Congested branch
    fig.add_trace(go.Scatter(
        x=k_cong, y=q_cong,
        mode='lines',
        name='Congested Regime (Stop-and-Go, Low Speed)',
        line=dict(color='#d62728', width=3, dash='dash')
    ))

    # Critical point annotation
    fig.add_trace(go.Scatter(
        x=[k_crit], y=[q_max],
        mode='markers+text',
        name='Roadway Capacity (q_max)',
        text=[f"Capacity q_max ({int(q_max)} veh/h) at k_crit"],
        textposition="top center",
        marker=dict(size=12, color='#ff7f0e')
    ))

    # Annotate low volume ambiguity
    fig.add_annotation(
        x=15, y=1300,
        text="Point A: Low Flow<br>Free-flowing road<br>(e.g. 3:00 AM)",
        showarrow=True, arrowhead=2, arrowcolor='#2ca02c', ax=-40, ay=-40
    )
    fig.add_annotation(
        x=105, y=1300,
        text="Point B: Same Low Flow!<br>Total Gridlock<br>(Speed ≈ 10 km/h)",
        showarrow=True, arrowhead=2, arrowcolor='#d62728', ax=40, ay=-40
    )

    fig.update_layout(
        title="Fundamental Diagram of Traffic Flow: Why High Flow ≠ Physical Congestion",
        xaxis=dict(title="Traffic Density k (vehicles / km)"),
        yaxis=dict(title="Traffic Flow q (vehicles / hour)"),
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_forecast_timeline_steps(
    current_label: str,
    horizon_label: str,
    current_val: float,
    forecast_val: float,
    unit_str: str,
    trend_str: str,
    delta_val: float
) -> go.Figure:
    """Generates a professional 2-step chronological forecast timeline card for the Operations Center."""
    fig = go.Figure()

    # Step positions
    x_pos = [1.0, 3.0]
    y_pos = [1.0, 1.0]

    trend_color = "#EF4444" if trend_str == "Rising" else ("#10B981" if trend_str == "Falling" else "#3B82F6")
    trend_icon = "↗️" if trend_str == "Rising" else ("↘️" if trend_str == "Falling" else "➡️")

    # Connecting directional arrow
    fig.add_annotation(
        x=2.0, y=1.0,
        ax=-0.8, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor=trend_color
    )

    # Text for the arrow
    fig.add_annotation(
        x=2.0, y=1.28,
        text=f"<b>{trend_icon} {trend_str.upper()} ({delta_val:+.2f} {unit_str})</b>",
        showarrow=False,
        font=dict(size=12, color=trend_color)
    )

    # Step 1: Current Observation
    curr_fmt = f"{current_val:.3f}" if current_val < 2.0 else f"{current_val:,.0f}"
    fig.add_trace(go.Scatter(
        x=[x_pos[0]], y=[y_pos[0]],
        mode='markers+text',
        text=[f"<b>{current_label}</b><br><span style='font-size:16px;'>{curr_fmt}</span><br><span style='font-size:11px; color:#64748B;'>{unit_str}</span>"],
        textposition="bottom center",
        marker=dict(size=26, color="#0F172A", line=dict(color="#38BDF8", width=3)),
        hoverinfo='skip',
        showlegend=False
    ))

    # Step 2: Forecast Horizon
    fcst_fmt = f"{forecast_val:.3f}" if forecast_val < 2.0 else f"{forecast_val:,.0f}"
    fig.add_trace(go.Scatter(
        x=[x_pos[1]], y=[y_pos[1]],
        mode='markers+text',
        text=[f"<b>{horizon_label}</b><br><span style='font-size:16px; color:{trend_color};'>{fcst_fmt}</span><br><span style='font-size:11px; color:#64748B;'>{unit_str}</span>"],
        textposition="bottom center",
        marker=dict(size=26, color=trend_color, line=dict(color="#0F172A", width=3)),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.2, 3.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.4, 1.6]),
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white"
    )
    return fig


def classify_demand_trend(delta: float, dataset: str = "A") -> Dict[str, str]:
    """Classifies forward forecast trend with documented, scientifically tested thresholds.

    Parameters:
    -----------
    delta : float
        Forecasted change (predicted value minus current observed value).
    dataset : str
        'A' for Dataset A (normalized flow [0, 1]).
        'B' for Dataset B (hourly traffic volume in veh/h).

    Thresholds & Rationale:
    -----------------------
    Dataset A (Normalized Flow [0.0 - 1.0]):
        - delta >= +0.040           -> 'Rising' (↗️ RISING DEMAND) [>=4% design capacity change]
        - +0.010 <= delta < +0.040  -> 'Slightly Higher' (↗️ SLIGHTLY HIGHER) [1% to 4% increase]
        - -0.010 < delta < +0.010   -> 'Steady' (➡️ STEADY FLOW) [<1% minor fluctuation]
        - -0.040 < delta <= -0.010  -> 'Slightly Lower' (↘️ SLIGHTLY LOWER) [1% to 4% decrease]
        - delta <= -0.040           -> 'Easing' (↘️ EASING DEMAND) [>=4% design capacity reduction]

    Dataset B (Hourly Traffic Volume in veh/h [0 - ~7,200]):
        - delta >= +300.0           -> 'Rising' (↗️ RISING DEMAND) [Substantial commuter surge]
        - +100.0 <= delta < +300.0  -> 'Slightly Higher' (↗️ SLIGHTLY HIGHER) [Moderate increase]
        - -100.0 < delta < +100.0   -> 'Steady' (➡️ STEADY FLOW) [Steady corridor volume]
        - -300.0 < delta <= -100.0  -> 'Slightly Lower' (↘️ SLIGHTLY LOWER) [Moderate decline]
        - delta <= -300.0           -> 'Easing' (↘️ EASING DEMAND) [Substantial post-peak reduction]
    """
    ds = dataset.upper()
    if ds == "A":
        if delta >= 0.040:
            status = "Rising"
            badge_class = "trend-rising"
            badge_html = f'<span class="trend-pill trend-rising">↗️ RISING DEMAND (+{delta:.3f})</span>'
            plain_text = f"Projected flow expands noticeably by +{delta:.3f} over the 30-minute forward horizon."
        elif delta >= 0.010:
            status = "Slightly Higher"
            badge_class = "trend-slightly-higher"
            badge_html = f'<span class="trend-pill trend-slightly-higher">↗️ SLIGHTLY HIGHER (+{delta:.3f})</span>'
            plain_text = f"Projected flow increases moderately by +{delta:.3f} over the 30-minute forward horizon."
        elif delta > -0.010:
            status = "Steady"
            badge_class = "trend-steady"
            badge_html = f'<span class="trend-pill trend-steady">➡️ STEADY FLOW ({delta:+.3f})</span>'
            plain_text = f"Projected flow remains steady within ±0.010 threshold ({delta:+.3f})."
        elif delta > -0.040:
            status = "Slightly Lower"
            badge_class = "trend-slightly-lower"
            badge_html = f'<span class="trend-pill trend-slightly-lower">↘️ SLIGHTLY LOWER ({delta:.3f})</span>'
            plain_text = f"Projected flow eases moderately by {delta:.3f} over the 30-minute forward horizon."
        else:
            status = "Easing"
            badge_class = "trend-easing"
            badge_html = f'<span class="trend-pill trend-easing">↘️ EASING DEMAND ({delta:.3f})</span>'
            plain_text = f"Projected flow contracts noticeably by {delta:.3f} as demand recedes."
    else:
        if delta >= 300.0:
            status = "Rising"
            badge_class = "trend-rising"
            badge_html = f'<span class="trend-pill trend-rising">↗️ RISING DEMAND (+{delta:,.0f} veh/h)</span>'
            plain_text = f"Traffic demand is surging by +{delta:,.0f} veh/h over the next hour."
        elif delta >= 100.0:
            status = "Slightly Higher"
            badge_class = "trend-slightly-higher"
            badge_html = f'<span class="trend-pill trend-slightly-higher">↗️ SLIGHTLY HIGHER (+{delta:,.0f} veh/h)</span>'
            plain_text = f"Traffic volume is trending moderately higher (+{delta:,.0f} veh/h) over the next hour."
        elif delta > -100.0:
            status = "Steady"
            badge_class = "trend-steady"
            badge_html = f'<span class="trend-pill trend-steady">➡️ STEADY FLOW ({delta:+,.0f} veh/h)</span>'
            plain_text = f"Corridor volume remains steady within ±100 veh/h ({delta:+,.0f} veh/h change)."
        elif delta > -300.0:
            status = "Slightly Lower"
            badge_class = "trend-slightly-lower"
            badge_html = f'<span class="trend-pill trend-slightly-lower">↘️ SLIGHTLY LOWER ({delta:,.0f} veh/h)</span>'
            plain_text = f"Traffic volume is easing moderately ({delta:,.0f} veh/h) over the next hour."
        else:
            status = "Easing"
            badge_class = "trend-easing"
            badge_html = f'<span class="trend-pill trend-easing">↘️ EASING DEMAND ({delta:,.0f} veh/h)</span>'
            plain_text = f"Traffic volume is receding noticeably ({delta:,.0f} veh/h) as peak hours conclude."

    return {
        "status": status,
        "badge_html": badge_html,
        "plain_text": plain_text,
        "description": plain_text,
        "badge_class": badge_class
    }


def plot_centerpiece_forecast_dataset_a(
    history_steps: List[int],
    history_flows: List[float],
    current_step: int,
    current_flow: float,
    forecast_step: int,
    forecast_flow: float,
    actual_target: float,
    sensor_id: int,
    trend_str: str,
    retrospective_mode: bool = False
) -> go.Figure:
    """Creates the prominent centerpiece time-series visualization for Dataset A (30-min horizon)."""
    fig = go.Figure()

    if trend_str in ["Rising", "Slightly Higher"]:
        trend_color = "#DC2626" if trend_str == "Rising" else "#D97706"
    elif trend_str in ["Easing", "Slightly Lower"]:
        trend_color = "#059669" if trend_str == "Easing" else "#10B981"
    else:
        trend_color = "#2563EB"

    # 1. Historical Observed Trajectory (stops strictly at current_step)
    fig.add_trace(go.Scatter(
        x=history_steps,
        y=history_flows,
        mode='lines+markers',
        name='Observed History',
        line=dict(color='#3B82F6', width=2.5),
        marker=dict(size=6, color='#1D4ED8'),
        hovertemplate="Step %{x}<br>Observed Flow: %{y:.3f} (Normalized)<extra></extra>"
    ))

    # 2. Current Observation Highlight at step t
    fig.add_trace(go.Scatter(
        x=[current_step],
        y=[current_flow],
        mode='markers+text',
        name=f'Current Obs (Step {current_step})',
        text=[f"Current: {current_flow:.3f}"],
        textposition="top left",
        textfont=dict(size=12, color='#0F172A', weight="bold"),
        marker=dict(size=14, color='#0F172A', line=dict(color='#38BDF8', width=3)),
        hovertemplate=f"Current Step {current_step}<br>Flow: {current_flow:.3f}<extra></extra>"
    ))

    # 3. Forecast Projection Line (from current step to forecast step)
    fig.add_trace(go.Scatter(
        x=[current_step, forecast_step],
        y=[current_flow, forecast_flow],
        mode='lines',
        name='Forecast Trajectory (+30m)',
        line=dict(color=trend_color, width=2.5, dash='dash'),
        hoverinfo='skip'
    ))

    # 4. Forecast Horizon Point at t+2 steps
    fig.add_trace(go.Scatter(
        x=[forecast_step],
        y=[forecast_flow],
        mode='markers+text',
        name='Model Forecast (t+30m)',
        text=[f"Fcst: {forecast_flow:.3f}"],
        textposition="top center",
        textfont=dict(size=12, color=trend_color, weight="bold"),
        marker=dict(size=15, color=trend_color, symbol='diamond', line=dict(color='#FFFFFF', width=2)),
        hovertemplate=f"Forecast Horizon (Step {forecast_step}, +30m)<br>Predicted Flow: {forecast_flow:.3f}<extra></extra>"
    ))

    # 5. Ground Truth Target & Residual ONLY in Retrospective Evaluation Mode
    if retrospective_mode:
        fig.add_trace(go.Scatter(
            x=[forecast_step],
            y=[actual_target],
            mode='markers+text',
            name='Actual Target (Retrospective)',
            text=[f"Actual: {actual_target:.3f}"],
            textposition="bottom right",
            textfont=dict(size=11, color='#059669'),
            marker=dict(size=11, color='#059669', symbol='circle-open', line=dict(color='#059669', width=3)),
            hovertemplate=f"Actual Ground Truth (Step {forecast_step})<br>Target Flow: {actual_target:.3f}<extra></extra>"
        ))
        residual = actual_target - forecast_flow
        fig.add_trace(go.Scatter(
            x=[forecast_step, forecast_step],
            y=[forecast_flow, actual_target],
            mode='lines',
            name=f'Residual Error ({residual:+.3f})',
            line=dict(color='#94A3B8', width=2, dash='dot'),
            hoverinfo='skip'
        ))

    # Background shading for Forecast Horizon Zone
    fig.add_vrect(
        x0=current_step,
        x1=forecast_step + 0.5,
        fillcolor="rgba(241, 245, 249, 0.6)",
        layer="below",
        line_width=0,
        annotation_text="+30M HORIZON",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#64748B")
    )

    all_vals = history_flows + [current_flow, forecast_flow]
    if retrospective_mode:
        all_vals.append(actual_target)
    y_min = max(0.0, min(all_vals) - 0.08)
    y_max = min(1.0, max(all_vals) + 0.12)

    title_suffix = " <span style='font-size:12px;color:#059669;font-weight:normal;'>[Retrospective Ground Truth Revealed]</span>" if retrospective_mode else ""

    fig.update_layout(
        title=dict(
            text=f"<b>Time-Series Trajectory & 30-Minute Forward Forecast</b> — Sensor S{sensor_id:02d} (Normalized Flow [0, 1]){title_suffix}",
            font=dict(size=14, color="#0F172A"),
            y=0.97,
            x=0.01
        ),
        xaxis=dict(
            title="Historical Observation Interval (15-Minute Cadence)",
            showgrid=True,
            gridcolor="#F1F5F9",
            dtick=1
        ),
        yaxis=dict(
            title="Normalized Traffic Flow [0.0 - 1.0]",
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="#F1F5F9"
        ),
        template="plotly_white",
        height=410,
        margin=dict(l=45, r=25, t=55, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5)
        )
    )
    return fig


def plot_centerpiece_forecast_dataset_b(
    history_times: List[str],
    history_vols: List[float],
    current_time: str,
    current_vol: float,
    forecast_time: str,
    forecast_vol: float,
    actual_target: float,
    trend_str: str,
    weather_str: str,
    retrospective_mode: bool = False
) -> go.Figure:
    """Creates the prominent centerpiece time-series visualization for Dataset B (1-hour horizon)."""
    fig = go.Figure()

    if trend_str in ["Rising", "Slightly Higher"]:
        trend_color = "#DC2626" if trend_str == "Rising" else "#D97706"
    elif trend_str in ["Easing", "Slightly Lower"]:
        trend_color = "#059669" if trend_str == "Easing" else "#10B981"
    else:
        trend_color = "#2563EB"

    # 1. Historical Observed Trajectory (Stops at current_time)
    fig.add_trace(go.Scatter(
        x=history_times,
        y=history_vols,
        mode='lines+markers',
        name='Observed Hourly Volume',
        line=dict(color='#2563EB', width=2.5),
        marker=dict(size=5, color='#1D4ED8'),
        hovertemplate="%{x}<br>Volume: %{y:,.0f} veh/h<extra></extra>"
    ))

    # 2. Current Observation Highlight at time t
    fig.add_trace(go.Scatter(
        x=[current_time],
        y=[current_vol],
        mode='markers+text',
        name='Current Observation (t)',
        text=[f"Current: {current_vol:,.0f}"],
        textposition="bottom left",
        textfont=dict(size=12, color='#0F172A', weight="bold"),
        marker=dict(size=14, color='#0F172A', line=dict(color='#38BDF8', width=3)),
        hovertemplate=f"Current: {current_time}<br>Observed: {current_vol:,.0f} veh/h<br>Weather: {weather_str}<extra></extra>"
    ))

    # 3. Forecast Projection Line (From t to t+1h forward)
    fig.add_trace(go.Scatter(
        x=[current_time, forecast_time],
        y=[current_vol, forecast_vol],
        mode='lines',
        name='Forecast Projection (+1h)',
        line=dict(color=trend_color, width=2.5, dash='dash'),
        hoverinfo='skip'
    ))

    # 4. Forecast Horizon Point at t+1h
    fig.add_trace(go.Scatter(
        x=[forecast_time],
        y=[forecast_vol],
        mode='markers+text',
        name='Model Forecast (t+1h)',
        text=[f"Fcst: {forecast_vol:,.0f}"],
        textposition="top center",
        textfont=dict(size=12, color=trend_color, weight="bold"),
        marker=dict(size=15, color=trend_color, symbol='diamond', line=dict(color='#FFFFFF', width=2)),
        hovertemplate=f"Forecast Horizon ({forecast_time})<br>Predicted Volume: {forecast_vol:,.0f} veh/h<extra></extra>"
    ))

    # 5. Ground Truth & Residual ONLY in Retrospective Evaluation Mode
    if retrospective_mode:
        fig.add_trace(go.Scatter(
            x=[forecast_time],
            y=[actual_target],
            mode='markers+text',
            name='Actual Target (Retrospective)',
            text=[f"Actual: {actual_target:,.0f}"],
            textposition="bottom right",
            textfont=dict(size=11, color='#059669'),
            marker=dict(size=11, color='#059669', symbol='circle-open', line=dict(color='#059669', width=3)),
            hovertemplate=f"Actual Ground Truth ({forecast_time})<br>Target Volume: {actual_target:,.0f} veh/h<extra></extra>"
        ))
        res_vol = actual_target - forecast_vol
        fig.add_trace(go.Scatter(
            x=[forecast_time, forecast_time],
            y=[forecast_vol, actual_target],
            mode='lines',
            name=f'Residual Error ({res_vol:+,.0f} veh/h)',
            line=dict(color='#94A3B8', width=2, dash='dot'),
            hoverinfo='skip'
        ))

    all_vols = history_vols + [current_vol, forecast_vol]
    if retrospective_mode:
        all_vols.append(actual_target)
    y_min = max(0, min(all_vols) - 400)
    y_max = max(all_vols) + 600

    title_suffix = " <span style='font-size:12px;color:#059669;font-weight:normal;'>[Retrospective Ground Truth Revealed]</span>" if retrospective_mode else ""

    fig.update_layout(
        title=dict(
            text=f"<b>Time-Series Trajectory & 1-Hour Forward Forecast</b> — Interstate 94 ATR 301 (veh/h){title_suffix}",
            font=dict(size=14, color="#0F172A"),
            y=0.97,
            x=0.01
        ),
        xaxis=dict(
            title="Hourly Timeline (Unique Chronological Observations)",
            showgrid=True,
            gridcolor="#F1F5F9",
            type='category'
        ),
        yaxis=dict(
            title="Traffic Volume (Vehicles / Hour)",
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="#F1F5F9"
        ),
        template="plotly_white",
        height=410,
        margin=dict(l=55, r=25, t=55, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5)
        )
    )
    return fig


def plot_forecasting_studio_timeseries(
    x_vals: Any,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
    retrospective_mode: bool = True,
    title: str = "Forecast vs Ground Truth Observations",
    y_label: str = "Traffic Measurement",
    x_label: str = "Time Interval",
    horizon_tag: str = "+30 min",
    unit_str: str = ""
) -> go.Figure:
    """Creates the primary interactive time-series chart for Forecasting Studio.

    In Retrospective Mode:
        - Top panel: Ground truth (green), Model forecast (blue), Persistence baseline (slate).
        - Bottom panel: Step-by-step prediction residuals (Forecast - Actual), color-coded by
          Overprediction (#EA580C Coral) vs Underprediction (#6366F1 Indigo).

    In Prospective Mode:
        - Single panel: Historical observations up to time t, current observation marker,
          and forward projection to t+h with ground truth strictly hidden.
    """
    x_list = list(x_vals)
    n_pts = len(x_list)

    if retrospective_mode:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=[0.68, 0.32],
            subplot_titles=(
                f"<b>{title}</b>",
                "<b>Step-by-Step Residual Error</b> [Forecast − Actual Ground Truth]"
            )
        )

        # 1. Ground Truth (Green)
        fig.add_trace(go.Scatter(
            x=x_list, y=y_actual,
            mode='lines+markers',
            name='Actual Ground Truth (y)',
            line=dict(color='#059669', width=2.5),
            marker=dict(size=5, color='#047857'),
            hovertemplate="Step %{x}<br>Actual: %{y:.3f} " + unit_str + "<extra></extra>"
        ), row=1, col=1)

        # 2. Model Forecast (Blue)
        fig.add_trace(go.Scatter(
            x=x_list, y=y_pred,
            mode='lines+markers',
            name=f'Model Forecast ({horizon_tag})',
            line=dict(color='#2563EB', width=2.2),
            marker=dict(size=5, symbol='diamond', color='#1D4ED8'),
            hovertemplate="Step %{x}<br>Forecast: %{y:.3f} " + unit_str + "<extra></extra>"
        ), row=1, col=1)

        # 3. Persistence Baseline (Slate)
        if y_baseline is not None:
            fig.add_trace(go.Scatter(
                x=x_list, y=y_baseline,
                mode='lines',
                name='Persistence Baseline (t)',
                line=dict(color='#64748B', width=1.5, dash='dot'),
                hovertemplate="Step %{x}<br>Baseline: %{y:.3f} " + unit_str + "<extra></extra>"
            ), row=1, col=1)

        # 4. Residual Error Bars (Forecast - Actual)
        residuals = y_pred - y_actual
        bar_colors = [
            '#EA580C' if r > 0 else ('#6366F1' if r < 0 else '#94A3B8')
            for r in residuals
        ]
        hover_residuals = [
            f"Step: {x_list[i]}<br>Fcst: {y_pred[i]:.3f}<br>Actual: {y_actual[i]:.3f}<br>"
            f"<b>{'⚠️ Overprediction (+)' if residuals[i] > 0 else ('📉 Underprediction (-)' if residuals[i] < 0 else 'Exact Match')}</b>: {residuals[i]:+.3f} {unit_str}"
            for i in range(n_pts)
        ]

        fig.add_trace(go.Bar(
            x=x_list, y=residuals,
            name='Residual Error (Forecast − Actual)',
            marker=dict(color=bar_colors),
            hoverinfo='text',
            hovertext=hover_residuals,
            showlegend=False
        ), row=2, col=1)

        # Zero reference line on residual plot
        fig.add_hline(
            y=0, line_dash="dash", line_color="#94A3B8", line_width=1.2,
            row=2, col=1
        )

        fig.update_layout(
            template="plotly_white",
            height=540,
            hovermode="x unified",
            margin=dict(l=55, r=25, t=55, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(size=11)
            )
        )
        fig.update_yaxes(title_text=y_label, row=1, col=1, gridcolor="#F1F5F9")
        fig.update_yaxes(title_text="Error (Residual)", row=2, col=1, gridcolor="#F1F5F9")
        fig.update_xaxes(title_text=x_label, row=2, col=1, gridcolor="#F1F5F9")

    else:
        # Prospective Mode: Future ground truth obscured
        fig = go.Figure()

        # Historical observed values up to time t
        obs_vals = y_baseline if y_baseline is not None else y_actual
        fig.add_trace(go.Scatter(
            x=x_list, y=obs_vals,
            mode='lines+markers',
            name='Observed Telemetry (History up to t)',
            line=dict(color='#2563EB', width=2.5),
            marker=dict(size=5, color='#1D4ED8'),
            hovertemplate="Observed: %{y:.3f} " + unit_str + "<extra></extra>"
        ))

        # Current observation at final step
        latest_x = x_list[-1]
        latest_obs = obs_vals[-1]
        latest_pred = y_pred[-1]

        fig.add_trace(go.Scatter(
            x=[latest_x], y=[latest_obs],
            mode='markers+text',
            name='Current Observation (t)',
            text=[f"Observed: {latest_obs:.3f}"],
            textposition="top left",
            textfont=dict(size=11, color='#0F172A', weight="bold"),
            marker=dict(size=13, color='#0F172A', line=dict(color='#38BDF8', width=2.5)),
            hovertemplate=f"Current Step {latest_x}<br>Observed: {latest_obs:.3f} {unit_str}<extra></extra>"
        ))

        # Projected horizon point
        fcst_x_label = f"{latest_x} ({horizon_tag})"
        fig.add_trace(go.Scatter(
            x=[latest_x, fcst_x_label],
            y=[latest_obs, latest_pred],
            mode='lines+markers',
            name=f'Model Forecast ({horizon_tag})',
            line=dict(color='#EA580C', width=2.5, dash='dash'),
            marker=dict(size=[6, 14], color=['#EA580C', '#EA580C'], symbol=['circle', 'diamond']),
            text=["", f"Fcst: {latest_pred:.3f}"],
            textposition="top right",
            textfont=dict(size=11, color='#EA580C', weight="bold"),
            hovertemplate=f"Horizon: {fcst_x_label}<br>Projected: {latest_pred:.3f} {unit_str}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b> <span style='font-size:12px; color:#D97706; font-weight:normal;'>[Prospective Live Mode — Future Ground Truth Obscured]</span>",
                font=dict(size=14, color="#0F172A"),
                y=0.97,
                x=0.01
            ),
            xaxis=dict(title=x_label, gridcolor="#F1F5F9"),
            yaxis=dict(title=y_label, gridcolor="#F1F5F9"),
            template="plotly_white",
            height=430,
            hovermode="closest",
            margin=dict(l=55, r=25, t=55, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(size=11)
            )
        )

    return fig


def plot_residual_distribution_histogram(
    residuals: np.ndarray,
    unit_str: str = "",
    title: str = "Residual Error Distribution (Forecast − Actual)"
) -> go.Figure:
    """Generates an error distribution chart separating overprediction from underprediction.

    Overpredictions (res > 0): Model forecast higher than actual ground truth (#EA580C Coral).
    Underpredictions (res < 0): Model forecast lower than actual ground truth (#6366F1 Indigo).
    """
    res = np.array(residuals, dtype=float)
    n_total = len(res)
    if n_total == 0:
        return go.Figure()

    mean_bias = float(np.mean(res))
    mae = float(np.mean(np.abs(res)))
    n_over = int(np.sum(res > 0))
    n_under = int(np.sum(res < 0))
    n_exact = int(np.sum(res == 0))

    fig = go.Figure()

    # Underpredictions (< 0)
    under_vals = res[res < 0]
    if len(under_vals) > 0:
        fig.add_trace(go.Histogram(
            x=under_vals,
            name=f"Underprediction ({n_under}/{n_total}, {n_under/n_total*100:.1f}%)",
            marker=dict(color='#6366F1', line=dict(color='#4338CA', width=1)),
            opacity=0.85
        ))

    # Overpredictions (> 0)
    over_vals = res[res > 0]
    if len(over_vals) > 0:
        fig.add_trace(go.Histogram(
            x=over_vals,
            name=f"Overprediction ({n_over}/{n_total}, {n_over/n_total*100:.1f}%)",
            marker=dict(color='#EA580C', line=dict(color='#C2410C', width=1)),
            opacity=0.85
        ))

    # Vertical reference line at Zero Error
    fig.add_vline(
        x=0, line_dash="solid", line_color="#0F172A", line_width=2,
        annotation_text="Zero Error Line", annotation_position="top left",
        annotation_font=dict(size=10, color="#0F172A")
    )

    # Vertical reference line at Mean Signed Error (Bias)
    bias_color = "#EA580C" if mean_bias > 0 else "#6366F1"
    fig.add_vline(
        x=mean_bias, line_dash="dash", line_color=bias_color, line_width=2,
        annotation_text=f"Mean Bias ({mean_bias:+.3f})", annotation_position="top right",
        annotation_font=dict(size=10, color=bias_color)
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color="#0F172A"),
            y=0.96,
            x=0.01
        ),
        xaxis=dict(
            title=f"Residual Error (Forecast − Actual) [{unit_str}]",
            gridcolor="#F1F5F9",
            zeroline=False
        ),
        yaxis=dict(
            title="Frequency Count (Observations)",
            gridcolor="#F1F5F9"
        ),
        barmode='overlay',
        template="plotly_white",
        height=320,
        margin=dict(l=45, r=25, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10)
        )
    )
    return fig




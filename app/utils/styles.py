"""Enterprise UI Design System & Styling Tokens for Smart Traffic TOC.

Provides shared CSS tokens, badge helpers, and executive metric card containers
optimized for clean visual hierarchy, low cognitive load, and high data density.
"""

def get_shared_css() -> str:
    """Returns the core enterprise design system CSS."""
    return """
<style>
    /* Global Reset & Typography */
    .stApp {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0F172A;
    }

    /* Streamlit Chrome Optimization */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1480px !important;
    }

    /* Sidebar Navigation Enhancement */
    [data-testid="stSidebarNav"] {
        padding-top: 0.5rem;
    }
    [data-testid="stSidebarNav"] ul {
        gap: 0.25rem;
    }
    [data-testid="stSidebarNav"] li a {
        padding: 0.5rem 0.85rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #334155 !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebarNav"] li a:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebarNav"] li a[aria-current="page"] {
        background: linear-gradient(90deg, #EFF6FF 0%, #DBEAFE 100%) !important;
        color: #1D4ED8 !important;
        border-left: 4px solid #2563EB !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebarNav"] li:first-child a {
        overflow: visible !important;
    }
    [data-testid="stSidebarNav"] li:first-child a span {
        visibility: hidden !important;
        position: relative !important;
        display: inline-block !important;
        min-width: 165px !important;
        overflow: visible !important;
    }
    [data-testid="stSidebarNav"] li:first-child a span::after {
        visibility: visible !important;
        position: absolute !important;
        top: 0;
        left: 0;
        content: "Operations Overview" !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }

    /* Enterprise Header Console Bar */
    .toc-header-bar {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #38BDF8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .toc-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .toc-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.32rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-demo {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .badge-online {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }

    /* Hero Telemetry Stat Cards */
    .hero-stat-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 9px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .hero-stat-card-forecast {
        background: linear-gradient(180deg, #FFFFFF 0%, #EFF6FF 100%);
        border: 2px solid #2563EB !important;
        border-radius: 9px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.12);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .hero-stat-tag {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 0.35rem;
    }
    .hero-stat-val {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.12;
        font-variant-numeric: tabular-nums;
        color: #0F172A;
    }
    .hero-stat-sub {
        font-size: 0.86rem;
        color: #475569;
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* Granular Trend Badges */
    .trend-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        padding: 0.26rem 0.62rem;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.84rem;
    }
    .trend-rising {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .trend-slightly-higher {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .trend-steady {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    .trend-slightly-lower {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .trend-easing {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }

    /* Tactical Decision Box */
    .tactical-decision-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 0.8rem;
    }
    .tactical-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.22rem 0.6rem;
        border-radius: 4px;
        background-color: #F1F5F9;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    .tactical-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .tactical-interp {
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 0.65rem;
    }
    .tactical-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.6rem;
    }
    .action-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.32rem 0.72rem;
        font-size: 0.84rem;
        font-weight: 700;
        color: #1E293B;
    }

    /* Action & Advisory Box Variants */
    .action-box {
        border-radius: 9px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .action-warning {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        color: #92400E;
    }
    .action-info {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        color: #1E40AF;
    }
    .action-normal {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        color: #065F46;
    }
    .action-badge-tag {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.22rem 0.58rem;
        border-radius: 4px;
        background-color: rgba(0, 0, 0, 0.06);
        color: #1E293B;
        margin-bottom: 0.5rem;
    }

    /* Visual Centerpiece Container */
    .viz-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .viz-panel-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Compact Footnote Callout */
    .sys-scope-note {
        font-size: 0.76rem;
        color: #64748B;
        border-top: 1px dashed #CBD5E1;
        padding-top: 0.45rem;
        margin-top: 0.45rem;
        line-height: 1.4;
    }

    /* Analytical Card & Explainer Support */
    .toc-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .toc-card-header {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .toc-card-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
    }
    .toc-card-sub {
        font-size: 0.8rem;
        color: #64748B;
    }
    .explainer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .disclaimer-banner {
        background-color: #F1F5F9;
        border-left: 4px solid #64748B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #334155;
        margin-bottom: 1rem;
    }
</style>
"""

def render_header_bar(title: str, subtitle: str) -> str:
    """Renders the standard enterprise header bar with certified status badges."""
    return f"""
<div class="toc-header-bar">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.6rem;">
        <div>
            <div class="toc-title">{title}</div>
            <div class="toc-subtitle">{subtitle}</div>
        </div>
        <div>
            <span class="status-badge badge-demo">● HISTORICAL DEMO MODE</span>
            <span class="status-badge badge-online">● TRAINED MODELS LOADED</span>
        </div>
    </div>
</div>
"""

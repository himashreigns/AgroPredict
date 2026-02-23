import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

// ── Plotly dark layout base ──────────────────────────────────────────────────
const DARK = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.03)',
    font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 11 },
    margin: { l: 50, r: 20, t: 36, b: 40 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.08)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.08)' },
    legend: { bgcolor: 'rgba(13,21,38,0.8)', bordercolor: 'rgba(99,102,241,0.2)', borderwidth: 1, font: { size: 11 } },
    colorway: ['#6366f1', '#10b981', '#f59e0b', '#f87171', '#06b6d4', '#a78bfa', '#34d399', '#fbbf24', '#fb923c', '#e879f9'],
}
const PLOT_CFG = { displayModeBar: false, responsive: true }
const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function useApi(url, deps = []) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        setLoading(true); setError(null)
        axios.get(url)
            .then(r => setData(r.data))
            .catch(e => setError(e.message))
            .finally(() => setLoading(false))
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, deps)

    return { data, loading, error }
}

function LoadingCard() {
    return <div className="loading"><div className="spinner" /><span>Loading…</span></div>
}
function ErrorCard({ msg }) {
    return <div className="error-box">⚠️ {msg}</div>
}

// ── KPI Cards ────────────────────────────────────────────────────────────────
function KpiCards() {
    const { data, loading, error } = useApi('/api/dashboard/kpis')
    if (loading) return <div className="grid-4"><LoadingCard /></div>
    if (error) return <div className="grid-4"><ErrorCard msg={error} /></div>

    const dp = data.avg_price_delta_pct
    const cards = [
        { label: 'Current Avg Market Price', value: `Rs ${data.avg_price?.toLocaleString()}`, sub: `${dp >= 0 ? '▲' : '▼'} ${Math.abs(dp)}% vs last week`, cls: dp >= 0 ? 'kpi-delta-up' : 'kpi-delta-down' },
        { label: 'Most Expensive Commodity', value: data.most_expensive, sub: 'Highest average price overall', cls: '' },
        { label: 'Most Price-Volatile', value: data.most_volatile, sub: 'Highest standard deviation', cls: '' },
        { label: 'Total Dataset Records', value: data.total_records?.toLocaleString(), sub: '7 years · 5 markets · 20 commodities', cls: '' },
    ]
    return (
        <div className="grid-4">
            {cards.map(c => (
                <div className="kpi-card" key={c.label}>
                    <div className="kpi-label">{c.label}</div>
                    <div className="kpi-value">{c.value}</div>
                    <div className={`kpi-sub ${c.cls}`}>{c.sub}</div>
                </div>
            ))}
        </div>
    )
}

// ── Price Trends Chart ────────────────────────────────────────────────────────
function PriceTrends({ filters }) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const { meta } = filters

    useEffect(() => {
        setLoading(true)
        const params = new URLSearchParams({
            commodities: filters.commodities.join(','),
            year_from: filters.yearFrom,
            year_to: filters.yearTo,
        })
        axios.get(`/api/dashboard/trends?${params}`)
            .then(r => { setData(r.data); setError(null) })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false))
    }, [filters.commodities, filters.yearFrom, filters.yearTo])

    if (loading) return <LoadingCard />
    if (error) return <ErrorCard msg={error} />
    if (!data?.length) return <div className="empty-state">No data for selection.</div>

    const commodities = [...new Set(data.map(d => d.commodity))]
    const traces = commodities.map(c => {
        const rows = data.filter(d => d.commodity === c).sort((a, b) => a.date.localeCompare(b.date))
        return { x: rows.map(r => r.date), y: rows.map(r => r.price_lkr), type: 'scatter', mode: 'lines', name: c, line: { width: 2 } }
    })

    // Add crisis annotation
    const shapes = [
        {
            type: 'rect', x0: '2022-03-01', x1: '2022-09-30', y0: 0, y1: 1,
            xref: 'x', yref: 'paper', fillcolor: 'rgba(248,113,113,0.07)',
            line: { color: 'rgba(248,113,113,0.2)', width: 1 },
        },
        {
            type: 'rect', x0: '2025-01-01', x1: '2025-12-31', y0: 0, y1: 1,
            xref: 'x', yref: 'paper', fillcolor: 'rgba(16,185,129,0.05)',
            line: { color: 'rgba(16,185,129,0.15)', width: 1 },
        },
    ]
    const annotations = [
        {
            x: '2022-06-01', y: 0.97, xref: 'x', yref: 'paper',
            text: '⚠️ 2022 Economic Crisis', showarrow: false,
            font: { color: '#fca5a5', size: 10 }, bgcolor: 'rgba(248,113,113,0.15)',
        },
        {
            x: '2025-07-01', y: 0.97, xref: 'x', yref: 'paper',
            text: '📗 2025 Data', showarrow: false,
            font: { color: '#6ee7b7', size: 10 }, bgcolor: 'rgba(16,185,129,0.15)',
        },
    ]

    return (
        <Plot
            data={traces}
            layout={{
                ...DARK,
                height: 340,
                margin: { l: 55, r: 20, t: 30, b: 50 },
                shapes, annotations,
                showlegend: true,
                legend: {
                    ...DARK.legend,
                    orientation: 'h',
                    x: 0, y: -0.22,
                    xanchor: 'left', yanchor: 'top',
                },
                yaxis: { ...DARK.yaxis, title: { text: 'Price (LKR / kg)', font: { size: 11 } } },
                xaxis: { ...DARK.xaxis, title: '' },
            }}
            config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler
        />
    )
}

// ── Market Comparison ─────────────────────────────────────────────────────────
function MarketComparison() {
    const { data, loading, error } = useApi('/api/dashboard/market-comparison')
    if (loading) return <LoadingCard />
    if (error) return <ErrorCard msg={error} />

    const wholesale = data.filter(d => d.price_type === 'Wholesale').sort((a, b) => a.market.localeCompare(b.market))
    const retail = data.filter(d => d.price_type === 'Retail').sort((a, b) => a.market.localeCompare(b.market))

    return (
        <Plot
            data={[
                { x: wholesale.map(d => d.market), y: wholesale.map(d => d.avg_price), name: 'Wholesale Price', type: 'bar', marker: { color: '#6366f1' } },
                { x: retail.map(d => d.market), y: retail.map(d => d.avg_price), name: 'Retail Price', type: 'bar', marker: { color: '#f59e0b' } },
            ]}
            layout={{
                ...DARK, height: 300, barmode: 'group',
                yaxis: { ...DARK.yaxis, title: { text: 'Avg Price (LKR/kg)', font: { size: 11 } } },
            }}
            config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler
        />
    )
}

// ── Seasonal Heatmap ──────────────────────────────────────────────────────────
function SeasonalHeatmap({ selCommodities }) {
    const { data, loading, error } = useApi('/api/dashboard/seasonal-heatmap')
    if (loading) return <LoadingCard />
    if (error) return <ErrorCard msg={error} />

    const commodities = selCommodities.length
        ? selCommodities
        : [...new Set(data.map(d => d.commodity))].sort()
    const months = MONTH_NAMES

    // Build z matrix
    const z = commodities.map(c => {
        const row = data.filter(d => d.commodity === c)
        return months.map((_, mi) => {
            const match = row.find(r => r.month === mi + 1)
            return match ? match.avg_price : null
        })
    })
    const zText = z.map(row => row.map(v => v ? `Rs ${v.toFixed(0)}` : ''))

    return (
        <Plot
            data={[{
                type: 'heatmap', z, x: months, y: commodities,
                colorscale: 'RdYlGn', reversescale: true,
                text: zText, texttemplate: '%{text}', textfont: { size: 9, color: '#fff' },
                colorbar: { title: { text: 'LKR/kg', font: { color: '#94a3b8' } }, tickfont: { color: '#94a3b8' }, len: 0.9 },
                hovertemplate: '<b>%{y}</b><br>%{x}: Rs %{z:.0f}/kg<extra></extra>',
            }]}
            layout={{
                ...DARK, height: 420, margin: { ...DARK.margin, l: 110 },
                xaxis: { ...DARK.xaxis, title: '' }, yaxis: { ...DARK.yaxis, title: '' },
            }}
            config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler
        />
    )
}


function CategoryDistribution() {
    const { data, loading, error } = useApi('/api/dashboard/category-distribution')
    if (loading) return <LoadingCard />
    if (error) return <ErrorCard msg={error} />

    const catColors = { Vegetable: '#10b981', Fruit: '#f59e0b' }
    const traces = data.map(d => ({
        type: 'box',
        name: d.category,
        q1: [d.q25], median: [d.median], q3: [d.q75],
        lowerfence: [d.q10], upperfence: [d.q90],
        mean: [d.mean],
        x: [d.category],
        marker: { color: catColors[d.category] || '#6366f1' },
        boxmean: 'sd',
        hovertemplate: `<b>${d.category}</b><br>Median: Rs %{median:.0f}<br>Mean: Rs %{mean:.0f}<extra></extra>`,
    }))

    return (
        <Plot
            data={traces}
            layout={{
                ...DARK, height: 300, showlegend: false,
                yaxis: { ...DARK.yaxis, title: { text: 'Price (LKR/kg)', font: { size: 11 } } },
            }}
            config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler
        />
    )
}

// ── Dashboard Page ────────────────────────────────────────────────────────────
export default function Dashboard() {
    const { data: meta } = useApi('/api/metadata')

    const allCommodities = meta?.commodities || []
    const [selCommodities, setSelCommodities] = useState(['Tomato', 'Carrot', 'Beans', 'Green Chilli'])
    const [yearFrom, setYearFrom] = useState(2019)
    const [yearTo, setYearTo] = useState(2025)

    const filters = { commodities: selCommodities, yearFrom, yearTo, meta }

    return (
        <div className="page-wrapper">
            {/* Hero */}
            <div className="hero-banner">
                <h1>📊 Sri Lanka Produce Price Dashboard</h1>
                <p>Historical weekly prices for 20 vegetables & fruits · 5 major markets · 2019–2024</p>
                <span className="hero-tag">🌿 Data from DOA · CBSL · HARTI · Statistics Dept.</span>
            </div>

            {/* KPIs */}
            <KpiCards />

            {/* Filters */}
            <div className="filter-row">
                <div className="filter-field" style={{ flex: '1 1 auto' }}>
                    <label>Commodities (Trend Chart &amp; Heatmap)</label>
                    <div style={{
                        display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6,
                        maxHeight: 90, overflowY: 'auto', padding: '4px 0',
                    }}>
                        {allCommodities.map(c => {
                            const active = selCommodities.includes(c)
                            return (
                                <button key={c}
                                    onClick={() => {
                                        if (active) {
                                            if (selCommodities.length > 1) setSelCommodities(s => s.filter(x => x !== c))
                                        } else {
                                            setSelCommodities(s => [...s, c])
                                        }
                                    }}
                                    style={{
                                        padding: '3px 10px',
                                        borderRadius: 20,
                                        fontSize: '0.73rem',
                                        fontWeight: 500,
                                        cursor: 'pointer',
                                        border: active ? '1.5px solid #6366f1' : '1.5px solid rgba(99,102,241,0.25)',
                                        background: active ? 'rgba(99,102,241,0.18)' : 'rgba(255,255,255,0.04)',
                                        color: active ? '#a5b4fc' : '#64748b',
                                        transition: 'all 0.15s',
                                    }}
                                >{c}</button>
                            )
                        })}
                    </div>
                </div>
                <div className="filter-field" style={{ minWidth: 110 }}>
                    <label>Year From</label>
                    <select value={yearFrom} onChange={e => setYearFrom(+e.target.value)}>
                        {[2019, 2020, 2021, 2022, 2023, 2024, 2025].map(y => <option key={y} value={y}>{y}</option>)}
                    </select>
                </div>
                <div className="filter-field" style={{ minWidth: 110 }}>
                    <label>Year To</label>
                    <select value={yearTo} onChange={e => setYearTo(+e.target.value)}>
                        {[2019, 2020, 2021, 2022, 2023, 2024, 2025].map(y => <option key={y} value={y}>{y}</option>)}
                    </select>
                </div>
            </div>

            {/* Trend Chart (full width) */}
            <div className="grid-1">
                <div className="chart-card">
                    <h3>📈 Price Trends Over Time</h3>
                    <p>Weekly average prices per commodity — select commodities & date range above. Red zone = 2022 economic crisis.</p>
                    <PriceTrends filters={filters} />
                </div>
            </div>

            {/* Market Comparison (full width) */}
            <div className="grid-1">
                <div className="chart-card">
                    <h3>🏪 Average Price by Market & Type</h3>
                    <p>Dambulla & Manning Market = wholesale hubs (lower prices); Narahenpita & Colombo Local = retail (higher prices)</p>
                    <MarketComparison />
                </div>
            </div>

            {/* Heatmap (full width) */}
            <div className="grid-1">
                <div className="chart-card">
                    <h3>🗓️ Seasonal Price Heatmap — Commodity × Month</h3>
                    <p>Green = harvest season (low prices / high supply), Red = scarcity (off-season / high demand). Based on Maha (Oct–Jan) & Yala (May–Aug) cultivation cycles.</p>
                    <SeasonalHeatmap selCommodities={selCommodities} />
                </div>
            </div>

            {/* Price Distribution */}
            <div className="grid-1">
                <div className="chart-card">
                    <h3>📦 Price Distribution — Vegetables vs Fruits</h3>
                    <p>Box plot showing median, interquartile range (Q1–Q3) and 10th–90th percentile spread for each category</p>
                    <CategoryDistribution />
                </div>
            </div>
        </div>
    )
}

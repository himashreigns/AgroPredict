import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

const DARK = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.03)',
    font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 11 },
    margin: { l: 55, r: 20, t: 36, b: 40 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    legend: { bgcolor: 'rgba(13,21,38,0.8)', bordercolor: 'rgba(99,102,241,0.2)', borderwidth: 1 },
    colorway: ['#6366f1', '#10b981', '#f59e0b', '#f87171', '#06b6d4'],
}
const PLOT_CFG = { displayModeBar: false, responsive: true }
const TODAY = new Date().toISOString().split('T')[0]  // dynamic today

function isFuture(dateStr) { return dateStr > TODAY }

// ── Sub-components ─────────────────────────────────────────────────────────────
function Loading() { return <div className="loading"><div className="spinner" /><span>Computing…</span></div> }
function Err({ msg }) { return <div className="error-box">⚠️ {msg}</div> }

// ── Feature Importance ────────────────────────────────────────────────────────
function FeatureImportance() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [err, setErr] = useState(null)
    useEffect(() => {
        axios.get('/api/explain/feature-importance')
            .then(r => setData(r.data)).catch(e => setErr(e.message)).finally(() => setLoading(false))
    }, [])
    if (loading) return <Loading />
    if (err) return <Err msg={err} />
    const top = data.slice(0, 12).reverse()
    return (
        <Plot data={[{
            type: 'bar', orientation: 'h',
            x: top.map(d => d.importance), y: top.map(d => d.label),
            marker: { color: '#6366f1', opacity: 0.85 },
            text: top.map(d => d.importance.toFixed(0)), textposition: 'outside',
            textfont: { color: '#e2e8f0', size: 10 },
            hovertemplate: '<b>%{y}</b><br>Score: %{x:.0f}<extra></extra>',
        }]}
            layout={{
                ...DARK, height: 400, margin: { ...DARK.margin, l: 230 },
                xaxis: { ...DARK.xaxis, title: { text: 'Feature Importance (gain)', font: { size: 11 } } },
                yaxis: { ...DARK.yaxis, automargin: true },
            }}
            config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler />
    )
}

// ── PDP Chart ────────────────────────────────────────────────────────────────
function Pdp({ feature }) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [err, setErr] = useState(null)
    useEffect(() => {
        setLoading(true)
        axios.get(`/api/explain/pdp?feature=${feature}`)
            .then(r => setData(r.data)).catch(e => setErr(e.message)).finally(() => setLoading(false))
    }, [feature])
    if (loading) return <Loading />
    if (err) return <Err msg={err} />
    const pts = data.points
    const upper = pts.map(p => p.mean_price + p.std_price)
    const lower = pts.map(p => p.mean_price - p.std_price)
    return (
        <Plot data={[
            {
                x: pts.map(p => p.x), y: pts.map(p => p.mean_price), type: 'scatter', mode: 'lines',
                name: 'Mean Price', line: { color: '#6366f1', width: 2.5 },
                hovertemplate: '<b>%{x}</b><br>Predicted: Rs %{y:.0f}<extra></extra>'
            },
            {
                x: [...pts.map(p => p.x), ...pts.map(p => p.x).reverse()],
                y: [...upper, ...lower.reverse()],
                fill: 'toself', fillcolor: 'rgba(99,102,241,0.1)',
                line: { color: 'transparent' }, showlegend: false, hoverinfo: 'skip'
            },
        ]}
            layout={{
                ...DARK, height: 250, showlegend: false,
                xaxis: { ...DARK.xaxis, title: { text: data.label, font: { size: 11 } } },
                yaxis: { ...DARK.yaxis, title: { text: 'Predicted Price (LKR/kg)', font: { size: 11 } } },
            }}
            config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler />
    )
}

// ── SHAP Waterfall ────────────────────────────────────────────────────────────
function ShapWaterfall({ contributions, baseValue, prediction }) {
    if (!contributions?.length) return null
    const top = contributions.slice(0, 10)
    const rest = contributions.slice(10).reduce((s, c) => s + c.shap_value, 0)
    const items = rest !== 0 ? [...top, { label: 'All Other Features', shap_value: rest, feature_value: '' }] : top
    const colors = items.map(c => c.shap_value >= 0 ? 'rgba(248,113,113,0.85)' : 'rgba(16,185,129,0.85)')
    return (
        <Plot data={[{
            type: 'bar', orientation: 'h',
            x: items.map(c => c.shap_value), y: items.map(c => c.label), base: baseValue,
            marker: { color: colors, line: { color: 'rgba(0,0,0,0.2)', width: 0.5 } },
            text: items.map(c => c.shap_value >= 0 ? `+Rs ${c.shap_value.toFixed(1)}` : `Rs ${c.shap_value.toFixed(1)}`),
            textposition: 'outside', textfont: { color: '#e2e8f0', size: 10 },
            hovertemplate: '<b>%{y}</b><br>Contribution: %{x:+.2f} Rs<extra></extra>',
        }]}
            layout={{
                ...DARK, height: 400, margin: { ...DARK.margin, l: 230 },
                shapes: [
                    {
                        type: 'line', x0: baseValue, x1: baseValue, y0: -0.5, y1: items.length - 0.5,
                        line: { color: 'rgba(99,102,241,0.7)', dash: 'dot', width: 1.5 }, xref: 'x', yref: 'y'
                    },
                    {
                        type: 'line', x0: prediction, x1: prediction, y0: -0.5, y1: items.length - 0.5,
                        line: { color: '#34d399', dash: 'dot', width: 1.5 }, xref: 'x', yref: 'y'
                    },
                ],
                annotations: [
                    {
                        x: baseValue, y: items.length - 0.1, xref: 'x', yref: 'y', text: `Base: Rs ${baseValue}`,
                        showarrow: false, font: { color: '#a5b4fc', size: 10 }, yanchor: 'bottom'
                    },
                    {
                        x: prediction, y: items.length - 0.1, xref: 'x', yref: 'y', text: `Predicted: Rs ${prediction}`,
                        showarrow: false, font: { color: '#34d399', size: 10 }, yanchor: 'bottom'
                    },
                ],
                xaxis: { ...DARK.xaxis, title: { text: 'SHAP Contribution to Predicted Price (Rs/kg)', font: { size: 11 } } },
                yaxis: { ...DARK.yaxis, automargin: true },
            }}
            config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler />
    )
}

// ── Future Forecast Chart ─────────────────────────────────────────────────────
function ForecastChart({ commodity, market, priceType, weeksAhead, inflationIndex }) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(false)
    const [err, setErr] = useState(null)

    const fetchForecast = useCallback(() => {
        setLoading(true); setErr(null)
        const p = new URLSearchParams({
            commodity, market, price_type: priceType,
            weeks: weeksAhead,
            inflation_index: inflationIndex,
        })
        axios.get(`/api/forecast?${p}`)
            .then(r => setData(r.data))
            .catch(e => setErr(e.response?.data?.detail || e.message))
            .finally(() => setLoading(false))
    }, [commodity, market, priceType, weeksAhead, inflationIndex])

    useEffect(() => { fetchForecast() }, [fetchForecast])

    if (loading) return <Loading />
    if (err) return <Err msg={err} />
    if (!data) return null

    const fc = data.forecasts
    const dates = fc.map(r => r.date)
    const prices = fc.map(r => r.predicted_price)
    const upper = fc.map(r => r.upper_bound)
    const lower = fc.map(r => r.lower_bound)

    // Festival markers
    const festAnnotations = fc
        .filter(r => r.is_festive)
        .map(r => ({
            x: r.date, y: r.predicted_price,
            text: '🎉', showarrow: false, font: { size: 16 },
        }))

    const seasonColors = { Maha: 'rgba(99,102,241,0.15)', Yala: 'rgba(16,185,129,0.15)', 'Off-Season': 'rgba(0,0,0,0)' }
    const seasonShapes = []
    let cur = fc[0]?.season
    let start = fc[0]?.date
    for (let i = 1; i <= fc.length; i++) {
        const s = fc[i]?.season
        if (s !== cur || i === fc.length) {
            seasonShapes.push({
                type: 'rect', x0: start, x1: fc[i - 1]?.date || dates[dates.length - 1],
                y0: 0, y1: 1, xref: 'x', yref: 'paper',
                fillcolor: seasonColors[cur] || 'rgba(0,0,0,0)', line: { width: 0 },
            })
            cur = s; start = fc[i]?.date
        }
    }

    return (
        <>
            {/* KPI Summary */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, marginBottom: 16 }}>
                <div className="metric-badge">
                    <div className="m-val">Rs {data.avg_forecast}</div>
                    <div className="m-lbl">Avg Forecast</div>
                    <div className="m-desc">{weeksAhead}-week average</div>
                </div>
                <div className="metric-badge">
                    <div className="m-val" style={{ color: '#34d399' }}>Rs {data.min_week.price}</div>
                    <div className="m-lbl">Lowest Price</div>
                    <div className="m-desc">{data.min_week.date}</div>
                </div>
                <div className="metric-badge">
                    <div className="m-val" style={{ color: '#f87171' }}>Rs {data.max_week.price}</div>
                    <div className="m-lbl">Highest Price</div>
                    <div className="m-desc">{data.max_week.date}</div>
                </div>
            </div>

            <Plot data={[
                {
                    x: [...dates, ...[...dates].reverse()],
                    y: [...upper, ...lower.reverse()],
                    fill: 'toself', fillcolor: 'rgba(99,102,241,0.1)',
                    line: { color: 'transparent' }, name: '±12% Range', showlegend: true,
                    hoverinfo: 'skip'
                },
                {
                    x: dates, y: prices, type: 'scatter', mode: 'lines+markers', name: 'Predicted Price',
                    line: { color: '#6366f1', width: 2.5 }, marker: { size: 5, color: '#6366f1' },
                    hovertemplate: '<b>%{x}</b><br>Rs %{y:.2f}/kg<extra></extra>'
                },
                {
                    x: dates, y: upper, type: 'scatter', mode: 'lines', name: 'Upper Bound',
                    line: { color: 'rgba(248,113,113,0.5)', dash: 'dot', width: 1 }, showlegend: false,
                    hovertemplate: 'Upper: Rs %{y:.2f}<extra></extra>'
                },
                {
                    x: dates, y: lower, type: 'scatter', mode: 'lines', name: 'Lower Bound',
                    line: { color: 'rgba(16,185,129,0.5)', dash: 'dot', width: 1 }, showlegend: false,
                    hovertemplate: 'Lower: Rs %{y:.2f}<extra></extra>'
                },
            ]}
                layout={{
                    ...DARK, height: 360,
                    shapes: seasonShapes,
                    annotations: festAnnotations,
                    yaxis: { ...DARK.yaxis, title: { text: 'Predicted Price (LKR/kg)', font: { size: 11 } } },
                    xaxis: { ...DARK.xaxis, title: '', tickangle: -35 },
                    legend: { ...DARK.legend, x: 0, y: 1.1, orientation: 'h' },
                }}
                config={PLOT_CFG} style={{ width: '100%' }} useResizeHandler />

            {/* Weekly table */}
            <div style={{ marginTop: 16, overflowX: 'auto' }}>
                <table className="algo-table">
                    <thead>
                        <tr>
                            <th>Week</th><th>Date</th><th>Predicted Price</th>
                            <th>Range (88%–112%)</th><th>Season</th><th>Inflation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {fc.map(r => (
                            <tr key={r.week} style={r.is_festive ? { backgroundColor: 'rgba(245,158,11,0.08)' } : {}}>
                                <td>W{r.week}</td>
                                <td>{r.date}</td>
                                <td style={{ fontWeight: 600, color: '#a5b4fc' }}>Rs {r.predicted_price}</td>
                                <td style={{ fontSize: '0.78rem', color: '#64748b' }}>Rs {r.lower_bound} – Rs {r.upper_bound}</td>
                                <td><span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{r.season}{r.is_festive ? ' 🎉' : ''}</span></td>
                                <td style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{r.inflation_used}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </>
    )
}


// ── Predict Page ──────────────────────────────────────────────────────────────
export default function Predict() {
    const [meta, setMeta] = useState(null)
    const [form, setForm] = useState({
        commodity: 'Tomato',
        market: 'Manning Market',
        date: TODAY,
        price_type: 'Wholesale',
        inflation_index: 1.50,
    })
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [activeTab, setActiveTab] = useState('forecast')
    const [weeksAhead, setWeeksAhead] = useState(12)

    useEffect(() => {
        axios.get('/api/metadata').then(r => {
            setMeta(r.data)
        })
    }, [])

    const set = (k, v) => setForm(f => ({ ...f, [k]: v }))
    const selectedDateIsFuture = isFuture(form.date)

    const handlePredict = async () => {
        setLoading(true); setError(null); setResult(null)
        try {
            const r = await axios.post('/api/predict', {
                ...form, inflation_index: +form.inflation_index,
            })
            setResult(r.data)
            setActiveTab('shap')
        } catch (e) {
            setError(e.response?.data?.detail || e.message)
        } finally {
            setLoading(false)
        }
    }

    const pct = ((form.inflation_index - 1.0) / (3.0 - 1.0) * 100).toFixed(0)
    const inflLabel = +form.inflation_index <= 1.2 ? '😊 Stable' :
        +form.inflation_index <= 1.8 ? '📈 Moderate' :
            +form.inflation_index <= 2.4 ? '⚠️ High' : '🔥 Crisis'

    return (
        <div className="page-wrapper">
            {/* Hero */}
            <div className="hero-banner">
                <h1>🔮 Price Prediction & AI Explanation</h1>
                <p>Predict or forecast prices for any commodity using LightGBM — includes future forecasting up to 52 weeks ahead.</p>
                <span className="hero-tag">⚡ LightGBM · SHAP Waterfall · Multi-Week Future Forecast · PDPs</span>
            </div>

            {/* Metrics */}
            {meta?.metrics && (
                <div className="metrics-row">
                    <div className="metric-badge">
                        <div className="m-val">Rs {meta.metrics.rmse?.toFixed(2)}</div>
                        <div className="m-lbl">RMSE</div>
                        <div className="m-desc">Root Mean Squared Error</div>
                    </div>
                    <div className="metric-badge">
                        <div className="m-val">Rs {meta.metrics.mae?.toFixed(2)}</div>
                        <div className="m-lbl">MAE</div>
                        <div className="m-desc">Mean Absolute Error</div>
                    </div>
                    <div className="metric-badge">
                        <div className="m-val">{(meta.metrics.r2 * 100)?.toFixed(2)}%</div>
                        <div className="m-lbl">R² Score</div>
                        <div className="m-desc">Variance explained</div>
                    </div>
                    <div className="metric-badge">
                        <div className="m-val">{meta.metrics.mape?.toFixed(2)}%</div>
                        <div className="m-lbl">MAPE</div>
                        <div className="m-desc">Mean Absolute % Error</div>
                    </div>
                </div>
            )}

            <div className="predict-layout">
                {/* ── Left: Form ── */}
                <div className="form-card">
                    <h2>⚙️ Prediction Parameters</h2>

                    <div className="form-group">
                        <label>Commodity</label>
                        <select value={form.commodity} onChange={e => set('commodity', e.target.value)}>
                            {(meta?.commodities || ['Tomato', 'Carrot']).map(c => <option key={c}>{c}</option>)}
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Market / Location</label>
                        <select value={form.market} onChange={e => set('market', e.target.value)}>
                            {(meta?.markets || ['Manning Market']).map(m => <option key={m}>{m}</option>)}
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Price Type</label>
                        <select value={form.price_type} onChange={e => set('price_type', e.target.value)}>
                            <option>Wholesale</option>
                            <option>Retail</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Prediction Date <span style={{ color: '#64748b', textTransform: 'none', fontWeight: 400, fontSize: '0.7rem' }}>(past or future)</span></label>
                        <input type="date" value={form.date} onChange={e => set('date', e.target.value)}
                            min="2019-01-01" max="2027-12-31" />
                    </div>

                    <div className="form-group">
                        <label>Inflation Index — {inflLabel}</label>
                        <div className="slider-wrap">
                            <input type="range" min="1.0" max="3.0" step="0.05"
                                value={form.inflation_index}
                                onChange={e => set('inflation_index', e.target.value)}
                                style={{ '--pct': `${pct}%` }} />
                            <span className="slider-val">{(+form.inflation_index).toFixed(2)}</span>
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: 4 }}>
                            1.0 = pre-crisis · 1.5 = current 2026 estimate · 2.8 = 2022 peak
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Weeks to Forecast Ahead (for Forecast tab)</label>
                        <div className="slider-wrap">
                            <input type="range" min="4" max="52" step="1"
                                value={weeksAhead}
                                onChange={e => setWeeksAhead(+e.target.value)}
                                style={{ '--pct': `${((weeksAhead - 4) / (52 - 4) * 100).toFixed(0)}%` }} />
                            <span className="slider-val">{weeksAhead}w</span>
                        </div>
                    </div>

                    {error && <div className="error-box">⚠️ {error}</div>}

                    <button className="btn-predict" onClick={handlePredict} disabled={loading}>
                        {loading ? '⏳  Predicting…' : '⚡  Predict Price'}
                    </button>

                </div>

                {/* ── Right: Results ── */}
                <div>
                    {/* ====== TABS ====== */}
                    <div className="chart-card" style={{ marginBottom: 20 }}>
                        <div className="tabs">
                            <button className={`tab-btn${activeTab === 'forecast' ? ' active' : ''}`} onClick={() => setActiveTab('forecast')}>📅 Future Forecast</button>
                            <button className={`tab-btn${activeTab === 'shap' ? ' active' : ''}`} onClick={() => setActiveTab('shap')}>🌊 SHAP Waterfall</button>
                            <button className={`tab-btn${activeTab === 'importance' ? ' active' : ''}`} onClick={() => setActiveTab('importance')}>📊 Feature Importance</button>
                        </div>

                        {/* ── Future Forecast Tab ── */}
                        {activeTab === 'forecast' && (
                            <>
                                <h3>📅 Multi-Week Future Price Forecast — from Today ({TODAY})</h3>
                                <div className="info-box" style={{ marginBottom: 16 }}>
                                    🔄 <b>Iterative forecasting:</b> Each week's predicted price feeds back as the next week's "Last Week's Price" input.
                                    This propagates uncertainty forward — the model re-uses its own outputs as lag inputs, similar to how an ARIMA or Prophet rolling forecast works.
                                    Shaded region = ±12% confidence band. 🟣 Maha season · 🟢 Yala season backgrounds shown.
                                </div>
                                <ForecastChart
                                    commodity={form.commodity}
                                    market={form.market}
                                    priceType={form.price_type}
                                    weeksAhead={weeksAhead}
                                    inflationIndex={+form.inflation_index}
                                />
                            </>
                        )}

                        {/* ── SHAP Waterfall Tab ── */}
                        {activeTab === 'shap' && (
                            <>
                                {result ? (
                                    <>
                                        {/* Result Hero Card */}
                                        <div className="result-hero" style={{ marginBottom: 16 }}>
                                            <div className="result-commodity">{result.commodity} · {result.market} · {result.price_type}</div>
                                            <div className="result-price">Rs {result.predicted_price?.toFixed(2)}</div>
                                            <div className="result-unit">per kilogram</div>
                                            <div className="result-range">📉 Confidence: Rs {result.lower_bound} – Rs {result.upper_bound}</div>
                                            <div className="result-tags">
                                                <span className="result-tag">📅 {result.date}</span>
                                                <span className="result-tag">🌱 {result.season}</span>
                                                {result.is_festive && <span className="result-tag">🎉 Festive</span>}
                                                {result.historical_avg && (
                                                    <span className="result-tag">
                                                        📊 Hist. Avg: Rs {result.historical_avg}
                                                        {result.predicted_price > result.historical_avg
                                                            ? ` ↑ +${(result.predicted_price - result.historical_avg).toFixed(0)}`
                                                            : ` ↓ -${(result.historical_avg - result.predicted_price).toFixed(0)}`}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                        <h3>SHAP Waterfall — What Drove This Prediction</h3>
                                        <p style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: 10 }}>
                                            🔴 Red bars = features that <b>raised</b> the price · 🟢 Green bars = features that <b>lowered</b> the price, relative to the model baseline of Rs {result.base_value}
                                        </p>
                                        <div className="info-box">
                                            💡 <b>How to read:</b> Start at the blue dotted line (historical average = Rs {result.base_value}).
                                            Each bar shifts the prediction left or right. The green dotted line marks the final predicted price of Rs {result.predicted_price}.
                                        </div>
                                        <ShapWaterfall contributions={result.shap_contributions} baseValue={result.base_value} prediction={result.predicted_price} />
                                    </>
                                ) : (
                                    <div className="empty-state" style={{ padding: '40px 0' }}>
                                        <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🌊</div>
                                        <p>Click <b>"Predict Price"</b> to generate a SHAP waterfall for your chosen date.</p>
                                        <div className="info-box" style={{ textAlign: 'left', maxWidth: 420, margin: '16px auto 0' }}>
                                            💡 <b>Tip:</b> Select any date — past or future — set your parameters, then click <b>Predict Price</b>.
                                        </div>
                                    </div>
                                )}
                            </>
                        )}

                        {/* ── Feature Importance Tab ── */}
                        {activeTab === 'importance' && (
                            <>
                                <h3>Global Feature Importance — What the Model Relies On Most</h3>
                                <p style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: 10 }}>Feature importance by total split gain across all 1,000 trees — higher score = more influence across all predictions</p>
                                <div className="info-box">
                                    💡 <b>Price lag features dominate</b> because produce prices are highly autocorrelated — last week's price is the strongest predictor of this week's price. Seasonal harvest factor captures supply-side cycles and inflation captures economic shocks.
                                </div>
                                <FeatureImportance />
                            </>
                        )}

                    </div>
                </div>
            </div>
        </div>
    )
}

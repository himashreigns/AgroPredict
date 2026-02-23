import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'

export default function App() {
    return (
        <BrowserRouter>
            <div className="app-layout">
                {/* ── Sidebar ── */}
                <aside className="sidebar">
                    <div className="sidebar-brand">
                        <span className="logo">🥦</span>
                        <h1>SL Price Predictor</h1>
                        <p>Powered by LightGBM + SHAP</p>
                    </div>

                    <nav className="sidebar-nav">
                        <div className="nav-label">Navigation</div>
                        <NavLink to="/dashboard" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
                            <span className="nav-icon">📊</span> Dashboard
                        </NavLink>
                        <NavLink to="/predict" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
                            <span className="nav-icon">🔮</span> Predict & Explain
                        </NavLink>
                    </nav>

                    <div className="sidebar-footer">
                        <b>Data Sources</b>
                        DOA / SHEP AgriInfoHub<br />
                        CBSL Economic Indicators<br />
                        HARTI Food Bulletins<br />
                        Dept. of Census & Statistics
                        <b>Algorithm</b>
                        LightGBM Regressor<br />
                        (Gradient Boosted Trees)
                        <b>Explainability</b>
                        SHAP + Partial Dependence Plots
                    </div>
                </aside>

                {/* ── Main ── */}
                <main className="main-content">
                    <Routes>
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/predict" element={<Predict />} />
                    </Routes>
                </main>
            </div>
        </BrowserRouter>
    )
}

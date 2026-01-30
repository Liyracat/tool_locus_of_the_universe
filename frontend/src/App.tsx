import { Link, Route, Routes } from "react-router-dom";
import TopPage from "./pages/Top";
import ImportPage from "./pages/Import";
import SettingsPage from "./pages/Settings";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title">Locus of the Universe</div>
        <nav className="app-nav">
          <Link to="/" className="nav-link">
            トップ
          </Link>
          <Link to="/import" className="nav-link">
            インポート
          </Link>
          <Link to="/settings" className="nav-link">
            設定
          </Link>
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<TopPage />} />
          <Route path="/import" element={<ImportPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}

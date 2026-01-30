import { Route, Routes } from "react-router-dom";
import TopPage from "./pages/Top";
import ImportPage from "./pages/Import";
import SettingsPage from "./pages/Settings";

export default function App() {
  return (
    <div className="app-shell">
      <Routes>
        <Route path="/" element={<TopPage />} />
        <Route path="/import" element={<ImportPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </div>
  );
}

import { Link } from "react-router-dom";

export default function TopPage() {
  return (
    <div className="page">
      <section className="panel">
        <div className="panel-row">
          <div className="breadcrumb">パンくず: トップ</div>
          <div className="panel-actions">
            <Link to="/import" className="button primary">
              インポート
            </Link>
            <Link to="/settings" className="button ghost">
              設定
            </Link>
          </div>
        </div>
        <div className="panel-row filters">
          <div className="filter-item">種別: (仮)</div>
          <div className="filter-item">キーワード: (仮)</div>
          <button className="button">フィルタ追加</button>
        </div>
      </section>

      <section className="galaxy-panel">
        <div className="galaxy-header">Galaxy Map Canvas (仮)</div>
        <div className="galaxy-canvas">
          <div className="galaxy-placeholder">
            星の描画は後続で実装予定です。
          </div>
        </div>
      </section>
    </div>
  );
}

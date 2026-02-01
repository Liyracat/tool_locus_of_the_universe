import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ImportPreviewResponse } from "../api";

export default function ImportPage() {
  const [rawText, setRawText] = useState("");
  const [preview, setPreview] = useState<ImportPreviewResponse | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const handlePreview = async () => {
    setStatus(null);
    setBusy(true);
    try {
      const result = await api.previewImport(rawText);
      setPreview(result);
      setStatus(`プレビュー件数: ${result.parts.length}`);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "プレビューに失敗しました");
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async () => {
    if (!preview) {
      setStatus("先にプレビューを実行してください。");
      return;
    }
    setStatus(null);
    setBusy(true);
    try {
      const result = await api.commitImport(preview);
      setStatus(`保存完了: thread_id=${result.thread_id}`);
      setRawText("");
      setPreview(null);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "保存に失敗しました");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page page-shell">
      <header className="page-header">
        <div className="breadcrumb">インポート</div>
        <Link to="/" className="text-link">
          戻る / トップ
        </Link>
      </header>

      <section className="panel">
        <div className="panel-row column">
          <label className="label">テキスト入力フォーム</label>
          <textarea
            className="textarea"
            value={rawText}
            onChange={(event) => setRawText(event.target.value)}
            placeholder="Speaker: 本文..."
            rows={10}
          />
          <div className="panel-actions">
            <button className="button primary" onClick={handlePreview} disabled={busy}>
              プレビュー
            </button>
            <button className="button" onClick={handleSave} disabled={busy}>
              保存
            </button>
          </div>
          {status && <div className="status-text">{status}</div>}
        </div>
      </section>

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">プレビュー一覧</div>
        </div>
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>speaker_name</th>
                <th>conversation_at</th>
                <th>contents</th>
              </tr>
            </thead>
            <tbody>
              {preview?.parts.map((part, index) => (
                <tr key={`${part.message_id}-${part.text_id}-${index}`}>
                  <td>{part.speaker_name}</td>
                  <td>{part.conversation_at ?? ""}</td>
                  <td className="contents-cell">{part.contents}</td>
                </tr>
              ))}
              {!preview?.parts.length && (
                <tr>
                  <td colSpan={3} className="empty-cell">
                    プレビュー結果がありません。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

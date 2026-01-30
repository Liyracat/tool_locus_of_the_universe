import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Speaker, UtteranceRole, WorkerJob } from "../api";

const emptySpeaker = {
  speaker_name: "",
  speaker_role: "",
  canonical_role: "other",
  speaker_type_detail: "",
};

const emptyRole = { utterance_role_name: "" };

export default function SettingsPage() {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [utteranceRoles, setUtteranceRoles] = useState<UtteranceRole[]>([]);
  const [workerJobs, setWorkerJobs] = useState<WorkerJob[]>([]);
  const [speakerForm, setSpeakerForm] = useState({ ...emptySpeaker });
  const [editingSpeakerId, setEditingSpeakerId] = useState<string | null>(null);
  const [roleForm, setRoleForm] = useState({ ...emptyRole });
  const [editingRoleId, setEditingRoleId] = useState<number | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const loadAll = async () => {
    try {
      const [speakerList, roleList, jobList] = await Promise.all([
        api.listSpeakers(),
        api.listUtteranceRoles(),
        api.listWorkerJobs(),
      ]);
      setSpeakers(speakerList);
      setUtteranceRoles(roleList);
      setWorkerJobs(jobList);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "データ取得に失敗しました");
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  const submitSpeaker = async () => {
    setStatus(null);
    try {
      if (editingSpeakerId) {
        await api.updateSpeaker(editingSpeakerId, {
          speaker_name: speakerForm.speaker_name,
          speaker_role: speakerForm.speaker_role || null,
          canonical_role: speakerForm.canonical_role,
          speaker_type_detail: speakerForm.speaker_type_detail || null,
        });
      } else {
        await api.createSpeaker({
          speaker_name: speakerForm.speaker_name,
          speaker_role: speakerForm.speaker_role || null,
          canonical_role: speakerForm.canonical_role,
          speaker_type_detail: speakerForm.speaker_type_detail || null,
        });
      }
      setSpeakerForm({ ...emptySpeaker });
      setEditingSpeakerId(null);
      await loadAll();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "スピーカー保存に失敗しました");
    }
  };

  const submitRole = async () => {
    setStatus(null);
    try {
      if (editingRoleId !== null) {
        await api.updateUtteranceRole(editingRoleId, { ...roleForm });
      } else {
        await api.createUtteranceRole({ ...roleForm });
      }
      setRoleForm({ ...emptyRole });
      setEditingRoleId(null);
      await loadAll();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "ロール保存に失敗しました");
    }
  };

  const retryJob = async (job_id: string) => {
    setStatus(null);
    try {
      await api.retryWorkerJob(job_id);
      await loadAll();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "再実施に失敗しました");
    }
  };

  return (
    <div className="page page-shell">
      <header className="page-header">
        <div className="breadcrumb">設定</div>
        <Link to="/" className="text-link">
          戻る / トップ
        </Link>
      </header>
      {status && <div className="status-text">{status}</div>}

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">speakers CRUD</div>
        </div>
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>speaker_id</th>
                <th>speaker_name</th>
                <th>speaker_role</th>
                <th>canonical_role</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {speakers.map((speaker) => (
                <tr key={speaker.speaker_id}>
                  <td className="mono">{speaker.speaker_id}</td>
                  <td>{speaker.speaker_name}</td>
                  <td>{speaker.speaker_role ?? ""}</td>
                  <td>{speaker.canonical_role}</td>
                  <td>
                    <button
                      className="button tiny"
                      onClick={() => {
                        setEditingSpeakerId(speaker.speaker_id);
                        setSpeakerForm({
                          speaker_name: speaker.speaker_name,
                          speaker_role: speaker.speaker_role ?? "",
                          canonical_role: speaker.canonical_role,
                          speaker_type_detail: speaker.speaker_type_detail ?? "",
                        });
                      }}
                    >
                      編集
                    </button>
                    <button
                      className="button tiny ghost"
                      onClick={() => api.deleteSpeaker(speaker.speaker_id).then(loadAll)}
                    >
                      削除
                    </button>
                  </td>
                </tr>
              ))}
              {!speakers.length && (
                <tr>
                  <td colSpan={5} className="empty-cell">
                    スピーカーが登録されていません。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="form-grid">
          <div className="field">
            <label className="label">speaker_name</label>
            <input
              className="input"
              value={speakerForm.speaker_name}
              onChange={(event) =>
                setSpeakerForm((prev) => ({ ...prev, speaker_name: event.target.value }))
              }
            />
          </div>
          <div className="field">
            <label className="label">speaker_role</label>
            <input
              className="input"
              value={speakerForm.speaker_role}
              onChange={(event) =>
                setSpeakerForm((prev) => ({ ...prev, speaker_role: event.target.value }))
              }
            />
          </div>
          <div className="field">
            <label className="label">canonical_role</label>
            <select
              className="input"
              value={speakerForm.canonical_role}
              onChange={(event) =>
                setSpeakerForm((prev) => ({ ...prev, canonical_role: event.target.value }))
              }
            >
              <option value="self">self</option>
              <option value="ai">ai</option>
              <option value="human">human</option>
              <option value="other">other</option>
            </select>
          </div>
          <div className="field">
            <label className="label">speaker_type_detail</label>
            <input
              className="input"
              value={speakerForm.speaker_type_detail}
              onChange={(event) =>
                setSpeakerForm((prev) => ({ ...prev, speaker_type_detail: event.target.value }))
              }
            />
          </div>
          <button className="button primary" onClick={submitSpeaker}>
            {editingSpeakerId ? "更新" : "保存"}
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">utterance_roles CRUD</div>
        </div>
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>utterance_role_id</th>
                <th>utterance_role_name</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {utteranceRoles.map((role) => (
                <tr key={role.utterance_role_id}>
                  <td>{role.utterance_role_id}</td>
                  <td>{role.utterance_role_name}</td>
                  <td>
                    <button
                      className="button tiny"
                      onClick={() => {
                        setEditingRoleId(role.utterance_role_id);
                        setRoleForm({ utterance_role_name: role.utterance_role_name });
                      }}
                    >
                      編集
                    </button>
                    <button
                      className="button tiny ghost"
                      onClick={() => api.deleteUtteranceRole(role.utterance_role_id).then(loadAll)}
                    >
                      削除
                    </button>
                  </td>
                </tr>
              ))}
              {!utteranceRoles.length && (
                <tr>
                  <td colSpan={3} className="empty-cell">
                    ロールが登録されていません。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="form-grid">
          <div className="field">
            <label className="label">utterance_role_name</label>
            <input
              className="input"
              value={roleForm.utterance_role_name}
              onChange={(event) => setRoleForm({ utterance_role_name: event.target.value })}
            />
          </div>
          <button className="button primary" onClick={submitRole}>
            {editingRoleId !== null ? "更新" : "保存"}
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">worker_jobs 一覧</div>
        </div>
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>job_id</th>
                <th>job_type</th>
                <th>target</th>
                <th>status</th>
                <th>updated_at</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {workerJobs.map((job) => (
                <tr key={job.job_id}>
                  <td className="mono">{job.job_id}</td>
                  <td>{job.job_type}</td>
                  <td>{job.target_table}:{job.target_id}</td>
                  <td>{job.status}</td>
                  <td>{job.updated_at}</td>
                  <td>
                    {(job.status === "processing" || job.status === "failed") && (
                      <button className="button tiny" onClick={() => retryJob(job.job_id)}>
                        再実施
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {!workerJobs.length && (
                <tr>
                  <td colSpan={6} className="empty-cell">
                    worker_jobs がありません。
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

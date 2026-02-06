import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  SeedMergeCandidateListItem,
  Speaker,
  UnreviewedSeed,
  UtteranceRole,
  WorkerJob,
  WorkerTargetInfo,
} from "../api";

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
  const [workerJobsTotal, setWorkerJobsTotal] = useState(0);
  const [showSpeakers, setShowSpeakers] = useState(false);
  const [showRoles, setShowRoles] = useState(false);
  const [jobPage, setJobPage] = useState(1);
  const [mergePage, setMergePage] = useState(1);
  const [seedPage, setSeedPage] = useState(1);
  const [splitTargetType, setSplitTargetType] = useState("");
  const [splitTargetId, setSplitTargetId] = useState("");
  const [splitUpperText, setSplitUpperText] = useState("");
  const [splitLowerText, setSplitLowerText] = useState("");
  const [splitUpperMeta, setSplitUpperMeta] = useState<WorkerTargetInfo | null>(null);
  const [speakerForm, setSpeakerForm] = useState({ ...emptySpeaker });
  const [editingSpeakerId, setEditingSpeakerId] = useState<string | null>(null);
  const [roleForm, setRoleForm] = useState({ ...emptyRole });
  const [editingRoleId, setEditingRoleId] = useState<number | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [mergeCandidates, setMergeCandidates] = useState<SeedMergeCandidateListItem[]>([]);
  const [mergeTotal, setMergeTotal] = useState(0);
  const [unreviewedSeeds, setUnreviewedSeeds] = useState<UnreviewedSeed[]>([]);
  const [unreviewedTotal, setUnreviewedTotal] = useState(0);
  const [unreviewedSeedBodies, setUnreviewedSeedBodies] = useState<Record<string, string>>({});

  const loadAll = async () => {
    try {
      const [speakerList, roleList] = await Promise.all([
        api.listSpeakers(),
        api.listUtteranceRoles(),
      ]);
      setSpeakers(speakerList);
      setUtteranceRoles(roleList);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "データ取得に失敗しました");
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  const jobsPerPage = 50;
  const totalJobPages = Math.max(1, Math.ceil(workerJobsTotal / jobsPerPage));
  const pagedJobs = workerJobs;
  const mergePerPage = 50;
  const seedPerPage = 50;
  const totalMergePages = Math.max(1, Math.ceil(mergeTotal / mergePerPage));
  const totalSeedPages = Math.max(1, Math.ceil(unreviewedTotal / seedPerPage));

  useEffect(() => {
    setJobPage((prev) => Math.min(prev, totalJobPages));
  }, [totalJobPages]);

  useEffect(() => {
    const loadJobs = async () => {
      try {
        const result = await api.listWorkerJobs(jobPage, jobsPerPage);
        setWorkerJobs(result.items);
        setWorkerJobsTotal(result.total);
      } catch (err) {
        setStatus(err instanceof Error ? err.message : "worker_jobsの取得に失敗しました");
      }
    };
    void loadJobs();
  }, [jobPage]);

  useEffect(() => {
    const loadMergeCandidates = async () => {
      try {
        const result = await api.listSeedMergeCandidatesAll(mergePage, mergePerPage);
        setMergeCandidates(result.items);
        setMergeTotal(result.total);
      } catch (err) {
        setStatus(err instanceof Error ? err.message : "統合候補の取得に失敗しました");
      }
    };
    void loadMergeCandidates();
  }, [mergePage]);

  useEffect(() => {
    const loadSeeds = async () => {
      try {
        const result = await api.listUnreviewedSeeds(seedPage, seedPerPage);
        setUnreviewedSeeds(result.items);
        setUnreviewedTotal(result.total);
        setUnreviewedSeedBodies((prev) => {
          const next = { ...prev };
          result.items.forEach((seed) => {
            if (!(seed.seed_id in next)) {
              next[seed.seed_id] = seed.body ?? "";
            }
          });
          return next;
        });
      } catch (err) {
        setStatus(err instanceof Error ? err.message : "未レビューseedの取得に失敗しました");
      }
    };
    void loadSeeds();
  }, [seedPage]);

  useEffect(() => {
    setMergePage((prev) => Math.min(prev, totalMergePages));
  }, [totalMergePages]);

  useEffect(() => {
    setSeedPage((prev) => Math.min(prev, totalSeedPages));
  }, [totalSeedPages]);

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
      const jobs = await api.listWorkerJobs(jobPage, jobsPerPage);
      setWorkerJobs(jobs.items);
      setWorkerJobsTotal(jobs.total);
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
      const jobs = await api.listWorkerJobs(jobPage, jobsPerPage);
      setWorkerJobs(jobs.items);
      setWorkerJobsTotal(jobs.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "ロール保存に失敗しました");
    }
  };

  const purgeUnusedData = async () => {
    setStatus(null);
    try {
      const result = await api.purgeUnusedData();
      const totalDeleted = Object.values(result.deleted ?? {}).reduce(
        (sum, value) => sum + (Number.isFinite(value) ? Number(value) : 0),
        0
      );
      setStatus(`不要データを削除しました（${totalDeleted}件）`);
      await loadAll();
      const jobs = await api.listWorkerJobs(jobPage, jobsPerPage);
      setWorkerJobs(jobs.items);
      setWorkerJobsTotal(jobs.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "不要データ削除に失敗しました");
    }
  };

  const reprioritizeWorkerJobs = async () => {
    setStatus(null);
    try {
      const result = await api.reprioritizeWorkerJobs();
      setStatus(`優先順位を更新しました（${result.updated}件）`);
      await loadAll();
      const jobs = await api.listWorkerJobs(jobPage, jobsPerPage);
      setWorkerJobs(jobs.items);
      setWorkerJobsTotal(jobs.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "優先順位更新に失敗しました");
    }
  };

  const seedRefetch = async () => {
    setStatus(null);
    try {
      const result = await api.seedRefetch();
      setStatus(`seed再取得を開始しました（${result.count}件）`);
      await loadAll();
      const jobs = await api.listWorkerJobs(jobPage, jobsPerPage);
      setWorkerJobs(jobs.items);
      setWorkerJobsTotal(jobs.total);
      const mergeResult = await api.listSeedMergeCandidatesAll(mergePage, mergePerPage);
      setMergeCandidates(mergeResult.items);
      setMergeTotal(mergeResult.total);
      const seedResult = await api.listUnreviewedSeeds(seedPage, seedPerPage);
      setUnreviewedSeeds(seedResult.items);
      setUnreviewedTotal(seedResult.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "seed再取得に失敗しました");
    }
  };

  const retryJob = async (job_id: string) => {
    setStatus(null);
    try {
      await api.retryWorkerJob(job_id);
      await loadAll();
      const jobs = await api.listWorkerJobs(jobPage, jobsPerPage);
      setWorkerJobs(jobs.items);
      setWorkerJobsTotal(jobs.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "再実施に失敗しました");
    }
  };

  const deleteSuccessJobs = async () => {
    setStatus(null);
    try {
      const result = await api.deleteSuccessWorkerJobs();
      setStatus(`successのジョブを${result.deleted}件削除しました`);
      await loadAll();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "successジョブ削除に失敗しました");
    }
  };

  const deleteSpeaker = async (speaker_id: string) => {
    setStatus(null);
    try {
      await api.deleteSpeaker(speaker_id);
      await loadAll();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "スピーカー削除に失敗しました");
    }
  };

  const deleteRole = async (utterance_role_id: number) => {
    setStatus(null);
    try {
      await api.deleteUtteranceRole(utterance_role_id);
      await loadAll();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "ロール削除に失敗しました");
    }
  };

  const handleSelectWorkerTarget = async (job: WorkerJob) => {
    if (job.target_table !== "utterance_split" && job.target_table !== "seed") return;
    setStatus(null);
    try {
      const info = await api.getWorkerTarget(job.target_table, job.target_id);
      setSplitTargetType(info.target_table);
      setSplitTargetId(info.target_id);
      setSplitUpperText(info.contents ?? "");
      setSplitLowerText("");
      setSplitUpperMeta(info);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "対象データの取得に失敗しました");
    }
  };

  const handleSplitSave = async () => {
    setStatus(null);
    try {
      if (splitTargetType === "utterance_split" && splitTargetId) {
        await api.saveUtteranceSplit({
          utterance_split_id: splitTargetId,
          contents_top: splitUpperText,
          contents_bottom: splitLowerText,
        });
        setSplitLowerText("");
        await loadAll();
        setStatus("utterance_splitを保存しました。");
        return;
      }
      if (splitTargetType === "seed" && splitTargetId) {
        await api.saveSeedSplit({
          seed_id: splitTargetId,
          body_top: splitUpperText,
          body_bottom: splitLowerText,
        });
        setSplitLowerText("");
        await loadAll();
        setStatus("seedを保存しました。");
        return;
      }
      setStatus("target_typeを選択してください。");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "保存に失敗しました");
    }
  };

  const handleMergeApprove = async (
    candidate: SeedMergeCandidateListItem,
    direction: "a" | "b"
  ) => {
    setStatus(null);
    try {
      const targetSeedId = direction === "a" ? candidate.seed_b_id : candidate.seed_a_id;
      const canonicalSeedId = direction === "a" ? candidate.seed_a_id : candidate.seed_b_id;
      const targetBody = direction === "a" ? candidate.seed_b_body : candidate.seed_a_body;
      const related = await api.listSeedMergeCandidates(targetSeedId);
      const rejectIds = related
        .map((item) => item.candidate_id)
        .filter((id) => id !== candidate.candidate_id);
      await api.updateSeed({
        seed_id: targetSeedId,
        seed_type: "seed",
        body: targetBody ?? "",
        canonical_seed_id: canonicalSeedId,
        review_status: null,
      });
      await api.resolveSeedMergeCandidates({
        merged_candidate_id: candidate.candidate_id,
        reject_candidate_ids: rejectIds,
      });
      await loadAll();
      const refreshed = await api.listSeedMergeCandidatesAll(mergePage, mergePerPage);
      setMergeCandidates(refreshed.items);
      setMergeTotal(refreshed.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "統合に失敗しました");
    }
  };

  const handleMergeReject = async (candidate: SeedMergeCandidateListItem) => {
    setStatus(null);
    try {
      await api.updateSeedMergeCandidate(candidate.candidate_id, "rejected");
      const refreshed = await api.listSeedMergeCandidatesAll(mergePage, mergePerPage);
      setMergeCandidates(refreshed.items);
      setMergeTotal(refreshed.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "却下に失敗しました");
    }
  };

  const handleSeedReview = async (seed: UnreviewedSeed, nextStatus: "reviewed" | "rejected") => {
    setStatus(null);
    try {
      const nextBody = unreviewedSeedBodies[seed.seed_id] ?? seed.body ?? "";
      await api.updateSeed({
        seed_id: seed.seed_id,
        seed_type: seed.seed_type,
        body: nextBody,
        canonical_seed_id: null,
        review_status: nextStatus,
      });
      const refreshed = await api.listUnreviewedSeeds(seedPage, seedPerPage);
      setUnreviewedSeeds(refreshed.items);
      setUnreviewedTotal(refreshed.total);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "更新に失敗しました");
    }
  };

  return (
    <div className="page page-shell">
      <header className="page-header">
        <div className="breadcrumb">設定</div>
        <div className="header-actions">
          <button type="button" className="button tiny ghost" onClick={reprioritizeWorkerJobs}>
            優先順位自動採番
          </button>
          <button type="button" className="button tiny ghost" onClick={seedRefetch}>
            seed再取得
          </button>
          <button type="button" className="button tiny ghost" onClick={purgeUnusedData}>
            不要データ削除
          </button>
          <Link to="/" className="text-link">
            戻る / トップ
          </Link>
        </div>
      </header>
      {status && <div className="status-text">{status}</div>}

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">speakers CRUD</div>
          <button
            type="button"
            className="button tiny ghost"
            onClick={() => setShowSpeakers((prev) => !prev)}
          >
            {showSpeakers ? "閉じる" : "開く"}
          </button>
        </div>
        {showSpeakers && (
          <>
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
                          type="button"
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
                          type="button"
                          className="button tiny ghost"
                          onClick={() => deleteSpeaker(speaker.speaker_id)}
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
              <button type="button" className="button primary" onClick={submitSpeaker}>
                {editingSpeakerId ? "更新" : "保存"}
              </button>
            </div>
          </>
        )}
      </section>

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">utterance_roles CRUD</div>
          <button
            type="button"
            className="button tiny ghost"
            onClick={() => setShowRoles((prev) => !prev)}
          >
            {showRoles ? "閉じる" : "開く"}
          </button>
        </div>
        {showRoles && (
          <>
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
                          type="button"
                          className="button tiny"
                          onClick={() => {
                            setEditingRoleId(role.utterance_role_id);
                            setRoleForm({ utterance_role_name: role.utterance_role_name });
                          }}
                        >
                          編集
                        </button>
                        <button
                          type="button"
                          className="button tiny ghost"
                          onClick={() => deleteRole(role.utterance_role_id)}
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
              <button type="button" className="button primary" onClick={submitRole}>
                {editingRoleId !== null ? "更新" : "保存"}
              </button>
            </div>
          </>
        )}
      </section>

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">worker_jobs 一覧</div>
          <button type="button" className="button tiny ghost" onClick={deleteSuccessJobs}>
            success/no_dataを削除
          </button>
        </div>
        <div className="panel-row column">
          <div className="section-title">utterance_split・seed 分割</div>
          <div className="split-form">
            <div className="split-form-actions">
              <button type="button" className="button primary" onClick={handleSplitSave}>
                保存
              </button>
            </div>
            <div className="form-grid full-width split-form-grid">
              <div className="field">
                <label className="label">target_type</label>
                <input className="input" value={splitTargetType} readOnly />
              </div>
              <div className="field">
                <label className="label">上段</label>
                <input
                  className="input"
                  value={splitUpperText}
                  onChange={(event) => setSplitUpperText(event.target.value)}
                />
              </div>
              <div className="field">
                <label className="label">下段</label>
                <input
                  className="input"
                  value={splitLowerText}
                  onChange={(event) => setSplitLowerText(event.target.value)}
                />
              </div>
            </div>
          </div>
          {splitUpperMeta?.target_table === "seed" && (
            <div className="form-hint">
              seed_type: {splitUpperMeta.seed_type ?? ""} / created_from: {splitUpperMeta.created_from ?? ""}
            </div>
          )}
          {splitUpperMeta?.target_table === "utterance_split" && (
            <div className="form-hint">utterance_id: {splitUpperMeta.utterance_id ?? ""}</div>
          )}
        </div>
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>job_id</th>
                <th>job_type</th>
                <th>target</th>
                <th>status</th>
                <th>error</th>
                <th>created_at</th>
                <th className="sticky-col">操作</th>
              </tr>
            </thead>
            <tbody>
              {pagedJobs.map((job) => (
                <tr key={job.job_id}>
                  <td className="mono">{job.job_id}</td>
                  <td>{job.job_type}</td>
                  <td>
                    <button
                      type="button"
                      className="text-link"
                      onClick={() => handleSelectWorkerTarget(job)}
                    >
                      {job.target_table}:{job.target_id}
                    </button>
                  </td>
                  <td>{job.status}</td>
                  <td>{job.error ?? ""}</td>
                  <td>{job.created_at}</td>
                  <td className="sticky-col">
                    {(job.status === "processing" || job.status === "failed") && (
                      <button type="button" className="button tiny" onClick={() => retryJob(job.job_id)}>
                        再実施
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {!workerJobs.length && (
                <tr>
                  <td colSpan={7} className="empty-cell">
                    worker_jobs がありません。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {totalJobPages > 1 && (
          <div className="panel-row">
            <button
              type="button"
              className="button tiny ghost"
              onClick={() => setJobPage((prev) => Math.max(1, prev - 1))}
              disabled={jobPage <= 1}
            >
              前へ
            </button>
            <div className="mono">
              {jobPage} / {totalJobPages}
            </div>
            <button
              type="button"
              className="button tiny ghost"
              onClick={() => setJobPage((prev) => Math.min(totalJobPages, prev + 1))}
              disabled={jobPage >= totalJobPages}
            >
              次へ
            </button>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">seed_merge_candidates 一覧</div>
        </div>
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>seed_a contents</th>
                <th>seed_b contents</th>
                <th>reason</th>
                <th>similarity</th>
                <th className="sticky-col">操作</th>
              </tr>
            </thead>
            <tbody>
              {mergeCandidates.map((item) => (
                <tr key={item.candidate_id}>
                  <td className="contents-cell">{item.seed_a_body ?? ""}</td>
                  <td className="contents-cell">{item.seed_b_body ?? ""}</td>
                  <td>{item.reason}</td>
                  <td>{typeof item.similarity === "number" ? item.similarity.toFixed(3) : ""}</td>
                  <td className="sticky-col">
                    <button
                      type="button"
                      className="button tiny"
                      onClick={() => handleMergeApprove(item, "a")}
                    >
                      aに統合
                    </button>
                    <button
                      type="button"
                      className="button tiny ghost"
                      onClick={() => handleMergeApprove(item, "b")}
                    >
                      bに統合
                    </button>
                    <button
                      type="button"
                      className="button tiny ghost"
                      onClick={() => handleMergeReject(item)}
                    >
                      却下
                    </button>
                  </td>
                </tr>
              ))}
              {!mergeCandidates.length && (
                <tr>
                  <td colSpan={5} className="empty-cell">
                    統合候補がありません。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {totalMergePages > 1 && (
          <div className="panel-row">
            <button
              type="button"
              className="button tiny ghost"
              onClick={() => setMergePage((prev) => Math.max(1, prev - 1))}
              disabled={mergePage <= 1}
            >
              前へ
            </button>
            <div className="mono">
              {mergePage} / {totalMergePages}
            </div>
            <button
              type="button"
              className="button tiny ghost"
              onClick={() => setMergePage((prev) => Math.min(totalMergePages, prev + 1))}
              disabled={mergePage >= totalMergePages}
            >
              次へ
            </button>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-row">
          <div className="section-title">未レビュー seeds 一覧</div>
        </div>
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>seed_type</th>
                <th>body</th>
                <th className="sticky-col">操作</th>
              </tr>
            </thead>
            <tbody>
              {unreviewedSeeds.map((seed) => (
                <tr key={seed.seed_id}>
                  <td>{seed.seed_type}</td>
                  <td className="contents-cell">
                    <textarea
                      className="textarea"
                      rows={3}
                      value={unreviewedSeedBodies[seed.seed_id] ?? seed.body ?? ""}
                      onChange={(event) =>
                        setUnreviewedSeedBodies((prev) => ({
                          ...prev,
                          [seed.seed_id]: event.target.value,
                        }))
                      }
                    />
                  </td>
                  <td className="sticky-col">
                    <button
                      type="button"
                      className="button tiny"
                      onClick={() => handleSeedReview(seed, "reviewed")}
                    >
                      承認
                    </button>
                    <button
                      type="button"
                      className="button tiny ghost"
                      onClick={() => handleSeedReview(seed, "rejected")}
                    >
                      却下
                    </button>
                  </td>
                </tr>
              ))}
              {!unreviewedSeeds.length && (
                <tr>
                  <td colSpan={3} className="empty-cell">
                    未レビューのseedがありません。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {totalSeedPages > 1 && (
          <div className="panel-row">
            <button
              type="button"
              className="button tiny ghost"
              onClick={() => setSeedPage((prev) => Math.max(1, prev - 1))}
              disabled={seedPage <= 1}
            >
              前へ
            </button>
            <div className="mono">
              {seedPage} / {totalSeedPages}
            </div>
            <button
              type="button"
              className="button tiny ghost"
              onClick={() => setSeedPage((prev) => Math.min(totalSeedPages, prev + 1))}
              disabled={seedPage >= totalSeedPages}
            >
              次へ
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

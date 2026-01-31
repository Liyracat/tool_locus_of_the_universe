PRAGMA foreign_keys = ON;

-- =========================================
-- speakers
-- =========================================
CREATE TABLE IF NOT EXISTS speakers (
  speaker_id           TEXT PRIMARY KEY,                -- UUID
  speaker_name         TEXT NOT NULL,                   -- 例: リラ, フウ
  speaker_role         TEXT,                            -- 生データ側の表示名: "あなた", "ChatGPT"など
  canonical_role       TEXT NOT NULL                    -- self / ai / human / other
    CHECK (canonical_role IN ('self','ai','human','other')),
  speaker_type_detail  TEXT,                            -- 例: other_book / other_blog / ai_chatgpt / human_direct など
  created_at           TEXT NOT NULL,
  updated_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_speakers_canonical_role
  ON speakers (canonical_role);

CREATE INDEX IF NOT EXISTS idx_speakers_type_detail
  ON speakers (speaker_type_detail);


-- =========================================
-- utterance_roles
-- =========================================
CREATE TABLE IF NOT EXISTS utterance_roles (
  utterance_role_id    INTEGER PRIMARY KEY,
  utterance_role_name  TEXT NOT NULL UNIQUE,            -- 例: 知識, 抽象概念, 目標, 主張...
  created_at           TEXT NOT NULL,
  updated_at           TEXT NOT NULL
);


-- =========================================
-- utterance
-- =========================================
CREATE TABLE IF NOT EXISTS utterance (
  utterance_id              TEXT PRIMARY KEY,           -- UUID
  thread_id                 TEXT NOT NULL,              -- UUID
  message_id                INTEGER NOT NULL,           -- スレッド内連番
  chunk_id                  INTEGER NOT NULL,           -- メッセージ分割の連番
  speaker_id                TEXT NOT NULL,
  conversation_at           TEXT,                       -- 発話時刻(元データ)
  contents                  TEXT NOT NULL,              -- chunkの生テキスト
  utterance_role_id         INTEGER,
  utterance_role_confidence REAL                         -- 0..1
    CHECK (utterance_role_confidence IS NULL OR (utterance_role_confidence >= 0.0 AND utterance_role_confidence <= 1.0)),
  
  hypothetical              REAL                         -- 0..1
    CHECK (hypothetical >= 0.0 AND hypothetical <= 1.0),
  confidence                REAL                         -- 0..1
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
  reinterpretation          REAL                         -- 0..1
    CHECK (reinterpretation >= 0.0 AND reinterpretation <= 1.0),
  resistance                REAL                         -- 0..1
    CHECK (resistance >= 0.0 AND resistance <= 1.0),
  direction                 REAL                         -- -1..1
    CHECK (direction >= -1.0 AND direction <= 1.0),
  did_asked_evaluation      INTEGER NOT NULL DEFAULT 0   -- 0/1
    CHECK (did_asked_evaluation IN (0,1)),
  did_asked_model           INTEGER NOT NULL DEFAULT 0   -- 0/1
    CHECK (did_asked_model IN (0,1)),
  did_asked_premise         INTEGER NOT NULL DEFAULT 0   -- 0/1
    CHECK (did_asked_premise IN (0,1)),
  did_asked_conversion      INTEGER NOT NULL DEFAULT 0   -- 0/1
    CHECK (did_asked_conversion IN (0,1)),
  did_asked_question        INTEGER NOT NULL DEFAULT 0   -- 0/1
    CHECK (did_asked_question IN (0,1)),
  did_asked_knowledge       INTEGER NOT NULL DEFAULT 0   -- 0/1
    CHECK (did_asked_knowledge IN (0,1)),
  created_at                TEXT NOT NULL,
  updated_at                TEXT NOT NULL,
  FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id) ON UPDATE CASCADE,
  FOREIGN KEY (utterance_role_id) REFERENCES utterance_roles(utterance_role_id) ON UPDATE CASCADE
);

-- よく使う検索パス
CREATE INDEX IF NOT EXISTS idx_utterance_thread_msg_chunk
  ON utterance (thread_id, message_id, chunk_id);

CREATE INDEX IF NOT EXISTS idx_utterance_speaker
  ON utterance (speaker_id);

CREATE INDEX IF NOT EXISTS idx_utterance_role
  ON utterance (utterance_role_id);

CREATE INDEX IF NOT EXISTS idx_utterance_conversation_at
  ON utterance (conversation_at);


-- =========================================
-- worker_jobs
-- =========================================
CREATE TABLE IF NOT EXISTS worker_jobs (
  job_id        TEXT PRIMARY KEY,                       -- UUID
  job_type      TEXT NOT NULL,                          -- 例: llm_classify / seed_extract / embed / umap / edge_assign ...
  target_table  TEXT NOT NULL,                          -- 例: utterance, seeds, clusters ...
  target_id     TEXT NOT NULL,                          -- 対象PK
  status        TEXT NOT NULL                            -- queued / processing / success / failed
    CHECK (status IN ('queued','processing','success','failed')),
  priority      INTEGER NOT NULL DEFAULT 100,           -- 小さいほど優先
  payload_json  TEXT,                                   -- 任意パラメータ
  locked_at     TEXT,
  lock_owner    TEXT,
  started_at    TEXT,
  finished_at   TEXT,
  error         TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  expires_at    TEXT                                     -- 物理削除用
);

CREATE INDEX IF NOT EXISTS idx_worker_jobs_status_priority
  ON worker_jobs (status, priority, created_at);

CREATE INDEX IF NOT EXISTS idx_worker_jobs_target
  ON worker_jobs (target_table, target_id);

CREATE INDEX IF NOT EXISTS idx_worker_jobs_expires_at
  ON worker_jobs (expires_at);


-- =========================================
-- seeds (星)
-- =========================================
CREATE TABLE IF NOT EXISTS seeds (
  seed_id       TEXT PRIMARY KEY,                       -- UUID
  seed_type     TEXT NOT NULL DEFAULT 'seed'             -- seed / external_knowledge / concept など
    CHECK (seed_type IN ('seed','external_knowledge','concept')),
  title         TEXT,                                   -- UI用の短い見出し
  body          TEXT NOT NULL,                           -- Seed本文
  created_from  TEXT NOT NULL DEFAULT 'utterance'        -- utterance / manual / import など
    CHECK (created_from IN ('utterance','manual','import')),
  avg_hypothetical              REAL                         -- 0..1
    CHECK (avg_hypothetical >= 0.0 AND avg_hypothetical <= 1.0),
  avg_confidence                REAL                         -- 0..1
    CHECK (avg_confidence >= 0.0 AND avg_confidence <= 1.0),
  avg_reinterpretation          REAL                         -- 0..1
    CHECK (avg_reinterpretation >= 0.0 AND avg_reinterpretation <= 1.0),
  avg_resistance                REAL                         -- 0..1
    CHECK (avg_resistance >= 0.0 AND avg_resistance <= 1.0),
  avg_direction                 REAL                         -- -1..1
    CHECK (avg_direction >= -1.0 AND avg_direction <= 1.0),
  review_status                 TEXT NOT NULL DEFAULT "auto" -- auto / reviewed / edited / rejected
    CHECK (review_status IN ('auto','reviewed','edited','rejected')),
  canonical_seed_id TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_seeds_seed_type
  ON seeds (seed_type);


-- =========================================
-- utterance_seeds (発言->Seed 多対多)
-- =========================================
CREATE TABLE IF NOT EXISTS utterance_seeds (
  utterance_id    TEXT NOT NULL,
  seed_id         TEXT NOT NULL,
  relation_type   TEXT NOT NULL DEFAULT 'derived_from'   -- derived_from / supports / contrasts / inspired_by ...
    CHECK (relation_type IN ('derived_from','supports','contrasts','inspired_by','same_topic')),
  confidence      REAL                                   -- 0..1
    CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
  created_at      TEXT NOT NULL,
  PRIMARY KEY (utterance_id, seed_id, relation_type),
  FOREIGN KEY (utterance_id) REFERENCES utterance(utterance_id) ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (seed_id) REFERENCES seeds(seed_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_utterance_seeds_seed
  ON utterance_seeds (seed_id);

CREATE INDEX IF NOT EXISTS idx_utterance_seeds_utterance
  ON utterance_seeds (utterance_id);


-- =========================================
-- clusters (銀河/クラスタ)
-- =========================================
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id     TEXT PRIMARY KEY,                      -- UUID
  cluster_name   TEXT,
  cluster_overview TEXT,
  cluster_level  TEXT NOT NULL DEFAULT 'cluster'         -- cluster / galaxy
    CHECK (cluster_level IN ('cluster','galaxy')),
  is_archived    INTEGER NOT NULL DEFAULT 0              -- 0/1
    CHECK (is_archived IN (0,1)),
  avg_hypothetical              REAL                         -- 0..1
    CHECK (avg_hypothetical >= 0.0 AND avg_hypothetical <= 1.0),
  avg_confidence                REAL                         -- 0..1
    CHECK (avg_confidence >= 0.0 AND avg_confidence <= 1.0),
  avg_reinterpretation          REAL                         -- 0..1
    CHECK (avg_reinterpretation >= 0.0 AND avg_reinterpretation <= 1.0),
  avg_resistance                REAL                         -- 0..1
    CHECK (avg_resistance >= 0.0 AND avg_resistance <= 1.0),
  avg_direction                 REAL                         -- -1..1
    CHECK (avg_direction >= -1.0 AND avg_direction <= 1.0),
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_clusters_level
  ON clusters (cluster_level);


-- =========================================
-- embeddings (埋め込み)
-- =========================================
CREATE TABLE IF NOT EXISTS embeddings (
  embedding_id       TEXT PRIMARY KEY,                  -- UUID
  target_type        TEXT NOT NULL                      -- utterance / seed / cluster
    CHECK (target_type IN ('utterance','seed','cluster')),
  target_id          TEXT NOT NULL,
  model_name         TEXT NOT NULL,                     -- 例: paraphrase-multilingual
  dims               INTEGER NOT NULL,
  vector             BLOB NOT NULL,                     -- float32配列などをバイナリ格納推奨
  is_l2_normalized   INTEGER NOT NULL DEFAULT 1          -- 0/1
    CHECK (is_l2_normalized IN (0,1)),
  faiss_index_id     INTEGER,                           -- FAISS側のID(任意)
  created_at         TEXT NOT NULL,
  UNIQUE (target_type, target_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_target
  ON embeddings (target_type, target_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_model
  ON embeddings (model_name);


-- レイアウト（地図）定義
CREATE TABLE layout_runs (
  layout_id TEXT PRIMARY KEY,
  algorithm TEXT NOT NULL,                  -- 'umap' など
  dims      INTEGER NOT NULL,               -- 2 or 3
  scope_type TEXT NOT NULL,                 -- 'global' or 'cluster'
  scope_cluster_id TEXT,                    -- scope_type='cluster' のとき必須
  params_json TEXT NOT NULL,                -- {"n_neighbors":..., "min_dist":...}
  is_active INTEGER NOT NULL DEFAULT 1     -- 0/1
    CHECK (is_active IN (0,1)),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- scopeの整合性（SQLiteのCHECKは弱いけど最低限）
CREATE INDEX idx_layout_runs_scope
  ON layout_runs(scope_type, scope_cluster_id);

-- レイアウト上の各点
CREATE TABLE layout_points (
  layout_id TEXT NOT NULL,
  target_type TEXT NOT NULL CHECK (target_type IN ('cluster','seed','utterance')),
  target_id   TEXT NOT NULL,
  x REAL NOT NULL,
  y REAL NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1    -- 0/1
    CHECK (is_active IN (0,1)),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (layout_id, target_type, target_id),
  FOREIGN KEY (layout_id) REFERENCES layout_runs(layout_id) ON DELETE CASCADE
);

CREATE INDEX idx_layout_points_target
  ON layout_points(target_type, target_id);


-- =========================================
-- edges (関係線)
-- =========================================
CREATE TABLE IF NOT EXISTS edges (
  edge_id      TEXT PRIMARY KEY,                        -- UUID
  src_type     TEXT NOT NULL                            -- seed / cluster / utterance
    CHECK (src_type IN ('seed','cluster','utterance')),
  src_id       TEXT NOT NULL,
  dst_type     TEXT NOT NULL
    CHECK (dst_type IN ('seed','cluster','utterance')),
  dst_id       TEXT NOT NULL,
  edge_type    TEXT NOT NULL                            -- near / part_of / supports / contrasts ...
    CHECK (edge_type IN ('near','part_of','supports','contrasts','inspired_by','same_topic')),
  weight       REAL,                                    -- 類似度や重み
  is_active    INTEGER NOT NULL DEFAULT 1                -- 0/1（張り替え思想）
    CHECK (is_active IN (0,1)),
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src
  ON edges (src_type, src_id, edge_type, is_active);

CREATE INDEX IF NOT EXISTS idx_edges_dst
  ON edges (dst_type, dst_id, edge_type, is_active);

CREATE INDEX IF NOT EXISTS idx_edges_type
  ON edges (edge_type, is_active);

-- =========================================
-- seed_merge_candidates
-- =========================================

CREATE TABLE IF NOT EXISTS seed_merge_candidates (
    candidate_id TEXT PRIMARY KEY,         -- UUID

    seed_a_id TEXT NOT NULL,                -- 統合先候補（canonical）
    seed_b_id TEXT NOT NULL,                -- 統合元候補（alias）

    reason TEXT NOT NULL,                   -- exact_text / near_duplicate / manual / other
    similarity REAL,                        -- embedding類似度（任意、exact_text時はNULL可）

    status TEXT NOT NULL DEFAULT 'proposed', 
        -- proposed / merged / rejected

    decided_at TEXT,                        -- merged or rejected になった時刻
    decided_by TEXT,                        -- self / system / ai（任意）

    note TEXT,                              -- 任意メモ（なぜ統合/却下したか）

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    -- 同一ペアの重複候補防止
    UNIQUE (seed_a_id, seed_b_id)
);

-- 検索用インデックス
CREATE INDEX IF NOT EXISTS idx_seed_merge_candidates_status
    ON seed_merge_candidates(status);

CREATE INDEX IF NOT EXISTS idx_seed_merge_candidates_seed_a
    ON seed_merge_candidates(seed_a_id);

CREATE INDEX IF NOT EXISTS idx_seed_merge_candidates_seed_b
    ON seed_merge_candidates(seed_b_id);

-- =========================================
-- all_seed_info
-- =========================================

CREATE TABLE IF NOT EXISTS all_seed_info (
    avg_seed_distance REAL,                -- 近傍距離の平均
    median_seed_distance REAL,             -- 近傍距離の中央値
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

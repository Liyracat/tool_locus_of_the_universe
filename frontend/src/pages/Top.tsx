import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, MapLink, MapNode, MapResponse } from "../api";
import GalaxyMapCanvas, { GalaxyLink, GalaxyNode } from "../components/GalaxyMapCanvas";

const buildMockMap = (): MapResponse => {
  const nodes: MapNode[] = [];
  const links: MapLink[] = [];
  const center = { x: 0, y: 0 };

  const addNode = (
    id: string,
    node_type: MapNode["node_type"],
    x: number,
    y: number,
    radius: number,
    color_key: string,
    label?: string
  ) => {
    nodes.push({
      id,
      node_type,
      x,
      y,
      radius,
      color_key,
      glow_intensity: node_type === "cluster" ? 0.8 : 0.5,
      label,
    });
  };

  addNode("cluster-core", "cluster", center.x, center.y, 16, "cluster", "アイデアの銀河");
  const ring = [
    { id: "seed-1", angle: 0.2, radius: 220, size: 8, color: "seed" },
    { id: "seed-2", angle: 0.9, radius: 180, size: 7, color: "seed" },
    { id: "seed-3", angle: 1.6, radius: 240, size: 9, color: "seed" },
    { id: "seed-4", angle: 2.4, radius: 210, size: 7, color: "seed" },
    { id: "seed-5", angle: 3.0, radius: 260, size: 8, color: "seed" },
    { id: "seed-6", angle: 4.2, radius: 200, size: 6, color: "seed" },
    { id: "seed-7", angle: 5.1, radius: 230, size: 7, color: "seed" },
  ];

  ring.forEach((item) => {
    addNode(
      item.id,
      "seed",
      center.x + Math.cos(item.angle) * item.radius,
      center.y + Math.sin(item.angle) * item.radius,
      item.size,
      item.color
    );
    links.push({
      id: `link-${item.id}`,
      src_id: "cluster-core",
      dst_id: item.id,
      link_type: "near",
      weight: 0.6,
      is_active: true,
      origin: "edge",
    });
  });

  const leftCluster = {
    id: "cluster-left",
    x: -420,
    y: -120,
    radius: 12,
  };
  addNode(leftCluster.id, "cluster", leftCluster.x, leftCluster.y, leftCluster.radius, "cluster");
  for (let i = 0; i < 18; i += 1) {
    const angle = (i / 18) * Math.PI * 2;
    const radius = 90 + Math.random() * 40;
    const id = `l-seed-${i}`;
    addNode(
      id,
      "seed",
      leftCluster.x + Math.cos(angle) * radius,
      leftCluster.y + Math.sin(angle) * radius,
      5 + (i % 3),
      i % 3 === 0 ? "accent" : "seed"
    );
    links.push({
      id: `link-${id}`,
      src_id: leftCluster.id,
      dst_id: id,
      link_type: "near",
      weight: 0.4 + (i % 4) * 0.1,
      is_active: true,
      origin: "edge",
    });
  }

  const rightCluster = {
    id: "cluster-right",
    x: 420,
    y: 140,
    radius: 12,
  };
  addNode(rightCluster.id, "cluster", rightCluster.x, rightCluster.y, rightCluster.radius, "cluster");
  for (let i = 0; i < 22; i += 1) {
    const angle = (i / 22) * Math.PI * 2;
    const radius = 100 + Math.random() * 50;
    const id = `r-seed-${i}`;
    addNode(
      id,
      "seed",
      rightCluster.x + Math.cos(angle) * radius,
      rightCluster.y + Math.sin(angle) * radius,
      5 + (i % 3),
      i % 4 === 0 ? "accent" : "seed"
    );
    links.push({
      id: `link-${id}`,
      src_id: rightCluster.id,
      dst_id: id,
      link_type: "near",
      weight: 0.35 + (i % 5) * 0.1,
      is_active: true,
      origin: "edge",
    });
  }

  links.push({
    id: "link-bridge-left",
    src_id: "cluster-core",
    dst_id: leftCluster.id,
    link_type: "near",
    weight: 0.7,
    is_active: true,
    origin: "edge",
  });
  links.push({
    id: "link-bridge-right",
    src_id: "cluster-core",
    dst_id: rightCluster.id,
    link_type: "near",
    weight: 0.7,
    is_active: true,
    origin: "edge",
  });

  return {
    breadcrumb: [
      { label: "全体ビュー", view: "global" },
      { label: "グラスタA", view: "cluster", cluster_id: "cluster-a" },
    ],
    filters: {
      category: "俯瞰",
      focus: "偽",
      tone: "善",
    },
    nodes,
    links,
  };
};

const convertNodes = (nodes: MapNode[], scale: number): GalaxyNode[] =>
  nodes.map((node) => ({
    id: node.id,
    node_type: node.node_type,
    x: node.x * scale,
    y: node.y * scale,
    radius: node.radius,
    glow_intensity: node.glow_intensity ?? 0.5,
    color_key: node.color_key,
    label: node.label ?? node.title,
    meta: node.meta,
  }));

const convertLinks = (links: MapLink[]): GalaxyLink[] =>
  links.map((link) => ({
    id: link.id,
    src_id: link.src_id,
    dst_id: link.dst_id,
    link_type: link.link_type,
    weight: link.weight,
    is_active: link.is_active,
    origin: link.origin,
  }));

export default function TopPage() {
  const [mapData, setMapData] = useState<MapResponse | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [useMock, setUseMock] = useState(false);

  useEffect(() => {
    const load = async () => {
      setStatus(null);
      try {
        const data = await api.getMap({ view: "global" });
        if (!data.nodes.length) {
          setUseMock(true);
          setMapData(buildMockMap());
          return;
        }
        setUseMock(false);
        setMapData(data);
      } catch (err) {
        setStatus(err instanceof Error ? err.message : "データ取得に失敗しました");
        setUseMock(true);
        setMapData(buildMockMap());
      }
    };
    void load();
  }, []);

  const breadcrumb = mapData?.breadcrumb
    ?.map((item) => item.label)
    .filter(Boolean)
    .join("  >  ") ?? "全体ビュー  >  グラスタA";

  const nodes = useMemo(
    () => convertNodes(mapData?.nodes ?? [], useMock ? 1 : 200),
    [mapData, useMock]
  );
  const links = useMemo(() => convertLinks(mapData?.links ?? []), [mapData]);

  return (
    <div className="top-page">
      <div className="top-overlay">
        <div className="top-toolbar">
          <div className="breadcrumb breadcrumb-dark">{breadcrumb}</div>
          <div className="top-actions">
            <Link to="/import" className="icon-button" aria-label="インポート">
              <span className="icon">＋</span>
            </Link>
            <Link to="/settings" className="icon-button" aria-label="設定">
              <span className="icon">⚙</span>
            </Link>
          </div>
        </div>
        <div className="filter-row">
          <button className="filter-chip">
            カテゴリ: <strong>俯瞰</strong>
          </button>
          <button className="filter-chip">
            顕在意識: <strong>偽</strong>
          </button>
          <button className="filter-chip">
            顕在意識: <strong>善</strong>
          </button>
          <button className="filter-chip add">＋ フィルタ</button>
        </div>
      </div>

      <div className="galaxy-stage">
        <GalaxyMapCanvas nodes={nodes} links={links} edgeMode="hover" />
        {status && <div className="status-banner">{status}</div>}
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import {
  Application,
  Assets,
  Color,
  Container,
  Graphics,
  Point,
  Sprite,
  Text,
  TextStyle,
} from "pixi.js";

export type GalaxyNode = {
  id: string;
  node_type: "seed" | "cluster" | "galaxy" | "utterance" | string;
  x: number;
  y: number;
  radius: number;
  glow_intensity: number;
  color_key: string;
  label?: string;
  meta?: Record<string, unknown>;
};

export type GalaxyLink = {
  id: string;
  src_id: string;
  dst_id: string;
  link_type: string;
  weight: number;
  is_active: boolean;
  origin: "edge" | "utterance_seed" | string;
};

export type GalaxyMapCanvasProps = {
  nodes: GalaxyNode[];
  links: GalaxyLink[];
  edgeMode?: "hover" | "topN";
  topN?: number;
  onNodeHover?: (node: GalaxyNode | null) => void;
  onNodeClick?: (node: GalaxyNode) => void;
};

const COLOR_MAP: Record<string, string> = {
  seed: "#89c7ff",
  cluster: "#ffd25f",
  galaxy: "#ffffff",
  utterance: "#b39bff",
  accent: "#9af3ff",
};

const EDGE_WIDTH: Record<string, number> = {
  near: 1.25,
  part_of: 1.8,
  derived_from: 1.4,
  supports: 1.6,
  contrasts: 1.6,
  inspired_by: 1.6,
  same_topic: 1.4,
};

const EDGE_COLOR: Record<string, string> = {
  near: "#c8e8ff",
  part_of: "#a5b4fc",
  derived_from: "#93c5fd",
  supports: "#fde68a",
  contrasts: "#fca5a5",
  inspired_by: "#fbcfe8",
  same_topic: "#99f6e4",
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export default function GalaxyMapCanvas({
  nodes,
  links,
  edgeMode = "hover",
  topN = 3,
  onNodeHover,
  onNodeClick,
}: GalaxyMapCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const worldRef = useRef<Container | null>(null);
  const panRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const edgesRef = useRef<Graphics | null>(null);
  const nodesLayerRef = useRef<Container | null>(null);
  const labelsLayerRef = useRef<Container | null>(null);
  const nodesRef = useRef<Map<string, Container>>(new Map());
  const labelRef = useRef<Map<string, Text>>(new Map());
  const sizeRef = useRef<{ width: number; height: number }>({ width: 0, height: 0 });
  const hoverRef = useRef<GalaxyNode | null>(null);
  const pulseRef = useRef<
    Array<{
      nodeId: string;
      halo: Sprite;
      bloom: Sprite;
      baseHaloAlpha: number;
      baseBloomAlpha: number;
      phase: number;
    }>
  >([]);
  const [tooltip, setTooltip] = useState<{
    label: string;
    type: string;
    x: number;
    y: number;
    visible: boolean;
  } | null>(null);
  
  const [pixiReady, setPixiReady] = useState(false);
  const setupCleanupRef = useRef<null | (() => void)>(null);

  useEffect(() => {
    console.log("useEffect start", containerRef.current);
    const host = containerRef.current;
    if (!host) return;

    let destroyed = false;
    const setup = async () => {
      const bounds = host.getBoundingClientRect();
      const width = Math.max(320, bounds.width);
      const height = Math.max(260, bounds.height);

      const hasInit = typeof (Application as unknown as { prototype?: { init?: unknown } }).prototype
        ?.init === "function";

      let app: Application;
      if (hasInit) {
        app = new Application();
        await (app as unknown as { init: (options: unknown) => Promise<void> }).init({
          width,
          height,
          backgroundAlpha: 0,
          antialias: true,
          resolution: window.devicePixelRatio || 1,
          autoDensity: true,
        });
      } else {
        app = new Application({
          width,
          height,
          backgroundAlpha: 0,
          antialias: true,
          resolution: window.devicePixelRatio || 1,
          autoDensity: true,
        });
      }

      if (destroyed) {
        app.destroy(true);
        return;
      }

      // --- Load star texture (radial gradient) ---
      // Place file at: frontend/public/radial_gradient.png
      // Access path: /radial_gradient.png
      await Assets.load("/radial_gradient.png");

      appRef.current = app;
      const canvas = ((app as unknown as { canvas?: HTMLCanvasElement }).canvas ??
        (app as unknown as { view?: HTMLCanvasElement }).view) as HTMLCanvasElement;
      host.appendChild(canvas);

      const world = new Container();
      worldRef.current = world;
      app.stage.addChild(world);
      world.x = width / 2;
      world.y = height / 2;
      sizeRef.current = { width, height };

      const edgesLayer = new Graphics();
      edgesRef.current = edgesLayer;
      world.addChild(edgesLayer);

      const nodesLayer = new Container();
      world.addChild(nodesLayer);
      nodesLayerRef.current = nodesLayer;

      const labelLayer = new Container();
      world.addChild(labelLayer);
      labelsLayerRef.current = labelLayer;
      
      setPixiReady(true);

      const interactionTarget = app.stage;
      if ("eventMode" in interactionTarget) {
        (interactionTarget as unknown as { eventMode: string }).eventMode = "static";
        (interactionTarget as unknown as { hitArea: unknown }).hitArea = app.screen;
      } else {
        (interactionTarget as unknown as { interactive: boolean }).interactive = true;
        (interactionTarget as unknown as { hitArea: unknown }).hitArea = app.screen;
      }

      let isDragging = false;
      const last = new Point();

      interactionTarget.on("pointerdown", (event: { global: Point }) => {
        isDragging = true;
        last.copyFrom(event.global);
      });

      interactionTarget.on("pointerup", () => {
        isDragging = false;
      });

      interactionTarget.on("pointerupoutside", () => {
        isDragging = false;
      });

      interactionTarget.on("pointermove", (event: { global: Point }) => {
        if (!isDragging || !worldRef.current) return;
        const current = event.global;
        const dx = current.x - last.x;
        const dy = current.y - last.y;
        worldRef.current.x += dx;
        worldRef.current.y += dy;
        panRef.current.x += dx;
        panRef.current.y += dy;
        last.copyFrom(current);
      });

      canvas.addEventListener(
        "wheel",
        (event) => {
          event.preventDefault();
          const world = worldRef.current;
          if (!world) return;
          const scaleFactor = event.deltaY > 0 ? 0.92 : 1.08;
          const newScale = clamp(world.scale.x * scaleFactor, 0.35, 2.4);
          const pointer = new Point(event.offsetX, event.offsetY);
          const before = world.toLocal(pointer);
          world.scale.set(newScale);
          const after = world.toLocal(pointer);
          world.x += (after.x - before.x) * world.scale.x;
          world.y += (after.y - before.y) * world.scale.y;
          panRef.current.x = world.x - sizeRef.current.width / 2;
          panRef.current.y = world.y - sizeRef.current.height / 2;
        },
        { passive: false }
      );

      const resizeObserver = new ResizeObserver(() => {
        if (!appRef.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
      
        appRef.current.renderer.resize(rect.width, rect.height);
      
        if (worldRef.current) {
          worldRef.current.x = rect.width / 2 + panRef.current.x;
          worldRef.current.y = rect.height / 2 + panRef.current.y;
        }
      
        sizeRef.current = { width: rect.width, height: rect.height };
      });
      resizeObserver.observe(host);

      app.ticker.add(() => {
        if (!hoverRef.current || !appRef.current || !worldRef.current) return;
        const node = hoverRef.current;
        const nodeGraphic = nodesRef.current.get(node.id);
        if (!nodeGraphic) return;
        const screenPoint = worldRef.current.toGlobal(nodeGraphic.position);
        setTooltip((prev) =>
          prev
            ? {
                ...prev,
                x: screenPoint.x,
                y: screenPoint.y,
              }
            : null
        );
      });

      // --- Pulsing glow (single ticker, cheap) ---
      const pulseTick = () => {
        const app = appRef.current;
        if (!app) return;
        const t = (app.ticker.lastTime ?? 0) / 1000;
        const hoveredId = hoverRef.current?.id ?? null;

        for (const p of pulseRef.current) {
          // Hover中の星は pointerover/out の強調があるので、脈動は止める（競合防止）
          if (hoveredId && p.nodeId === hoveredId) continue;

          // ゆっくり揺れる。位相ずらしで全体が同期しないようにする。
          const w1 = 0.6;
          const w2 = 0.45;
          const haloMul = 0.92 + 0.08 * Math.sin(t * w1 + p.phase);
          const bloomMul = 0.94 + 0.06 * Math.sin(t * w2 + p.phase * 1.31);

          p.halo.alpha = clamp(p.baseHaloAlpha * haloMul, 0, 1);
          p.bloom.alpha = clamp(p.baseBloomAlpha * bloomMul, 0, 0.45);
        }
      };

      app.ticker.add(pulseTick);

      return () => {
        if (app?.ticker) {
          app.ticker.remove(pulseTick);
        }
        resizeObserver.disconnect();
      };
    };

    void setup().then((cleanup) => {
      // setupが完了した後、useEffectのcleanupで呼べるように保持
      if (typeof cleanup === "function") setupCleanupRef.current = cleanup;
    });

    return () => {
      destroyed = true;
      setPixiReady(false);
      // setup内で登録したticker/observerの後始末
      setupCleanupRef.current?.();
      setupCleanupRef.current = null;
      if (appRef.current) {
        appRef.current.destroy(true);
        appRef.current = null;
      }
      nodesRef.current.clear();
      labelRef.current.clear();
      worldRef.current = null;
      edgesRef.current = null;
      nodesLayerRef.current = null;
      labelsLayerRef.current = null;
      pulseRef.current = [];
    };
  }, []);

  useEffect(() => {
    const app = appRef.current;
    const world = worldRef.current;
    const edgesLayer = edgesRef.current;
    if (!pixiReady || !app || !world || !edgesLayer) return;

    const nodesLayer = nodesLayerRef.current ?? undefined;
    const labelsLayer = labelsLayerRef.current ?? undefined;
    console.log("nodesLayer", nodesLayer, "labelsLayer", labelsLayer);
    if (!nodesLayer || !labelsLayer) return;

    const tex = Assets.get("/radial_gradient.png");
    if (!tex) {
      console.warn("radial_gradient.png texture not ready");
      return;
    }

    nodesLayer.removeChildren();
    labelsLayer.removeChildren();
    nodesRef.current.clear();
    labelRef.current.clear();
    pulseRef.current = [];

    const labelStyle = new TextStyle({
      fill: "#dbeafe",
      fontFamily: "\"Shippori Mincho\", \"Hiragino Mincho ProN\", serif",
      fontSize: 12,
      dropShadow: true,
      dropShadowAlpha: 0.6,
      dropShadowDistance: 2,
    });

    nodes.forEach((node) => {
      const color = COLOR_MAP[node.color_key] || COLOR_MAP[node.node_type] || "#ffffff";
      const phase = Math.random() * Math.PI * 2;

      const star = new Container();

      if ("eventMode" in star) {
        (star as unknown as { eventMode: string }).eventMode = "static";
      } else {
        (star as unknown as { interactive: boolean }).interactive = true;
      }
      (star as any).cursor = "pointer"; // Containerにも効く

      const baseColor = new Color(color).toNumber();

      const bloomAlpha = clamp(0.10 + node.glow_intensity * 0.10, 0.06, 0.30);
      const haloAlpha = clamp(0.25 + node.glow_intensity * 0.25, 0.18, 0.85);
      const coreAlpha = 0.92;

      // 1) bloom（外側、薄い、ADD）
      const bloom = new Sprite(tex);
      bloom.anchor.set(0.5);
      bloom.tint = baseColor;
      bloom.blendMode = "add";
      bloom.alpha = 0.10 + node.glow_intensity * 0.10;
      const bloomSize = (node.radius * 2) * (4.0 + node.glow_intensity * 1.8);
      bloom.width = bloomSize;
      bloom.height = bloomSize;

      // 2) halo（中間、ADD）
      const halo = new Sprite(tex);
      halo.anchor.set(0.5);
      halo.tint = baseColor;
      halo.blendMode = "add";
      halo.alpha = 0.25 + node.glow_intensity * 0.25;
      const haloSize = (node.radius * 2) * (2.2 + node.glow_intensity * 1.2);
      halo.width = haloSize;
      halo.height = haloSize;

      // 3) core（中心、白寄り、Normal）
      const core = new Sprite(tex);
      core.anchor.set(0.5);
      // コアは白っぽくした方が“眩しさ”になる（色はhaloに任せる）
      core.tint = 0xffffff;
      core.alpha = 0.9;
      const coreSize = node.radius * 2;
      core.width = coreSize;
      core.height = coreSize;

      star.addChild(bloom, halo, core);

      star.x = node.x;
      star.y = node.y;
      star.name = node.id;

      star.on("pointerover", () => {
        hoverRef.current = node;
        halo.alpha = clamp(haloAlpha + 0.15, 0, 1);
        bloom.alpha = clamp(bloomAlpha + 0.08, 0, 0.45);
        core.alpha = clamp(coreAlpha + 0.04, 0, 1);
        star.scale.set(1.12);
        const screenPoint = world.toGlobal(star.position);
        const label = node.label || node.meta?.["title"]?.toString() || node.id;
        setTooltip({
          label,
          type: node.node_type,
          x: screenPoint.x,
          y: screenPoint.y,
          visible: true,
        });
        onNodeHover?.(node);
        if (edgeMode === "hover") {
          drawEdgesForNode(node.id, links, edgesLayer, nodesRef.current);
        }
      });

      star.on("pointerout", () => {
        hoverRef.current = null;
        halo.alpha = haloAlpha;
        bloom.alpha = bloomAlpha;
        core.alpha = coreAlpha;
        setTooltip(null);
        onNodeHover?.(null);
        if (edgeMode === "hover") {
          edgesLayer.clear();
        }
      });

      star.on("pointertap", () => {
        onNodeClick?.(node);
      });

      nodesLayer.addChild(star);
      nodesRef.current.set(node.id, star);

      // Register for pulsing (single ticker uses this list)
      // Phase is randomized per star to avoid synchronized flicker.
      pulseRef.current.push({
        nodeId: node.id,
        halo,
        bloom,
        baseHaloAlpha: haloAlpha,
        baseBloomAlpha: bloomAlpha,
        phase: Math.random() * Math.PI * 2,
      });

      if (node.label) {
        const labelText = new Text({ text: node.label, style: labelStyle });
        labelText.x = node.x + node.radius + 6;
        labelText.y = node.y - node.radius - 6;
        labelsLayer.addChild(labelText);
        labelRef.current.set(node.id, labelText);
      }
    });

    if (edgeMode === "topN") {
      drawTopEdges(nodes, links, edgesLayer, topN);
    } else {
      edgesLayer.clear();
    }

    const cull = () => {
      const viewWidth = app.renderer.width;
      const viewHeight = app.renderer.height;
      nodesRef.current.forEach((graphic) => {
        const point = world.toGlobal(graphic.position);
        const margin = 120;
        const visible =
          point.x > -margin &&
          point.x < viewWidth + margin &&
          point.y > -margin &&
          point.y < viewHeight + margin;
        graphic.visible = visible;
      });
    };

    app.ticker.add(cull);

    return () => {
      if (app?.ticker) {
        app.ticker.remove(cull);
      }
    };
  }, [pixiReady, nodes, links, edgeMode, topN, onNodeHover, onNodeClick]);

  return (
    <div className="galaxy-canvas-shell" ref={containerRef}>
      {tooltip?.visible && (
        <div
          className="galaxy-tooltip"
          style={{
            transform: `translate(${tooltip.x + 16}px, ${tooltip.y - 24}px)`,
          }}
        >
          <div className="tooltip-title">{tooltip.label}</div>
          <div className="tooltip-meta">{tooltip.type}</div>
        </div>
      )}
    </div>
  );
}

function drawEdgesForNode(
  nodeId: string,
  links: GalaxyLink[],
  graphics: Graphics,
  nodeGraphics: Map<string, Container>
) {
  graphics.clear();
  graphics.blendMode = "add";
  for (const link of links) {
    if (!link.is_active) continue;
    if (link.link_type !== "near") continue;
    if (link.src_id !== nodeId && link.dst_id !== nodeId) continue;
    const srcNodeGraphic = nodeGraphics.get(link.src_id);
    const dstNodeGraphic = nodeGraphics.get(link.dst_id);

    if (!srcNodeGraphic || !dstNodeGraphic) continue;

    const color = new Color(EDGE_COLOR[link.link_type] || "#c7d2fe").toNumber();
    const width = EDGE_WIDTH[link.link_type] ?? 1.2;
    const alpha = clamp(link.weight * 0.9, 0.15, 0.75);
    drawGlowLine(
      graphics,
      srcNodeGraphic.x,
      srcNodeGraphic.y,
      dstNodeGraphic.x,
      dstNodeGraphic.y,
      color,
      width,
      alpha
    );
  }
}

function drawTopEdges(
  nodes: GalaxyNode[],
  links: GalaxyLink[],
  graphics: Graphics,
  topN: number
) {
  graphics.clear();
  graphics.blendMode = "add";
  const adjacency = new Map<string, GalaxyLink[]>();
  links.forEach((link) => {
    if (!link.is_active) return;
    if (link.link_type !== "near") return;
    const list = adjacency.get(link.src_id) ?? [];
    list.push(link);
    adjacency.set(link.src_id, list);
  });

  const nodeMap = new Map<string, GalaxyNode>();
  nodes.forEach((node) => nodeMap.set(node.id, node));

  for (const node of nodes) {
    const list = adjacency.get(node.id);
    if (!list) continue;
    const top = [...list].sort((a, b) => b.weight - a.weight).slice(0, topN);
    top.forEach((link) => {
      const src = nodeMap.get(link.src_id);
      const dst = nodeMap.get(link.dst_id);
      if (!src || !dst) return;
      const width = EDGE_WIDTH[link.link_type] ?? 1.2;
      const color = new Color(EDGE_COLOR[link.link_type] || "#c7d2fe").toNumber();
      const alpha = clamp(link.weight * 0.9, 0.15, 0.65);
      drawGlowLine(graphics, src.x, src.y, dst.x, dst.y, color, width, alpha);
    });
  }
}

function drawGlowLine(
  g: Graphics,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: number,
  baseWidth: number,
  alpha: number
) {
  // outer glow
  g.lineStyle({
    width: baseWidth * 2.6,
    color,
    alpha: clamp(alpha * 0.22, 0.03, 0.25),
    cap: "round",
    join: "round",
  });
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);

  // mid glow
  g.lineStyle({
    width: baseWidth * 1.7,
    color,
    alpha: clamp(alpha * 0.35, 0.05, 0.35),
    cap: "round",
    join: "round",
  });
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);

  // core
  g.lineStyle({
    width: baseWidth,
    color,
    alpha: clamp(alpha, 0.08, 0.75),
    cap: "round",
    join: "round",
  });
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
}

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  DROUGHT_COLOR_BY_CODE,
  droughtCategoryValue,
  droughtLabelForCategory,
  normalizeDroughtCategory,
  type CoordinateCheckResult,
  type DroughtMapFeatureProperties,
  type ForecastLayer,
  type GeoJsonFeature,
  type GeoJsonFeatureCollection,
  type GeoJsonGeometry,
  type GeoJsonPosition
} from "@/lib/types";

type MapLibreMap = import("maplibre-gl").Map;
type MapLibreModule = typeof import("maplibre-gl");
type MapLayerMouseEvent = import("maplibre-gl").MapLayerMouseEvent;
type GeoJsonSourceLike = {
  setData: (data: GeoJsonFeatureCollection) => void;
};

const SOURCE_ID = "forecast-layer";
const CHECK_SOURCE_ID = "coordinate-check";

const EMPTY_FEATURE_COLLECTION: GeoJsonFeatureCollection = {
  type: "FeatureCollection",
  features: []
};

const BASEMAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

interface DroughtMapProps {
  geojson: GeoJsonFeatureCollection | null;
  layer?: ForecastLayer;
  isLoading?: boolean;
  error?: string | null;
  selectedPoint?: CoordinateCheckResult | null;
  emptyMessage?: string;
}

const VIRIDIS_COLORS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
const DROUGHT_PATTERN_NAMES = {
  None: "drought-pattern-none",
  D0: "drought-pattern-d0",
  D1: "drought-pattern-d1",
  D2: "drought-pattern-d2",
  D3: "drought-pattern-d3",
  D4: "drought-pattern-d4"
} as const;

function rgbFromHex(hex: string): [number, number, number] {
  const normalized = hex.replace("#", "");
  return [
    Number.parseInt(normalized.slice(0, 2), 16),
    Number.parseInt(normalized.slice(2, 4), 16),
    Number.parseInt(normalized.slice(4, 6), 16)
  ];
}

function droughtPatternImage(
  background: string,
  foreground: string,
  pattern: "none" | "horizontal" | "diagonal" | "cross" | "dots" | "reverse"
): { width: number; height: number; data: Uint8Array } {
  const size = 16;
  const data = new Uint8Array(size * size * 4);
  const bg = rgbFromHex(background);
  const fg = rgbFromHex(foreground);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const useForeground =
        pattern === "horizontal" ? y % 8 === 0 :
        pattern === "diagonal" ? (x + y) % 10 === 0 :
        pattern === "cross" ? (x + y) % 12 === 0 || (x - y + size) % 12 === 0 :
        pattern === "dots" ? x % 8 === 2 && y % 8 === 2 :
        pattern === "reverse" ? (x - y + size) % 10 === 0 :
        false;
      const [red, green, blue] = useForeground ? fg : bg;
      const offset = (y * size + x) * 4;
      data[offset] = red;
      data[offset + 1] = green;
      data[offset + 2] = blue;
      data[offset + 3] = 255;
    }
  }
  return { width: size, height: size, data };
}

function ensureDroughtPatterns(map: MapLibreMap): void {
  const patterns = [
    [DROUGHT_PATTERN_NAMES.None, "#ffffff", "#637067", "none"],
    [DROUGHT_PATTERN_NAMES.D0, "#ffff00", "#585800", "horizontal"],
    [DROUGHT_PATTERN_NAMES.D1, "#fcd37f", "#765414", "diagonal"],
    [DROUGHT_PATTERN_NAMES.D2, "#ffaa00", "#663f00", "cross"],
    [DROUGHT_PATTERN_NAMES.D3, "#e60000", "#ffffff", "dots"],
    [DROUGHT_PATTERN_NAMES.D4, "#730000", "#ffffff", "reverse"]
  ] as const;
  patterns.forEach(([name, background, foreground, pattern]) => {
    if (!map.hasImage(name)) {
      map.addImage(name, droughtPatternImage(background, foreground, pattern) as never, { pixelRatio: 1 });
    }
  });
}

function enhanceBasemapLegibility(map: MapLibreMap): void {
  const styleLayers = map.getStyle().layers ?? [];
  styleLayers.forEach((item) => {
    if (item.type === "symbol" && map.getLayoutProperty(item.id, "text-field") !== undefined) {
      map.setPaintProperty(item.id, "text-color", "#354139");
      map.setPaintProperty(item.id, "text-halo-color", "rgba(255,255,255,0.98)");
      map.setPaintProperty(item.id, "text-halo-width", 1.5);
      map.setPaintProperty(item.id, "text-halo-blur", 0.15);
    }
    if (item.type === "line" && /(boundary|admin)/i.test(item.id)) {
      map.setPaintProperty(item.id, "line-color", "#59665e");
      map.setPaintProperty(item.id, "line-opacity", 0.82);
      map.setPaintProperty(item.id, "line-width", 1.1);
    }
  });
}

function scalarColor(value: number, layer: Exclude<ForecastLayer, "drought">): string {
  const [minimum, maximum] = layer === "et" ? [0, 4] : [0.05, 0.45];
  const normalized = Math.max(0, Math.min(0.999999, (value - minimum) / (maximum - minimum)));
  return VIRIDIS_COLORS[Math.floor(normalized * VIRIDIS_COLORS.length)];
}

function categoryFromProperties(properties: Record<string, unknown> | null): unknown {
  if (!properties) {
    return undefined;
  }

  return (
    properties.droughtCategory ??
    properties.drought_category ??
    properties.category_label ??
    properties.categoryLabel ??
    properties.category ??
    properties.cat ??
    properties.usdm ??
    properties.risk ??
    properties.risk_label
  );
}

function normalizeFeature(feature: GeoJsonFeature, layer: ForecastLayer): GeoJsonFeature {
  const properties = feature.properties ?? {};
  if (layer !== "drought") {
    const value = typeof properties.value === "number" ? properties.value : Number(properties.value);
    const finiteValue = Number.isFinite(value) ? value : 0;
    return {
      ...feature,
      properties: {
        ...properties,
        mapColor: scalarColor(finiteValue, layer),
        mapOpacity: 0.86,
        mapOutlineColor: "#ffffff",
        mapOutlineOpacity: 0.24,
        mapOutlineWidth: 0.35,
        mapValue: finiteValue,
        mapLabel: layer === "et" ? "Evapotranspiration" : "Soil moisture",
        mapKind: "scalar"
      }
    };
  }

  const category = normalizeDroughtCategory(categoryFromProperties(properties));
  const categoryValue = droughtCategoryValue(category);

  const mapProperties: DroughtMapFeatureProperties = {
    ...properties,
    droughtCategory: category,
    droughtCategoryValue: categoryValue,
    droughtColor: DROUGHT_COLOR_BY_CODE[category],
    droughtOpacity: category === "None" ? 0.16 : 0.82,
    droughtOutlineColor: categoryValue >= 4 ? "#f2f5ef" : DROUGHT_COLOR_BY_CODE[category],
    droughtOutlineOpacity: categoryValue >= 4 ? 0.5 : 0.28,
    droughtOutlineWidth: categoryValue >= 4 ? 0.55 : 0.25,
    droughtLabel: droughtLabelForCategory(category),
    mapColor: DROUGHT_COLOR_BY_CODE[category],
    mapOpacity: category === "None" ? 0.16 : 0.82,
    mapOutlineColor: categoryValue >= 4 ? "#f2f5ef" : "#3f493f",
    mapOutlineOpacity: category === "None" ? 0.1 : categoryValue >= 4 ? 0.42 : 0.24,
    mapOutlineWidth: category === "None" ? 0.12 : categoryValue >= 4 ? 0.55 : 0.24,
    mapLabel: droughtLabelForCategory(category),
    mapKind: "drought",
    mapPattern: DROUGHT_PATTERN_NAMES[category]
  };

  return {
    ...feature,
    properties: mapProperties
  };
}

function collectPositionsFromGeometry(geometry: GeoJsonGeometry | null, out: GeoJsonPosition[]): void {
  if (!geometry) {
    return;
  }

  switch (geometry.type) {
    case "Point":
      out.push(geometry.coordinates);
      break;
    case "MultiPoint":
    case "LineString":
      geometry.coordinates.forEach((position) => out.push(position));
      break;
    case "MultiLineString":
    case "Polygon":
      geometry.coordinates.forEach((line) => line.forEach((position) => out.push(position)));
      break;
    case "MultiPolygon":
      geometry.coordinates.forEach((polygon) =>
        polygon.forEach((line) => line.forEach((position) => out.push(position)))
      );
      break;
    case "GeometryCollection":
      geometry.geometries.forEach((child) => collectPositionsFromGeometry(child, out));
      break;
  }
}

function featureCollectionBounds(collection: GeoJsonFeatureCollection): [[number, number], [number, number]] | null {
  const positions: GeoJsonPosition[] = [];
  collection.features.forEach((feature) => collectPositionsFromGeometry(feature.geometry, positions));

  let minLon = Number.POSITIVE_INFINITY;
  let minLat = Number.POSITIVE_INFINITY;
  let maxLon = Number.NEGATIVE_INFINITY;
  let maxLat = Number.NEGATIVE_INFINITY;

  positions.forEach(([lon, lat]) => {
    if (Number.isFinite(lon) && Number.isFinite(lat)) {
      minLon = Math.min(minLon, lon);
      minLat = Math.min(minLat, lat);
      maxLon = Math.max(maxLon, lon);
      maxLat = Math.max(maxLat, lat);
    }
  });

  if (!Number.isFinite(minLon) || !Number.isFinite(minLat) || !Number.isFinite(maxLon) || !Number.isFinite(maxLat)) {
    return null;
  }

  if (minLon === maxLon) {
    minLon -= 0.25;
    maxLon += 0.25;
  }
  if (minLat === maxLat) {
    minLat -= 0.25;
    maxLat += 0.25;
  }

  return [
    [minLon, minLat],
    [maxLon, maxLat]
  ];
}

function ensureForecastLayers(map: MapLibreMap): void {
  ensureDroughtPatterns(map);
  const styleLayers = map.getStyle().layers ?? [];
  const beforeLayerId =
    styleLayers.find((item) => item.type === "line" && item.id.toLowerCase().includes("boundary"))?.id ??
    styleLayers.find((item) => item.type === "symbol")?.id;
  if (!map.getSource(SOURCE_ID)) {
    map.addSource(SOURCE_ID, {
      type: "geojson",
      data: EMPTY_FEATURE_COLLECTION as never
    });
  }

  if (!map.getLayer("forecast-layer-fill")) {
    map.addLayer({
      id: "forecast-layer-fill",
      type: "fill",
      source: SOURCE_ID,
      filter: ["all", ["==", ["geometry-type"], "Polygon"], ["!=", ["get", "mapKind"], "drought"]],
      paint: {
        "fill-color": ["coalesce", ["get", "mapColor"], "#8A8F98"],
        "fill-opacity": ["coalesce", ["get", "mapOpacity"], 0.78]
      }
    }, beforeLayerId);
  }

  if (!map.getLayer("forecast-drought-pattern")) {
    map.addLayer({
      id: "forecast-drought-pattern",
      type: "fill",
      source: SOURCE_ID,
      filter: ["all", ["==", ["geometry-type"], "Polygon"], ["==", ["get", "mapKind"], "drought"]],
      paint: {
        "fill-pattern": ["get", "mapPattern"],
        "fill-opacity": ["coalesce", ["get", "mapOpacity"], 0.82]
      }
    }, beforeLayerId);
  }

  if (!map.getLayer("forecast-layer-outline")) {
    map.addLayer({
      id: "forecast-layer-outline",
      type: "line",
      source: SOURCE_ID,
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: {
        "line-color": ["coalesce", ["get", "mapOutlineColor"], "#6B7280"],
        "line-opacity": ["coalesce", ["get", "mapOutlineOpacity"], 0.55],
        "line-width": ["coalesce", ["get", "mapOutlineWidth"], 0.7]
      }
    }, beforeLayerId);
  }

  if (!map.getLayer("forecast-layer-points")) {
    map.addLayer({
      id: "forecast-layer-points",
      type: "circle",
      source: SOURCE_ID,
      filter: ["==", ["geometry-type"], "Point"],
      paint: {
        "circle-color": ["coalesce", ["get", "mapColor"], "#8A8F98"],
        "circle-opacity": ["coalesce", ["get", "mapOpacity"], 0.85],
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 4, 7, 7, 10, 10],
        "circle-stroke-color": ["coalesce", ["get", "mapOutlineColor"], "#ffffff"],
        "circle-stroke-opacity": 0.72,
        "circle-stroke-width": 0.9
      }
    }, beforeLayerId);
  }

  if (!map.getSource(CHECK_SOURCE_ID)) {
    map.addSource(CHECK_SOURCE_ID, {
      type: "geojson",
      data: EMPTY_FEATURE_COLLECTION as never
    });
  }

  if (!map.getLayer("coordinate-check-point")) {
    map.addLayer({
      id: "coordinate-check-point",
      type: "circle",
      source: CHECK_SOURCE_ID,
      paint: {
        "circle-color": "#397a48",
        "circle-radius": 6,
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 2
      }
    });
  }
}

function setGeoJsonSourceData(map: MapLibreMap, sourceId: string, data: GeoJsonFeatureCollection): void {
  const source = map.getSource(sourceId) as GeoJsonSourceLike | undefined;
  source?.setData(data);
}

function popupHtml(properties: Record<string, unknown> | null, layer: ForecastLayer): string {
  if (layer !== "drought") {
    const value = typeof properties?.value === "number" ? properties.value : undefined;
    const units = typeof properties?.units === "string" ? properties.units : layer === "et" ? "mm/day" : "m3/m3";
    const label = layer === "et" ? "Evapotranspiration" : "Soil moisture";
    return `
      <div style="min-width: 150px">
        <div style="font-weight: 700; color: #172019">${label}</div>
        <div style="font-size: 12px; color: #172019; margin-top: 6px">${value === undefined ? "No data" : `${value.toFixed(layer === "et" ? 3 : 4)} ${units}`}</div>
      </div>
    `;
  }

  const category = normalizeDroughtCategory(properties?.droughtCategory ?? properties?.category_label ?? properties?.category);
  const label = droughtLabelForCategory(category);
  const pdry = typeof properties?.pdry === "number" ? `${(properties.pdry * 100).toFixed(1)}%` : undefined;
  const pdryPct = typeof properties?.pdry_pct === "number" ? `${properties.pdry_pct.toFixed(1)}%` : undefined;

  return `
    <div style="min-width: 150px">
      <div style="font-weight: 700; color: #172019">${category === "None" ? "None" : category}</div>
      <div style="font-size: 12px; color: #5c685f; margin-top: 2px">${label}</div>
      ${
        pdryPct || pdry
          ? `<div style="font-size: 12px; color: #172019; margin-top: 6px">ET-SM dryness percentile: ${pdryPct ?? pdry}</div>`
          : ""
      }
    </div>
  `;
}

function featureSummary(properties: Record<string, unknown> | null, layer: ForecastLayer): string {
  if (layer !== "drought") {
    const value = typeof properties?.value === "number" ? properties.value : undefined;
    const units = typeof properties?.units === "string" ? properties.units : layer === "et" ? "mm/day" : "m3/m3";
    const label = layer === "et" ? "evapotranspiration" : "soil moisture";
    return `Map feature selected: ${label}, ${value === undefined ? "no data" : `${value.toFixed(layer === "et" ? 3 : 4)} ${units}`}.`;
  }

  const category = normalizeDroughtCategory(properties?.droughtCategory ?? properties?.category_label ?? properties?.category);
  const label = droughtLabelForCategory(category);
  const pdry = typeof properties?.pdry === "number" ? `${(properties.pdry * 100).toFixed(1)}%` : undefined;
  const pdryPct = typeof properties?.pdry_pct === "number" ? `${properties.pdry_pct.toFixed(1)}%` : undefined;
  return `Map feature selected: ${category === "None" ? "None" : category}, ${label}${
    pdryPct || pdry ? `, ET-SM dryness percentile ${pdryPct ?? pdry}` : ""
  }.`;
}

export default function DroughtMap({
  geojson,
  layer = "drought",
  isLoading = false,
  error,
  selectedPoint,
  emptyMessage = "Run the selected forecast to load this layer."
}: DroughtMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const maplibreRef = useRef<MapLibreModule | null>(null);
  const layerRef = useRef<ForecastLayer>(layer);
  const fitKeyRef = useRef<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [inspectedFeature, setInspectedFeature] = useState<string | null>(null);

  const normalizedGeoJson = useMemo<GeoJsonFeatureCollection>(() => {
    if (!geojson) {
      return EMPTY_FEATURE_COLLECTION;
    }

    return {
      ...geojson,
      features: geojson.features.map((feature) => normalizeFeature(feature, layer))
    };
  }, [geojson, layer]);

  useEffect(() => {
    layerRef.current = layer;
  }, [layer]);

  useEffect(() => {
    let cancelled = false;
    let resizeObserver: ResizeObserver | null = null;

    async function createMap() {
      if (!containerRef.current || mapRef.current) {
        return;
      }

      try {
        const maplibre = await import("maplibre-gl");
        if (cancelled || !containerRef.current) {
          return;
        }

        maplibreRef.current = maplibre;
        const map = new maplibre.Map({
          container: containerRef.current,
          style: BASEMAP_STYLE,
          bounds: [[-107.4, 25.7], [-94.5, 44.1]],
          fitBoundsOptions: { padding: 28 },
          attributionControl: false
        });

        mapRef.current = map;
        map.addControl(new maplibre.NavigationControl({ showCompass: false }), "top-right");
        map.addControl(new maplibre.AttributionControl({ compact: true }), "bottom-right");

        resizeObserver = new ResizeObserver(() => map.resize());
        resizeObserver.observe(containerRef.current);

        const handleClick = (event: MapLayerMouseEvent) => {
          const feature = event.features?.[0];
          if (!feature) {
            return;
          }
          const properties = (feature.properties as Record<string, unknown> | null) ?? null;
          setInspectedFeature(featureSummary(properties, layerRef.current));
          new maplibre.Popup({ closeButton: true, closeOnClick: true })
            .setLngLat(event.lngLat)
            .setHTML(popupHtml(properties, layerRef.current))
            .addTo(map);
        };

        map.on("load", () => {
          enhanceBasemapLegibility(map);
          ensureForecastLayers(map);
          ["forecast-layer-fill", "forecast-drought-pattern", "forecast-layer-points"].forEach((layerId) => {
            map.on("click", layerId, handleClick);
            map.on("mouseenter", layerId, () => {
              map.getCanvas().style.cursor = "pointer";
            });
            map.on("mouseleave", layerId, () => {
              map.getCanvas().style.cursor = "";
            });
          });
          setIsReady(true);
        });
      } catch (caught) {
        setMapError(caught instanceof Error ? caught.message : "Map failed to initialize");
      }
    }

    createMap();

    return () => {
      cancelled = true;
      resizeObserver?.disconnect();
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isReady) {
      return;
    }

    ensureForecastLayers(map);
    setGeoJsonSourceData(map, SOURCE_ID, normalizedGeoJson);

    const bounds = featureCollectionBounds(normalizedGeoJson);
    const fitKey = bounds ? bounds.flat().join(",") : null;
    if (bounds && fitKey !== fitKeyRef.current) {
      fitKeyRef.current = fitKey;
      map.fitBounds(bounds, {
        padding: { top: 40, right: 40, bottom: window.innerWidth < 640 ? 76 : 40, left: 40 },
        maxZoom: 8,
        duration: 700
      });
    }
  }, [isReady, normalizedGeoJson]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isReady) {
      return;
    }

    ensureForecastLayers(map);
    const hasPoint = selectedPoint?.gridLat !== undefined && selectedPoint?.gridLon !== undefined;
    const pointGeoJson: GeoJsonFeatureCollection = hasPoint
      ? {
          type: "FeatureCollection",
          features: [
            {
              type: "Feature",
              geometry: {
                type: "Point",
                coordinates: [selectedPoint.gridLon as number, selectedPoint.gridLat as number]
              },
              properties: {}
            }
          ]
        }
      : EMPTY_FEATURE_COLLECTION;

    setGeoJsonSourceData(map, CHECK_SOURCE_ID, pointGeoJson);
  }, [isReady, selectedPoint]);

  const empty = !geojson || geojson.features.length === 0;
  const selectedPointCategory = normalizeDroughtCategory(selectedPoint?.categoryLabel ?? selectedPoint?.category);
  const selectedPointSummary =
    selectedPoint?.gridLat !== undefined && selectedPoint?.gridLon !== undefined
      ? `Coordinate marker selected at grid ${selectedPoint.gridLat.toFixed(3)}, ${selectedPoint.gridLon.toFixed(3)} with category ${selectedPointCategory}, ${droughtLabelForCategory(selectedPointCategory)}.`
      : null;

  return (
    <section
      className="relative h-[420px] min-h-0 overflow-hidden bg-[#e9eee7] sm:h-[480px] xl:h-full"
      aria-label={layer === "drought" ? "ET and soil-moisture-derived drought map" : layer === "et" ? "Evapotranspiration endpoint map" : "Soil-moisture endpoint map"}
      data-map-ready={isReady ? "true" : "false"}
    >
      <div ref={containerRef} className="absolute inset-0" />
      <div className="sr-only" role="status" aria-live="polite">
        {inspectedFeature ?? selectedPointSummary ?? (empty ? "No forecast map data loaded." : "Forecast map data loaded.")}
      </div>

      {isLoading ? (
        <div
          className="absolute left-4 top-4 rounded-md border border-accent/25 bg-white px-3 py-2 text-sm font-medium text-accent-strong shadow-panel"
          role="status"
          aria-live="polite"
        >
          Loading map data
        </div>
      ) : null}

      {empty && !isLoading ? (
        <div className="absolute left-4 right-16 top-4 max-w-xs rounded-md border border-line bg-white px-3 py-2 text-sm text-muted shadow-panel sm:right-auto">
          {emptyMessage}
        </div>
      ) : null}

      {error || mapError ? (
        <div
          className="absolute bottom-4 left-4 right-4 rounded-md border border-danger/30 bg-[#fff2f0]/95 px-3 py-2 text-sm font-medium text-danger shadow-panel"
          role="alert"
        >
          {error ?? mapError}
        </div>
      ) : null}
    </section>
  );
}

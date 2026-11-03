/* Seyd-Yaar georeferencing — one grid convention for raster, markers and popups.
 * The BIN is row-major, row 0 north, and grid extrema are CELL CENTRES (EPSG:4326).
 * Leaflet's EPSG:3857 display requires an explicit latitude-to-Mercator resample.
 */
const SeydGeo = (() => {
  const RAD = Math.PI / 180;
  const MAX_MERCATOR_LAT = 85.0511287798066;

  function validateGrid(g) {
    if (!g || !Number.isInteger(g.width) || !Number.isInteger(g.height) ||
        g.width < 2 || g.height < 2 ||
        ![g.lon_min, g.lon_max, g.lat_min, g.lat_max].every(Number.isFinite) ||
        g.lon_max <= g.lon_min || g.lat_max <= g.lat_min ||
        Math.abs(g.lat_min) >= MAX_MERCATOR_LAT || Math.abs(g.lat_max) >= MAX_MERCATOR_LAT) {
      throw new Error('Invalid geographic grid: expected W,H >= 2 and EPSG:4326 cell-center bounds within Mercator limits');
    }
    return g;
  }

  function spacing(g) {
    validateGrid(g);
    return {dx: (g.lon_max - g.lon_min) / (g.width - 1),
            dy: (g.lat_max - g.lat_min) / (g.height - 1)};
  }

  function bounds(g, latOffset = 0, lonOffset = 0) {
    const {dx, dy} = spacing(g);
    const south = g.lat_min - dy/2 + latOffset;
    const north = g.lat_max + dy/2 + latOffset;
    if (Math.abs(south) >= MAX_MERCATOR_LAT || Math.abs(north) >= MAX_MERCATOR_LAT) {
      throw new Error('Grid edges exceed the Web Mercator latitude domain');
    }
    return [[south, g.lon_min - dx/2 + lonOffset],
            [north, g.lon_max + dx/2 + lonOffset]];
  }

  function xyToLatLon(g, x, y, northFirst = true, latOffset = 0, lonOffset = 0) {
    const {dx, dy} = spacing(g);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    const xx = Math.max(0, Math.min(g.width - 1, x));
    const yy = Math.max(0, Math.min(g.height - 1, y));
    const yNorth = northFirst ? yy : g.height - 1 - yy;
    return {lat: g.lat_max - yNorth * dy + latOffset,
            lon: g.lon_min + xx * dx + lonOffset};
  }

  function latLonToXY(g, lat, lon, northFirst = true, latOffset = 0, lonOffset = 0) {
    const {dx, dy} = spacing(g);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    const latBase = lat - latOffset, lonBase = lon - lonOffset;
    const eps = 1e-10;
    if (latBase < g.lat_min - dy/2 - eps || latBase > g.lat_max + dy/2 + eps ||
        lonBase < g.lon_min - dx/2 - eps || lonBase > g.lon_max + dx/2 + eps) return null;
    const x = Math.max(0, Math.min(g.width - 1,
      Math.floor((lonBase - (g.lon_min - dx/2)) / dx)));
    const yNorth = Math.max(0, Math.min(g.height - 1,
      Math.floor(((g.lat_max + dy/2) - latBase) / dy)));
    const y = northFirst ? yNorth : g.height - 1 - yNorth;
    return {x, y, i: y * g.width + x};
  }

  function mercatorY(lat) {
    const r = Math.max(-MAX_MERCATOR_LAT, Math.min(MAX_MERCATOR_LAT, lat)) * RAD;
    return Math.asinh(Math.tan(r));
  }

  function inverseMercatorY(y) {
    return Math.atan(Math.sinh(y)) / RAD;
  }

  // Destination PNG is uniformly sampled in EPSG:3857, not latitude. Nearest
  // grid-cell sampling matches latLonToXY, including near islands and coastlines.
  function rasterLookups(g, rasterWidth, rasterHeight, northFirst = true,
                         latOffset = 0, lonOffset = 0) {
    if (!(rasterWidth > 0 && rasterHeight > 0)) throw new Error('Invalid PNG size');
    const b = bounds(g, latOffset, lonOffset);
    const northY = mercatorY(b[1][0]);
    const southY = mercatorY(b[0][0]);
    const xIndices = new Int32Array(rasterWidth);
    const yIndices = new Int32Array(rasterHeight);
    for (let x = 0; x < rasterWidth; x++) {
      const lon = b[0][1] + ((x + 0.5) / rasterWidth) * (b[1][1] - b[0][1]);
      const p = latLonToXY(g, g.lat_max + latOffset, lon, northFirst, latOffset, lonOffset);
      xIndices[x] = p ? p.x : -1;
    }
    for (let y = 0; y < rasterHeight; y++) {
      const my = northY + ((y + 0.5) / rasterHeight) * (southY - northY);
      const lat = inverseMercatorY(my);
      const p = latLonToXY(g, lat, g.lon_min + lonOffset, northFirst, latOffset, lonOffset);
      yIndices[y] = p ? p.y : -1;
    }
    return {xIndices, yIndices, bounds: b};
  }

  return {validateGrid, spacing, bounds, xyToLatLon, latLonToXY,
          mercatorY, inverseMercatorY, rasterLookups};
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SeydGeo;

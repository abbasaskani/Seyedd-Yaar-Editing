/* Run: node tools/test_georef.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const geo = require('../docs/georef.js');
const root = path.resolve(__dirname, '..');
const raw = JSON.parse(fs.readFileSync(path.join(root, 'backend/config/aoi.geojson')));
const ring = raw.features[0].geometry.coordinates[0];
const lons = ring.map(p=>p[0]), lats = ring.map(p=>p[1]);
const g = {
  lon_min: Math.min(...lons), lon_max: Math.max(...lons),
  lat_min: Math.min(...lats), lat_max: Math.max(...lats),
  width: 220, height: 220
};
const {dx,dy} = geo.spacing(g);
let passed = 0;
function check(c, message) { assert.ok(c, message); passed++; }
const b=geo.bounds(g);
check(Math.abs(b[0][0]-(g.lat_min-dy/2))<1e-12, 'true south edge');
check(Math.abs(b[1][1]-(g.lon_max+dx/2))<1e-12, 'true east edge');
for (const northFirst of [true, false]) {
  for (let y=0; y<g.height; y++) {
    for (let x=0; x<g.width; x++) {
      const ll=geo.xyToLatLon(g,x,y,northFirst);
      const xy=geo.latLonToXY(g,ll.lat,ll.lon,northFirst);
      check(xy?.x===x&&xy?.y===y, `cell centres roundtrip (${northFirst}) ${x},${y}`);
    }
  }
  check(geo.latLonToXY(g,g.lat_max+dy/2+1e-4, 54,northFirst)===null, 'out of bounds');
  check(geo.latLonToXY(g,g.lat_min-dy/2-1e-4, 54,northFirst)===null, 'out of bounds south');
}
for (const [lat,lon] of [[12.5,54],[0,42],[-6,80],[20,60]]) {
  check(Math.abs(geo.inverseMercatorY(geo.mercatorY(lat))-lat)<1e-12,'Mercator inverse');
}
const ratio=4;
const W=g.width*ratio, H=g.height*ratio;
const look = geo.rasterLookups(g,W,H);
let matching=0, compared=0;
for (let k=0;k<20000;k++) {
  const xx=((k*1009)%W), yy=((k*2003)%H);
  const lat=geo.inverseMercatorY(geo.mercatorY(b[1][0]) + ((yy+.5)/H)*(geo.mercatorY(b[0][0])-geo.mercatorY(b[1][0])));
  const lon=b[0][1]+((xx+.5)/W)*(b[1][1]-b[0][1]);
  const ix=geo.latLonToXY(g,lat,lon);
  const sampleY=look.yIndices[yy], sampleX=look.xIndices[xx];
  compared++;
  if(ix.x===sampleX && ix.y===sampleY) matching++;
}
check(matching===compared,`Mercator PNG samples match popup: ${matching}/${compared}`);
const first=g.width*8+11;
const ll=geo.xyToLatLon(g,11,8,true,0.2,-0.01);
check(geo.latLonToXY(g,ll.lat,ll.lon,true,0.2,-0.01).i===first,'diagnostic offset consistent');
const html=fs.readFileSync(path.join(root,'docs/app.html'),'utf8');
check(html.indexOf('id="timeSlider"')<html.indexOf('src="app.js"'),'slider exists before script');
check(html.indexOf('src="georef.js"')<html.indexOf('src="app.js"'),'georef loads first');
const app = fs.readFileSync(path.join(root,'docs/app.js'),'utf8');
check(app.includes('state.lastComputed.arrShown = arrShown'),'popup/display share filtered array');
check(app.includes('timeIsos = displayed?.timeIsos'),'popup uses displayed time');
check(app.includes('SeydGeo.rasterLookups'),'Mercator reprojection active');
check(!app.includes('applyManualOffsets(-0.35'),'old offset cleared');
console.log(`PASS ${passed} assertions; ${matching}/${compared} PNG pixels matched popup sampling; cell size ~${dy.toFixed(4)}° lat`);

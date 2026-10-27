#!/usr/bin/env node
// Backtest for the buck model in deercam/web/static/predict.js. Internal only.
//
// Rolling origin: for every buck with at least 3 visits, and every visit after his 2nd, train on what was known
// just after his previous visit (his earlier visits, the weather and camera uptime up to then) and predict the
// next 7 days. Score:
//   top-5 hit  did a real visit land in one of the 5 best "windows" (a camera × a part of one day: dawn,
//              daytime, dusk, early night, late night)? That's what a hunter actually picks.
//   rank       where the real visit's window ranked among all windows (0% = top, 100% = last).
//   Brier      mean squared error of the hourly chances (lower is better), and skill against the null model
//              (> 0 means better than "same rate every camera-hour").
//
//   node tools/backtest.js [base URL, default http://localhost:8601]   (the local deerdar.com mirror works)
//   COOKIE='session=...' node tools/backtest.js https://deerdar.com
const M = require('../deercam/web/static/predict.js');
const BASE = process.argv[2] || 'http://localhost:8601', HORIZON = 7 * M.DAY, MIN_VISITS = 3;
const get = p => fetch(BASE + p, { headers: process.env.COOKIE ? { cookie: process.env.COOKIE } : {} }).then(r => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });

// The models compared: the same machinery with different conditions switched on.
const MODELS = {
  'null (base rate only)': [],
  'place + time': ['cam', 'sun'],
  'place + time + weather': null,  // everything, with the default weights
  'old page (% match)': 'old',     // what the page did before: his raw share of each condition, no exposure, no place
};
// The previous page's score for an hour: his raw share of the hour's option in each condition, averaged.
function oldScore(facts, f) {
  let s = 0, n = 0;
  for (const k of M.FACTORS) { if (k === 'cam' || !f[k]) continue; const known = facts.filter(x => x[k]); if (!known.length) continue; s += known.filter(x => x[k] === f[k]).length / known.length; n++; }
  return n ? s / n : 0;
}

async function main() {
  const [bucks, tl] = await Promise.all([get('/api/bucks?season=all'), get('/api/timeline?days=365')]);
  const cams = Object.fromEntries(tl.cameras.map(c => [c.camera_id, c])), ids = tl.cameras.map(c => c.camera_id);
  const placed = tl.cameras.filter(c => c.lat != null);
  const where = { lat: placed.reduce((a, c) => a + c.lat, 0) / placed.length, lon: placed.reduce((a, c) => a + c.lon, 0) / placed.length };
  const tzMin = (tl.offsets || []).at(-1)?.[1] ?? -300;
  const byCam = {}; for (const [t, ci] of tl.photos) (byCam[ids[ci]] ??= []).push(t);
  const up = M.uptime(byCam);
  const first = Math.min(...tl.photos.map(p => p[0])), pastDays = Math.min(92, Math.ceil((Date.now() - first) / M.DAY) + 2);
  const om = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${where.lat.toFixed(3)}&longitude=${where.lon.toFixed(3)}&hourly=temperature_2m,wind_direction_10m,wind_speed_10m,pressure_msl&temperature_unit=fahrenheit&wind_speed_unit=mph&timeformat=unixtime&timezone=GMT&past_days=${pastDays}&forecast_days=1`).then(r => r.json());
  const hs = om.hourly.time.map((t, i) => ({ t: t * 1000, temp: om.hourly.temperature_2m[i], deg: om.hourly.wind_direction_10m[i], mph: om.hourly.wind_speed_10m[i], p: om.hourly.pressure_msl[i] * 0.02953 }));
  const ctx = { hs, up, cams, where, tzMin, legalMin: 30 };

  const named = bucks.filter(b => !b.generic);
  const cases = [];
  for (const b of named) {
    const pat = await get(`/api/bucks/${b.id}/patterns?season=all`);
    const vs = (pat.recent || []).map(v => ({ ...v, t: Date.parse(v.start) })).filter(v => v.t >= hs[0].t).sort((a, c) => a.t - c.t);
    if (vs.length < MIN_VISITS) continue;
    for (let k = 2; k < vs.length; k++) cases.push({ buck: b.name, train: vs.slice(0, k), test: vs.slice(k), cutoff: vs[k - 1].t + M.HOUR });
  }
  console.log(`${named.length} named bucks, ${cases.length} test cases from bucks with ${MIN_VISITS}+ visits; weather ${new Date(hs[0].t).toISOString().slice(0, 10)} to now, ${Object.keys(up).length} cameras\n`);
  if (!cases.length) return console.log('Not enough visits to backtest yet.');

  const rows = [];
  for (const [name, only] of Object.entries(MODELS)) {
    let hits = 0, rankSum = 0, ranked = 0, brier = 0, brierNull = 0, slots = 0;
    for (const c of cases) {
      const train = { ...ctx, until: c.cutoff };
      const model = M.fit(c.train, train);
      if (Array.isArray(only)) for (const k of Object.keys(model.prefs)) if (!only.includes(k)) delete model.prefs[k];
      const t1 = c.cutoff + HORIZON, preds = M.predict(model, ctx, c.cutoff, t1).filter(p => M.isUp(up[p.cam], p.t));
      if (only === 'old') for (const p of preds) { p.lam = oldScore(model.facts, p.f); p.p = null; }
      const hit = p => c.test.some(v => v.camera_id === p.cam && v.t >= p.t && v.t < p.t + M.HOUR);
      if (only !== 'old') for (const p of preds) { const y = hit(p) ? 1 : 0; brier += (p.p - y) ** 2; brierNull += (1 - Math.exp(-model.rate) - y) ** 2; slots++; }
      // windows: camera × part of a day
      const win = {};
      for (const p of preds) { const day = Math.floor((p.t + tzMin * 60e3) / M.DAY), key = `${p.cam}|${day}|${p.f.sun}`; (win[key] ??= { lam: 0, hit: false }).lam += p.lam; if (hit(p)) win[key].hit = true; }
      const order = Object.values(win).sort((a, b) => b.lam - a.lam), at = order.findIndex(w => w.hit);
      if (at < 0) continue;  // the next visit was more than 7 days out
      hits += at < 5 ? 1 : 0; rankSum += at / Math.max(1, order.length - 1); ranked++;
    }
    rows.push({ name, hit5: ranked ? hits / ranked : 0, rank: ranked ? rankSum / ranked : 0, ranked, brier: slots ? brier / slots : NaN, skill: slots ? 1 - brier / brierNull : NaN });
  }
  const pct = v => `${(100 * v).toFixed(0)}%`.padStart(5);
  console.log('model'.padEnd(26), 'top-5 hit', ' avg rank', '   Brier', '  skill vs null', ' cases');
  for (const r of rows) console.log(r.name.padEnd(26), pct(r.hit5).padStart(9), pct(r.rank).padStart(9), (isNaN(r.brier) ? '—' : r.brier.toFixed(5)).padStart(8), (isNaN(r.skill) ? '—' : pct(r.skill)).padStart(15), String(r.ranked).padStart(6));
  console.log('\ntop-5 hit: higher is better · avg rank: lower is better (50% = coin flip) · skill: above 0% beats the null model');
}
main().catch(e => { console.error(e); process.exit(1); });

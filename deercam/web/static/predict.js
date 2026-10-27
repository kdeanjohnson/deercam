// Deerdar's buck model, shared by the Predictability page and the backtest (tools/backtest.js), so what the
// backtest scores is exactly what the page shows. Plain functions, no DOM.
//
// The idea: a buck's visits are rare events at a camera. For every camera-hour we know the conditions (time of
// day against the sun, wind, temperature, pressure, moon) and whether the camera was up. A buck's preference
// for a condition is how much more of his visits fall in it than the camera-hours do (exposure), pulled toward
// "no preference" when there are few visits. Predicting an hour at a camera multiplies his base rate by his
// preferences for that camera, that time of day, and (more softly) that weather.
(function (root) {
  const HOUR = 3600e3, DAY = 864e5, rad = d => d * Math.PI / 180, deg = r => r * 180 / Math.PI;

  // ------------------------------------------------------------------ sun and moon
  function sunTimes(ms, lat, lon) {
    const local = new Date(ms + lon / 15 * HOUR);  // local solar day, as in sun.light()
    const y = local.getUTCFullYear(), m = local.getUTCMonth(), d = local.getUTCDate();
    const n = Math.round((Date.UTC(y, m, d) - Date.UTC(y, 0, 1)) / DAY) + 1;
    const g = 2 * Math.PI / 365 * (n - 1);
    const eq = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g) - 0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));
    const decl = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g)
               + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
    const phi = rad(lat);
    const cosHa = Math.cos(rad(90.833)) / (Math.cos(phi) * Math.cos(decl)) - Math.tan(phi) * Math.tan(decl);
    if (cosHa < -1 || cosHa > 1) return null;
    const ha = deg(Math.acos(cosHa)), mid = Date.UTC(y, m, d);
    return { rise: mid + (720 - 4 * (lon + ha) - eq) * 60e3, set: mid + (720 - 4 * (lon - ha) - eq) * 60e3 };
  }
  function sunPos(ms, lat, lon) {
    const d = ms / DAY - 10957.5;  // days since J2000
    const g = rad(357.529 + 0.98560028 * d), q = 280.459 + 0.98564736 * d;
    const L = rad(q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)), e = rad(23.439 - 0.00000036 * d);
    const ra = Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L)), dec = Math.asin(Math.sin(e) * Math.sin(L));
    const ha = rad(((18.697374558 + 24.06570982441908 * d) % 24) * 15 + lon) - ra, phi = rad(lat);
    const alt = Math.asin(Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(ha));
    const az = Math.atan2(-Math.sin(ha), Math.tan(dec) * Math.cos(phi) - Math.sin(phi) * Math.cos(ha));
    return { alt: deg(alt), az: (deg(az) + 360) % 360 };
  }
  const SYNODIC = 29.530588853, NEW_MOON = Date.UTC(2000, 0, 6, 18, 14);
  const MOON8 = ['New moon', 'Waxing crescent', 'First quarter', 'Waxing gibbous', 'Full moon', 'Waning gibbous', 'Last quarter', 'Waning crescent'];
  function moonAt(ms) {
    const age = (((ms - NEW_MOON) / DAY) % SYNODIC + SYNODIC) % SYNODIC;
    const illum = (1 - Math.cos(2 * Math.PI * age / SYNODIC)) / 2, waxing = age < SYNODIC / 2;
    return { illum, waxing, name: MOON8[Math.floor((age / SYNODIC) * 8 + 0.5) % 8] };
  }

  // ------------------------------------------------------------------ conditions of one hour
  const W8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const SUN = { dawn: 'Dawn (±1 h sunrise)', dusk: 'Dusk (±1 h sunset)', day: 'Daytime', early: 'Early night', late: 'Late night' };
  // Time of day for a moment at a place. `tzMin` is the local clock offset, for splitting the night at midnight.
  function sunBucket(t, lat, lon, tzMin) {
    const s = sunTimes(t, lat, lon);
    if (!s) return null;
    const mr = (t - s.rise) / 60e3, ms = (t - s.set) / 60e3;
    if (Math.abs(mr) <= 60) return SUN.dawn;
    if (Math.abs(ms) <= 60) return SUN.dusk;
    if (mr > 0 && ms < 0) return SUN.day;
    return new Date(t + tzMin * 60e3).getUTCHours() >= 12 ? SUN.early : SUN.late;
  }
  function isLegal(t, lat, lon, legalMin) {
    const s = sunTimes(t, lat, lon);
    return !!s && t >= s.rise - legalMin * 60e3 && t <= s.set + legalMin * 60e3;
  }
  // `hs` is the hourly weather series [{t, temp (°F), deg (wind from), mph, p (inHg)}], one entry per hour.
  function hourFacts(hs, i) {
    const h = hs[i], at = k => hs[i - k], f = {};
    if (h.deg != null) f.wdir = W8[Math.round(h.deg / 45) % 8];
    if (h.mph != null) f.wspd = h.mph < 3 ? 'Calm (<3 mph)' : h.mph < 8 ? 'Light (3–7)' : h.mph < 15 ? 'Breezy (8–14)' : 'Windy (15+)';
    if (at(6) && h.deg != null) { const d = Math.abs(((h.deg - at(6).deg) % 360 + 540) % 360 - 180); f.wshift = d < 45 ? 'Held steady' : d < 90 ? 'Shifted 45–90°' : 'Swung 90°+'; }
    if (h.temp != null) f.temp = `${Math.floor(h.temp / 10) * 10}s °F`;
    if (at(24)) { const d = h.temp - at(24).temp; f.tdep = d <= -5 ? 'Colder (5°+)' : d >= 5 ? 'Warmer (5°+)' : 'About the same'; }
    if (at(3)) { const d = h.p - at(3).p; f.ptrend = d >= 0.02 ? 'Rising' : d <= -0.02 ? 'Falling' : 'Steady'; }
    if (at(24)) { const d = h.p - at(24).p; f.pdep = d <= -0.05 ? 'Lower (0.05+)' : d >= 0.05 ? 'Higher (0.05+)' : 'About the same'; }
    f.moon = MOON8.find(n => n.toLowerCase() === moonAt(h.t).name.toLowerCase());
    return f;
  }
  // The series index of the hour containing t (the series is hourly and sorted), or -1.
  function hourIndex(hs, t) {
    if (!hs.length) return -1;
    const i = Math.floor((t - hs[0].t) / HOUR);
    return i >= 0 && i < hs.length ? i : -1;
  }

  // ------------------------------------------------------------------ when each camera was up
  // A camera is up from its first photo to its last, except stretches of more than DOWN_GAP with no photo at
  // all (dead battery, full card, pulled). Returns {cameraId: [[from, to], ...]}.
  const DOWN_GAP = 48 * HOUR;
  function uptime(photoTimesByCam) {
    const out = {};
    for (const [cam, ts] of Object.entries(photoTimesByCam)) {
      const s = ts.slice().sort((a, b) => a - b), spans = [];
      if (!s.length) continue;
      let a = s[0], z = s[0];
      for (const t of s.slice(1)) { if (t - z > DOWN_GAP) { spans.push([a, z + HOUR]); a = t; } z = t; }
      spans.push([a, z + HOUR]);
      out[cam] = spans;
    }
    return out;
  }
  const isUp = (spans, t) => !!spans && spans.some(([a, z]) => t >= a && t < z);

  // ------------------------------------------------------------------ the model
  // How strongly each kind of condition counts in a prediction (an exponent on the preference: 1 = full,
  // 0 = ignored). Place and time of day lead; weather is softer; moon barely counts. Starting values, to be
  // tuned with the backtest.
  const WEIGHTS = { cam: 1, sun: 1, temp: .5, tdep: .5, ptrend: .4, pdep: .3, wdir: .3, wshift: .2, wspd: .3, moon: .15 };
  // How many visits' worth of "no preference" each preference starts from: the more, the more visits it takes
  // to move it. Time of day and place learn fastest; weather needs more evidence.
  const PRIOR = { cam: 2, sun: 2, default: 6 };
  const FACTORS = ['sun', 'cam', 'wdir', 'wshift', 'temp', 'tdep', 'ptrend', 'pdep', 'moon'];

  // Everything the model needs from the property: the hourly weather, when each camera was up, and where each
  // camera is. `until` limits it all to the past (the backtest trains on what was known then).
  function exposure({ hs, up, cams, where, tzMin, until = Infinity }) {
    const counts = {}, total = { n: 0 }, byCam = {};
    for (let i = 0; i < hs.length; i++) {
      const t = hs[i].t; if (t >= until) break;
      const f = hourFacts(hs, i);
      for (const c of Object.keys(up)) {
        if (!isUp(up[c], t)) continue;
        const at = cams[c]?.lat != null ? cams[c] : where;
        const ff = { ...f, sun: sunBucket(t + HOUR / 2, at.lat, at.lon, tzMin), cam: c };
        total.n++; byCam[c] = (byCam[c] || 0) + 1;
        for (const k of FACTORS) { const v = ff[k]; if (!v) continue; (counts[k] ??= { n: 0 })[v] = (counts[k][v] || 0) + 1; counts[k].n++; }
      }
    }
    return { counts, hours: total.n, byCam };
  }
  // A visit's conditions: the weather of its hour, time of day at its camera's exact moment, the camera.
  function visitFacts(v, { hs, cams, where, tzMin }) {
    const i = hourIndex(hs, v.t), at = cams[v.camera_id]?.lat != null ? cams[v.camera_id] : where;
    return { ...(i >= 0 ? hourFacts(hs, i) : {}), sun: at ? sunBucket(v.t, at.lat, at.lon, tzMin) : null, cam: v.camera_id };
  }
  // A buck's preferences: for each kind of condition and each option, how many times more likely his visits
  // are to fall in it than the camera-hours are, after starting from PRIOR visits of no preference.
  function preferences(facts, exp) {
    const out = {};
    for (const k of FACTORS) {
      const base = exp.counts[k]; if (!base?.n) continue;
      const known = facts.filter(f => f[k]); const n = known.length, a = PRIOR[k] ?? PRIOR.default;
      const opts = {};
      for (const [opt, e] of Object.entries(base)) {
        if (opt === 'n') continue;
        const b = e / base.n, c = known.filter(f => f[k] === opt).length;
        const post = (c + a * b) / (n + a);
        opts[opt] = { visits: c, of: n, usual: b, share: n ? c / n : 0, post, pref: post / b };
      }
      out[k] = opts;
    }
    return out;
  }
  // Fit a buck: his preferences and his base rate (visits per camera-hour).
  function fit(visits, ctx, exp = exposure(ctx)) {  // pass `exp` to share one exposure across bucks
    const facts = visits.map(v => visitFacts(v, ctx));
    return { exp, facts, prefs: preferences(facts, exp), rate: exp.hours ? visits.length / exp.hours : 0, n: visits.length };
  }
  // Expected visits in one hour at one camera, given that hour's conditions.
  function rate(model, f) {
    let r = model.rate;
    for (const k of FACTORS) {
      const p = model.prefs[k]?.[f[k]]?.pref; if (p == null) continue;
      r *= Math.pow(p, WEIGHTS[k] ?? .3);
    }
    return r;
  }
  // Every (camera, hour) from t0 to t1: the expected visits, the chance of at least one, and the conditions.
  function predict(model, ctx, t0, t1) {
    const out = [];
    for (let i = Math.max(0, hourIndex(ctx.hs, t0)); i < ctx.hs.length && ctx.hs[i].t < t1; i++) {
      const t = ctx.hs[i].t, f = hourFacts(ctx.hs, i);
      for (const c of Object.keys(ctx.cams)) {
        const at = ctx.cams[c]?.lat != null ? ctx.cams[c] : ctx.where;
        const ff = { ...f, sun: sunBucket(t + HOUR / 2, at.lat, at.lon, ctx.tzMin), cam: c };
        const lam = rate(model, ff);
        out.push({ t, cam: c, f: ff, lam, p: 1 - Math.exp(-lam), legal: isLegal(t + HOUR / 2, at.lat, at.lon, ctx.legalMin ?? 30) });
      }
    }
    return out;
  }
  // His leaning in one kind of condition: the option that rises most above how often it comes up (so one visit
  // on a rare wind doesn't beat five on a common one), and whether he favors it clearly (at least 1.5× usual,
  // after the pull toward no preference, and on 2+ visits).
  function favorite(model, k) {
    const opts = model.prefs[k]; if (!opts) return null;
    const best = Object.entries(opts).sort((a, b) => (b[1].post - b[1].usual) - (a[1].post - a[1].usual))[0];
    return best ? { key: best[0], ...best[1], clear: best[1].pref >= 1.5 && best[1].visits >= 2 } : null;
  }
  // Daylight ratio: share of his visits in legal light.
  function daylight(visits) {
    if (!visits.length) return null;
    const d = visits.filter(v => v.light && v.light !== 'night').length / visits.length;
    return { ratio: d, label: d < .35 ? 'Mostly nocturnal' : d < .65 ? 'Day and night' : 'Daylight mover' };
  }

  const api = { HOUR, DAY, W8, SUN, MOON8, FACTORS, WEIGHTS, PRIOR, DOWN_GAP, sunTimes, sunPos, moonAt, sunBucket, isLegal,
                hourFacts, hourIndex, uptime, isUp, exposure, visitFacts, preferences, fit, rate, predict, favorite, daylight };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DeerModel = api;
})(typeof self !== 'undefined' ? self : this);

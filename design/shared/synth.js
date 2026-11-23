/* SYNTHETIC data for design prototypes only. Not a real person, not a real result.
   The shapes mirror what /api/session returns (findings, reps, timelines) so a chosen concept
   can be wired to the real response without changing the layout. */
(function () {
  const D2R = Math.PI / 180, DUR = 12;
  const REPS = [{start_s: 0, end_s: 4, peak: 22}, {start_s: 4, end_s: 8, peak: 11}, {start_s: 8, end_s: 12, peak: 21}];
  const ease = x => x * x * (3 - 2 * x);
  const k = u => u < 0.6 ? Math.pow(Math.sin(Math.PI / 2 * u / 0.6), 2) : Math.pow(Math.cos(Math.PI / 2 * (u - 0.6) / 0.4), 2);  // slow rise, quicker return
  const rot = (a, x, y) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];
  const add = (p, q) => [p[0] + q[0], p[1] + q[1]];

  function repAt(t) { const tt = Math.min(Math.max(t, 0), DUR - 1e-6), i = Math.min(2, Math.floor(tt / 4)); return {i, r: REPS[i], u: (tt - REPS[i].start_s) / 4}; }
  function armUp(u) { return u < 0.6 ? ease(u / 0.6) : 1 - ease((u - 0.6) / 0.4); }   // the arm travels with the bend, so poses within one repetition differ

  // Body units: hip centre = (0,0), torso length = 1, y grows downwards. Person faces the camera.
  function pose(t) {
    const {r, u} = repAt(t), a = -r.peak * k(u) * D2R, up = armUp(u);
    const Sc = add([0, 0], rot(a, 0, -1)), LS = add(Sc, rot(a, 0.27, 0)), RS = add(Sc, rot(a, -0.27, 0));
    const th = up * 168 * D2R, th2 = th + up * 20 * D2R;
    const LE = add(LS, rot(a, 0.55 * Math.sin(th), 0.55 * Math.cos(th)));
    const LW = add(LE, rot(a, 0.5 * Math.sin(th2), 0.5 * Math.cos(th2)));
    const RE = add(RS, [-0.55 * Math.sin(8 * D2R), 0.55 * Math.cos(8 * D2R)]);
    const RW = add(RE, [-0.5 * Math.sin(3 * D2R), 0.5 * Math.cos(3 * D2R)]);
    return {a, lean: Math.abs(a) / D2R, up, Sc, head: add(Sc, rot(a * 0.7, 0, -0.34)), LS, RS, LE, LW, RE, RW,
      LH: [0.19, 0], RH: [-0.19, 0], LK: [0.2, 0.78], RK: [-0.2, 0.78], LA: [0.21, 1.55], RA: [-0.21, 1.55]};
  }
  const EDGES = [["LS", "LE"], ["LE", "LW"], ["RS", "RE"], ["RE", "RW"], ["LS", "RS"], ["LS", "LH"], ["RS", "RH"], ["LH", "RH"],
    ["LH", "LK"], ["LK", "LA"], ["RH", "RK"], ["RK", "RA"]];
  const LEFT = new Set(["LS", "LE", "LW", "LH", "LK", "LA"]);

  // view = {cx, hy, s}: screen position of the hip centre and pixels per body unit
  const at = (p, v) => [v.cx + p[0] * v.s, v.hy + p[1] * v.s];
  function line(g, P, v, o) {
    g.save(); g.globalAlpha = o.alpha ?? 1; g.lineCap = "round"; g.lineJoin = "round"; g.lineWidth = o.width;
    for (const [a, b] of EDGES) {
      g.strokeStyle = o.color || (LEFT.has(a) && LEFT.has(b) ? o.left : o.right);
      const A = at(P[a], v), B = at(P[b], v); g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
    }
    const h = at(P.head, v); g.strokeStyle = o.color || o.left; g.beginPath(); g.arc(h[0], h[1], 0.17 * v.s, 0, 7); if (o.fillHead) { g.fillStyle = o.fillHead; g.fill(); } g.stroke();
    if (o.joints) { g.fillStyle = o.joints; for (const n of ["LS", "RS", "LE", "RE", "LW", "RW", "LH", "RH"]) { const q = at(P[n], v); g.beginPath(); g.arc(q[0], q[1], o.width * 0.9, 0, 7); g.fill(); } }
    g.restore();
  }

  const timeline = []; for (let i = 0; i <= DUR * 10; i++) timeline.push(+pose(i / 10).lean.toFixed(2));
  const ghostTimes = [0.3, 0.6, 0.85].map(f => 4 + 4 * f);   // three moments of the selected (second) repetition
  const text = "Bend your upper body further to the side: the sideways bend of your upper body reached 11° (most clearly in the repetition around 4.0–8.0 s), while correct repetitions typically reach about 22°.";
  const RESULT = {
    ok: true, status: "ok", exercise: "ELK", name: "Flank stretch", strictness: "standard", duration_s: DUR, person_found: 1.0,
    feedback: {text, sentences: [{text, cites: ["F1"]}], source: "template_only"},
    findings: [{id: "F1", status: "error", severity: "small", issue: "body not tilted", measure_label: "sideways bend of your upper body", unit: "°",
      quotable: true, value: 11, typical: 22, threshold: 14.4, start_s: 4.0, end_s: 8.0, window_source: "worst_rep", reps: REPS,
      evidence: {loso_sensitivity: 0.89, loso_specificity: 0.91, kappa_system_A: 0.51, kappa_system_B: 0.54, kappa_clinicians: 0.43}}],
    timelines: {F1: timeline},
  };
  const q = new URLSearchParams(location.search);
  window.SYNTH = {pose, line, at, RESULT, DUR, REPS, ghostTimes, timeline, repAt, D2R,
    param: (n, d) => q.has(n) ? q.get(n) : d,
    still: q.has("still"), reduced: matchMedia("(prefers-reduced-motion: reduce)").matches || q.has("still")};
})();

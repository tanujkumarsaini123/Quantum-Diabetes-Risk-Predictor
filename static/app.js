const $ = (s, r = document) => r.querySelector(s);
const FIELDS = [
  {k:"age", label:"Age", unit:"years", min:18, max:90, step:1, def:35, hint:"Risk rises with age"},
  {k:"glucose", label:"Blood glucose", unit:"mg/dL", min:40, max:300, step:1, def:110, hint:"Plasma glucose, 2 hours after a glucose drink"},
  {k:"bmi", label:"Body mass index", unit:"kg/m²", min:12, max:65, step:0.1, def:27, hint:"Weight divided by height squared"},
  {k:"blood_pressure", label:"Blood pressure (diastolic)", unit:"mm Hg", min:30, max:140, step:1, def:72, hint:"The lower number in a reading"},
  {k:"pregnancies", label:"Pregnancies", unit:"count", min:0, max:17, step:1, def:1, hint:"Enter 0 if not applicable"},
  {k:"family_index", label:"Family history index", unit:"0.05 to 2.5", min:0.05, max:2.5, step:0.01, def:0.45, hint:"Higher means more relatives with diabetes"},
];
const PRESETS = {
  low:{age:24,glucose:92,bmi:22.5,blood_pressure:66,pregnancies:0,family_index:0.2},
  high:{age:52,glucose:178,bmi:36.4,blood_pressure:84,pregnancies:4,family_index:1.1},
};
const COLORS = {Low:"var(--low)", Moderate:"var(--mod)", High:"var(--high)"};
const pct = v => Math.round(v * 100) + "%";

function toast(msg){const t=$("#toast");t.textContent=msg;t.classList.add("on");clearTimeout(toast.t);toast.t=setTimeout(()=>t.classList.remove("on"),2600)}

/* ---------- navigation ---------- */
$("#nav").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  document.querySelectorAll("#nav button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("on", v.id === "view-" + b.dataset.view));
  if (b.dataset.view === "history") loadHistory();
  if (b.dataset.view === "lab") loadLab();
  window.scrollTo({top:0});
});

/* ---------- form ---------- */
const fieldsEl = $("#fields");
FIELDS.forEach(f => {
  fieldsEl.insertAdjacentHTML("beforeend", `<div class="field"><div class="top"><div><label for="n_${f.k}">${f.label}</label><small>${f.hint}</small></div>
  <div><input class="num" id="n_${f.k}" name="${f.k}" type="number" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.def}" required></div></div>
  <input type="range" id="r_${f.k}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.def}" aria-label="${f.label} slider"></div>`);
  const n = $("#n_" + f.k), r = $("#r_" + f.k);
  r.oninput = () => n.value = r.value;
  n.oninput = () => r.value = n.value;
});
function setValues(o){for(const k in o){$("#n_"+k).value=o[k];$("#r_"+k).value=o[k]}}
document.querySelectorAll("[data-preset]").forEach(b => b.onclick = () => { setValues(PRESETS[b.dataset.preset]); toast("Example loaded. Press Run quantum assessment."); });
$("#bmiGo").onclick = () => {
  const h = +$("#h").value / 100, w = +$("#wt").value;
  if (!h || !w) return toast("Enter both height and weight.");
  setValues({bmi: Math.min(65, Math.max(12, +(w / (h * h)).toFixed(1)))});
  toast("BMI updated.");
};

/* ---------- running animation ---------- */
function drawWires(){
  let s = "";
  for (let i = 0; i < 6; i++) {
    const y = 20 + i * 22;
    s += `<line class="wire" x1="14" x2="306" y1="${y}" y2="${y}"/><line class="flow" x1="14" x2="306" y1="${y}" y2="${y}" style="animation-delay:${i*.08}s"/>`;
    for (let g = 0; g < 3; g++) s += `<rect class="gate" x="${60 + g * 86}" y="${y - 7}" width="14" height="14" rx="4" style="animation-delay:${(i + g) * .12}s"/>`;
  }
  for (let g = 0; g < 3; g++) s += `<line x1="${67 + g * 86 + 36}" x2="${67 + g * 86 + 36}" y1="20" y2="130" stroke="var(--q)" stroke-width="2" opacity=".5"/>`;
  $("#wires").innerHTML = s;
}

/* ---------- predict ---------- */
$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(e.target)); $("#err").textContent = "";
  $("#go").disabled = true; $("#go").textContent = "Running circuit…";
  $("#empty").hidden = true; $("#out").hidden = true; $("#running").hidden = false; drawWires();
  try {
    const [res] = await Promise.all([fetch("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)}), new Promise(r => setTimeout(r, 900))]);
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || "Something went wrong.");
    show(d, body);
  } catch (err) {
    $("#running").hidden = true; $("#empty").hidden = false; $("#err").textContent = err.message;
  } finally { $("#go").disabled = false; $("#go").textContent = "Run quantum assessment"; }
});

function gauge(p, lvl){
  const cx=120, cy=120, r=92, pt = a => [cx + r * Math.cos(Math.PI - a), cy - r * Math.sin(Math.PI - a)];
  const arc = (a0,a1) => {const [x0,y0]=pt(a0),[x1,y1]=pt(a1);return `M${x0} ${y0} A${r} ${r} 0 0 1 ${x1} ${y1}`};
  const [nx, ny] = pt(Math.PI * p);
  $("#gauge").innerHTML = `<path d="${arc(0.001,Math.PI*.35)}" stroke="var(--low)" opacity=".28" fill="none" stroke-width="16"/><path d="${arc(Math.PI*.35,Math.PI*.6)}" stroke="var(--mod)" opacity=".28" fill="none" stroke-width="16"/><path d="${arc(Math.PI*.6,Math.PI-.001)}" stroke="var(--high)" opacity=".28" fill="none" stroke-width="16"/>
  <path d="${arc(0.001, Math.max(0.02, Math.PI * p))}" stroke="${COLORS[lvl]}" fill="none" stroke-width="16" stroke-linecap="round"/><circle cx="${nx}" cy="${ny}" r="9" fill="var(--card)" stroke="${COLORS[lvl]}" stroke-width="4"/>`;
}
const VERDICT = {
  Low:"The quantum model sees a low likelihood of diabetes from these measurements.",
  Moderate:"These measurements show a moderate likelihood. A routine check-up would be worthwhile.",
  High:"These measurements show a high likelihood of diabetes. Please see a clinician for a confirmatory test.",
};
const NEXT = {
  Low:["Keep up regular activity and a balanced diet.","Re-check yearly, or sooner if symptoms appear."],
  Moderate:["Book a fasting glucose or HbA1c test.","Aim for 150 minutes of activity a week.","Recheck in 3 to 6 months."],
  High:["Arrange an HbA1c or fasting glucose test soon.","Discuss results with a doctor before changing any treatment.","Watch for thirst, frequent urination, blurred vision."],
};
function show(d, inputs){
  $("#running").hidden = true; $("#out").hidden = false;
  gauge(d.quantum_prob, d.risk_level);
  $("#pct").textContent = pct(d.quantum_prob);
  const chip = $("#lvl"); chip.textContent = d.risk_level + " risk"; chip.className = "chip " + d.risk_level;
  $("#verdict").textContent = VERDICT[d.risk_level] + (d.agree ? " The classical model agrees." : " The classical model rates it differently, so treat this as a prompt for a proper test.");
  $("#qv").textContent = pct(d.quantum_prob); $("#cv").textContent = pct(d.classical_prob);
  requestAnimationFrame(() => { $("#qbar").style.width = pct(d.quantum_prob); $("#cbar").style.width = pct(d.classical_prob); });
  const max = Math.max(0.08, ...d.contributions.map(c => Math.abs(c.impact)));
  $("#qubits").innerHTML = d.contributions.map((c, i) => `<div class="q ${c.impact >= 0 ? "up" : "dn"}"><span class="dot">q${i+1}</span><span>${c.feature}</span><div class="track"><div class="fill" data-w="${Math.abs(c.impact) / max * 50}"></div></div><b>${c.impact >= 0 ? "+" : "−"}${Math.round(Math.abs(c.impact) * 100)}</b></div>`).join("");
  requestAnimationFrame(() => document.querySelectorAll(".fill").forEach(f => f.style.width = f.dataset.w + "%"));
  $("#steps").innerHTML = `<b>Suggested next steps</b><ul>${NEXT[d.risk_level].map(x => `<li>${x}</li>`).join("")}</ul>`;
  $("#saved").textContent = `Saved to patient history as record #${d.id}.`;
  if (window.innerWidth < 980) $("#result").scrollIntoView({behavior:"smooth"});
}

/* ---------- history ---------- */
async function loadHistory(){
  const d = await (await fetch("/api/history")).json();
  $("#stats").innerHTML = `<div class="stat"><b>${d.stats.total}</b><span>Assessments saved</span></div><div class="stat"><b>${d.stats.high}</b><span>High-risk results</span></div><div class="stat"><b>${pct(d.stats.avg)}</b><span>Average quantum risk</span></div>`;
  $("#noHist").hidden = d.rows.length > 0; $("#table").hidden = d.rows.length === 0;
  $("#table tbody").innerHTML = d.rows.map(r => `<tr><td>${new Date(r.created_at.replace(" ", "T") + "Z").toLocaleString([], {dateStyle:"medium", timeStyle:"short"})}</td><td>${esc(r.patient_name)}</td><td>${r.glucose}</td><td>${r.bmi}</td><td>${r.age}</td><td><b>${pct(r.quantum_prob)}</b></td><td><span class="chip ${r.risk_level}">${r.risk_level}</span></td><td><button data-del="${r.id}" aria-label="Delete record ${r.id}">Delete</button></td></tr>`).join("");
}
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
$("#table").addEventListener("click", async e => {
  const b = e.target.closest("[data-del]"); if (!b) return;
  await fetch("/api/history/" + b.dataset.del, {method:"DELETE"}); toast("Record deleted."); loadHistory();
});

/* ---------- model lab ---------- */
function drawCircuit(){
  const names = ["Glucose","BMI","Age","Blood pressure","Family history","Pregnancies"]; let s = "";
  names.forEach((n, i) => { const y = 30 + i * 42;
    s += `<text x="0" y="${y+5}" font-size="13" fill="var(--mute)" font-family="DM Sans">q${i+1} ${n}</text><line x1="130" x2="750" y1="${y}" y2="${y}" stroke="var(--line)" stroke-width="2"/>`;
    for (let b = 0; b < 3; b++) { const x = 150 + b * 200;
      s += `<rect x="${x}" y="${y-13}" width="40" height="26" rx="7" fill="var(--q)"/><text x="${x+20}" y="${y+4}" text-anchor="middle" fill="#fff" font-size="12" font-weight="600">RY</text>`;
      s += `<rect x="${x+54}" y="${y-13}" width="46" height="26" rx="7" fill="var(--q2)" stroke="var(--q)"/><text x="${x+77}" y="${y+4}" text-anchor="middle" fill="var(--q)" font-size="11" font-weight="600">Rot</text>`;
      if (i < 5) s += `<line x1="${x+125}" x2="${x+125}" y1="${y}" y2="${y+42}" stroke="var(--q)" stroke-width="2"/><circle cx="${x+125}" cy="${y}" r="4" fill="var(--q)"/><circle cx="${x+125}" cy="${y+42}" r="8" fill="none" stroke="var(--q)" stroke-width="2"/>`;
    }});
  s += `<rect x="730" y="17" width="26" height="26" rx="6" fill="var(--ink)"/><text x="743" y="35" text-anchor="middle" fill="var(--paper)" font-size="13" font-weight="700">Z</text>`;
  s += `<text x="150" y="268" font-size="12" fill="var(--mute)">RY = feature encoding · Rot + CNOT ring = trainable entangling layer · repeated 3 times · measure ⟨Z⟩ on q1</text>`;
  $("#circuit").innerHTML = s;
}
let labLoaded = false;
async function loadLab(){
  drawCircuit();
  const d = await (await fetch("/api/model")).json(), R = d.metrics.results;
  $("#cmp").innerHTML = [["roc_auc","ROC AUC","How well risk scores rank sick above healthy"],["recall","Recall","Share of diabetic patients caught"],["accuracy","Accuracy",""],["f1","F1 score",""]].map(([k,t,sub]) =>
    `<div class="metric"><h3>${t}${sub ? ` <span style="font-weight:400">· ${sub}</span>` : ""}</h3>${Object.keys(R).map(n => `<div class="mrow ${n.startsWith("Quantum") ? "qm" : ""}"><span>${n}</span><div class="bar"><i style="width:${R[n][k]*100}%"></i></div><b>${R[n][k].toFixed(2)}</b></div>`).join("")}</div>`).join("");
  const h = R["Quantum VQC"].loss_history, mx = Math.max(...h), mn = Math.min(...h);
  const pts = h.map((v, i) => `${36 + i / (h.length - 1) * 310},${16 + (1 - (v - mn) / (mx - mn || 1)) * 150}`).join(" ");
  $("#loss").innerHTML = `<line x1="36" x2="346" y1="170" y2="170" stroke="var(--line)"/><line x1="36" x2="36" y1="10" y2="170" stroke="var(--line)"/><polyline points="${pts}" fill="none" stroke="var(--q)" stroke-width="2.5" stroke-linejoin="round"/><text x="36" y="190" font-size="11" fill="var(--mute)">epoch 1</text><text x="346" y="190" text-anchor="end" font-size="11" fill="var(--mute)">epoch ${h.length}</text><text x="30" y="20" text-anchor="end" font-size="11" fill="var(--mute)">${mx.toFixed(2)}</text><text x="30" y="168" text-anchor="end" font-size="11" fill="var(--mute)">${mn.toFixed(2)}</text>`;
  const c = R["Quantum VQC"].confusion;
  $("#cm").innerHTML = `<div class="ok"><b>${c.tn}</b><small>Correctly cleared</small></div><div><b>${c.fp}</b><small>False alarm</small></div><div><b>${c.fn}</b><small>Missed case</small></div><div class="ok"><b>${c.tp}</b><small>Correctly flagged</small></div>`;
  $("#runs tbody").innerHTML = d.runs.map(r => `<tr><td>${r.created_at}</td><td>${r.model}</td><td>${r.accuracy.toFixed(3)}</td><td>${r.roc_auc.toFixed(3)}</td><td>${r.recall.toFixed(3)}</td><td>${r.f1.toFixed(3)}</td></tr>`).join("");
}

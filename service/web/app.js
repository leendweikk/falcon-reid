// Falcon ReID web client: talks only to the documented API (/api/docs).
const $ = (id) => document.getElementById(id);
let file = null, img = null, scale = 1, drag = null, lastResult = null, galleryOffset = 0;
const PAGE = 24;

// ---------- status ----------
async function refreshStatus() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("status").textContent = `модель: ${h.models.join(" + ")} (${h.mode}) · ${h.device} · порог отказа ${h.threshold} · в галерее ${h.gallery_size} ТС`;
  } catch { $("status").textContent = "API недоступен"; }
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  $("tab-search").hidden = b.dataset.tab !== "search";
  $("tab-gallery").hidden = b.dataset.tab !== "gallery";
  if (b.dataset.tab === "gallery") loadGallery();
});

// ---------- image + bbox drawing ----------
$("file").onchange = (e) => {
  file = e.target.files[0];
  if (!file) return;
  img = new Image();
  img.onload = () => {
    const c = $("canvas");
    scale = Math.min(1, 1100 / img.width);
    c.width = img.width * scale; c.height = img.height * scale;
    ["bx", "by", "bw", "bh"].forEach((k) => $(k).value = "");
    draw(); updateButtons();
  };
  img.src = URL.createObjectURL(file);
};

function box() {
  const v = ["bx", "by", "bw", "bh"].map((k) => parseInt($(k).value, 10));
  return v.every((n) => Number.isFinite(n)) && v[2] > 0 && v[3] > 0 ? v : null;
}
function draw() {
  if (!img) return;
  const c = $("canvas"), ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, c.width, c.height);
  const b = box();
  if (b) {
    ctx.strokeStyle = "#5b2bd6"; ctx.lineWidth = 3;
    ctx.strokeRect(b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale);
  }
}
function updateButtons() {
  const ready = !!(file && box());
  $("btn-search").disabled = !ready;
  $("btn-add").disabled = !ready;
}
["bx", "by", "bw", "bh"].forEach((k) => $(k).oninput = () => { draw(); updateButtons(); });

const pos = (e) => { const r = $("canvas").getBoundingClientRect();
  const k = $("canvas").width / r.width;
  return [(e.clientX - r.left) * k / scale, (e.clientY - r.top) * k / scale]; };
$("canvas").onmousedown = (e) => { if (img) drag = pos(e); };
$("canvas").onmousemove = (e) => {
  if (!drag) return;
  const [x, y] = pos(e);
  const x0 = Math.max(0, Math.min(drag[0], x)), y0 = Math.max(0, Math.min(drag[1], y));
  const x1 = Math.min(img.width, Math.max(drag[0], x)), y1 = Math.min(img.height, Math.max(drag[1], y));
  $("bx").value = Math.round(x0); $("by").value = Math.round(y0);
  $("bw").value = Math.round(x1 - x0); $("bh").value = Math.round(y1 - y0);
  draw();
};
window.onmouseup = () => { if (drag) { drag = null; updateButtons(); } };

function form(extra = {}) {
  const [x, y, w, h] = box();
  const f = new FormData();
  f.append("file", file); f.append("x", x); f.append("y", y); f.append("w", w); f.append("h", h);
  Object.entries(extra).forEach(([k, v]) => { if (v !== "" && v != null) f.append(k, v); });
  return f;
}
async function call(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : r.statusText);
  return body;
}

// ---------- search ----------
$("btn-search").onclick = async () => {
  $("btn-search").disabled = true;
  try {
    const res = await call("/api/search", { method: "POST", body: form({ top_k: $("topk").value }) });
    lastResult = res; showResult(res);
  } catch (e) { alert("Ошибка: " + e.message); }
  updateButtons();
};

function card(m, best) {
  const d = document.createElement("div");
  d.className = "item" + (best ? " best" : "");
  d.innerHTML = `<img src="${m.image_url}" alt="">
    <div class="meta"><b>#${m.rank}</b> · сходство ${m.similarity.toFixed(3)}<br>
    <small>${m.gallery_id}${m.camera ? " · камера " + m.camera : ""}${m.label ? " · " + m.label : ""}</small></div>`;
  return d;
}
function showResult(r) {
  $("result-card").hidden = false;
  const v = $("verdict");
  if (r.refused) {
    v.className = "verdict no";
    v.textContent = `Уверенного совпадения нет (отказ): уверенность ${r.confidence} < порога ${r.threshold}. Кандидаты ниже показаны только для справки.`;
  } else {
    v.className = "verdict ok";
    v.textContent = `Найдено совпадение: ${r.top_match} (уверенность ${r.confidence} ≥ порога ${r.threshold})`;
  }
  $("timing").textContent = `формирование признака ${r.extract_ms} мс` + (r.rescore_ms ? ` · второй этап (ансамбль) ${r.rescore_ms} мс` : "") + ` · поиск + переранжирование ${r.search_ms} мс · галерея ${r.gallery_size} ТС`;
  const grid = $("results");
  grid.className = "grid" + (r.refused ? " results-refused" : "");
  grid.innerHTML = "";
  r.results.forEach((m, i) => grid.appendChild(card(m, i === 0 && !r.refused)));
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
}
$("btn-json").onclick = () => lastResult && download(`search_${lastResult.search_id}.json`, JSON.stringify(lastResult, null, 2), "application/json");
$("btn-csv").onclick = () => {
  if (!lastResult) return;
  const rows = [["search_id", "refused", "confidence", "threshold", "rank", "gallery_id", "similarity", "camera", "label"]];
  lastResult.results.forEach((m) => rows.push([lastResult.search_id, lastResult.refused, lastResult.confidence,
    lastResult.threshold, m.rank, m.gallery_id, m.similarity, m.camera ?? "", m.label ?? ""]));
  download(`search_${lastResult.search_id}.csv`, rows.map((r) => r.join(",")).join("\n"), "text/csv");
};

// ---------- gallery ----------
$("btn-add").onclick = async () => {
  try {
    const r = await call("/api/gallery", { method: "POST",
      body: form({ gallery_id: $("g-id").value, camera: $("g-cam").value, label: $("g-label").value }) });
    $("add-msg").textContent = `Добавлено: ${r.gallery_id}`;
    loadGallery(); refreshStatus();
  } catch (e) { $("add-msg").textContent = "Ошибка: " + e.message; }
};
async function loadGallery() {
  const g = await call(`/api/gallery?limit=${PAGE}&offset=${galleryOffset}`);
  $("g-total").textContent = `(${g.total})`;
  $("g-page").textContent = ` ${galleryOffset + 1}–${Math.min(galleryOffset + PAGE, g.total)} `;
  const grid = $("gallery"); grid.innerHTML = "";
  g.items.forEach((it) => {
    const d = document.createElement("div"); d.className = "item";
    d.innerHTML = `<img src="${it.image_url}" alt=""><div class="meta"><small>${it.gallery_id}</small><br>
      <button data-id="${it.gallery_id}">удалить</button></div>`;
    d.querySelector("button").onclick = async () => {
      if (!confirm(`Удалить ${it.gallery_id}?`)) return;
      await call(`/api/gallery/${encodeURIComponent(it.gallery_id)}`, { method: "DELETE" });
      loadGallery(); refreshStatus();
    };
    grid.appendChild(d);
  });
  $("g-prev").disabled = galleryOffset === 0;
  $("g-next").disabled = galleryOffset + PAGE >= g.total;
}
$("g-prev").onclick = () => { galleryOffset = Math.max(0, galleryOffset - PAGE); loadGallery(); };
$("g-next").onclick = () => { galleryOffset += PAGE; loadGallery(); };

refreshStatus();

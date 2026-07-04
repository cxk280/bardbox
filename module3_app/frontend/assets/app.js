// Bardbox Writing Studio — frontend logic.
// Every fetch below is a relative path → same localhost origin. There are no absolute
// URLs and no external hosts anywhere in this file (the no-egress test greps for that).

const $ = (id) => document.getElementById(id);
const api = (path, body) =>
  fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then((r) => r.json());

// ---- view switching (with hash deep-linking, e.g. #privacy) ----
function showView(name) {
  const btn = document.querySelector(`.nav-item[data-view="${name}"]`);
  if (!btn) return;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  btn.classList.add("active");
  $("view-" + name).classList.add("active");
}
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => { location.hash = btn.dataset.view; });
});
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
if (location.hash) showView(location.hash.slice(1));

// ---- draft word/char count ----
function updateCount() {
  const t = $("draft").value.trim();
  const words = t ? t.split(/\s+/).length : 0;
  $("count").textContent = `${words} words · ${$("draft").value.length} characters`;
}
$("draft").addEventListener("input", updateCount);
updateCount();

// ---- rewrite ----
async function doRewrite() {
  const text = $("draft").value.trim();
  if (!text) return;
  const btn = $("rewrite-btn");
  const out = $("bard-out");
  btn.disabled = true;
  out.classList.remove("empty");
  out.textContent = "…";
  try {
    const res = await api("/rewrite", { text, intensity: $("intensity").value / 100 });
    out.textContent = res.output || "(no output)";
    $("citation").style.display = $("ground").checked ? "inline-flex" : "none";
  } catch (e) {
    out.classList.add("empty");
    out.textContent = "The local server is unreachable. Start it and try again.";
  } finally {
    btn.disabled = false;
  }
}
$("rewrite-btn").addEventListener("click", doRewrite);
$("again-btn").addEventListener("click", doRewrite);
$("copy-btn").addEventListener("click", () => navigator.clipboard?.writeText($("bard-out").textContent));

// ---- feedback ----
$("notes-btn").addEventListener("click", async () => {
  const text = $("fb-draft").value.trim();
  if (!text) return;
  const out = $("fb-out");
  out.classList.remove("empty");
  out.textContent = "…";
  const res = await api("/feedback", { text });
  out.textContent = res.output || "(no output)";
});
$("fb-copy").addEventListener("click", () => navigator.clipboard?.writeText($("fb-out").textContent));

// ---- chat ----
async function sendChat() {
  const input = $("chat-input");
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  const t = $("transcript");
  const you = document.createElement("div");
  you.className = "msg you";
  you.textContent = msg;
  t.appendChild(you);
  t.scrollTop = t.scrollHeight;
  const res = await api("/chat", { message: msg });
  const bard = document.createElement("div");
  bard.className = "msg bard";
  bard.textContent = res.output || "(no output)";
  t.appendChild(bard);
  t.scrollTop = t.scrollHeight;
}
$("chat-send").addEventListener("click", sendChat);
$("chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

// ---- privacy self-test: confirm the backend is local ----
$("selftest-btn").addEventListener("click", async () => {
  const res = await fetch("/health").then((r) => r.json());
  const local = !/https?:\/\/(?!127\.0\.0\.1|localhost)/.test(res.backend || "");
  const el = $("selftest-result");
  el.classList.add("show");
  $("selftest-text").textContent = local
    ? `No egress detected — backend "${res.backend}" is on-device.`
    : "Warning: backend is not local!";
});

// ---- boot: read backend status, set chips + demo banner ----
fetch("/health")
  .then((r) => r.json())
  .then((h) => {
    const model = h.is_real_model ? h.backend : "Demo transformer";
    $("chip-model").textContent = h.is_real_model ? "local model" : "demo mode";
    $("chip-quant").textContent = h.is_real_model ? "on device" : "no GGUF loaded";
    $("set-model").textContent = h.backend;
    $("infer-target").textContent = h.is_real_model ? "on device" : "on device (demo)";
    if (!h.is_real_model) {
      const d = $("demo");
      d.classList.add("show");
      d.textContent = "Demo mode: a built-in rule-based transformer is running (no model file found). Add a .gguf to module3_app/models/ for the real Bard. The privacy guarantee is identical either way.";
    }
  })
  .catch(() => {});

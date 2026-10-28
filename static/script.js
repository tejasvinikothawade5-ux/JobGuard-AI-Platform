(function () {
const form = document.getElementById("analyzer-form");
if (!form) return; // other pages (login, history...) have no analyzer
const resultsSection = document.getElementById("results");
const formError = document.getElementById("form-error");
const scanBtn = form.querySelector(".scan-btn");

const GAUGE_CIRCUMFERENCE = 440; // 2 * PI * r(70), matches stroke-dasharray in CSS

const EXAMPLES = {
  fake: {
    title: "Online Data Entry Executive - Immediate Joining",
    company: "Bright Future Careers",
    url: "http://brightfuturecareers.wixsite.com/apply-now",
    description:
      "URGENT HIRING!!! No experience needed, immediate joining! Earn up to Rs. 45,000 per week working just 2 hours a day from home. Limited seats available, apply now before offer closes. Selected candidates must pay a refundable registration fee of Rs.999 to confirm their slot. Contact us on WhatsApp only, no calls please. 100% guaranteed income, no boss, no targets!",
    requirements: "No requirements, anyone can apply, 18+ age, smartphone needed.",
    company_profile: "We are a fast growing company offering flexible online jobs.",
  },
  real: {
    title: "Backend Developer",
    company: "Persistent Systems",
    url: "https://www.persistentsystems.com/careers/apply",
    description:
      "Persistent Systems is looking for a Backend Developer to join our growing team. The ideal candidate will design, develop and maintain scalable backend services; participate in code reviews and maintain coding standards. Salary: Rs. 9,00,000 per annum (based on experience). Interested candidates can apply through our official careers portal.",
    requirements: "Bachelor's degree in Computer Science or related field, 2+ years of relevant industry experience, proficiency in Python, SQL and data structures.",
    company_profile: "Persistent Systems is an established organization with a strong presence in the industry, known for its professional work culture and employee growth programs.",
  },
};

document.querySelectorAll(".chip[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const ex = EXAMPLES[btn.dataset.example];
    for (const key in ex) {
      const field = form.elements[key];
      if (field) field.value = ex[key];
    }
    formError.hidden = true;
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  formError.hidden = true;

  const payload = {
    title: form.title.value,
    company: form.company.value,
    url: form.url.value,
    description: form.description.value,
    requirements: form.requirements.value,
    company_profile: form.company_profile.value,
  };

  if (!payload.description.trim() && !payload.title.trim()) {
    formError.textContent = "Please enter at least a job title or description to analyze.";
    formError.hidden = false;
    return;
  }

  scanBtn.classList.add("loading");
  scanBtn.querySelector("span").textContent = "Analyzing...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      formError.textContent = data.error || "Something went wrong. Please try again.";
      formError.hidden = false;
      return;
    }

    renderResults(data);
  } catch (err) {
    formError.textContent = "Could not reach the JobGuard AI server. Is app.py running?";
    formError.hidden = false;
  } finally {
    scanBtn.classList.remove("loading");
    scanBtn.querySelector("span").textContent = "Analyze listing";
  }
});

function riskClass(level) {
  if (level === "High Risk") return "high";
  if (level === "Medium Risk") return "medium";
  return "low";
}

function renderResults(data) {
  resultsSection.hidden = false;

  const cls = riskClass(data.risk_level);
  const gaugeFill = document.getElementById("gauge-fill");
  const offset = GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * data.risk_score) / 100;
  const colorMap = { high: "#D0313A", medium: "#D98A00", low: "#0B7F70" };

  // reset then animate on next frame so the CSS transition fires
  gaugeFill.style.transition = "none";
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  requestAnimationFrame(() => {
    gaugeFill.style.transition = "";
    gaugeFill.style.stroke = colorMap[cls];
    gaugeFill.style.strokeDashoffset = offset;
  });

  document.getElementById("risk-score").textContent = data.risk_score;

  const badge = document.getElementById("risk-level-badge");
  badge.textContent = data.risk_level;
  badge.className = "risk-badge " + cls;

  document.getElementById("verdict-title").textContent = data.recommendation.verdict;
  document.getElementById("verdict-advice").textContent = data.recommendation.advice;

  document.getElementById("prob-fraud-val").textContent = data.probability_fraud + "%";
  document.getElementById("prob-genuine-val").textContent = data.probability_genuine + "%";
  requestAnimationFrame(() => {
    document.getElementById("prob-fraud-bar").style.width = data.probability_fraud + "%";
    document.getElementById("prob-genuine-bar").style.width = data.probability_genuine + "%";
  });

  const flagsList = document.getElementById("flags-list");
  const flagsEmpty = document.getElementById("flags-empty");
  flagsList.innerHTML = "";
  if (data.red_flags.length === 0) {
    flagsEmpty.hidden = false;
  } else {
    flagsEmpty.hidden = true;
    data.red_flags.forEach((f) => {
      const li = document.createElement("li");
      const dot = document.createElement("span");
      dot.className = "flag-dot " + f.severity;
      const text = document.createElement("span");
      text.textContent = f.message;
      li.appendChild(dot);
      li.appendChild(text);
      flagsList.appendChild(li);
    });
  }

  document.getElementById("tech-readout").textContent = JSON.stringify(data.feature_snapshot, null, 2);

  setupHistoryBar(data);

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.getElementById("tech-toggle").addEventListener("click", (e) => {
  const readout = document.getElementById("tech-readout");
  const isHidden = readout.hidden;
  readout.hidden = !isHidden;
  e.target.textContent = isHidden ? "Hide technical feature readout" : "Show technical feature readout";
});

// Load model metrics for the footer
fetch("/model-info")
  .then((r) => r.json())
  .then((info) => {
    const acc = (info.metrics.accuracy * 100).toFixed(1);
    document.getElementById("model-meta").textContent =
      ` · Model: ${info.model_type} · Test accuracy: ${acc}% · ${info.feature_count_text} text features + ${info.feature_count_engineered} engineered signals.`;
  })
  .catch(() => {});

// ---- History bar (only shown when the server saved this analysis to the logged-in user's history) ----
let currentHistoryId = null;

function setupHistoryBar(data) {
  const bar = document.getElementById("history-bar");
  const msg = document.getElementById("history-msg");
  if (!bar) return;
  if (!data.history_id) { bar.hidden = true; currentHistoryId = null; return; }
  currentHistoryId = data.history_id;
  bar.hidden = false;
  msg.className = "history-msg";
  msg.textContent = data.history_status && data.history_status !== "analyzed"
    ? "Already in your history as: " + data.history_status + "."
    : "Saved to your history. Did you apply?";
  bar.querySelectorAll("button[data-status]").forEach((b) => (b.disabled = false));
}

const historyActions = document.getElementById("history-actions");
if (historyActions) {
  historyActions.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-status]");
    if (!btn || !currentHistoryId) return;
    const msg = document.getElementById("history-msg");
    const token = document.querySelector('meta[name="csrf-token"]').content;
    btn.disabled = true;
    try {
      const res = await fetch("/api/history/" + currentHistoryId + "/status", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
        body: JSON.stringify({ status: btn.dataset.status }),
      });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || "failed");
      msg.className = "history-msg" + (out.warning ? " warn" : "");
      msg.textContent = out.warning || "Marked as " + out.label + ".";
    } catch (err) {
      msg.className = "history-msg warn";
      msg.textContent = "Could not update your history. Please try again.";
    } finally {
      btn.disabled = false;
    }
  });
}
})();

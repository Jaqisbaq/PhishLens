// PhishLens: plain JS, no build step, no third-party libraries.
// Submits the form to POST /api/analyze and renders the result.
// This file never sends the submitted URL anywhere except that endpoint,
// and never attempts to fetch or navigate to it.

(function () {
  "use strict";

  const form = document.getElementById("analyze-form");
  const submitBtn = document.getElementById("submit-btn");
  const errorsBox = document.getElementById("form-errors");
  const errorsList = document.getElementById("form-errors-list");
  const resultPanel = document.getElementById("result-panel");
  const outcomeBadge = document.getElementById("outcome-badge");
  const fusedScoreEl = document.getElementById("fused-score");
  const provisionalNotice = document.getElementById("provisional-notice");
  const branchTableBody = document.getElementById("branch-table-body");
  const explanationsEl = document.getElementById("explanations");
  const caveatEl = document.getElementById("caveat");

  function showErrors(messages) {
    errorsList.innerHTML = "";
    messages.forEach(function (m) {
      const li = document.createElement("li");
      li.textContent = m;
      errorsList.appendChild(li);
    });
    errorsBox.classList.remove("hidden");
  }

  function hideErrors() {
    errorsBox.classList.add("hidden");
    errorsList.innerHTML = "";
  }

  function fmtScore(v) {
    return v === null || v === undefined ? "n/a" : v.toFixed(2);
  }

  function outcomeClass(outcome) {
    return "outcome-" + outcome.toLowerCase().replace(/\s+/g, "-");
  }

  function renderResult(result) {
    resultPanel.classList.remove("hidden");
    outcomeBadge.textContent = result.outcome;
    outcomeBadge.className = "outcome-badge " + outcomeClass(result.outcome);
    fusedScoreEl.textContent =
      "Fused score: " + fmtScore(result.fused_score) +
      " (method: " + result.fusion.method + ", branches used: " +
      (result.fusion.branches_used.join(", ") || "none") + ")";

    if (result.provisional_settings) {
      provisionalNotice.textContent = result.provisional_notice ||
        "Provisional settings are in use.";
      provisionalNotice.classList.remove("hidden");
    } else {
      provisionalNotice.classList.add("hidden");
    }

    branchTableBody.innerHTML = "";
    result.branches.forEach(function (b) {
      const row = document.createElement("tr");
      const cells = [
        b.branch,
        result.inputs_provided[b.branch] ? "provided" : "missing",
        b.status,
        b.probability === null || b.probability === undefined ? "n/a" : b.probability.toFixed(2),
        b.model_id ? (b.model_id + (b.revision ? " @ " + b.revision.slice(0, 8) : "")) : "n/a",
        b.latency_ms === null || b.latency_ms === undefined ? "n/a" : String(b.latency_ms)
      ];
      cells.forEach(function (text) {
        const td = document.createElement("td");
        td.textContent = text;
        row.appendChild(td);
      });
      branchTableBody.appendChild(row);
    });

    explanationsEl.innerHTML = "";
    (result.explanations || []).forEach(function (sentence) {
      const li = document.createElement("li");
      li.textContent = sentence;
      explanationsEl.appendChild(li);
    });

    caveatEl.textContent = result.caveat || "";
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    hideErrors();
    resultPanel.classList.add("hidden");
    submitBtn.disabled = true;
    submitBtn.textContent = "Analyzing...";

    try {
      const formData = new FormData(form);
      const response = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await response.json();

      if (response.status === 422 && data.errors) {
        showErrors(data.errors.map(function (e) { return e.field + ": " + e.message; }));
        return;
      }
      if (!response.ok) {
        showErrors(["The server returned an unexpected error (" + response.status + ")."]);
        return;
      }
      renderResult(data);
    } catch (err) {
      showErrors(["Could not reach the server: " + err.message]);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Analyze";
    }
  });
})();

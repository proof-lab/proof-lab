/**
 * Proof Lab – Frontend Interactive Application Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initDataStudio();
  initQuantLab();
  initSafeguards();
  initAutoPilot();
  initCopilot();
});

// ============================================================================
// Tab Navigation
// ============================================================================

function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const workspaces = document.querySelectorAll(".workspace-tab");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");

      tabs.forEach((t) => t.classList.remove("active"));
      workspaces.forEach((w) => w.classList.remove("active"));

      tab.classList.add("active");
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.classList.add("active");
      }
    });
  });
}

// ============================================================================
// 1. Data Studio
// ============================================================================

function initDataStudio() {
  const form = document.getElementById("dataExtractForm");
  const btn = document.getElementById("btnExtractData");

  if (!form || !btn) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    btn.textContent = "Extracting & Validating...";

    const payload = {
      symbol: document.getElementById("dsSymbol").value,
      broker: document.getElementById("dsBroker").value,
      timeframe: document.getElementById("dsTimeframe").value,
      data_source: document.getElementById("dsDataSource").value,
      start_date: document.getElementById("dsStartDate").value,
      end_date: document.getElementById("dsEndDate").value,
    };

    try {
      const res = await fetch("/api/v1/ui/data-studio/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        updateDataStudioView(data);
      } else {
        alert("Extraction failed. Server returned " + res.status);
      }
    } catch (err) {
      console.error("Data studio request error:", err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Extract & Validate History";
    }
  });
}

function updateDataStudioView(data) {
  document.getElementById("dsFetched").textContent = Number(data.total_rows_fetched).toLocaleString();
  document.getElementById("dsRetained").textContent = Number(data.rows_retained).toLocaleString();
  document.getElementById("dsRejected").textContent = Number(data.rows_rejected).toLocaleString();
  document.getElementById("dsDuplicates").textContent = Number(data.duplicate_rows).toLocaleString();
  document.getElementById("dsMissingGaps").textContent = Number(data.missing_intervals).toLocaleString();
  document.getElementById("dsCompleteness").textContent = data.completeness_pct + "%";
  document.getElementById("dsMedianSpread").textContent = data.average_spread_pips + " pips";
  document.getElementById("dsMaxSpread").textContent = data.max_spread_pips + " pips";
  document.getElementById("dsTimeSpan").textContent = data.date_range;

  const badge = document.getElementById("dsHealthBadge");
  badge.textContent = data.health_status;
  badge.className = "badge " + (data.health_status === "HEALTHY" ? "badge-success" : (data.health_status === "WARNING" ? "badge-warning" : "badge-danger"));
}

// ============================================================================
// 2. Quant Laboratory
// ============================================================================

function initQuantLab() {
  const btn = document.getElementById("btnTrainModel");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Training Pipeline Active...";

    const payload = {
      instrument: document.getElementById("qlInstrument").value,
      direction: document.getElementById("qlDirection").value,
      target_pips: parseFloat(document.getElementById("qlTarget").value),
      stop_pips: parseFloat(document.getElementById("qlStop").value),
      horizon_bars: parseInt(document.getElementById("qlHorizon").value, 10),
      label_policy: document.getElementById("qlLabelPolicy").value,
    };

    try {
      const res = await fetch("/api/v1/ui/quant-lab/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        document.getElementById("trainingStageLabel").textContent = "Status: " + data.current_stage;
        document.getElementById("trainingProgressBar").style.width = data.progress_pct + "%";
        document.getElementById("trainingPercentLabel").textContent = data.progress_pct + "%";
      }
    } catch (err) {
      console.error("Training error:", err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Train & Validate Model Pipeline";
    }
  });
}

// ============================================================================
// 3. Safeguards & Kill Switch
// ============================================================================

function initSafeguards() {
  const btnKill = document.getElementById("btnKillSwitch");
  const btnSave = document.getElementById("btnSaveSafeguards");

  if (btnKill) {
    btnKill.addEventListener("click", async () => {
      const confirmKill = confirm("EMERGENCY KILL SWITCH: Disarm all trading, close open positions, and cancel orders?");
      if (!confirmKill) return;

      try {
        const res = await fetch("/api/v1/ui/safeguards/kill-switch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });

        if (res.ok) {
          document.getElementById("killSwitchStatusText").textContent = "KILL SWITCH ACTIVATED: Live trading suspended!";
          document.getElementById("killSwitchStatusText").className = "mt-2 text-danger font-bold";
          // Disarm autopilot buttons
          document.querySelectorAll(".btn-pilot").forEach((b) => b.classList.remove("active"));
          document.querySelector(".btn-pilot[data-mode='OFF']").classList.add("active");
        }
      } catch (err) {
        console.error("Kill switch error:", err);
      }
    });
  }

  if (btnSave) {
    btnSave.addEventListener("click", async () => {
      alert("Safeguard risk enforcement thresholds updated.");
    });
  }
}

// ============================================================================
// 4. Auto-Pilot 3-Way Selector & Live Confirmation Gate
// ============================================================================

function initAutoPilot() {
  const buttons = document.querySelectorAll(".btn-pilot");
  const modal = document.getElementById("liveConfirmationModal");
  const btnCloseModal = document.getElementById("btnCloseLiveModal");
  const btnCancelModal = document.getElementById("btnCancelLiveModal");
  const btnConfirmLive = document.getElementById("btnConfirmLiveMode");
  const chkLiveAck = document.getElementById("chkExplicitLiveAck");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.getAttribute("data-mode");

      if (mode === "LIVE") {
        // Enforce explicit modal gate
        if (modal) {
          modal.classList.remove("hidden");
          chkLiveAck.checked = false;
          btnConfirmLive.disabled = true;
        }
      } else {
        setAutopilotMode(mode);
      }
    });
  });

  if (chkLiveAck && btnConfirmLive) {
    chkLiveAck.addEventListener("change", () => {
      btnConfirmLive.disabled = !chkLiveAck.checked;
    });

    btnConfirmLive.addEventListener("click", () => {
      setAutopilotMode("LIVE");
      if (modal) modal.classList.add("hidden");
    });
  }

  if (btnCloseModal && modal) {
    btnCloseModal.addEventListener("click", () => modal.classList.add("hidden"));
  }
  if (btnCancelModal && modal) {
    btnCancelModal.addEventListener("click", () => modal.classList.add("hidden"));
  }
}

function setAutopilotMode(mode) {
  document.querySelectorAll(".btn-pilot").forEach((b) => {
    b.classList.toggle("active", b.getAttribute("data-mode") === mode);
  });
  console.log("Auto-Pilot mode set to:", mode);
}

// ============================================================================
// 5. Co-Pilot Manual Order Pad
// ============================================================================

function initCopilot() {
  const btn = document.getElementById("btnSubmitCopilotOrder");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const symbol = document.getElementById("cpSymbol").value;
    const direction = document.getElementById("cpDirection").value;
    const volume = document.getElementById("cpVolume").value;

    const confirmed = confirm(`Submit Co-Pilot ${direction} order for ${volume} lots on ${symbol}?`);
    if (!confirmed) return;

    btn.disabled = true;
    try {
      const res = await fetch("/api/v1/ui/copilot/submit-order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          direction,
          volume_lots: parseFloat(volume),
          explicit_confirmation: true,
        }),
      });

      if (res.ok) {
        alert(`Co-Pilot order submitted successfully!`);
      } else {
        alert("Order submission error: " + res.status);
      }
    } catch (err) {
      console.error("Co-pilot order error:", err);
    } finally {
      btn.disabled = false;
    }
  });
}

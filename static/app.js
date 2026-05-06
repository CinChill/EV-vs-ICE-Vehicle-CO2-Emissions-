const form = document.querySelector("#predictionForm");
const emptyState = document.querySelector("#emptyState");
const resultStack = document.querySelector("#resultStack");
const errorBox = document.querySelector("#errorBox");
const categoryResult = document.querySelector("#categoryResult");
const co2Result = document.querySelector("#co2Result");
const impactResult = document.querySelector("#impactResult");
const impactCard = document.querySelector("#impactCard");
const annualCo2Result = document.querySelector("#annualCo2Result");
const fiveYearCo2Result = document.querySelector("#fiveYearCo2Result");
const categoryProbabilities = document.querySelector("#categoryProbabilities");
const impactProbabilities = document.querySelector("#impactProbabilities");

function setFormValues(values) {
  Object.entries(values).forEach(([name, value]) => {
    const field = form.elements.namedItem(name);
    if (field) {
      field.value = value;
    }
  });
}

function formPayload() {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function formatNumber(value, digits = 0) {
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function renderBars(container, rows) {
  container.innerHTML = "";
  if (!rows || rows.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No probability data.";
    container.appendChild(empty);
    return;
  }

  rows.slice(0, 5).forEach((row) => {
    const percent = Math.round(row.probability * 100);
    const item = document.createElement("div");
    item.className = "bar-row";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.title = row.label;
    label.textContent = row.label;

    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = `${percent}%`;
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = `${percent}%`;

    item.append(label, track, value);
    container.appendChild(item);
  });
}

function renderResult(payload) {
  emptyState.classList.add("hidden");
  resultStack.classList.remove("hidden");

  categoryResult.textContent = payload.vehicle_category;
  co2Result.textContent = formatNumber(payload.predicted_co2_g_per_mile, 1);
  impactResult.textContent = payload.co2_based_impact_level;
  impactCard.dataset.impact = payload.co2_based_impact_level;
  annualCo2Result.textContent = formatNumber(payload.annual_kg_co2, 0);
  fiveYearCo2Result.textContent = formatNumber(payload.five_year_kg_co2, 0);
  renderBars(categoryProbabilities, payload.category_probabilities);
  renderBars(impactProbabilities, payload.impact_probabilities);
}

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    const preset = window.vehicleOptions.presets[button.dataset.preset];
    if (preset) {
      setFormValues(preset);
    }
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.textContent = "Calculating";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      const message = payload.error || "Model files were not found.";
      showError(message);
      return;
    }
    renderResult(payload);
  } catch (error) {
    showError(error.message);
  } finally {
    submitButton.disabled = !window.modelStatus.ready;
    submitButton.textContent = "Predict";
  }
});

form.addEventListener("reset", () => {
  window.setTimeout(() => setFormValues(window.vehicleOptions.defaults), 0);
  emptyState.classList.remove("hidden");
  resultStack.classList.add("hidden");
  clearError();
});

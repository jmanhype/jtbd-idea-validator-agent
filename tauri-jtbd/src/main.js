const { invoke } = window.__TAURI__.core;

let validateForm;
let loadingEl;
let resultsEl;
let errorEl;
let settingsModal;
let apiKeyInput;
let settingsBtn;
let closeSettingsBtn;
let saveSettingsBtn;
let apiStatus;

// LocalStorage key for API key
const API_KEY_STORAGE = "jtbd_api_key";

// Load API key from localStorage
function getApiKey() {
  return localStorage.getItem(API_KEY_STORAGE) || "";
}

// Save API key to localStorage
function saveApiKey(apiKey) {
  localStorage.setItem(API_KEY_STORAGE, apiKey);
}

// Show settings modal
function showSettings() {
  apiKeyInput.value = getApiKey();
  settingsModal.style.display = "flex";
  apiStatus.textContent = "";
  apiStatus.className = "api-status";
}

// Hide settings modal
function hideSettings() {
  settingsModal.style.display = "none";
}

// Save settings
function saveSettings() {
  const apiKey = apiKeyInput.value.trim();

  if (!apiKey) {
    apiStatus.textContent = "⚠️ Please enter an API key";
    apiStatus.className = "api-status error";
    return;
  }

  if (!apiKey.startsWith("sk-")) {
    apiStatus.textContent = "⚠️ API key should start with 'sk-'";
    apiStatus.className = "api-status error";
    return;
  }

  saveApiKey(apiKey);
  apiStatus.textContent = "✓ API key saved successfully!";
  apiStatus.className = "api-status success";

  setTimeout(() => {
    hideSettings();
  }, 1000);
}

async function validateIdea(idea) {
  try {
    // Check for API key
    const apiKey = getApiKey();
    if (!apiKey) {
      errorEl.textContent = "⚠️ Please set your API key in Settings first";
      errorEl.style.display = "block";
      showSettings();
      return;
    }

    // Show loading state
    loadingEl.style.display = "block";
    resultsEl.style.display = "none";
    errorEl.style.display = "none";

    // Call Rust backend with API key
    const result = await invoke("validate_idea", { idea, apiKey });

    // Hide loading
    loadingEl.style.display = "none";

    // Display results
    document.getElementById("score").textContent = result.score.toFixed(1);
    document.getElementById("rationale").textContent = result.rationale;

    // Clear and populate assumptions
    const assumptionsList = document.getElementById("assumptions");
    assumptionsList.innerHTML = "";
    result.key_assumptions.forEach((assumption) => {
      const li = document.createElement("li");
      li.textContent = assumption;
      assumptionsList.appendChild(li);
    });

    // Clear and populate recommendations
    const recommendationsList = document.getElementById("recommendations");
    recommendationsList.innerHTML = "";
    result.recommendations.forEach((recommendation) => {
      const li = document.createElement("li");
      li.textContent = recommendation;
      recommendationsList.appendChild(li);
    });

    // Show results
    resultsEl.style.display = "block";

    // Scroll to results
    resultsEl.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    loadingEl.style.display = "none";
    errorEl.textContent = `Error: ${error}`;
    errorEl.style.display = "block";
    console.error("Validation error:", error);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  // Get all elements
  validateForm = document.querySelector("#validate-form");
  loadingEl = document.querySelector("#loading");
  resultsEl = document.querySelector("#results");
  errorEl = document.querySelector("#error");
  settingsModal = document.querySelector("#settings-modal");
  apiKeyInput = document.querySelector("#api-key");
  settingsBtn = document.querySelector("#settings-btn");
  closeSettingsBtn = document.querySelector("#close-settings");
  saveSettingsBtn = document.querySelector("#save-settings");
  apiStatus = document.querySelector("#api-status");

  // Settings event listeners
  settingsBtn.addEventListener("click", showSettings);
  closeSettingsBtn.addEventListener("click", hideSettings);
  saveSettingsBtn.addEventListener("click", saveSettings);

  // Close modal when clicking outside
  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
      hideSettings();
    }
  });

  // Close modal with Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && settingsModal.style.display === "flex") {
      hideSettings();
    }
  });

  // Validate form submission
  validateForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const idea = {
      title: document.getElementById("title").value,
      problem_statement: document.getElementById("problem").value,
      solution_overview: document.getElementById("solution").value,
      target_customer: document.getElementById("target").value,
    };

    validateIdea(idea);
  });

  // Show settings on first run if no API key
  if (!getApiKey()) {
    setTimeout(() => {
      showSettings();
    }, 500);
  }
});

const { invoke } = window.__TAURI__.core;

let validateForm;
let loadingEl;
let resultsEl;
let errorEl;

async function validateIdea(idea) {
  try {
    // Show loading state
    loadingEl.style.display = "block";
    resultsEl.style.display = "none";
    errorEl.style.display = "none";

    // Call Rust backend
    const result = await invoke("validate_idea", { idea });

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
  validateForm = document.querySelector("#validate-form");
  loadingEl = document.querySelector("#loading");
  resultsEl = document.querySelector("#results");
  errorEl = document.querySelector("#error");

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
});

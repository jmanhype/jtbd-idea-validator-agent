use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BusinessIdea {
    pub title: String,
    pub problem_statement: String,
    pub solution_overview: String,
    pub target_customer: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub score: f64,
    pub rationale: String,
    pub key_assumptions: Vec<String>,
    pub recommendations: Vec<String>,
}

// OpenAI API structures
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessageResponse,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessageResponse {
    content: String,
}

// DSPy-inspired Signature trait
trait Signature {
    fn get_instruction(&self) -> String;
    fn format_input(&self) -> String;
}

// Judge Signature (DSPy-style)
struct JudgeScoreSig {
    summary: String,
}

impl Signature for JudgeScoreSig {
    fn get_instruction(&self) -> String {
        "Evaluate this business idea across 5 criteria:
1. Underserved Opportunity (0-2)
2. Strategic Impact (0-2)
3. Market Scale (0-2)
4. Solution Differentiability (0-2)
5. Business Model Innovation (0-2)

Return ONLY valid JSON with \"score\" (0-10 total) and \"rationale\" (brief explanation).
Example: {\"score\": 7.5, \"rationale\": \"Strong market opportunity but...\"}\n".to_string()
    }

    fn format_input(&self) -> String {
        format!("Business Idea:\n{}", self.summary)
    }
}

// Deconstruct Signature (DSPy-style)
struct DeconstructSig {
    idea: String,
}

impl Signature for DeconstructSig {
    fn get_instruction(&self) -> String {
        "Extract 3-5 key business assumptions from this idea.
Return ONLY valid JSON array like: [{\"text\": \"assumption text\", \"confidence\": \"high\"}, ...]
No additional text before or after the JSON.\n".to_string()
    }

    fn format_input(&self) -> String {
        format!("Business Idea:\n{}", self.idea)
    }
}

// DSPy-style Predict module
async fn predict<T: Signature>(signature: T, api_key: &str) -> Result<String> {
    let client = reqwest::Client::new();

    let system_message = OpenAIMessage {
        role: "system".to_string(),
        content: signature.get_instruction(),
    };

    let user_message = OpenAIMessage {
        role: "user".to_string(),
        content: signature.format_input(),
    };

    let request = OpenAIRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![system_message, user_message],
        temperature: 0.3,
    };

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await?
        .json::<OpenAIResponse>()
        .await?;

    Ok(response.choices[0].message.content.clone())
}

#[tauri::command]
async fn validate_idea(idea: BusinessIdea) -> Result<ValidationResult, String> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY environment variable not set. Please set it before running the app.".to_string())?;

    // Create idea summary
    let summary = format!(
        "Title: {}\nProblem: {}\nSolution: {}\nTarget: {}",
        idea.title,
        idea.problem_statement,
        idea.solution_overview,
        idea.target_customer
    );

    // Step 1: Extract assumptions using DSPy-style signature
    let deconstruct_sig = DeconstructSig {
        idea: summary.clone(),
    };

    let assumptions_json = predict(deconstruct_sig, &api_key).await
        .map_err(|e| format!("Failed to extract assumptions: {}", e))?;

    // Parse assumptions
    let assumptions: Vec<String> = serde_json::from_str::<Vec<serde_json::Value>>(&assumptions_json)
        .ok()
        .map(|arr| arr.iter()
            .filter_map(|a| a.get("text").and_then(|t| t.as_str()).map(String::from))
            .collect())
        .unwrap_or_else(|| vec![
            "AI-generated assumption extraction (see full response for details)".to_string()
        ]);

    // Step 2: Score the idea using DSPy-style signature
    let judge_sig = JudgeScoreSig {
        summary: summary.clone(),
    };

    let scorecard_json = predict(judge_sig, &api_key).await
        .map_err(|e| format!("Failed to score idea: {}", e))?;

    // Parse scorecard
    let scorecard: serde_json::Value = serde_json::from_str(&scorecard_json)
        .map_err(|e| format!("Failed to parse scorecard JSON: {}. Response was: {}", e, scorecard_json))?;

    let score = scorecard.get("score")
        .or_else(|| scorecard.get("total"))
        .and_then(|s| s.as_f64())
        .unwrap_or(5.0);

    let rationale = scorecard.get("rationale")
        .or_else(|| scorecard.get("reasoning"))
        .and_then(|r| r.as_str())
        .unwrap_or("No rationale provided")
        .to_string();

    // Generate recommendations based on score
    let recommendations = if score < 4.0 {
        vec![
            "Consider pivoting to a more underserved market".to_string(),
            "Strengthen your unique value proposition".to_string(),
            "Validate core assumptions with customer interviews".to_string(),
        ]
    } else if score < 7.0 {
        vec![
            "Test key assumptions with MVPs".to_string(),
            "Research competitive differentiation more deeply".to_string(),
            "Develop a clearer go-to-market strategy".to_string(),
        ]
    } else {
        vec![
            "Proceed with prototype development".to_string(),
            "Set up metrics to track early traction".to_string(),
            "Begin building strategic partnerships".to_string(),
        ]
    };

    Ok(ValidationResult {
        score,
        rationale,
        key_assumptions: assumptions,
        recommendations,
    })
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, validate_idea])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

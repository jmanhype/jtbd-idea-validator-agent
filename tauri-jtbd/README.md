# JTBD Idea Validator - Tauri + Rust Prototype

A **Tauri desktop app** that validates business ideas using **DSPy-inspired Rust architecture**. This prototype demonstrates how to port DSPy concepts from Python to Rust for building LLM-powered applications.

## 🚀 What It Does

This app performs Jobs-to-Be-Done (JTBD) business idea validation:

- **Assumption Extraction**: Identifies 3-5 key business assumptions
- **Scoring**: Evaluates ideas across 5 criteria (0-10 scale):
  - Underserved Opportunity (0-2)
  - Strategic Impact (0-2)
  - Market Scale (0-2)
  - Solution Differentiability (0-2)
  - Business Model Innovation (0-2)
- **Recommendations**: Provides actionable next steps based on score

## 🏗️ Architecture

### DSPy-Inspired Rust Pattern

The Rust implementation follows DSPy's **Signature** and **Predict** pattern:

```rust
// DSPy-style Signature trait
trait Signature {
    fn get_instruction(&self) -> String;
    fn format_input(&self) -> String;
}

// Judge Signature (like DSPy's JudgeScoreSig)
struct JudgeScoreSig {
    summary: String,
}

impl Signature for JudgeScoreSig {
    fn get_instruction(&self) -> String {
        "Evaluate this business idea across 5 criteria:
        1. Underserved Opportunity (0-2)
        ..."
    }

    fn format_input(&self) -> String {
        format!("Business Idea:\n{}", self.summary)
    }
}

// DSPy-style Predict module
async fn predict<T: Signature>(signature: T, api_key: &str) -> Result<String> {
    // Makes OpenAI API call with formatted signature
}
```

### Tech Stack

- **Frontend**: Vanilla JavaScript + HTML/CSS
- **Backend**: Rust with Tauri
- **LLM**: OpenAI GPT-4o-mini via OpenRouter or OpenAI directly
- **Pattern**: DSPy-inspired Signatures for structured LLM prompting
- **API**: OpenRouter (cost-effective multi-model access) or OpenAI

## 📦 Setup

### Prerequisites

- Rust (latest stable)
- Node.js and npm
- OpenAI API key

### Installation

```bash
cd tauri-jtbd
npm install
```

### Configuration

Set your API key as an environment variable. The app supports both OpenRouter and OpenAI:

**OpenRouter (recommended for cost-effective access to multiple models):**
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

**OpenAI (alternative):**
```bash
export OPENAI_API_KEY=sk-...
```

## 🎯 Running the App

### Development Mode

```bash
npm run tauri dev
```

This will:
1. Build the Rust backend
2. Start the frontend dev server
3. Launch the desktop app

### Production Build

```bash
npm run tauri build
```

## 💡 Usage

1. **Enter Your Business Idea**:
   - Title
   - Problem Statement
   - Solution Overview
   - Target Customer

2. **Click "Validate Idea"**

3. **Review Results**:
   - Overall score (0-10)
   - Detailed rationale
   - Key assumptions identified
   - Actionable recommendations

## 🔧 Code Structure

```
tauri-jtbd/
├── src/                    # Frontend
│   ├── index.html         # Main UI
│   ├── main.js            # JavaScript logic
│   └── styles.css         # Styling
└── src-tauri/             # Rust backend
    ├── Cargo.toml         # Dependencies
    └── src/
        └── lib.rs         # DSPy-inspired validator
```

### Key Files

- **`src-tauri/src/lib.rs`**: Core validation logic with DSPy-inspired Signature pattern
- **`src/main.js`**: Frontend logic for calling Rust backend via Tauri commands
- **`src/index.html`**: User interface for idea input and results display

## 🎓 Learning Notes

### DSPy Concepts in Rust

This prototype demonstrates:

1. **Signatures**: Trait-based system for defining LLM input/output specifications
2. **Predict Module**: Generic function that takes any Signature and executes it
3. **Composable Pipeline**: Multiple signatures chained together (Deconstruct → Judge)

### Differences from Python DSPy

- **No Macros**: Uses Rust traits instead of Python decorators
- **Type Safety**: Compile-time checking for signature definitions
- **Direct API Calls**: Simplified implementation without full DSPy framework
- **Async/Await**: Tokio async runtime for non-blocking LLM calls

## 🚧 Future Enhancements

To make this a full DSPy implementation in Rust:

1. **Add DSRs Integration**: Once DSRs stabilizes for stable Rust
2. **Prompt Optimization**: Implement GEPA-style evolutionary optimization
3. **Chain-of-Thought**: Add reasoning traces
4. **Module System**: Build composable modules like Python DSPy
5. **Example System**: Add few-shot learning capabilities

## 📝 Notes

- This is a **prototype** demonstrating DSPy concepts in Rust
- Uses direct OpenAI API calls instead of full DSRs library (which requires Rust nightly)
- Simplified implementation focusing on core Signature/Predict pattern
- Production use would benefit from the full DSRs framework when it stabilizes

## 🔗 Related

- Original Python DSPy: https://github.com/stanfordnlp/dspy
- DSRs (Rust DSPy): https://github.com/krypticmouse/DSRs
- Tauri: https://tauri.app

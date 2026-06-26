import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# Client Initialization

# Initialize OpenRouter client
try:
    api_key_or = os.environ.get(
        "OPENROUTER_API_KEY") #or "key "
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_or,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "LLM Joke Evaluator"
        }
    )
except Exception as e:
    st.error(f"Error initializing OpenRouter client: {e}")

# Initialize Native OpenAI client
try:

    api_key_oa = os.environ.get("OPENAI_API_KEY") #or "key "
    openai_client = OpenAI(api_key=api_key_oa)
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")


#  Unified Router Function ---

def fetch_joke(provider, model_id):
    """
    Routes the request to the correct API client based on the provider.
    """
    if provider == "openai_native":
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=1.0
        )
    else:
        # Defaults to OpenRouter for all other multi-provider models
        response = openrouter_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=1.0
        )
    return response.choices[0].message.content


# UI Setup & Configuration ---

st.set_page_config(page_title="LLM Diversity & Joke Evaluator", layout="wide")
st.title("🃏 LLM Open-Endedness Evaluator")
st.write(
    "Evaluate how diverse and repetitive different frontier models are when prompted with the exact same request repeatedly."
)

st.sidebar.header("Configuration")
iterations = st.sidebar.slider("Number of Loops (Iterations)", min_value=2, max_value=10, value=5)

# Mapping models to their provider types and actual API IDs
models_to_test = {
    "Google: Gemma 4 26b (Free)": ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "Meta: Llama 3.3 70b": ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "OpenAI: GPT-4o Mini": ("openrouter", "openai/gpt-4o-mini"),
    "OpenAI: GPT-OSS 120b (Free)": ("openrouter", "openai/gpt-oss-120b:free"),
    "OpenAI: GPT-OSS 20b (Free)": ("openrouter", "openai/gpt-oss-20b:free"),
    "NVIDIA: Nemotron 3 Super (Free)": ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free")
}

selected_models = st.sidebar.multiselect(
    "Select Models to Benchmark",
    options=list(models_to_test.keys()),
    default=list(models_to_test.keys())[:3]
)

#  Execution Loop ---

if st.sidebar.button("Run Evaluation", type="primary"):
    if not selected_models:
        st.warning("Please select at least one model to run.")
    else:
        results = {model: [] for model in selected_models}

        progress_bar = st.progress(0)
        total_steps = len(selected_models) * iterations
        step = 0

        cols = st.columns(len(selected_models))

        for idx, model_label in enumerate(selected_models):
            provider, model_id = models_to_test[model_label]
            with cols[idx]:
                st.subheader(model_label)
                placeholder = st.empty()

                jokes_accumulated = []
                for i in range(iterations):
                    with st.spinner(f"Fetching Loop {i + 1}..."):
                        try:
                            # Passes both the provider designation and the model ID string
                            joke = fetch_joke(provider, model_id)
                            jokes_accumulated.append(joke)
                            results[model_label].append(joke)
                        except Exception as e:
                            # Print the raw error to the PyCharm terminal console for quick debugging
                            print(f"CRASH LOG [{model_label}]: {e}")
                            error_msg = f"Error: {str(e)[:50]}..."
                            jokes_accumulated.append(error_msg)
                            results[model_label].append(error_msg)

                    with placeholder.container():
                        for run_idx, jk in enumerate(jokes_accumulated):
                            st.markdown(f"**Run {run_idx + 1}:** {jk}")
                            st.write("---")

                    step += 1
                    progress_bar.progress(step / total_steps)

        st.success("Evaluation complete!")

        #  Data Metrics Table ---
        st.header("📊 Diversity Metrics Dashboard")

        metrics_data = []
        for model_label, joke_list in results.items():
            total_runs = len(joke_list)
            # Filter out error statements before calculating uniqueness metrics
            valid_jokes = [j for j in joke_list if not j.startswith("Error:")]

            if valid_jokes:
                unique_jokes = len(set([j.strip().lower() for j in valid_jokes]))
                diversity_score = (unique_jokes / len(valid_jokes)) * 100
                score_str = f"{diversity_score:.1f}%"
            else:
                unique_jokes = 0
                score_str = "N/A (All Runs Failed)"

            metrics_data.append({
                "Model": model_label,
                "Total Runs": total_runs,
                "Unique Jokes": unique_jokes,
                "Diversity Score": score_str
            })

        st.table(pd.DataFrame(metrics_data))
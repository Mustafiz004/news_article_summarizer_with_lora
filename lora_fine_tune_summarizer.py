import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATH = "Mustafiz004/fined_tuned_lora_model"


if "summary" not in st.session_state:
    st.session_state.summary = ""

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide"
)

# ------------------------------
# Custom CSS for better UI
# ------------------------------




st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(120deg, #f6f9fc, #eef2f7);
}

/* Big title */
.big-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #ff4b4b, #ff8c00, #1f77ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    text-align: center;
    color: #555;
    margin-bottom: 40px;
}

/* Article box */
.article-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #1f77ff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

/* Example button */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #1f77ff, #4CAF50);
    color: white;
    font-size: 16px;
    padding: 10px 25px;
    border-radius: 8px;
    border: none;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #4CAF50, #1f77ff);
    transform: scale(1.05);
    transition: 0.3s;
}

</style>
""", unsafe_allow_html=True)



# ------------------------------
# Load Model
# ------------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    return tokenizer, model, device



if "model_loaded" not in st.session_state:

    loading_text = st.empty()

    progress_bar = st.progress(0)

    loading_text.info(
        "🚀 AI model is starting for the first time.\n\n"
        "This may take several minutes..."
    )

    progress_bar.progress(20)

    tokenizer, model, device = load_model()

    progress_bar.progress(100)

    loading_text.success("✅ Model Ready!")

    st.session_state.model_loaded = True

else:

    tokenizer, model, device = load_model()




# ------------------------------
# Sidebar Controls
# ------------------------------

st.sidebar.title("⚙️ Settings")

min_len = st.sidebar.slider(
    "Minimum Summary Length",
    min_value=50,
    max_value=200,
    value=100,
    step=10
)

max_len = st.sidebar.slider(
    "Maximum Summary Length",
    min_value=120,
    max_value=400,
    value=220,
    step=20
)

if min_len >= max_len:
    st.error("Minimum length must be smaller than maximum length.")
    st.stop()

beam_size = st.sidebar.slider(
    "Beam Search Size",
    1, 8, 4
)

st.sidebar.markdown("---")

st.sidebar.write("Model Device:")
st.sidebar.success(device.upper())

# ------------------------------
# Main Page
# ------------------------------

st.markdown(
    '<p class="big-title">📰 AI News Article Summarizer</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Paste a long news article and let AI generate a concise summary using advanced NLP models.</p>',
    unsafe_allow_html=True
)

# Info banner
st.info("✨ Powered by Transformer Models | LoRA + QLoRA Fine-tuning")

st.markdown("---")



# Example article button

st.markdown("### 📄 Try an Example Article")

example_article = """
Ever noticed how plane seats appear to be getting smaller and smaller? 
With increasing numbers of people taking to the skies, some experts are questioning 
if having such packed out planes is putting passengers at risk. They say that the 
shrinking space on aeroplanes is not only uncomfortable but may also pose health 
and safety concerns for passengers.
"""

col1, col2 = st.columns([1,2])

with col1:

    if st.button("🚀 Load Example Article"):
        st.session_state["article"] = example_article

with col2:
    st.success("Click the button to auto-fill a sample article.")


# ------------------------------
# Article Input Section
# ------------------------------

st.markdown("### ✍️ Enter Your News Article")

st.markdown(
"""
Paste a **long article, blog, or report**, and the AI will generate a concise summary.
"""
)

article = st.text_area(
    label="",
    height=300,
    value=st.session_state.get("article", ""),
    placeholder="Paste your full article here..."
)

# ------------------------------
# Article Statistics
# ------------------------------

word_count = len(article.split())
char_count = len(article)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📝 Words", word_count)

with col2:
    st.metric("🔤 Characters", char_count)

with col3:
    estimated_time = max(1, word_count // 200)
    st.metric("⏱ Reading Time (min)", estimated_time)



st.markdown("---")

generate_button = st.button("🚀 Generate AI Summary", use_container_width=True)


# ------------------------------
# Generate Summary
# ------------------------------

if generate_button:

    if article.strip() == "":
        st.warning("⚠️ Please enter article text before generating summary.")

    else:

        st.markdown("### 🤖 AI Processing")

        progress = st.progress(0)

        status = st.empty()

        with st.spinner("AI model is analyzing the article..."):

            status.info("Step 1/4 — Tokenizing article")
            progress.progress(20)

            inputs = tokenizer(
                article,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )

            status.info("Step 2/4 — Preparing tensors")
            progress.progress(40)

            inputs = {k: v.to(device) for k, v in inputs.items()}

            status.info("Step 3/4 — Running Transformer model")
            progress.progress(70)

            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_len,
                min_length=min_len,
                num_beams=beam_size,
                early_stopping=True
            )

            status.info("Step 4/4 — Decoding summary")
            progress.progress(90)

            st.session_state.summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

            progress.progress(100)

        status.success("✅ Summary generated successfully!")


# ------------------------------
# Output Section
# ------------------------------

if st.session_state.summary:

    st.markdown("## 📄 Generated Summary")

    st.markdown(
        f"""
        <div class="result-box">
        {st.session_state.summary}
        </div>
        """,
        unsafe_allow_html=True
    )

    summary_words = len(st.session_state.summary.split())

    compression = round((summary_words / max(1, word_count)) * 100, 2)

    st.markdown("---")

    st.markdown("### 📊 Summary Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg,#4facfe,#00f2fe);
                padding:20px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:20px;
                font-weight:bold;">
                📝 Summary Words
                <h1 style="margin:5px">{summary_words}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg,#43e97b,#38f9d7);
                padding:20px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:20px;
                font-weight:bold;">
                📉 Compression Rate
                <h1 style="margin:5px">{compression}%</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📉 Visual Compression")

    st.progress(min(int(compression), 100))

    st.caption(
        f"Original article: **{word_count} words** → "
        f"Summary: **{summary_words} words** "
        f"({compression}% of original length)"
    )


st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.download_button(
        label="⬇ Download Summary",
        data=st.session_state.summary,
        file_name="ai_summary.txt",
        use_container_width=True
    )

with col2:

    st.code(st.session_state.summary, language="text")

st.success("🎉 Your AI-generated summary is ready!")


# ------------------------------
# Footer
# ------------------------------

st.markdown("---")
st.markdown("All Right Reserved ❤️ Md. Mustafizur Rahman")
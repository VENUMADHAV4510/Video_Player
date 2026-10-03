import streamlit as st
from faster_whisper import WhisperModel
import torch
import os
import tempfile
import subprocess
import json
import html

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Whisper Subtitle Player",
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle-text {
    text-align: center;
    color: #888;
    margin-bottom: 30px;
}

.stButton button {
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Whisper Subtitle Player</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">'
    'Upload a video and generate synchronized English subtitles'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "large-v3-turbo"
LANGUAGE = "en"

# ============================================================
# GPU
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    DEVICE = "cuda"
    COMPUTE_TYPE = "int8_float16"
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

# ============================================================
# LOAD WHISPER MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    return model


# ============================================================
# GET VIDEO DURATION
# ============================================================

def get_duration(video_path):

    try:

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                video_path
            ],
            capture_output=True,
            text=True
        )

        data = json.loads(result.stdout)

        return float(data["format"]["duration"])

    except Exception:

        return 0


# ============================================================
# SRT TIME
# ============================================================

def srt_time(seconds):

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(seconds % 60)

    milliseconds = int(
        round(
            (seconds - int(seconds)) * 1000
        )
    )

    if milliseconds >= 1000:

        milliseconds = 0
        secs += 1

    if secs >= 60:

        secs = 0
        minutes += 1

    if minutes >= 60:

        minutes = 0
        hours += 1

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


# ============================================================
# SRT TO WEBVTT
# ============================================================

def srt_to_vtt(srt_text):

    lines = srt_text.splitlines()

    output = ["WEBVTT", ""]

    for line in lines:

        if "-->" in line:

            line = line.replace(",", ".")

        output.append(line)

    return "\n".join(output)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        f"**Model:** `{MODEL_NAME}`"
    )

    st.write(
        f"**Language:** `{LANGUAGE}`"
    )

    if CUDA_AVAILABLE:

        st.success(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        st.warning(
            "CUDA not available — using CPU"
        )

# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "📁 Upload Video",
    type=[
        "mp4",
        "mkv",
        "avi",
        "mov",
        "webm",
        "mpeg",
        "mpg"
    ]
)

# ============================================================
# MAIN
# ============================================================

if uploaded_video:

    # ========================================================
    # SAVE VIDEO
    # ========================================================

    video_suffix = os.path.splitext(
        uploaded_video.name
    )[1]

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=video_suffix
    )

    temp_video.write(
        uploaded_video.read()
    )

    temp_video.close()

    video_path = temp_video.name

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    duration = get_duration(video_path)

    st.info(
        f"🎥 **{uploaded_video.name}**  |  "
        f"Duration: **{duration:.2f} seconds**"
    )

    # ========================================================
    # TRANSCRIPTION BUTTON
    # ========================================================

    if st.button(
        "🎙️ Generate Subtitles",
        type="primary"
    ):

        try:

            # ================================================
            # LOAD MODEL
            # ================================================

            with st.spinner(
                "Loading Whisper model..."
            ):

                model = load_model()

            st.success(
                "Whisper model loaded."
            )

            # ================================================
            # TRANSCRIBE
            # ================================================

            st.write(
                "### 🎙️ Transcribing..."
            )

            progress = st.progress(0)

            status = st.empty()

            segments, info = model.transcribe(

                video_path,

                language=LANGUAGE,

                beam_size=5,

                vad_filter=False,

                condition_on_previous_text=True,

                temperature=0.0,

                word_timestamps=False
            )

            # ================================================
            # CREATE SRT
            # ================================================

            srt_output = ""

            subtitle_number = 1

            for segment in segments:

                text = segment.text.strip()

                if not text:
                    continue

                start = segment.start
                end = segment.end

                srt_output += (
                    f"{subtitle_number}\n"
                )

                srt_output += (
                    f"{srt_time(start)} --> "
                    f"{srt_time(end)}\n"
                )

                srt_output += (
                    f"{text}\n\n"
                )

                subtitle_number += 1

                # =========================================
                # PROGRESS
                # =========================================

                if duration > 0:

                    percentage = min(
                        int(
                            (end / duration) * 100
                        ),
                        100
                    )

                    progress.progress(
                        percentage
                    )

                    status.write(
                        f"Transcribing: "
                        f"{percentage}%"
                    )

            progress.progress(100)

            status.success(
                "Transcription complete!"
            )

            # ================================================
            # SAVE SRT
            # ================================================

            srt_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".srt",
                mode="w",
                encoding="utf-8"
            )

            srt_file.write(
                srt_output
            )

            srt_file.close()

            # ================================================
            # SAVE TO SESSION
            # ================================================

            st.session_state["video_path"] = video_path

            st.session_state["srt"] = srt_output

            st.session_state["srt_path"] = srt_file.name

            st.session_state["video_name"] = uploaded_video.name

            st.session_state["language"] = info.language

            st.session_state["language_probability"] = (
                info.language_probability
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Transcription error: {e}"
            )

# ============================================================
# PLAYER
# ============================================================

if (
    "video_path" in st.session_state
    and
    "srt" in st.session_state
):

    st.markdown("---")

    st.subheader("▶️ Video Player")

    video_path = st.session_state["video_path"]

    srt_text = st.session_state["srt"]

    # ========================================================
    # CONVERT SRT TO WEBVTT
    # ========================================================

    vtt_text = srt_to_vtt(
        srt_text
    )

    # ========================================================
    # CREATE VTT FILE
    # ========================================================

    vtt_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".vtt",
        mode="w",
        encoding="utf-8"
    )

    vtt_file.write(
        vtt_text
    )

    vtt_file.close()

    # ========================================================
    # READ VIDEO
    # ========================================================

    with open(
        video_path,
        "rb"
    ) as video_file:

        video_bytes = video_file.read()

    # ========================================================
    # READ VTT
    # ========================================================

    with open(
        vtt_file.name,
        "rb"
    ) as subtitle_file:

        subtitle_bytes = subtitle_file.read()

    # ========================================================
    # VIDEO PLAYER
    # ========================================================

    st.video(
        video_bytes,
        subtitles=subtitle_bytes
    )

    # ========================================================
    # DOWNLOAD SRT
    # ========================================================

    st.download_button(
        label="⬇️ Download SRT Subtitles",
        data=srt_text,
        file_name="transcript.srt",
        mime="text/plain"
    )

    # ========================================================
    # TRANSCRIPTION INFO
    # ========================================================

    st.markdown("---")

    st.subheader("📊 Transcription Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Language",
            st.session_state.get(
                "language",
                "en"
            )
        )

    with col2:

        st.metric(
            "Probability",
            f"{st.session_state.get('language_probability', 0):.2%}"
        )

    with col3:

        subtitle_count = len(
            [
                x
                for x in srt_text.split("\n\n")
                if x.strip()
            ]
        )

        st.metric(
            "Subtitles",
            subtitle_count
        )

    # ========================================================
    # TRANSCRIPT
    # ========================================================

    st.markdown("---")

    st.subheader("📝 Transcript")

    st.text_area(
        "Generated subtitles",
        srt_text,
        height=400
    )

else:

    st.info(
        "👆 Upload a video above to get started."
    )

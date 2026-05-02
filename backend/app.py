import json
import uuid
from pathlib import Path

import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/chat"
CONVERSATIONS_FILE = Path("conversations.json")
DEV_MODE = False
SHOW_AB_IN_CHAT = False

# خليها False للمستخدم العادي
IS_ADMIN = False


st.set_page_config(
    page_title="Scam Phishing Chatbot",
    page_icon="🛡️",
    layout="centered"
)


# =========================
# Basic Storage
# =========================
def load_conversations():
    if CONVERSATIONS_FILE.exists():
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversations(conversations):
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def create_new_conversation():
    return {
        "id": str(uuid.uuid4()),
        "title": "محادثة جديدة",
        "messages": []
    }


# =========================
# UI Helpers
# =========================
def get_badge(classification, confidence):
    percent = int(float(confidence) * 100)

    if classification == "Phishing":
        return f"🔴 **Phishing — {percent}%**"
    elif classification == "Suspicious":
        return f"🟡 **Suspicious — {percent}%**"
    elif classification == "Needs Review":
        return f"⚪ **Needs Review — {percent}%**"
    return f"🟢 **Safe — {percent}%**"


def is_arabic_text(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in str(text or ""))


def get_ui_labels(user_message):
    if is_arabic_text(user_message):
        return {
            "reason": "السبب",
            "red_flags": "العلامات الخطيرة",
            "advice": "النصيحة",
            "no_red_flags": "لا توجد علامات واضحة.",
            "examples": "أمثلة",
            "practical_tips": "نصائح عملية",
            "summary": "الخلاصة",
            "uploaded_files": "📎 الملفات:",
            "checking_file": "تم رفع ملف/صورة للتحقق منها.",
            "spinner": "جاري التحليل...",
        }

    return {
        "reason": "Reason",
        "red_flags": "Red Flags",
        "advice": "Advice",
        "no_red_flags": "No clear red flags.",
        "examples": "Examples",
        "practical_tips": "Practical Tips",
        "summary": "Summary",
        "uploaded_files": "📎 Files:",
        "checking_file": "A file/image was uploaded for analysis.",
        "spinner": "Analyzing...",
    }


def get_model_info(data, answer=None):
    answer = answer or {}

    model_used = data.get("model_used") or answer.get("model_used") or "N/A"
    ab_variant = data.get("ab_variant") or answer.get("ab_variant") or "N/A"
    ab_enabled = data.get("ab_test_enabled") or answer.get("ab_test_enabled") or False

    return model_used, ab_variant, ab_enabled


def render_ab_card(model_used, ab_variant, latency=None, mode=None):
    if not SHOW_AB_IN_CHAT:
        return

    st.info(f"Model: {model_used} | Variant: {ab_variant} | Mode: {mode} | Latency: {latency}s")


def build_educational_saved_text(answer, labels=None):
    labels = labels or get_ui_labels("")

    title = answer.get("title", "Educational Explanation")
    explanation = answer.get("explanation", "")
    examples = answer.get("examples", [])
    tips = answer.get("practical_tips", [])
    short_summary = answer.get("short_summary", "")

    parts = []

    if title:
        parts.append(f"## {title}")

    if explanation:
        parts.append(explanation)

    if examples:
        examples_text = "\n".join([f"- {ex}" for ex in examples])
        parts.append(f"### {labels['examples']}\n{examples_text}")

    if tips:
        tips_text = "\n".join([f"- {tip}" for tip in tips])
        parts.append(f"### {labels['practical_tips']}\n{tips_text}")

    if short_summary:
        parts.append(f"### {labels['summary']}\n{short_summary}")

    return "\n\n".join(parts).strip()


def post_to_backend(user_input, messages_without_current_user, multipart_files, ab_model, timeout=180):
    payload = {
        "message": user_input,
        "chat_history": json.dumps(messages_without_current_user, ensure_ascii=False),
        "ab_model": ab_model
    }

    response = requests.post(
        API_URL,
        data=payload,
        files=multipart_files if multipart_files else None,
        timeout=timeout
    )

    if response.status_code != 200:
        raise RuntimeError(f"Backend Error {response.status_code}: {response.text}")

    return response.json()


def run_backend_request(user_input, messages_without_current_user, multipart_files, ab_choice):
    if ab_choice == "both":
        gpt_data = post_to_backend(
            user_input=user_input,
            messages_without_current_user=messages_without_current_user,
            multipart_files=multipart_files,
            ab_model="gpt",
            timeout=180
        )

        try:
            phi_data = post_to_backend(
                user_input=user_input,
                messages_without_current_user=messages_without_current_user,
                multipart_files=multipart_files,
                ab_model="phi",
                timeout=240
            )
            gpt_data["_admin_comparison_also_ran"] = True
            gpt_data["_admin_phi_logged"] = True
            gpt_data["_phi_status"] = "logged"
            gpt_data["_phi_mode"] = phi_data.get("mode", "unknown")
        except Exception as e:
            gpt_data["_admin_comparison_also_ran"] = False
            gpt_data["_admin_phi_logged"] = False
            gpt_data["_phi_error"] = str(e)

        return gpt_data

    return post_to_backend(
        user_input=user_input,
        messages_without_current_user=messages_without_current_user,
        multipart_files=multipart_files,
        ab_model=ab_choice,
        timeout=180
    )


# =========================
# Session State
# =========================
if "conversations" not in st.session_state:
    old_conversations = load_conversations()

    new_conv = create_new_conversation()
    st.session_state.conversations = [new_conv] + old_conversations
    st.session_state.current_conversation_id = new_conv["id"]

    save_conversations(st.session_state.conversations)

if not st.session_state.conversations:
    new_conv = create_new_conversation()
    st.session_state.conversations.append(new_conv)
    st.session_state.current_conversation_id = new_conv["id"]
    save_conversations(st.session_state.conversations)

if "current_conversation_id" not in st.session_state:
    new_conv = create_new_conversation()
    st.session_state.conversations.insert(0, new_conv)
    st.session_state.current_conversation_id = new_conv["id"]
    save_conversations(st.session_state.conversations)

if "ab_model_choice" not in st.session_state:
    st.session_state.ab_model_choice = "both"


def get_current_conversation():
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_conversation_id:
            return conv

    new_conv = create_new_conversation()
    st.session_state.conversations.insert(0, new_conv)
    st.session_state.current_conversation_id = new_conv["id"]
    save_conversations(st.session_state.conversations)
    return new_conv


def update_current_conversation(updated_conv):
    for i, conv in enumerate(st.session_state.conversations):
        if conv["id"] == updated_conv["id"]:
            st.session_state.conversations[i] = updated_conv
            break

    save_conversations(st.session_state.conversations)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("💬 المحادثات")

    if st.button("➕ محادثة جديدة"):
        new_conv = create_new_conversation()
        st.session_state.conversations.insert(0, new_conv)
        st.session_state.current_conversation_id = new_conv["id"]
        save_conversations(st.session_state.conversations)
        st.rerun()

    for conv_item in st.session_state.conversations:
        title = conv_item.get("title", "محادثة")
        if st.button(title, key=conv_item["id"]):
            st.session_state.current_conversation_id = conv_item["id"]
            st.rerun()

    if IS_ADMIN:
        st.divider()
        st.subheader("🧪 A/B Testing - Admin")
        st.session_state.ab_model_choice = st.selectbox(
            "Model routing",
            options=["both", "auto", "gpt", "phi"],
            index=["both", "auto", "gpt", "phi"].index(st.session_state.ab_model_choice),
            help=(
                "both = نفس السؤال يذهب إلى GPT و Phi ويسجل الاثنين في Dashboard. "
                "auto = يختار موديل واحد عشوائيًا. "
                "gpt/phi = إجبار موديل واحد."
            )
        )

    st.divider()

    if st.button("🗑️ مسح المحادثة الحالية"):
        conv = get_current_conversation()
        conv["messages"] = []
        conv["title"] = "محادثة جديدة"
        update_current_conversation(conv)
        st.rerun()

    if DEV_MODE:
        show_sources = st.checkbox("Show Retrieved Sources", value=False)
        show_raw = st.checkbox("Show Raw JSON", value=False)
        show_extracted = st.checkbox("Show Extracted Text", value=False)
    else:
        show_sources = False
        show_raw = False
        show_extracted = False


# =========================
# Main Page
# =========================
conv = get_current_conversation()
messages = conv["messages"]

st.title("🛡️ Scam & Phishing Chatbot")
st.caption("يدعم رفع جميع أنواع الملفات للتحليل الأمني.")

if not messages:
    st.markdown("### كيف أستطيع مساعدتكم؟")


for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        meta = msg.get("metadata") or {}
        if SHOW_AB_IN_CHAT and meta.get("model_used"):
            render_ab_card(
                model_used=meta.get("model_used", "N/A"),
                ab_variant=meta.get("ab_variant", "N/A"),
                latency=meta.get("latency"),
                mode=meta.get("mode")
            )

        if msg.get("files"):
            st.caption("📎 الملفات:")
            for file_name in msg["files"]:
                st.write(f"- {file_name}")


prompt = st.chat_input(
    "اكتب الرسالة أو ارفع أي ملف/صورة للتحقق...",
    accept_file="multiple"
)

if prompt:
    user_input = prompt.text or ""
    uploaded_files = prompt.files or []
    labels = get_ui_labels(user_input)

    file_names = [f.name for f in uploaded_files]

    user_display = user_input.strip()
    if not user_display and file_names:
        user_display = labels["checking_file"]

    messages.append({
        "role": "user",
        "content": user_display,
        "files": file_names
    })

    if conv["title"] == "محادثة جديدة":
        conv["title"] = user_display[:28] + "..." if len(user_display) > 28 else user_display

    update_current_conversation(conv)

    with st.chat_message("user"):
        st.write(user_display)
        if file_names:
            st.caption(labels["uploaded_files"])
            for name in file_names:
                st.write(f"- {name}")

    multipart_files = []

    for f in uploaded_files:
        multipart_files.append(
            (
                "files",
                (
                    f.name,
                    f.getvalue(),
                    f.type or "application/octet-stream"
                )
            )
        )

    try:
        with st.spinner(labels["spinner"]):
            data = run_backend_request(
                user_input=user_input,
                messages_without_current_user=messages[:-1],
                multipart_files=multipart_files,
                ab_choice=st.session_state.ab_model_choice
            )

        mode = data.get("mode", "unknown")
        latency = data.get("latency", 0)
        intent = data.get("intent", "unknown")
        answer = data.get("answer", {})
        model_used, ab_variant, ab_enabled = get_model_info(data, answer)

        with st.chat_message("assistant"):

            if SHOW_AB_IN_CHAT:
                render_ab_card(
                    model_used=model_used,
                    ab_variant=ab_variant,
                    latency=latency,
                    mode=mode
                )

            if mode == "MULTI_FILE_IMAGE_ANALYSIS":
                st.caption(f"Mode: {mode} | Latency: {latency}s")

                file_results = data.get("file_results", [])
                assistant_parts = []

                for item in file_results:
                    filename = item.get("filename", "file")
                    file_mode = item.get("mode", "")
                    file_answer = item.get("answer", {})

                    classification = file_answer.get("classification", "Suspicious")
                    confidence = float(file_answer.get("confidence", 0.5))
                    reason = file_answer.get("reason", "")
                    red_flags = file_answer.get("red_flags", [])
                    advice = file_answer.get("advice", "")

                    st.divider()
                    st.subheader(f"📎 {filename}")
                    st.caption(f"File Mode: {file_mode}")
                    st.markdown(get_badge(classification, confidence))

                    st.write(f"**{labels['reason']}:**")
                    st.write(reason)

                    st.write(f"**{labels['red_flags']}:**")
                    if red_flags:
                        for flag in red_flags:
                            st.write(f"- {flag}")
                    else:
                        st.write(labels["no_red_flags"])

                    st.write(f"**{labels['advice']}:**")
                    st.write(advice)

                    if show_extracted:
                        with st.expander("Extracted Text Preview"):
                            st.text(item.get("extracted_preview", ""))

                    assistant_parts.append(f"""
File: {filename}
Result: {classification} — {int(confidence * 100)}%

{labels['reason']}:
{reason}

{labels['advice']}:
{advice}
""".strip())

                assistant_text = "\n\n---\n\n".join(assistant_parts)

            elif mode == "LLM_EDUCATIONAL":
                st.caption(f"Mode: {mode} | Intent: {intent} | Latency: {latency}s")

                st.subheader(answer.get("title", "Educational Explanation"))
                st.write(answer.get("explanation", ""))

                examples = answer.get("examples", [])
                if examples:
                    st.subheader(labels["examples"])
                    for ex in examples:
                        st.write(f"- {ex}")

                tips = answer.get("practical_tips", [])
                if tips:
                    st.subheader(labels["practical_tips"])
                    for tip in tips:
                        st.write(f"- {tip}")

                st.subheader(labels["summary"])
                st.write(answer.get("short_summary", ""))

                assistant_text = build_educational_saved_text(answer, labels)

            else:
                classification = answer.get("classification", "Suspicious")
                confidence = float(answer.get("confidence", 0.5))
                reason = answer.get("reason", "")
                red_flags = answer.get("red_flags", [])
                advice = answer.get("advice", "")
                top_score = answer.get("retrieval_top_score", 0)
                threshold = data.get("rag_threshold", "-")

                st.markdown(get_badge(classification, confidence))

                st.caption(
                    f"Mode: {mode} | Intent: {intent} | "
                    f"Top Score: {top_score} | Threshold: {threshold} | "
                    f"Latency: {latency}s"
                )

                st.subheader(labels["reason"])
                st.write(reason)

                st.subheader(labels["red_flags"])
                if red_flags:
                    for flag in red_flags:
                        st.write(f"- {flag}")
                else:
                    st.write(labels["no_red_flags"])

                st.subheader(labels["advice"])
                st.write(advice)

                if show_sources:
                    with st.expander("Retrieved Sources"):
                        st.json(data.get("retrieved_sources", []))

                assistant_text = f"""
{classification} — {int(confidence * 100)}%

Mode: {mode}
Latency: {latency}s

{labels['reason']}:
{reason}

{labels['red_flags']}:
{chr(10).join([f'- {flag}' for flag in red_flags]) if red_flags else labels['no_red_flags']}

{labels['advice']}:
{advice}
""".strip()

            if show_raw:
                with st.expander("Raw API Response"):
                    st.json(data)

        messages.append({
            "role": "assistant",
            "content": assistant_text,
            "metadata": {
                "model_used": model_used,
                "ab_variant": ab_variant,
                "ab_test_enabled": ab_enabled,
                "latency": latency,
                "mode": mode,
                "intent": intent,
                "admin_comparison_mode": st.session_state.ab_model_choice == "both",
                "phi_logged_in_dashboard": data.get("_admin_phi_logged", False),
            }
        })

        conv["messages"] = messages
        update_current_conversation(conv)

    except Exception as e:
        st.error(f"Error: {e}")
import os
import json
from datetime import datetime
from collections import Counter
from typing import List, Dict

import streamlit as st
from langchain_core.documents import Document as LCDocument


def format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_folder_size(folder_path: str) -> int:
    total_size = 0

    if not os.path.exists(folder_path):
        return total_size

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass

    return total_size


def load_collection_documents(collection_path: str, documents_json: str = "documents.json"):
    json_path = os.path.join(collection_path, documents_json)

    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []


def collect_dashboard_stats(collections_dir: str, documents_json: str = "documents.json") -> Dict:
    stats = {
        "total_collections": 0,
        "searchable_collections": 0,
        "total_chunks": 0,
        "total_storage_bytes": 0,
        "file_type_counter": Counter(),
        "collection_rows": []
    }

    if not os.path.exists(collections_dir):
        return stats

    for collection_name in sorted(os.listdir(collections_dir)):
        collection_path = os.path.join(collections_dir, collection_name)

        if not os.path.isdir(collection_path):
            continue

        stats["total_collections"] += 1

        documents = load_collection_documents(
            collection_path=collection_path,
            documents_json=documents_json
        )

        has_index = (
            os.path.exists(os.path.join(collection_path, "index.faiss"))
        )

        if documents and has_index:
            stats["searchable_collections"] += 1

        chunk_count = len(documents)
        storage_bytes = get_folder_size(collection_path)

        file_names = set()
        file_types = Counter()

        for item in documents:
            metadata = item.get("metadata", {})
            file_name = metadata.get("file_name", "Unknown")
            file_type = metadata.get("file_type", "Unknown")

            file_names.add(file_name)
            file_types[file_type] += 1
            stats["file_type_counter"][file_type] += 1

        modified_time = datetime.fromtimestamp(
            os.path.getmtime(collection_path)
        ).strftime("%d %b %Y %I:%M %p")

        stats["total_chunks"] += chunk_count
        stats["total_storage_bytes"] += storage_bytes

        stats["collection_rows"].append(
            {
                "Collection": collection_name,
                "Searchable": "Yes" if documents and has_index else "No",
                "Files": len(file_names),
                "Chunks": chunk_count,
                "Storage": format_bytes(storage_bytes),
                "Updated": modified_time,
                "Top Types": ", ".join(
                    [f"{key}:{value}" for key, value in file_types.most_common(3)]
                ) or "-"
            }
        )

    return stats


def render_collection_dashboard(collections_dir: str, documents_json: str = "documents.json"):
    stats = collect_dashboard_stats(
        collections_dir=collections_dir,
        documents_json=documents_json
    )

    st.markdown("## 📊 Collection Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{stats["total_collections"]}</div>
                <div class="metric-label">Collections</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{stats["searchable_collections"]}</div>
                <div class="metric-label">Searchable</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{stats["total_chunks"]}</div>
                <div class="metric-label">Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{format_bytes(stats["total_storage_bytes"])}</div>
                <div class="metric-label">Storage</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if stats["file_type_counter"]:
        st.markdown("### 📁 File Type Breakdown")
        type_cols = st.columns(4)

        for index, (file_type, count) in enumerate(stats["file_type_counter"].most_common()):
            with type_cols[index % 4]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{count}</div>
                        <div class="metric-label">{file_type}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    if stats["collection_rows"]:
        st.markdown("### 🗂️ Collection Details")
        st.dataframe(
            stats["collection_rows"],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No collections found yet. Process files first.")


def build_semantic_chat_context(history: list, max_messages: int = 5) -> str:
    if not history:
        return "No previous chat history."

    recent_items = history[-max_messages:]
    lines = []

    for index, item in enumerate(recent_items, start=1):
        question = item.get("question", "")
        answer = item.get("answer", "")

        lines.append(f"Previous Question {index}: {question}")
        lines.append(f"Previous Answer {index}: {answer}")
        lines.append("")

    return "\n".join(lines).strip()


def render_chat_memory_status(history: list, max_messages: int = 5):
    total_messages = len(history)
    used_messages = min(total_messages, max_messages)

    st.markdown(
        f"""
        <div class="status-box">
            🧠 Semantic Chat Memory Active<br>
            <span class="small-note">
                Using last <b>{used_messages}</b> of <b>{total_messages}</b> saved chat messages for follow-up context.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


def build_suggested_questions_prompt(answer: str, source_docs: List[LCDocument]) -> str:
    source_preview = ""

    for index, doc in enumerate(source_docs[:3], start=1):
        metadata = doc.metadata
        file_name = metadata.get("file_name", "Unknown file")
        file_type = metadata.get("file_type", "Unknown")
        content = doc.page_content[:600].replace("\n", " ")

        source_preview += f"\nSource {index}: {file_name} ({file_type})\n{content}\n"

    return f"""
Based on the answer and source context below, suggest 5 short follow-up questions the user may ask next.

Rules:
- Return only questions.
- One question per line.
- Do not number them.
- Keep questions specific to the document content.

Answer:
{answer}

Source context:
{source_preview}
""".strip()


def generate_suggested_questions(model, answer: str, source_docs: List[LCDocument]):
    if not answer or not source_docs:
        return []

    prompt = build_suggested_questions_prompt(
        answer=answer,
        source_docs=source_docs
    )

    try:
        response = model.invoke(prompt)
        raw_text = getattr(response, "content", str(response))

        questions = []

        for line in raw_text.splitlines():
            clean_line = line.strip()
            clean_line = clean_line.lstrip("-•0123456789. ").strip()

            if clean_line and clean_line.endswith("?"):
                questions.append(clean_line)

        return questions[:5]

    except Exception as e:
        st.warning(f"Could not generate suggested questions: {e}")
        return []


def render_suggested_questions(questions: List[str]):
    if not questions:
        return

    st.markdown("### 💡 Suggested Questions")

    for question in questions:
        st.markdown(
            f"""
            <div class="file-card">
                {question}
            </div>
            """,
            unsafe_allow_html=True
        )

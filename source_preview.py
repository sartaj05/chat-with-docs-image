import re
import html
import streamlit as st


def get_source_label(metadata: dict) -> str:
    file_name = metadata.get("file_name", "Unknown file")
    file_type = metadata.get("file_type", "Unknown")
    page = metadata.get("page")
    chunk = metadata.get("chunk")

    if page:
        return f"{file_name} | {file_type} | Page {page}"

    return f"{file_name} | {file_type} | Chunk {chunk}"


def extract_keywords(query: str):
    if not query:
        return []

    words = re.findall(r"\b\w+\b", query.lower())

    stop_words = {
        "the", "is", "are", "was", "were", "a", "an",
        "and", "or", "of", "to", "in", "for", "with",
        "on", "from", "by", "this", "that", "what",
        "which", "how", "why", "when", "where", "tell",
        "me", "about", "summary", "summarize", "document"
    }

    keywords = [
        word
        for word in words
        if len(word) > 2 and word not in stop_words
    ]

    return list(dict.fromkeys(keywords))


def highlight_text(text: str, query: str):
    safe_text = html.escape(text)

    keywords = extract_keywords(query)

    for keyword in keywords:
        pattern = re.compile(
            rf"({re.escape(keyword)})",
            re.IGNORECASE
        )

        safe_text = pattern.sub(
            r"<mark>\1</mark>",
            safe_text
        )

    return safe_text


def render_source_preview(
    docs,
    query: str = "",
    title: str = "Sources"
):
    if not docs:
        return

    st.markdown(f"### 🔎 {title}")

    seen = set()
    unique_docs = []

    for doc in docs:
        collection = doc.metadata.get("collection", "unknown")
        label = get_source_label(doc.metadata)
        full_label = f"{collection} | {label}"

        if full_label not in seen:
            seen.add(full_label)
            unique_docs.append(doc)

    for index, doc in enumerate(unique_docs, start=1):
        metadata = doc.metadata

        label = get_source_label(metadata)
        collection = metadata.get("collection", "unknown")
        file_type = metadata.get("file_type", "Unknown")
        final_score = metadata.get("final_score", 0)
        vector_score = metadata.get("vector_score", 0)
        keyword_score = metadata.get("keyword_score", 0)

        preview = doc.page_content[:1200].strip()
        highlighted_preview = highlight_text(preview, query)
        anchor_id = f"source-{index}-{abs(hash(label))}"

        st.markdown(
            f'<a href="#{anchor_id}">↘ Jump to source {index}</a>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div id="{anchor_id}" class="source-card">
                <b>{index}. {html.escape(label)}</b>
                <span class="collection-pill">{html.escape(collection)}</span>
                <span class="score-pill">Score: {final_score}</span>
                <br>
                <span class="small-note">
                    Type: {html.escape(file_type)} ·
                    Vector: {vector_score} ·
                    Keyword: {keyword_score}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            f"View highlighted source preview {index}",
            expanded=False
        ):
            st.markdown(
                f"""
                <div class="highlight-preview">
                    {highlighted_preview}
                </div>
                """,
                unsafe_allow_html=True
            )

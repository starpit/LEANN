pub mod ast;
pub mod sentence;

/// Split text into chunks by sentences with overlap.
pub fn chunk_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let sentences = sentence::split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_len = 0;

    for sent in &sentences {
        let sent_len = sent.len();

        if current_len + sent_len > chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());

            // Handle overlap: keep trailing sentences
            if chunk_overlap > 0 {
                let overlap_text = get_overlap_text(&current_chunk, chunk_overlap);
                current_chunk = overlap_text;
                current_len = current_chunk.len();
            } else {
                current_chunk.clear();
                current_len = 0;
            }
        }

        if !current_chunk.is_empty() {
            current_chunk.push(' ');
            current_len += 1;
        }
        current_chunk.push_str(sent);
        current_len += sent_len;
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

fn get_overlap_text(text: &str, overlap_chars: usize) -> String {
    if text.len() <= overlap_chars {
        return text.to_string();
    }
    text[text.len() - overlap_chars..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_basic() {
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = chunk_text(text, 40, 0);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 100, 0);
        assert!(chunks.is_empty());
    }
}

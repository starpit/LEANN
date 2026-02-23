/// Split text into sentences using simple heuristics.
///
/// Handles common sentence-ending punctuation (. ! ?) while avoiding
/// false splits on abbreviations and decimal numbers.
/// Uses byte-level boundary detection to avoid allocating a Vec<char>.
pub fn split_sentences(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0; // byte offset of current sentence start

    for (byte_pos, ch) in text.char_indices() {
        if ch == '.' || ch == '!' || ch == '?' {
            let is_boundary = if ch == '.' {
                is_sentence_boundary_bytes(bytes, byte_pos)
            } else {
                true
            };

            if is_boundary {
                let end = byte_pos + 1; // . ! ? are single-byte ASCII
                let trimmed = text[start..end].trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                start = end;
            }
        }
    }

    // Add remaining text as last sentence
    let trimmed = text[start..].trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Check if a period at byte position `pos` is a sentence boundary.
/// All checks use ASCII-level byte inspection, which is correct because
/// the sentence-ending punctuation (.) and the surrounding context
/// characters we care about (digits, uppercase letters, whitespace) are ASCII.
fn is_sentence_boundary_bytes(bytes: &[u8], pos: usize) -> bool {
    let len = bytes.len();

    // Not a boundary if followed by a digit (decimal number: "3.14")
    if pos + 1 < len && bytes[pos + 1].is_ascii_digit() {
        return false;
    }

    // Not a boundary if preceded by a single uppercase letter (abbreviation: "U.S.")
    if pos >= 1
        && bytes[pos - 1].is_ascii_uppercase()
        && (pos < 2 || !bytes[pos - 2].is_ascii_alphanumeric())
    {
        return false;
    }

    // Is a boundary if at end of text
    if pos + 1 >= len {
        return true;
    }

    // Is a boundary if followed by whitespace + uppercase
    if pos + 2 < len && bytes[pos + 1].is_ascii_whitespace() && bytes[pos + 2].is_ascii_uppercase()
    {
        return true;
    }

    // Is a boundary if followed by whitespace
    if bytes[pos + 1].is_ascii_whitespace() {
        return true;
    }

    // Default: followed by newline counts
    if bytes[pos + 1] == b'\n' {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_basic_sentences() {
        let text = "Hello world. This is a test. Another sentence here.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test.");
        assert_eq!(sentences[2], "Another sentence here.");
    }

    #[test]
    fn test_split_with_exclamation() {
        let text = "Wow! That is amazing. Really?";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_decimal_not_split() {
        let text = "The value is 3.14 and it matters.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_empty_text() {
        let sentences = split_sentences("");
        assert!(sentences.is_empty());
    }
}

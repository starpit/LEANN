/// Split text into sentences using simple heuristics.
///
/// Handles common sentence-ending punctuation (. ! ?) while avoiding
/// false splits on abbreviations and decimal numbers.
pub fn split_sentences(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        let ch = chars[i];
        current.push(ch);

        if (ch == '.' || ch == '!' || ch == '?') && !current.trim().is_empty() {
            // Check if this is a real sentence boundary
            let is_boundary = if ch == '.' {
                is_sentence_boundary(&chars, i)
            } else {
                true
            };

            if is_boundary {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        i += 1;
    }

    // Add remaining text as last sentence
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Check if a period at position `pos` is a sentence boundary.
fn is_sentence_boundary(chars: &[char], pos: usize) -> bool {
    let len = chars.len();

    // Not a boundary if followed by a digit (decimal number)
    if pos + 1 < len && chars[pos + 1].is_ascii_digit() {
        return false;
    }

    // Not a boundary if preceded by a single uppercase letter (abbreviation like "U.S.")
    if pos >= 1 && chars[pos - 1].is_ascii_uppercase() {
        if pos < 2 || !chars[pos - 2].is_alphanumeric() {
            return false;
        }
    }

    // Is a boundary if followed by whitespace + uppercase, or end of text
    if pos + 1 >= len {
        return true;
    }

    if pos + 2 < len && chars[pos + 1].is_whitespace() && chars[pos + 2].is_uppercase() {
        return true;
    }

    // Is a boundary if followed by whitespace
    if chars[pos + 1].is_whitespace() {
        return true;
    }

    // Default: followed by newline counts
    if chars[pos + 1] == '\n' {
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

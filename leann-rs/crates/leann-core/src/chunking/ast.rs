use std::collections::HashMap;

/// Language detection from file extension.
pub fn detect_language(filename: &str) -> Option<&'static str> {
    let ext = filename.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "py" => Some("python"),
        "rs" => Some("rust"),
        "js" | "jsx" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        "java" => Some("java"),
        "go" => Some("go"),
        "c" | "h" => Some("c"),
        "cpp" | "cxx" | "cc" | "hpp" => Some("cpp"),
        "rb" => Some("ruby"),
        "sh" | "bash" => Some("bash"),
        _ => None,
    }
}

/// A code chunk extracted from AST analysis.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub text: String,
    pub chunk_type: String, // "function", "class", "method", "module", etc.
    pub name: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Chunk source code using a simple AST-like approach.
///
/// This uses heuristic-based parsing to identify function/class boundaries
/// rather than a full AST parser. For a full tree-sitter implementation,
/// add the tree-sitter crate and language grammars.
pub fn chunk_code(source: &str, filename: &str, max_chunk_size: usize) -> Vec<CodeChunk> {
    let language = detect_language(filename).unwrap_or("unknown");

    match language {
        "python" => chunk_python(source, filename, max_chunk_size),
        "rust" => chunk_rust(source, filename, max_chunk_size),
        "javascript" | "typescript" => chunk_js_ts(source, filename, max_chunk_size),
        _ => chunk_generic(source, filename, language, max_chunk_size),
    }
}

/// Chunk Python source code by detecting function and class definitions.
fn chunk_python(source: &str, filename: &str, max_chunk_size: usize) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Detect function or class definition
        if trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("async def ")
        {
            let indent = line.len() - line.trim_start().len();
            let chunk_type = if trimmed.starts_with("class ") {
                "class"
            } else {
                "function"
            };

            let name = extract_name(trimmed);
            let start_line = i;

            // Find the end of this block (next line at same or lower indentation)
            let mut end_line = i + 1;
            while end_line < lines.len() {
                let next = lines[end_line];
                if next.trim().is_empty() {
                    end_line += 1;
                    continue;
                }
                let next_indent = next.len() - next.trim_start().len();
                if next_indent <= indent && !next.trim().is_empty() {
                    // Check if this is a decorator for the next function
                    if next.trim().starts_with('@') {
                        break;
                    }
                    // Same or lower indent means block ended
                    break;
                }
                end_line += 1;
            }

            let text: String = lines[start_line..end_line].join("\n");
            if text.len() <= max_chunk_size {
                chunks.push(CodeChunk {
                    text,
                    chunk_type: chunk_type.to_string(),
                    name: Some(name),
                    start_line: start_line + 1,
                    end_line,
                    language: "python".to_string(),
                    metadata: make_metadata(filename, start_line + 1, end_line),
                });
            } else {
                // Split large blocks
                let sub_chunks = split_large_block(&lines[start_line..end_line], max_chunk_size);
                for sub in sub_chunks {
                    chunks.push(CodeChunk {
                        text: sub,
                        chunk_type: format!("{}_part", chunk_type),
                        name: None,
                        start_line: start_line + 1,
                        end_line,
                        language: "python".to_string(),
                        metadata: make_metadata(filename, start_line + 1, end_line),
                    });
                }
            }

            i = end_line;
        } else {
            i += 1;
        }
    }

    // If no chunks were found, fall back to generic chunking
    if chunks.is_empty() {
        return chunk_generic(source, filename, "python", max_chunk_size);
    }

    chunks
}

/// Chunk Rust source code by detecting fn, struct, impl, enum blocks.
fn chunk_rust(source: &str, filename: &str, max_chunk_size: usize) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        let is_block_start = trimmed.starts_with("pub fn ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("pub impl ")
            || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("trait ")
            || trimmed.starts_with("pub mod ")
            || trimmed.starts_with("mod ");

        if is_block_start {
            let chunk_type = if trimmed.contains("fn ") {
                "function"
            } else if trimmed.contains("struct ") {
                "struct"
            } else if trimmed.contains("enum ") {
                "enum"
            } else if trimmed.contains("impl ") {
                "impl"
            } else if trimmed.contains("trait ") {
                "trait"
            } else {
                "module"
            };

            let name = extract_rust_name(trimmed);
            let start_line = i;

            // Find matching closing brace using brace counting
            let mut brace_count = 0;
            let mut end_line = i;
            let mut found_open = false;

            for (j, line) in lines.iter().enumerate().skip(i) {
                for ch in line.chars() {
                    if ch == '{' {
                        brace_count += 1;
                        found_open = true;
                    } else if ch == '}' {
                        brace_count -= 1;
                    }
                }
                end_line = j + 1;
                if found_open && brace_count == 0 {
                    break;
                }
            }

            let text: String = lines[start_line..end_line].join("\n");
            if text.len() <= max_chunk_size {
                chunks.push(CodeChunk {
                    text,
                    chunk_type: chunk_type.to_string(),
                    name: Some(name),
                    start_line: start_line + 1,
                    end_line,
                    language: "rust".to_string(),
                    metadata: make_metadata(filename, start_line + 1, end_line),
                });
            } else {
                let sub_chunks = split_large_block(&lines[start_line..end_line], max_chunk_size);
                for sub in sub_chunks {
                    chunks.push(CodeChunk {
                        text: sub,
                        chunk_type: format!("{}_part", chunk_type),
                        name: None,
                        start_line: start_line + 1,
                        end_line,
                        language: "rust".to_string(),
                        metadata: make_metadata(filename, start_line + 1, end_line),
                    });
                }
            }

            i = end_line;
        } else {
            i += 1;
        }
    }

    if chunks.is_empty() {
        return chunk_generic(source, filename, "rust", max_chunk_size);
    }

    chunks
}

/// Chunk JavaScript/TypeScript source code.
fn chunk_js_ts(source: &str, filename: &str, max_chunk_size: usize) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    let language = detect_language(filename).unwrap_or("javascript");

    while i < lines.len() {
        let trimmed = lines[i].trim();

        let is_block_start = trimmed.starts_with("function ")
            || trimmed.starts_with("async function ")
            || trimmed.starts_with("export function ")
            || trimmed.starts_with("export async function ")
            || trimmed.starts_with("export default function ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("export class ")
            || trimmed.starts_with("export default class ")
            || trimmed.contains("=> {");

        if is_block_start {
            let chunk_type = if trimmed.contains("class ") {
                "class"
            } else {
                "function"
            };

            let start_line = i;
            let mut brace_count = 0;
            let mut end_line = i;
            let mut found_open = false;

            for (j, line) in lines.iter().enumerate().skip(i) {
                for ch in line.chars() {
                    if ch == '{' {
                        brace_count += 1;
                        found_open = true;
                    } else if ch == '}' {
                        brace_count -= 1;
                    }
                }
                end_line = j + 1;
                if found_open && brace_count == 0 {
                    break;
                }
            }

            let text: String = lines[start_line..end_line].join("\n");
            if text.len() <= max_chunk_size {
                chunks.push(CodeChunk {
                    text,
                    chunk_type: chunk_type.to_string(),
                    name: None,
                    start_line: start_line + 1,
                    end_line,
                    language: language.to_string(),
                    metadata: make_metadata(filename, start_line + 1, end_line),
                });
            }

            i = end_line;
        } else {
            i += 1;
        }
    }

    if chunks.is_empty() {
        return chunk_generic(source, filename, language, max_chunk_size);
    }

    chunks
}

/// Generic line-based chunking for unsupported languages.
fn chunk_generic(
    source: &str,
    filename: &str,
    language: &str,
    max_chunk_size: usize,
) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut start_line = 0;

    for (i, line) in lines.iter().enumerate() {
        if current.len() + line.len() + 1 > max_chunk_size && !current.is_empty() {
            chunks.push(CodeChunk {
                text: current.clone(),
                chunk_type: "block".to_string(),
                name: None,
                start_line: start_line + 1,
                end_line: i,
                language: language.to_string(),
                metadata: make_metadata(filename, start_line + 1, i),
            });
            current.clear();
            start_line = i;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    if !current.trim().is_empty() {
        chunks.push(CodeChunk {
            text: current,
            chunk_type: "block".to_string(),
            name: None,
            start_line: start_line + 1,
            end_line: lines.len(),
            language: language.to_string(),
            metadata: make_metadata(filename, start_line + 1, lines.len()),
        });
    }

    chunks
}

fn extract_name(definition_line: &str) -> String {
    let trimmed = definition_line.trim();
    // "def foo(..." or "class Foo:" or "async def bar(..."
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    for (i, &part) in parts.iter().enumerate() {
        if (part == "def" || part == "class")
            && let Some(name) = parts.get(i + 1)
        {
            return name.trim_end_matches('(').trim_end_matches(':').to_string();
        }
    }
    "unknown".to_string()
}

fn extract_rust_name(definition_line: &str) -> String {
    let trimmed = definition_line.trim();
    let keywords = ["fn", "struct", "enum", "impl", "trait", "mod"];
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    for (i, &part) in parts.iter().enumerate() {
        if keywords.contains(&part)
            && let Some(name) = parts.get(i + 1)
        {
            return name
                .trim_end_matches('{')
                .trim_end_matches('<')
                .trim_end_matches('(')
                .to_string();
        }
    }
    "unknown".to_string()
}

fn split_large_block(lines: &[&str], max_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in lines {
        if current.len() + line.len() + 1 > max_size && !current.is_empty() {
            chunks.push(current.clone());
            current.clear();
        }

        // If a single line exceeds max_size, split it by characters
        if line.len() > max_size && current.is_empty() {
            let mut offset = 0;
            while offset < line.len() {
                let end = (offset + max_size).min(line.len());
                chunks.push(line[offset..end].to_string());
                offset = end;
            }
            continue;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    if !current.trim().is_empty() {
        chunks.push(current);
    }

    chunks
}

fn make_metadata(
    filename: &str,
    start_line: usize,
    end_line: usize,
) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();
    m.insert("source".to_string(), serde_json::json!(filename));
    m.insert("start_line".to_string(), serde_json::json!(start_line));
    m.insert("end_line".to_string(), serde_json::json!(end_line));
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("foo.py"), Some("python"));
        assert_eq!(detect_language("bar.rs"), Some("rust"));
        assert_eq!(detect_language("baz.js"), Some("javascript"));
        assert_eq!(detect_language("qux.txt"), None);
    }

    #[test]
    fn test_chunk_python() {
        let source = r#"
def hello():
    print("hello")

def world():
    print("world")

class Foo:
    def bar(self):
        pass
"#;
        let chunks = chunk_code(source, "test.py", 1000);
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_chunk_rust() {
        let source = r#"
fn hello() {
    println!("hello");
}

fn world() {
    println!("world");
}

struct Foo {
    bar: i32,
}
"#;
        let chunks = chunk_code(source, "test.rs", 1000);
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_chunk_generic() {
        let source = "line 1\nline 2\nline 3\nline 4\nline 5";
        let chunks = chunk_code(source, "test.txt", 20);
        assert!(!chunks.is_empty());
    }
}

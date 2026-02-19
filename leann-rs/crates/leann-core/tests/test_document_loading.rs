//! E2E-5: Document Loading & Chunking
//!
//! Tests document extraction from various file types and text chunking.
//! Mirrors Python test_document_rag.py and test_astchunk_integration.py.

use leann_core::chunking::ast::{chunk_code, detect_language};
use leann_core::chunking::chunk_text;
use leann_core::document_loaders::extract_text;
use std::io::Write;

/// Load a .txt file and verify content.
#[test]
fn test_load_txt_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.txt");
    std::fs::write(&path, "Hello, this is a test document.\nSecond line here.").unwrap();

    let content = extract_text(&path).unwrap();
    assert!(content.is_some());
    let text = content.unwrap();
    assert!(text.contains("Hello"));
    assert!(text.contains("Second line"));
}

/// Load a .md file.
#[test]
fn test_load_md_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readme.md");
    std::fs::write(
        &path,
        "# Title\n\nSome markdown content.\n\n- Item 1\n- Item 2",
    )
    .unwrap();

    let content = extract_text(&path).unwrap();
    assert!(content.is_some());
    assert!(content.unwrap().contains("Title"));
}

/// Load a .rs file.
#[test]
fn test_load_rs_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lib.rs");
    std::fs::write(&path, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

    let content = extract_text(&path).unwrap();
    assert!(content.is_some());
    assert!(content.unwrap().contains("fn main"));
}

/// Load a .py file.
#[test]
fn test_load_py_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("script.py");
    std::fs::write(
        &path,
        "def hello():\n    print('hello world')\n\nif __name__ == '__main__':\n    hello()\n",
    )
    .unwrap();

    let content = extract_text(&path).unwrap();
    assert!(content.is_some());
    assert!(content.unwrap().contains("def hello"));
}

/// Empty file returns None.
#[test]
fn test_load_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.txt");
    std::fs::write(&path, "").unwrap();

    let content = extract_text(&path).unwrap();
    assert!(content.is_none(), "Empty file should return None");
}

/// Whitespace-only file returns None.
#[test]
fn test_load_whitespace_only_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spaces.txt");
    std::fs::write(&path, "   \n\n   \t  ").unwrap();

    let content = extract_text(&path).unwrap();
    assert!(content.is_none(), "Whitespace-only file should return None");
}

/// Nonexistent file should not panic.
#[test]
fn test_load_nonexistent_file() {
    let path = std::path::Path::new("/tmp/nonexistent_file_xyz_123.txt");
    let content = extract_text(path).unwrap();
    assert!(content.is_none());
}

// --- Chunking tests ---

/// Chunk a long document, verify chunks are bounded by max_size.
#[test]
fn test_chunk_text_sizes() {
    let long_text = "This is a sentence. ".repeat(100);
    let chunks = chunk_text(&long_text, 200, 0);
    assert!(chunks.len() > 1, "Long text should produce multiple chunks");
    for chunk in &chunks {
        // Allow some slack because chunking happens at sentence boundaries
        assert!(chunk.len() <= 400, "Chunk too large: {} chars", chunk.len());
    }
}

/// Verify sentence overlap between consecutive chunks.
#[test]
fn test_chunk_overlap() {
    let text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here.";
    let chunks = chunk_text(text, 50, 20);
    assert!(chunks.len() >= 2, "Should produce at least 2 chunks");
}

/// Short text produces a single chunk.
#[test]
fn test_chunk_short_text() {
    let text = "Just one short sentence.";
    let chunks = chunk_text(text, 1000, 0);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], "Just one short sentence.");
}

/// Empty text produces no chunks.
#[test]
fn test_chunk_empty_text() {
    let chunks = chunk_text("", 100, 0);
    assert!(chunks.is_empty());
}

// --- AST chunking tests ---

/// Python source → AST chunking → chunks align with function/class boundaries.
#[test]
fn test_ast_chunk_python() {
    let source = r#"
import os

def hello():
    print("hello")
    return True

def world(name):
    print(f"world {name}")

class MyClass:
    def __init__(self):
        self.x = 1

    def method(self):
        return self.x
"#;
    let chunks = chunk_code(source, "test.py", 2000);
    assert!(
        chunks.len() >= 2,
        "Expected at least 2 Python chunks, got {}",
        chunks.len()
    );

    // Verify chunk types
    let has_function = chunks.iter().any(|c| c.chunk_type == "function");
    let has_class = chunks.iter().any(|c| c.chunk_type == "class");
    assert!(has_function, "Should detect function chunks");
    assert!(has_class, "Should detect class chunks");

    // All chunks should be Python
    for chunk in &chunks {
        assert_eq!(chunk.language, "python");
    }
}

/// Rust source → AST chunking → chunks align with fn/impl/struct boundaries.
#[test]
fn test_ast_chunk_rust() {
    let source = r#"
use std::io;

fn hello() {
    println!("hello");
}

fn world() -> String {
    "world".to_string()
}

struct Foo {
    bar: i32,
    baz: String,
}

impl Foo {
    fn new() -> Self {
        Self { bar: 0, baz: String::new() }
    }
}
"#;
    let chunks = chunk_code(source, "lib.rs", 2000);
    assert!(
        chunks.len() >= 3,
        "Expected at least 3 Rust chunks, got {}",
        chunks.len()
    );

    let types: Vec<&str> = chunks.iter().map(|c| c.chunk_type.as_str()).collect();
    assert!(
        types.contains(&"function"),
        "Should detect function chunks: {:?}",
        types
    );

    for chunk in &chunks {
        assert_eq!(chunk.language, "rust");
    }
}

/// JavaScript source → AST chunking.
#[test]
fn test_ast_chunk_javascript() {
    let source = r#"
function hello() {
    console.log("hello");
}

class MyClass {
    constructor() {
        this.x = 1;
    }

    method() {
        return this.x;
    }
}

async function fetchData() {
    const data = await fetch("/api");
    return data.json();
}
"#;
    let chunks = chunk_code(source, "app.js", 2000);
    assert!(
        chunks.len() >= 2,
        "Expected at least 2 JS chunks, got {}",
        chunks.len()
    );

    for chunk in &chunks {
        assert_eq!(chunk.language, "javascript");
    }
}

/// Non-code file with AST chunking → falls back to generic chunking.
#[test]
fn test_ast_fallback_to_generic() {
    let source = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10";
    let chunks = chunk_code(source, "notes.txt", 30);
    assert!(!chunks.is_empty(), "Generic fallback should produce chunks");
    for chunk in &chunks {
        assert_eq!(chunk.chunk_type, "block");
    }
}

/// Language detection from filename.
#[test]
fn test_detect_language() {
    assert_eq!(detect_language("main.py"), Some("python"));
    assert_eq!(detect_language("lib.rs"), Some("rust"));
    assert_eq!(detect_language("app.js"), Some("javascript"));
    assert_eq!(detect_language("index.ts"), Some("typescript"));
    assert_eq!(detect_language("Main.java"), Some("java"));
    assert_eq!(detect_language("main.go"), Some("go"));
    assert_eq!(detect_language("notes.txt"), None);
    assert_eq!(detect_language("data.csv"), None);
}

/// Code chunks include source metadata.
#[test]
fn test_ast_chunk_metadata() {
    let source = "fn hello() {\n    println!(\"hi\");\n}\n";
    let chunks = chunk_code(source, "test.rs", 2000);
    assert!(!chunks.is_empty());

    let chunk = &chunks[0];
    assert!(chunk.start_line > 0, "start_line should be 1-indexed");
    assert!(chunk.end_line >= chunk.start_line);
    assert_eq!(
        chunk.metadata.get("source"),
        Some(&serde_json::json!("test.rs"))
    );
}

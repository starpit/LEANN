//! Tree-sitter based AST chunking for accurate, grammar-aware code splitting.
//!
//! This module is gated behind per-language feature flags (e.g. `tree-sitter-python`).
//! When enabled, it replaces the heuristic-based chunker in `ast.rs` for supported
//! languages, falling back to heuristics for unsupported languages or parse failures.

use tree_sitter::{Node, Parser};

use super::ast::{CodeChunk, make_metadata, split_large_block};

/// Configuration for a language's AST chunking behavior.
struct LanguageConfig {
    language: tree_sitter::Language,
    /// Node types that represent top-level definitions to extract as chunks.
    definition_types: &'static [&'static str],
    /// Human-readable language name for `CodeChunk::language`.
    name: &'static str,
}

/// Try to chunk source code using tree-sitter. Returns `None` if the language
/// is not supported by any enabled feature, or if parsing fails.
pub fn chunk_code_tree_sitter(
    source: &str,
    filename: &str,
    max_chunk_size: usize,
) -> Option<Vec<CodeChunk>> {
    let config = get_language_config(filename)?;

    let mut parser = Parser::new();
    parser.set_language(&config.language).ok()?;
    let tree = parser.parse(source, None)?;
    let root = tree.root_node();

    let source_bytes = source.as_bytes();
    let mut chunks = Vec::new();
    let mut covered_end: usize = 0;

    collect_definitions(
        root,
        &config,
        source,
        source_bytes,
        filename,
        max_chunk_size,
        &mut chunks,
        &mut covered_end,
    );

    // Capture any trailing module-level code after the last definition.
    if covered_end < source_bytes.len() {
        let gap_text = &source[covered_end..];
        if !gap_text.trim().is_empty() {
            push_block_chunks(
                gap_text,
                covered_end,
                source,
                filename,
                config.name,
                max_chunk_size,
                &mut chunks,
            );
        }
    }

    Some(chunks)
}

/// Recursively walk the tree, collecting definition nodes as chunks and
/// module-level gaps as block chunks. Non-definition children are walked
/// through transparently so that definitions nested inside body/block nodes
/// (e.g. methods inside a class body) are discovered.
fn collect_definitions(
    node: Node<'_>,
    config: &LanguageConfig,
    source: &str,
    source_bytes: &[u8],
    filename: &str,
    max_chunk_size: usize,
    chunks: &mut Vec<CodeChunk>,
    covered_end: &mut usize,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();

        if !config.definition_types.contains(&kind) {
            // Walk through non-definition nodes (e.g. `block`, `class_body`)
            // to find definitions nested deeper in the tree.
            if child.child_count() > 0 {
                collect_definitions(
                    child,
                    config,
                    source,
                    source_bytes,
                    filename,
                    max_chunk_size,
                    chunks,
                    covered_end,
                );
            }
            continue;
        }

        // --- This child is a definition node. ---
        {
            let start_byte = child.start_byte();
            let end_byte = child.end_byte();

            // Emit gap before this definition as a "block" chunk.
            if start_byte > *covered_end {
                let gap_text = &source[*covered_end..start_byte];
                if !gap_text.trim().is_empty() {
                    push_block_chunks(
                        gap_text,
                        *covered_end,
                        source,
                        filename,
                        config.name,
                        max_chunk_size,
                        chunks,
                    );
                }
            }

            let text = &source[start_byte..end_byte];
            let start_line = child.start_position().row + 1;
            let end_line = child.end_position().row + 1;
            let chunk_type = node_kind_to_chunk_type(kind);
            let name = extract_definition_name(child, source_bytes);

            if text.len() <= max_chunk_size {
                chunks.push(CodeChunk {
                    text: text.to_string(),
                    chunk_type: chunk_type.to_string(),
                    name,
                    start_line,
                    end_line,
                    language: config.name.to_string(),
                    metadata: make_metadata(filename, start_line, end_line),
                });
            } else {
                // Try recursing into children first for compound definitions
                // (e.g. a class with methods).
                let mut sub_chunks = Vec::new();
                let mut sub_covered = start_byte;
                collect_definitions(
                    child,
                    config,
                    source,
                    source_bytes,
                    filename,
                    max_chunk_size,
                    &mut sub_chunks,
                    &mut sub_covered,
                );

                if sub_chunks.is_empty() {
                    // No sub-definitions found; fall back to line-based splitting.
                    let lines: Vec<&str> = text.lines().collect();
                    for sub in split_large_block(&lines, max_chunk_size) {
                        chunks.push(CodeChunk {
                            text: sub,
                            chunk_type: format!("{chunk_type}_part"),
                            name: None,
                            start_line,
                            end_line,
                            language: config.name.to_string(),
                            metadata: make_metadata(filename, start_line, end_line),
                        });
                    }
                } else {
                    // Capture any trailing gap inside this node.
                    if sub_covered < end_byte {
                        let gap = &source[sub_covered..end_byte];
                        if !gap.trim().is_empty() {
                            push_block_chunks(
                                gap,
                                sub_covered,
                                source,
                                filename,
                                config.name,
                                max_chunk_size,
                                &mut sub_chunks,
                            );
                        }
                    }
                    chunks.extend(sub_chunks);
                }
            }

            *covered_end = end_byte;
        }
    }
}

/// Push gap text as one or more "block" chunks, splitting if necessary.
fn push_block_chunks(
    text: &str,
    byte_offset: usize,
    source: &str,
    filename: &str,
    language: &str,
    max_chunk_size: usize,
    chunks: &mut Vec<CodeChunk>,
) {
    let start_line = source[..byte_offset].lines().count() + 1;
    let end_line = start_line + text.lines().count().saturating_sub(1);

    if text.len() <= max_chunk_size {
        chunks.push(CodeChunk {
            text: text.to_string(),
            chunk_type: "block".to_string(),
            name: None,
            start_line,
            end_line,
            language: language.to_string(),
            metadata: make_metadata(filename, start_line, end_line),
        });
    } else {
        let lines: Vec<&str> = text.lines().collect();
        for sub in split_large_block(&lines, max_chunk_size) {
            chunks.push(CodeChunk {
                text: sub,
                chunk_type: "block".to_string(),
                name: None,
                start_line,
                end_line,
                language: language.to_string(),
                metadata: make_metadata(filename, start_line, end_line),
            });
        }
    }
}

/// Map tree-sitter node kinds to chunk type names.
fn node_kind_to_chunk_type(kind: &str) -> &'static str {
    match kind {
        "function_definition"
        | "function_declaration"
        | "method_declaration"
        | "method_definition"
        | "arrow_function" => "function",
        "class_definition" | "class_declaration" => "class",
        "interface_declaration" => "interface",
        "decorated_definition" => "function", // unwrap decorator to get inner type
        _ => "block",
    }
}

/// Try to extract the name identifier from a definition node.
fn extract_definition_name(node: Node<'_>, source: &[u8]) -> Option<String> {
    // For decorated_definition, look at the inner definition's name.
    if node.kind() == "decorated_definition" {
        if let Some(def) = node.child_by_field_name("definition") {
            return extract_definition_name(def, source);
        }
    }

    // Most definition nodes have a "name" field.
    if let Some(name_node) = node.child_by_field_name("name") {
        let name = &source[name_node.start_byte()..name_node.end_byte()];
        return Some(String::from_utf8_lossy(name).into_owned());
    }

    None
}

/// Get the tree-sitter language configuration for a file, based on extension
/// and enabled features.
fn get_language_config(filename: &str) -> Option<LanguageConfig> {
    let ext = filename.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        #[cfg(feature = "tree-sitter-python")]
        "py" => Some(LanguageConfig {
            language: tree_sitter_python::LANGUAGE.into(),
            definition_types: &[
                "function_definition",
                "class_definition",
                "decorated_definition",
            ],
            name: "python",
        }),

        #[cfg(feature = "tree-sitter-java")]
        "java" => Some(LanguageConfig {
            language: tree_sitter_java::LANGUAGE.into(),
            definition_types: &[
                "method_declaration",
                "class_declaration",
                "interface_declaration",
            ],
            name: "java",
        }),

        #[cfg(feature = "tree-sitter-c-sharp")]
        "cs" => Some(LanguageConfig {
            language: tree_sitter_c_sharp::LANGUAGE.into(),
            definition_types: &[
                "method_declaration",
                "class_declaration",
                "interface_declaration",
            ],
            name: "csharp",
        }),

        #[cfg(feature = "tree-sitter-typescript")]
        "ts" => Some(LanguageConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            definition_types: &[
                "function_declaration",
                "class_declaration",
                "arrow_function",
                "method_definition",
            ],
            name: "typescript",
        }),

        #[cfg(feature = "tree-sitter-typescript")]
        "tsx" => Some(LanguageConfig {
            language: tree_sitter_typescript::LANGUAGE_TSX.into(),
            definition_types: &[
                "function_declaration",
                "class_declaration",
                "arrow_function",
                "method_definition",
            ],
            name: "typescript",
        }),

        #[cfg(feature = "tree-sitter-javascript")]
        "js" | "jsx" => Some(LanguageConfig {
            language: tree_sitter_javascript::LANGUAGE.into(),
            definition_types: &[
                "function_declaration",
                "class_declaration",
                "arrow_function",
                "method_definition",
            ],
            name: "javascript",
        }),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_python_chunking() {
        let source = r#"
import os

def hello():
    print("hello")

def world():
    print("world")

class Foo:
    def bar(self):
        pass

    def baz(self):
        return 42
"#;
        let chunks = chunk_code_tree_sitter(source, "test.py", 1000).unwrap();
        assert!(
            chunks.len() >= 3,
            "Expected at least 3 chunks (2 functions + 1 class), got {}: {:?}",
            chunks.len(),
            chunks
                .iter()
                .map(|c| (&c.chunk_type, &c.name))
                .collect::<Vec<_>>()
        );

        // Verify we got the expected definition names.
        let names: Vec<_> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"hello"), "Missing 'hello' function");
        assert!(names.contains(&"world"), "Missing 'world' function");
        assert!(names.contains(&"Foo"), "Missing 'Foo' class");
    }

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_python_decorated() {
        let source = r#"
@decorator
def decorated_fn():
    pass

@property
def prop(self):
    return self._val
"#;
        let chunks = chunk_code_tree_sitter(source, "test.py", 1000).unwrap();
        let names: Vec<_> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"decorated_fn"), "Missing 'decorated_fn'");
        assert!(names.contains(&"prop"), "Missing 'prop'");
    }

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_python_large_function_splits() {
        // Generate a function larger than max_chunk_size.
        let mut source = "def big_fn():\n".to_string();
        for i in 0..100 {
            source.push_str(&format!("    x_{i} = {i}\n"));
        }
        let chunks = chunk_code_tree_sitter(&source, "test.py", 200).unwrap();
        assert!(
            chunks.len() > 1,
            "Large function should be split into multiple chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    #[cfg(feature = "tree-sitter-java")]
    fn test_java_chunking() {
        let source = r#"
public class MyClass {
    public void hello() {
        System.out.println("hello");
    }

    public int add(int a, int b) {
        return a + b;
    }
}
"#;
        let chunks = chunk_code_tree_sitter(source, "MyClass.java", 1000).unwrap();
        assert!(
            !chunks.is_empty(),
            "Expected at least 1 chunk for Java class"
        );
        let names: Vec<_> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"MyClass"), "Missing 'MyClass' class");
    }

    #[test]
    #[cfg(feature = "tree-sitter-javascript")]
    fn test_javascript_chunking() {
        let source = r#"
function greet(name) {
    return `Hello, ${name}!`;
}

class Animal {
    constructor(name) {
        this.name = name;
    }

    speak() {
        return `${this.name} makes a noise.`;
    }
}
"#;
        let chunks = chunk_code_tree_sitter(source, "test.js", 1000).unwrap();
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks (function + class), got {}",
            chunks.len()
        );
        let names: Vec<_> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"greet"), "Missing 'greet' function");
        assert!(names.contains(&"Animal"), "Missing 'Animal' class");
    }

    #[test]
    #[cfg(feature = "tree-sitter-typescript")]
    fn test_typescript_chunking() {
        let source = r#"
function add(a: number, b: number): number {
    return a + b;
}

class Counter {
    private count: number = 0;

    increment(): void {
        this.count++;
    }
}
"#;
        let chunks = chunk_code_tree_sitter(source, "test.ts", 1000).unwrap();
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    #[cfg(feature = "tree-sitter-c-sharp")]
    fn test_csharp_chunking() {
        let source = r#"
public class Calculator {
    public int Add(int a, int b) {
        return a + b;
    }
}
"#;
        let chunks = chunk_code_tree_sitter(source, "Calculator.cs", 1000).unwrap();
        assert!(!chunks.is_empty(), "Expected chunks for C# code");
    }

    #[test]
    fn test_unsupported_extension_returns_none() {
        let result = chunk_code_tree_sitter("hello", "test.txt", 1000);
        assert!(result.is_none());
    }

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_empty_source() {
        let chunks = chunk_code_tree_sitter("", "test.py", 1000).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_module_level_code_captured() {
        let source = r#"
import os
import sys

X = 42

def foo():
    pass

Y = 99
"#;
        let chunks = chunk_code_tree_sitter(source, "test.py", 1000).unwrap();
        // Should have block chunks for the imports/assignments and a function chunk.
        let block_chunks: Vec<_> = chunks.iter().filter(|c| c.chunk_type == "block").collect();
        assert!(
            !block_chunks.is_empty(),
            "Module-level code should produce block chunks"
        );
        let fn_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == "function")
            .collect();
        assert!(
            !fn_chunks.is_empty(),
            "Should have at least one function chunk"
        );
    }

    // --- TSX ---

    #[test]
    #[cfg(feature = "tree-sitter-typescript")]
    fn test_tsx_chunking() {
        let source = r#"
function App(): JSX.Element {
    return <div>Hello</div>;
}

class Widget {
    render() {
        return <span />;
    }
}
"#;
        let chunks = chunk_code_tree_sitter(source, "App.tsx", 1000).unwrap();
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks (function + class), got {}",
            chunks.len()
        );
        let names: Vec<_> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"App"), "Missing 'App' function");
        assert!(names.contains(&"Widget"), "Missing 'Widget' class");
    }

    // --- Dispatch integration ---

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_dispatch_routes_to_tree_sitter() {
        // chunk_code (the public API in ast.rs) should use tree-sitter when
        // the feature is enabled, producing named chunks that heuristics also
        // find — but tree-sitter chunks should carry correct AST names.
        let source = "def greet():\n    return 'hi'\n";
        let chunks = crate::chunking::ast::chunk_code(source, "test.py", 1000);
        assert!(!chunks.is_empty());
        // Tree-sitter extracts the name; heuristic also does, so this works
        // either way — but the chunk_type being "function" (not "function_part")
        // confirms a clean parse.
        assert_eq!(chunks[0].chunk_type, "function");
        assert_eq!(chunks[0].name.as_deref(), Some("greet"));
    }

    // --- Fallback path ---

    #[test]
    fn test_fallback_to_heuristic_for_unsupported_lang() {
        // Rust files have no tree-sitter grammar in our feature set, so
        // chunk_code should fall back to the heuristic chunker.
        let source = "fn main() {\n    println!(\"hello\");\n}\n";
        let chunks = crate::chunking::ast::chunk_code(source, "test.rs", 1000);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].language, "rust");
    }

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_fallback_on_only_whitespace() {
        // Source with only whitespace — tree-sitter parses it but produces no
        // definition chunks (empty vec), so chunk_code should fall through to
        // heuristic. Heuristic also returns empty for whitespace-only.
        let chunks = crate::chunking::ast::chunk_code("   \n\n  \n", "test.py", 1000);
        assert!(chunks.is_empty());
    }

    // --- Nested classes ---

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_python_nested_class() {
        let source = r#"
class Outer:
    class Inner:
        def method(self):
            pass

    def outer_method(self):
        return 1
"#;
        let chunks = chunk_code_tree_sitter(source, "test.py", 1000).unwrap();
        let names: Vec<_> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"Outer"), "Missing 'Outer' class");
    }

    #[test]
    #[cfg(feature = "tree-sitter-java")]
    fn test_java_inner_class() {
        let source = r#"
public class Outer {
    public void outerMethod() {
        System.out.println("outer");
    }

    public class Inner {
        public void innerMethod() {
            System.out.println("inner");
        }
    }
}
"#;
        let chunks = chunk_code_tree_sitter(source, "Outer.java", 1000).unwrap();
        let names: Vec<_> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"Outer"), "Missing 'Outer' class");
    }

    // --- Line number accuracy ---

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_line_numbers_accurate() {
        // Lines are 1-indexed in CodeChunk.
        let source = "def foo():\n    pass\n\ndef bar():\n    pass\n";
        //            ^line 1        ^line 2  ^3   ^line 4        ^line 5
        let chunks = chunk_code_tree_sitter(source, "test.py", 1000).unwrap();
        let fns: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == "function")
            .collect();
        assert_eq!(fns.len(), 2);

        assert_eq!(fns[0].name.as_deref(), Some("foo"));
        assert_eq!(fns[0].start_line, 1);
        assert_eq!(fns[0].end_line, 2);

        assert_eq!(fns[1].name.as_deref(), Some("bar"));
        assert_eq!(fns[1].start_line, 4);
        assert_eq!(fns[1].end_line, 5);
    }

    #[test]
    #[cfg(feature = "tree-sitter-javascript")]
    fn test_js_line_numbers() {
        let source = "function a() {\n  return 1;\n}\n\nfunction b() {\n  return 2;\n}\n";
        let chunks = chunk_code_tree_sitter(source, "test.js", 1000).unwrap();
        let fns: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == "function")
            .collect();
        assert_eq!(fns.len(), 2);
        assert_eq!(fns[0].name.as_deref(), Some("a"));
        assert_eq!(fns[0].start_line, 1);
        assert_eq!(fns[0].end_line, 3);
        assert_eq!(fns[1].name.as_deref(), Some("b"));
        assert_eq!(fns[1].start_line, 5);
        assert_eq!(fns[1].end_line, 7);
    }

    // --- Large class with child methods (recursive descent) ---

    #[test]
    #[cfg(feature = "tree-sitter-python")]
    fn test_large_class_recurses_into_methods() {
        // Build a class whose total text exceeds max_chunk_size, but each
        // individual method fits. The chunker should recurse and emit
        // per-method chunks rather than a single split blob.
        let mut source = "class Big:\n".to_string();
        for i in 0..10 {
            source.push_str(&format!(
                "    def method_{i}(self):\n        x = {i}\n        return x\n\n"
            ));
        }
        // Each method is ~50 bytes; 10 methods + class header ≈ 550 bytes.
        // Set max to 200 so the class as a whole doesn't fit.
        let chunks = chunk_code_tree_sitter(&source, "test.py", 200).unwrap();
        // We should get individual method chunks, not just "class_part" blobs.
        let fn_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == "function")
            .collect();
        assert!(
            fn_chunks.len() >= 5,
            "Expected most methods to become individual function chunks, got {} function chunks out of {} total",
            fn_chunks.len(),
            chunks.len()
        );
        // None should exceed max_chunk_size.
        for c in &chunks {
            assert!(
                c.text.len() <= 200,
                "Chunk '{}' exceeds max_chunk_size: {} bytes",
                c.name.as_deref().unwrap_or("<unnamed>"),
                c.text.len()
            );
        }
    }

    #[test]
    #[cfg(feature = "tree-sitter-java")]
    fn test_large_java_class_recurses() {
        let mut source = "public class Big {\n".to_string();
        for i in 0..10 {
            source.push_str(&format!("    public int m{i}() {{ return {i}; }}\n"));
        }
        source.push_str("}\n");
        let chunks = chunk_code_tree_sitter(&source, "Big.java", 150).unwrap();
        let method_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == "function")
            .collect();
        assert!(
            method_chunks.len() >= 5,
            "Expected individual method chunks from recursive descent, got {}",
            method_chunks.len()
        );
    }
}

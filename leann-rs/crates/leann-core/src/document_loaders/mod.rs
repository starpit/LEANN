//! Document loaders for various file formats.
//!
//! Provides text extraction from binary document formats (PDF, etc.)
//! that cannot be loaded with simple `read_to_string`.

#[cfg(feature = "pdf")]
pub mod pdf;

use anyhow::Result;
use std::path::Path;

/// Extract text content from a file, handling both text and binary formats.
///
/// For text files, reads the file as UTF-8. For supported binary formats
/// (PDF when the `pdf` feature is enabled), uses specialized extractors.
///
/// Returns `None` if the file format is not supported or extraction fails.
pub fn extract_text(path: &Path) -> Result<Option<String>> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        #[cfg(feature = "pdf")]
        "pdf" => match pdf::extract_pdf_text(path) {
            Ok(text) if !text.trim().is_empty() => Ok(Some(text)),
            Ok(_) => {
                tracing::warn!("PDF has no extractable text: {}", path.display());
                Ok(None)
            }
            Err(e) => {
                tracing::warn!("Failed to extract text from PDF {}: {}", path.display(), e);
                Ok(None)
            }
        },
        #[cfg(not(feature = "pdf"))]
        "pdf" => {
            tracing::warn!(
                "PDF support not enabled. Rebuild with `pdf` feature to load: {}",
                path.display()
            );
            Ok(None)
        }
        _ => {
            // Text-based file: try read_to_string
            match std::fs::read_to_string(path) {
                Ok(content) if !content.trim().is_empty() => Ok(Some(content)),
                Ok(_) => Ok(None),
                Err(e) => {
                    tracing::debug!("Could not read {} as text: {}", path.display(), e);
                    Ok(None)
                }
            }
        }
    }
}

/// Returns true if the file extension is a binary format that requires
/// a specialized loader (not `read_to_string`).
pub fn is_binary_document(path: &Path) -> bool {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    matches!(ext.as_str(), "pdf")
}

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;

/// Convert a serde_json::Value to a Python object.
fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyObject {
    match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_pyobject(py).unwrap().to_owned().into_any().unbind(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py).unwrap().into_any().unbind()
            } else if let Some(f) = n.as_f64() {
                f.into_pyobject(py).unwrap().into_any().unbind()
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_py(py, item)).unwrap();
            }
            list.into_any().unbind()
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_py(py, v)).unwrap();
            }
            dict.into_any().unbind()
        }
    }
}

/// Convert a Python dict to HashMap<String, serde_json::Value>.
fn py_dict_to_metadata(dict: &Bound<'_, PyDict>) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    for (key, value) in dict.iter() {
        if let Ok(key_str) = key.extract::<String>() {
            let json_val = py_to_json_value(&value);
            map.insert(key_str, json_val);
        }
    }
    map
}

/// Convert a Python object to serde_json::Value.
fn py_to_json_value(obj: &Bound<'_, PyAny>) -> serde_json::Value {
    if obj.is_none() {
        serde_json::Value::Null
    } else if let Ok(b) = obj.extract::<bool>() {
        serde_json::Value::Bool(b)
    } else if let Ok(i) = obj.extract::<i64>() {
        serde_json::json!(i)
    } else if let Ok(f) = obj.extract::<f64>() {
        serde_json::json!(f)
    } else if let Ok(s) = obj.extract::<String>() {
        serde_json::Value::String(s)
    } else {
        // Fallback: convert to string representation
        serde_json::Value::String(format!("{}", obj))
    }
}

/// Search result returned from LEANN queries.
#[pyclass]
#[derive(Clone)]
struct SearchResult {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    text: String,
    metadata_inner: HashMap<String, serde_json::Value>,
}

#[pymethods]
impl SearchResult {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        for (k, v) in &self.metadata_inner {
            dict.set_item(k, json_value_to_py(py, v)).unwrap();
        }
        dict.into_pyobject(py).unwrap().into_any().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', score={:.4}, text='{}...')",
            self.id,
            self.score,
            &self.text[..self.text.len().min(50)]
        )
    }
}

impl From<leann_core::SearchResult> for SearchResult {
    fn from(r: leann_core::SearchResult) -> Self {
        Self {
            id: r.id,
            score: r.score,
            text: r.text,
            metadata_inner: r.metadata,
        }
    }
}

/// Builder for creating LEANN indexes.
#[pyclass]
struct LeannBuilder {
    inner: leann_core::LeannBuilder,
}

#[pymethods]
impl LeannBuilder {
    #[new]
    #[pyo3(signature = (embedding_model, dimensions=None, embedding_mode="sentence-transformers", **kwargs))]
    fn new(
        embedding_model: &str,
        dimensions: Option<usize>,
        embedding_mode: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut builder = leann_core::LeannBuilder::new(
            embedding_model,
            dimensions,
            embedding_mode,
        );

        if let Some(kw) = kwargs {
            if let Ok(Some(m)) = kw.get_item("m") {
                if let Ok(val) = m.extract::<usize>() {
                    builder = builder.with_m(val);
                }
            }
            if let Ok(Some(ef)) = kw.get_item("ef_construction") {
                if let Ok(val) = ef.extract::<usize>() {
                    builder = builder.with_ef_construction(val);
                }
            }
            if let Ok(Some(compact)) = kw.get_item("compact") {
                if let Ok(val) = compact.extract::<bool>() {
                    builder = builder.with_compact(val);
                }
            }
            if let Ok(Some(recompute)) = kw.get_item("recompute") {
                if let Ok(val) = recompute.extract::<bool>() {
                    builder = builder.with_recompute(val);
                }
            }
        }

        Ok(Self { inner: builder })
    }

    /// Add a text chunk with optional metadata.
    #[pyo3(signature = (text, metadata=None))]
    fn add_text(&mut self, text: &str, metadata: Option<&Bound<'_, PyDict>>) {
        let meta = metadata
            .map(py_dict_to_metadata)
            .unwrap_or_default();
        self.inner.add_text(text, meta);
    }

    /// Build the index at the given path.
    fn build_index(&mut self, py: Python<'_>, index_path: &str) -> PyResult<()> {
        py.allow_threads(|| {
            // Use Ollama as default provider for the builder
            let provider = leann_core::embedding::ollama::OllamaEmbedding::new(
                "nomic-embed-text",
                None,
            );
            self.inner
                .build_index(&PathBuf::from(index_path), &provider)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
        })
    }
}

/// Searcher for querying LEANN indexes.
#[pyclass]
struct LeannSearcher {
    inner: leann_core::LeannSearcher,
}

#[pymethods]
impl LeannSearcher {
    #[new]
    #[pyo3(signature = (index_path, **_kwargs))]
    fn new(index_path: &str, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let searcher = leann_core::LeannSearcher::open(&PathBuf::from(index_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        Ok(Self { inner: searcher })
    }

    /// Search the index.
    #[pyo3(signature = (query, top_k=5, **_kwargs))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<SearchResult>> {
        let query_owned = query.to_string();
        py.allow_threads(|| {
            let results = self
                .inner
                .search(&query_owned, top_k)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

            Ok(results.into_iter().map(SearchResult::from).collect())
        })
    }

    fn cleanup(&mut self) {
        self.inner.cleanup();
    }
}

/// RAG chat interface combining search + LLM.
#[pyclass]
struct LeannChat {
    inner: leann_core::chat::LeannChat,
}

#[pymethods]
impl LeannChat {
    #[new]
    #[pyo3(signature = (index_path, llm_config=None, **_kwargs))]
    fn new(
        index_path: &str,
        llm_config: Option<&Bound<'_, PyDict>>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let searcher = leann_core::LeannSearcher::open(&PathBuf::from(index_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let config = if let Some(cfg) = llm_config {
            let llm_type = cfg
                .get_item("type")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "openai".to_string());
            let model = cfg
                .get_item("model")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok());
            let api_key = cfg
                .get_item("api_key")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok());

            Some(leann_core::chat::LlmConfig {
                llm_type,
                model,
                api_key,
                base_url: None,
                host: None,
            })
        } else {
            None
        };

        let chat = leann_core::chat::LeannChat::new(searcher, config.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        Ok(Self { inner: chat })
    }

    /// Ask a question using RAG.
    #[pyo3(signature = (question, top_k=5, **_kwargs))]
    fn ask(
        &self,
        py: Python<'_>,
        question: &str,
        top_k: usize,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let question_owned = question.to_string();
        py.allow_threads(|| {
            self.inner
                .ask(&question_owned, top_k)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
        })
    }
}

/// Multi-turn reasoning agent.
#[pyclass]
struct ReActAgent {
    searcher_path: String,
    max_iterations: usize,
}

#[pymethods]
impl ReActAgent {
    #[new]
    #[pyo3(signature = (index_path, max_iterations=5))]
    fn new(index_path: &str, max_iterations: usize) -> PyResult<Self> {
        Ok(Self {
            searcher_path: index_path.to_string(),
            max_iterations,
        })
    }

    /// Run the agent on a question.
    #[pyo3(signature = (question, top_k=5))]
    fn run(&self, py: Python<'_>, question: &str, top_k: usize) -> PyResult<String> {
        let searcher = leann_core::LeannSearcher::open(&PathBuf::from(&self.searcher_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let config = leann_core::chat::LlmConfig::default();

        let mut agent = leann_core::react_agent::ReActAgent::new(
            searcher,
            Some(&config),
            self.max_iterations,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let question_owned = question.to_string();
        py.allow_threads(|| {
            agent
                .run(&question_owned, top_k)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
        })
    }
}

/// LEANN Python module.
#[pymodule]
fn leann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchResult>()?;
    m.add_class::<LeannBuilder>()?;
    m.add_class::<LeannSearcher>()?;
    m.add_class::<LeannChat>()?;
    m.add_class::<ReActAgent>()?;
    Ok(())
}

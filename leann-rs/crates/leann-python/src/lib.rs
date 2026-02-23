use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;

use leann_core::searcher::SearchConfig;

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

/// Extract a kwarg by trying multiple key names in order.
/// Returns the first match found, or None if no key matched.
fn extract_kwarg<'py, T: pyo3::FromPyObject<'py>>(
    kw: &Bound<'py, PyDict>,
    keys: &[&str],
) -> Option<T> {
    for key in keys {
        if let Ok(Some(val)) = kw.get_item(*key) {
            if let Ok(extracted) = val.extract::<T>() {
                return Some(extracted);
            }
        }
    }
    None
}

/// Convert an anyhow::Error to the appropriate Python exception.
fn anyhow_to_pyerr(e: anyhow::Error) -> PyErr {
    let msg = format!("{}", e);
    // Map common error patterns to appropriate Python exception types
    if msg.contains("not found")
        || msg.contains("No such file")
        || msg.contains("does not exist")
    {
        pyo3::exceptions::PyFileNotFoundError::new_err(msg)
    } else if msg.contains("Mismatch")
        || msg.contains("Invalid")
        || msg.contains("empty")
        || msg.contains("not supported")
    {
        pyo3::exceptions::PyValueError::new_err(msg)
    } else {
        pyo3::exceptions::PyRuntimeError::new_err(msg)
    }
}

/// Extract LlmParams from Python kwargs dict.
///
/// Pulls `temperature`, `max_tokens`, `top_p` from the dict.
/// Any remaining keys that aren't search-config keys go into `extra`.
fn extract_llm_params(kw: Option<&Bound<'_, PyDict>>) -> leann_core::chat::LlmParams {
    let Some(kw) = kw else {
        return leann_core::chat::LlmParams::default();
    };

    let mut params = leann_core::chat::LlmParams::default();

    if let Some(v) = extract_kwarg::<f64>(kw, &["temperature"]) {
        params.temperature = Some(v);
    }
    if let Some(v) = extract_kwarg::<usize>(kw, &["max_tokens"]) {
        params.max_tokens = Some(v);
    }
    if let Some(v) = extract_kwarg::<f64>(kw, &["top_p"]) {
        params.top_p = Some(v);
    }

    // Remaining keys that aren't search-config or llm-param keys go into extra.
    let known_keys: &[&str] = &[
        "temperature",
        "max_tokens",
        "top_p",
        "complexity",
        "beam_width",
        "prune_ratio",
        "batch_size",
        "use_grep",
        "gemma",
        "expected_zmq_port",
        "zmq_port",
        "pruning_strategy",
        "metadata_filters",
        "provider_options",
    ];
    for (key, value) in kw.iter() {
        if let Ok(key_str) = key.extract::<String>() {
            if !known_keys.contains(&key_str.as_str()) {
                params.extra.insert(key_str, py_to_json_value(&value));
            }
        }
    }

    params
}

/// Extract a SearchConfig from Python kwargs dict.
fn extract_search_config(kw: Option<&Bound<'_, PyDict>>) -> SearchConfig {
    let Some(kw) = kw else {
        return SearchConfig::default();
    };

    let mut config = SearchConfig::default();

    if let Some(v) = extract_kwarg::<usize>(kw, &["complexity"]) {
        config.complexity = v;
    }
    if let Some(v) = extract_kwarg::<usize>(kw, &["beam_width"]) {
        config.beam_width = v;
    }
    if let Some(v) = extract_kwarg::<f64>(kw, &["prune_ratio"]) {
        config.prune_ratio = v;
    }
    if let Some(v) = extract_kwarg::<usize>(kw, &["batch_size"]) {
        config.batch_size = v;
    }
    if let Some(v) = extract_kwarg::<bool>(kw, &["use_grep"]) {
        config.use_grep = v;
    }
    if let Some(v) = extract_kwarg::<f64>(kw, &["gemma"]) {
        config.gemma = v;
    }
    if let Some(v) = extract_kwarg::<u16>(kw, &["expected_zmq_port", "zmq_port"]) {
        config.zmq_port = Some(v);
    }
    if let Some(v) = extract_kwarg::<String>(kw, &["pruning_strategy"]) {
        config.pruning_strategy = Some(v);
    }

    // Extract provider_options: dict[str, Any]
    if let Ok(Some(po_obj)) = kw.get_item("provider_options")
        && let Ok(po_dict) = po_obj.downcast::<PyDict>()
    {
        config.provider_options = Some(py_dict_to_metadata(po_dict));
    }

    // Extract metadata_filters: dict[str, dict[str, Any]]
    if let Ok(Some(filters_obj)) = kw.get_item("metadata_filters") {
        if let Ok(filters_dict) = filters_obj.downcast::<PyDict>() {
            let mut filters = HashMap::new();
            for (field_key, field_val) in filters_dict.iter() {
                if let Ok(field_name) = field_key.extract::<String>() {
                    if let Ok(spec_dict) = field_val.downcast::<PyDict>() {
                        let mut spec = HashMap::new();
                        for (op_key, op_val) in spec_dict.iter() {
                            if let Ok(op_name) = op_key.extract::<String>() {
                                spec.insert(op_name, py_to_json_value(&op_val));
                            }
                        }
                        filters.insert(field_name, spec);
                    }
                }
            }
            if !filters.is_empty() {
                config.metadata_filters = Some(filters);
            }
        }
    }

    config
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
///
/// Signature matches the Python `LeannBuilder`:
///   LeannBuilder(backend_name, embedding_model=..., dimensions=...,
///                embedding_mode=..., embedding_options=..., **backend_kwargs)
///
/// `backend_kwargs` accepts both Python-style names (M, efConstruction,
/// is_compact, is_recompute) and Rust-style names (m, ef_construction,
/// compact, recompute). Python-style names take precedence when both are given.
#[pyclass]
struct LeannBuilder {
    inner: leann_core::LeannBuilder,
}

#[pymethods]
impl LeannBuilder {
    #[new]
    #[pyo3(signature = (backend_name="hnsw", embedding_model="facebook/contriever", dimensions=None, embedding_mode="sentence-transformers", embedding_options=None, **kwargs))]
    fn new(
        backend_name: &str,
        embedding_model: &str,
        dimensions: Option<usize>,
        embedding_mode: &str,
        embedding_options: Option<&Bound<'_, PyDict>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if backend_name != "hnsw" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Backend '{}' is not supported in the Rust implementation. Only 'hnsw' is available.",
                backend_name
            )));
        }

        let mut builder =
            leann_core::LeannBuilder::new(embedding_model, dimensions, embedding_mode);

        if let Some(opts) = embedding_options {
            builder = builder.with_embedding_options(py_dict_to_metadata(opts));
        }

        if let Some(kw) = kwargs {
            // Accept Python-style (M, efConstruction, is_compact, is_recompute)
            // and Rust-style (m, ef_construction, compact, recompute).
            // Python-style checked first so it wins when both are present.
            if let Some(val) = extract_kwarg::<usize>(kw, &["M", "m"]) {
                builder = builder.with_m(val);
            }
            if let Some(val) = extract_kwarg::<usize>(kw, &["efConstruction", "ef_construction"]) {
                builder = builder.with_ef_construction(val);
            }
            if let Some(val) = extract_kwarg::<bool>(kw, &["is_compact", "compact"]) {
                builder = builder.with_compact(val);
            }
            if let Some(val) = extract_kwarg::<bool>(kw, &["is_recompute", "recompute"]) {
                builder = builder.with_recompute(val);
            }
            if let Some(val) = extract_kwarg::<String>(kw, &["distance_metric"]) {
                builder = builder.with_distance_metric(
                    leann_core::index::DistanceMetric::from_str_lossy(&val),
                );
            }
        }

        Ok(Self { inner: builder })
    }

    /// Add a text chunk with optional metadata.
    #[pyo3(signature = (text, metadata=None))]
    fn add_text(&mut self, text: &str, metadata: Option<&Bound<'_, PyDict>>) {
        let meta = metadata.map(py_dict_to_metadata).unwrap_or_default();
        self.inner.add_text(text, meta);
    }

    /// Build the index at the given path.
    ///
    /// Uses the builder's `embedding_mode` and `embedding_model` to select
    /// the embedding provider (ollama, openai, gemini, or sentence-transformers/zmq).
    fn build_index(&mut self, py: Python<'_>, index_path: &str) -> PyResult<()> {
        py.allow_threads(|| {
            let provider = self.inner.create_embedding_provider().map_err(anyhow_to_pyerr)?;
            self.inner
                .build_index(&PathBuf::from(index_path), provider.as_ref())
                .map_err(anyhow_to_pyerr)
        })
    }

    /// Build the index from pre-computed embeddings (no embedding server needed).
    fn build_index_from_embeddings(
        &mut self,
        py: Python<'_>,
        index_path: &str,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
    ) -> PyResult<()> {
        if embeddings.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "embeddings list is empty",
            ));
        }
        let ncols = embeddings[0].len();
        let nrows = embeddings.len();
        let flat: Vec<f32> = embeddings.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((nrows, ncols), flat).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid embedding shape: {}", e))
        })?;

        py.allow_threads(|| {
            self.inner
                .build_index_from_embeddings(&PathBuf::from(index_path), &ids, &arr)
                .map_err(anyhow_to_pyerr)
        })
    }
}

/// Searcher for querying LEANN indexes.
///
/// Signature matches the Python `LeannSearcher`:
///   LeannSearcher(index_path, enable_warmup=True, recompute_embeddings=True, **kwargs)
#[pyclass]
struct LeannSearcher {
    inner: leann_core::LeannSearcher,
    index_path: String,
}

#[pymethods]
impl LeannSearcher {
    #[new]
    #[pyo3(signature = (index_path, enable_warmup=true, recompute_embeddings=true, **kwargs))]
    #[allow(unused_variables)]
    fn new(
        index_path: &str,
        enable_warmup: bool,
        recompute_embeddings: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let options = leann_core::SearcherOptions {
            recompute_embeddings: Some(recompute_embeddings),
            enable_warmup,
        };
        let searcher =
            leann_core::LeannSearcher::open_with_options(&PathBuf::from(index_path), &options)
                .map_err(anyhow_to_pyerr)?;

        Ok(Self {
            inner: searcher,
            index_path: index_path.to_string(),
        })
    }

    /// Search the index with optional configuration kwargs.
    ///
    /// Supported kwargs: complexity, beam_width, prune_ratio, metadata_filters,
    /// batch_size, use_grep, gemma, expected_zmq_port.
    #[pyo3(signature = (query, top_k=5, **kwargs))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<SearchResult>> {
        let config = extract_search_config(kwargs);
        let query_owned = query.to_string();
        py.allow_threads(|| {
            let results = self
                .inner
                .search_with_params(&query_owned, top_k, &config)
                .map_err(anyhow_to_pyerr)?;

            Ok(results.into_iter().map(SearchResult::from).collect())
        })
    }

    fn cleanup(&mut self) {
        self.inner.cleanup();
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) {
        self.inner.cleanup();
    }
}

/// RAG chat interface combining search + LLM.
///
/// Signature matches the Python `LeannChat`:
///   LeannChat(index_path, llm_config=None, enable_warmup=False, searcher=None, **kwargs)
#[pyclass]
struct LeannChat {
    inner: leann_core::chat::LeannChat,
}

/// Extract an LlmConfig from a Python dict.
fn extract_llm_config(cfg: &Bound<'_, PyDict>) -> leann_core::chat::LlmConfig {
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
    let base_url = cfg
        .get_item("base_url")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok());
    let host = cfg
        .get_item("host")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok());

    leann_core::chat::LlmConfig {
        llm_type,
        model,
        api_key,
        base_url,
        host,
    }
}

#[pymethods]
impl LeannChat {
    #[new]
    #[pyo3(signature = (index_path, llm_config=None, enable_warmup=false, searcher=None, **kwargs))]
    #[allow(unused_variables)]
    fn new(
        index_path: &str,
        llm_config: Option<&Bound<'_, PyDict>>,
        enable_warmup: bool,
        searcher: Option<&Bound<'_, LeannSearcher>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let config = llm_config.map(extract_llm_config);

        // If a searcher is provided, use its index_path (same pattern as ReActAgent).
        let effective_path = if let Some(s) = searcher {
            s.borrow().index_path.clone()
        } else {
            index_path.to_string()
        };

        let searcher_options = leann_core::SearcherOptions {
            recompute_embeddings: None,
            enable_warmup,
        };
        let chat = leann_core::chat::LeannChat::new_with_options(
            &PathBuf::from(&effective_path),
            config.as_ref(),
            &searcher_options,
        )
        .map_err(anyhow_to_pyerr)?;

        Ok(Self { inner: chat })
    }

    /// Ask a question using RAG with optional search and LLM configuration kwargs.
    ///
    /// Search kwargs: complexity, beam_width, prune_ratio, metadata_filters,
    /// batch_size, use_grep, gemma, expected_zmq_port.
    /// LLM kwargs: temperature, max_tokens, top_p (plus any extras).
    #[pyo3(signature = (question, top_k=5, **kwargs))]
    fn ask(
        &self,
        py: Python<'_>,
        question: &str,
        top_k: usize,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let config = extract_search_config(kwargs);
        let llm_params = extract_llm_params(kwargs);
        let question_owned = question.to_string();
        py.allow_threads(|| {
            self.inner
                .ask_with_params(&question_owned, top_k, &config, &llm_params)
                .map_err(anyhow_to_pyerr)
        })
    }

    /// Start an interactive REPL loop, reading questions from stdin.
    #[pyo3(signature = (top_k=5))]
    fn start_interactive(&self, py: Python<'_>, top_k: usize) -> PyResult<()> {
        use std::io::{BufRead, Write};

        println!("LEANN interactive mode (type 'quit' or 'exit' to stop)");
        loop {
            print!("\nQuestion: ");
            std::io::stdout().flush().unwrap();

            let mut line = String::new();
            let bytes = std::io::stdin().lock().read_line(&mut line).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("stdin read error: {e}"))
            })?;
            if bytes == 0 {
                break; // EOF
            }

            let question = line.trim();
            if question.is_empty() {
                continue;
            }
            if question == "quit" || question == "exit" {
                break;
            }

            let q = question.to_string();
            match py.allow_threads(|| self.inner.ask(&q, top_k)) {
                Ok(answer) => println!("\n{answer}"),
                Err(e) => println!("\nError: {e}"),
            }
        }
        Ok(())
    }

    fn cleanup(&mut self) {
        self.inner.cleanup();
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) {
        self.inner.cleanup();
    }
}

/// Multi-turn reasoning agent.
///
/// Signature matches the Python `ReActAgent`:
///   ReActAgent(searcher, llm=None, llm_config=None, max_iterations=5)
///
/// The first argument can be a `LeannSearcher` instance (matching Python)
/// or a string index path (convenience).
#[pyclass]
struct ReActAgent {
    searcher_path: String,
    max_iterations: usize,
}

#[pymethods]
impl ReActAgent {
    #[new]
    #[pyo3(signature = (searcher, llm=None, llm_config=None, max_iterations=5))]
    #[allow(unused_variables)]
    fn new(
        searcher: &Bound<'_, PyAny>,
        llm: Option<&Bound<'_, PyAny>>,
        llm_config: Option<&Bound<'_, PyDict>>,
        max_iterations: usize,
    ) -> PyResult<Self> {
        let path = if let Ok(s) = searcher.extract::<String>() {
            s
        } else if let Ok(s) = searcher.downcast::<LeannSearcher>() {
            s.borrow().index_path.clone()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "First argument must be a LeannSearcher or a string index path",
            ));
        };

        Ok(Self {
            searcher_path: path,
            max_iterations,
        })
    }

    /// Run the agent on a question.
    #[pyo3(signature = (question, top_k=5))]
    fn run(&self, py: Python<'_>, question: &str, top_k: usize) -> PyResult<String> {
        let searcher = leann_core::LeannSearcher::open(&PathBuf::from(&self.searcher_path))
            .map_err(anyhow_to_pyerr)?;

        let config = leann_core::chat::LlmConfig::default();

        let mut agent = leann_core::react_agent::ReActAgent::new(
            searcher,
            Some(&config),
            self.max_iterations,
        )
        .map_err(anyhow_to_pyerr)?;

        let question_owned = question.to_string();
        py.allow_threads(|| agent.run(&question_owned, top_k).map_err(anyhow_to_pyerr))
    }

    /// Search the index directly (without the ReAct reasoning loop).
    #[pyo3(signature = (query, top_k=5))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
    ) -> PyResult<Vec<SearchResult>> {
        let searcher = leann_core::LeannSearcher::open(&PathBuf::from(&self.searcher_path))
            .map_err(anyhow_to_pyerr)?;
        let query_owned = query.to_string();
        py.allow_threads(|| {
            let results = searcher
                .search(&query_owned, top_k)
                .map_err(anyhow_to_pyerr)?;
            Ok(results.into_iter().map(SearchResult::from).collect())
        })
    }
}

/// Get list of registered backend names.
#[pyfunction]
fn get_registered_backends() -> Vec<String> {
    vec!["hnsw".to_string()]
}

/// Convenience factory matching Python's `create_react_agent()`.
///
/// Signature: `create_react_agent(index_path, llm_config=None, max_iterations=5, **searcher_kwargs)`
#[pyfunction]
#[pyo3(signature = (index_path, llm_config=None, max_iterations=5, **searcher_kwargs))]
#[allow(unused_variables)]
fn create_react_agent(
    index_path: &str,
    llm_config: Option<&Bound<'_, PyDict>>,
    max_iterations: usize,
    searcher_kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<ReActAgent> {
    Ok(ReActAgent {
        searcher_path: index_path.to_string(),
        max_iterations,
    })
}

/// LEANN Python module.
#[pymodule]
fn leann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchResult>()?;
    m.add_class::<LeannBuilder>()?;
    m.add_class::<LeannSearcher>()?;
    m.add_class::<LeannChat>()?;
    m.add_class::<ReActAgent>()?;
    m.add_function(wrap_pyfunction!(get_registered_backends, m)?)?;
    m.add_function(wrap_pyfunction!(create_react_agent, m)?)?;

    // Expose BACKEND_REGISTRY dict matching Python's module-level export.
    let registry = PyDict::new(m.py());
    registry.set_item("hnsw", "hnsw")?;
    m.add("BACKEND_REGISTRY", registry)?;

    Ok(())
}

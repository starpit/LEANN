#![allow(unused_variables)]

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "leann",
    version,
    about = "LEANN - Lightweight vector database and RAG system"
)]
struct Cli {
    /// Show detailed output including backend logs
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    /// Suppress all non-essential output
    #[arg(short = 'q', long, global = true, conflicts_with = "verbose")]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build document index
    Build {
        /// Index name (default: current directory name)
        index_name: Option<String>,

        /// Documents directories and/or files (default: current directory)
        #[arg(long, num_args = 1.., default_values_t = vec![".".to_string()])]
        docs: Vec<String>,

        /// Backend to use
        #[arg(long, default_value = "hnsw", value_parser = ["hnsw", "diskann"])]
        backend_name: String,

        /// Embedding model
        #[arg(long, default_value = "facebook/contriever")]
        embedding_model: String,

        /// Embedding backend mode
        #[arg(long, default_value = "sentence-transformers", value_parser = ["sentence-transformers", "openai", "mlx", "ollama"])]
        embedding_mode: String,

        /// Override Ollama-compatible embedding host
        #[arg(long)]
        embedding_host: Option<String>,

        /// Base URL for OpenAI-compatible embedding services
        #[arg(long)]
        embedding_api_base: Option<String>,

        /// API key for embedding service (defaults to OPENAI_API_KEY)
        #[arg(long)]
        embedding_api_key: Option<String>,

        /// Prompt template to prepend to all texts for embedding
        #[arg(long)]
        embedding_prompt_template: Option<String>,

        /// Prompt template for queries (different from build template)
        #[arg(long)]
        query_prompt_template: Option<String>,

        /// Force rebuild existing index
        #[arg(long, short = 'f')]
        force: bool,

        /// Graph degree (default: 32)
        #[arg(long, default_value = "32")]
        graph_degree: usize,

        /// Build complexity (default: 64)
        #[arg(long, default_value = "64")]
        complexity: usize,

        /// Number of threads for HNSW construction (default: all cores)
        #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))]
        num_threads: usize,

        /// Use compact storage (default: true). Must use --no-compact for --no-recompute build.
        #[arg(long)]
        compact: bool,

        /// Disable compact storage
        #[arg(long = "no-compact")]
        no_compact: bool,

        /// Enable recomputation (default: true)
        #[arg(long)]
        recompute: bool,

        /// Disable recomputation
        #[arg(long = "no-recompute")]
        no_recompute: bool,

        /// Comma-separated list of file extensions to include (e.g., '.txt,.pdf,.pptx'). If not specified, uses default supported types.
        #[arg(long)]
        file_types: Option<String>,

        /// Include hidden files and directories (default: false, disable with --no-include-hidden)
        #[arg(long)]
        include_hidden: bool,

        /// Document chunk size in tokens (default: 256)
        #[arg(long, default_value = "256")]
        doc_chunk_size: usize,

        /// Document chunk overlap in tokens (default: 128)
        #[arg(long, default_value = "128")]
        doc_chunk_overlap: usize,

        /// Code chunk size in tokens (default: 512)
        #[arg(long, default_value = "512")]
        code_chunk_size: usize,

        /// Code chunk overlap in tokens (default: 50)
        #[arg(long, default_value = "50")]
        code_chunk_overlap: usize,

        /// Enable AST-aware chunking for code files
        #[arg(long)]
        use_ast_chunking: bool,

        /// AST chunk size in characters (default: 300)
        #[arg(long, default_value = "300")]
        ast_chunk_size: usize,

        /// AST chunk overlap in characters (default: 64)
        #[arg(long, default_value = "64")]
        ast_chunk_overlap: usize,

        /// Fall back to traditional chunking if AST chunking fails (default: true)
        #[arg(long)]
        ast_fallback_traditional: bool,
    },

    /// Compare current files against last checkpoint and report changes
    Watch {
        /// Index name
        index_name: String,
    },

    /// Search documents
    Search {
        /// Index name
        index_name: String,
        /// Search query
        query: String,
        /// Number of results (default: 5)
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Search complexity (default: 64)
        #[arg(long, default_value = "64")]
        complexity: usize,
        /// Beam width
        #[arg(long, default_value = "1")]
        beam_width: usize,
        /// Prune ratio
        #[arg(long, default_value = "0.0")]
        prune_ratio: f64,
        /// Enable/disable embedding recomputation (default: true)
        #[arg(long)]
        recompute: bool,
        /// Disable embedding recomputation
        #[arg(long = "no-recompute")]
        no_recompute: bool,
        /// Pruning strategy
        #[arg(long, default_value = "global", value_parser = ["global", "local", "proportional"])]
        pruning_strategy: String,
        /// Non-interactive mode: automatically select index without prompting
        #[arg(long)]
        non_interactive: bool,
        /// Display file paths and metadata in search results
        #[arg(long)]
        show_metadata: bool,
        /// Prompt template to prepend to query for embedding
        #[arg(long)]
        embedding_prompt_template: Option<String>,
    },

    /// Ask questions
    Ask {
        /// Index name
        index_name: String,
        /// Question to ask (omit for prompt or when using --interactive)
        query: Option<String>,
        /// LLM provider
        #[arg(long, default_value = "ollama", value_parser = ["simulated", "ollama", "hf", "openai", "anthropic"])]
        llm: String,
        /// Model name
        #[arg(long, default_value = "qwen3:8b")]
        model: String,
        /// Override Ollama-compatible host
        #[arg(long)]
        host: Option<String>,
        /// Interactive chat mode
        #[arg(long, short = 'i')]
        interactive: bool,
        /// Retrieval count (default: 20)
        #[arg(long, default_value = "20")]
        top_k: usize,
        /// Search complexity
        #[arg(long, default_value = "32")]
        complexity: usize,
        /// Beam width
        #[arg(long, default_value = "1")]
        beam_width: usize,
        /// Prune ratio
        #[arg(long, default_value = "0.0")]
        prune_ratio: f64,
        /// Enable/disable embedding recomputation (default: true)
        #[arg(long)]
        recompute: bool,
        /// Disable embedding recomputation
        #[arg(long = "no-recompute")]
        no_recompute: bool,
        /// Pruning strategy
        #[arg(long, default_value = "global", value_parser = ["global", "local", "proportional"])]
        pruning_strategy: String,
        /// Thinking budget for reasoning models (supported by GPT-Oss:20b and other reasoning models)
        #[arg(long, value_parser = ["low", "medium", "high"])]
        thinking_budget: Option<String>,
        /// Base URL for OpenAI-compatible APIs
        #[arg(long)]
        api_base: Option<String>,
        /// API key for cloud LLM providers
        #[arg(long)]
        api_key: Option<String>,
    },

    /// Use ReAct agent for multiturn retrieval and reasoning
    React {
        /// Index name
        index_name: String,
        /// Question to research
        query: String,
        /// LLM provider
        #[arg(long, default_value = "ollama", value_parser = ["simulated", "ollama", "hf", "openai", "anthropic"])]
        llm: String,
        /// Model name
        #[arg(long, default_value = "qwen3:8b")]
        model: String,
        /// Override Ollama-compatible host
        #[arg(long)]
        host: Option<String>,
        /// Number of results per search (default: 5)
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Maximum number of search iterations (default: 5)
        #[arg(long, default_value = "5")]
        max_iterations: usize,
        /// Base URL for OpenAI-compatible APIs
        #[arg(long)]
        api_base: Option<String>,
        /// API key for cloud LLM providers
        #[arg(long)]
        api_key: Option<String>,
    },

    /// List all indexes
    List,

    /// Remove an index
    Remove {
        /// Index name to remove
        index_name: String,
        /// Force removal without confirmation
        #[arg(long, short = 'f')]
        force: bool,
    },

    /// Start HTTP API server for LEANN vector DB
    Serve {
        /// Host to bind to (default: 0.0.0.0)
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Port to bind to (default: 8000)
        #[arg(long, default_value = "8000")]
        port: u16,
    },
}

// ---------------------------------------------------------------------------
// Index path resolution (matches Python CLI convention)
// ---------------------------------------------------------------------------

/// Get the indexes directory: .leann/indexes/ under current working directory.
fn indexes_dir() -> PathBuf {
    let dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".leann")
        .join("indexes");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Resolve the index base path for a given index name.
/// Convention: .leann/indexes/<index_name>/documents.leann
fn get_index_path(index_name: &str) -> PathBuf {
    indexes_dir().join(index_name).join("documents.leann")
}

/// Check if an index exists.
fn index_exists(index_name: &str) -> bool {
    let meta = indexes_dir()
        .join(index_name)
        .join("documents.leann.meta.json");
    meta.exists()
}

/// Derive index name from current directory if not provided.
fn resolve_index_name(name: Option<&str>) -> String {
    if let Some(n) = name {
        n.to_string()
    } else {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "index".to_string())
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            index_name,
            docs,
            backend_name,
            embedding_model,
            embedding_mode,
            embedding_host,
            embedding_api_base,
            embedding_api_key,
            embedding_prompt_template,
            query_prompt_template,
            force,
            graph_degree,
            complexity,
            num_threads,
            compact: _,
            no_compact,
            recompute: _,
            no_recompute,
            file_types,
            include_hidden,
            doc_chunk_size,
            doc_chunk_overlap,
            code_chunk_size,
            code_chunk_overlap,
            use_ast_chunking,
            ast_chunk_size,
            ast_chunk_overlap,
            ast_fallback_traditional: _,
        } => {
            let name = resolve_index_name(index_name.as_deref());
            cmd_build(BuildArgs {
                index_name: name,
                docs,
                embedding_model,
                embedding_mode,
                embedding_host,
                embedding_api_base,
                embedding_api_key,
                force,
                graph_degree,
                complexity,
                num_threads,
                compact: !no_compact,     // default true unless --no-compact
                recompute: !no_recompute, // default true unless --no-recompute
                file_types,
                include_hidden,
                doc_chunk_size,
                doc_chunk_overlap,
                code_chunk_size,
                code_chunk_overlap,
                use_ast_chunking,
                ast_chunk_size,
                _ast_chunk_overlap: ast_chunk_overlap,
                _ast_fallback_traditional: true, // default true per Python CLI (flag only confirms)
            })?;
        }
        Commands::Search {
            index_name,
            query,
            top_k,
            complexity,
            beam_width,
            prune_ratio,
            recompute: _,
            no_recompute,
            pruning_strategy,
            non_interactive,
            show_metadata,
            embedding_prompt_template,
        } => {
            let _recompute = !no_recompute;
            cmd_search(
                &index_name,
                &query,
                top_k,
                complexity,
                beam_width,
                prune_ratio,
                show_metadata,
            )?;
        }
        Commands::Ask {
            index_name,
            query,
            llm,
            model,
            host,
            interactive,
            top_k,
            complexity,
            beam_width,
            prune_ratio,
            recompute: _,
            no_recompute,
            pruning_strategy,
            thinking_budget,
            api_base,
            api_key,
        } => {
            let _recompute = !no_recompute;
            if interactive {
                cmd_interactive_ask(
                    &index_name,
                    top_k,
                    &llm,
                    &model,
                    host.as_deref(),
                    api_base.as_deref(),
                    api_key.as_deref(),
                )?;
            } else if let Some(q) = query {
                cmd_ask(
                    &index_name,
                    &q,
                    top_k,
                    &llm,
                    &model,
                    host.as_deref(),
                    api_base.as_deref(),
                    api_key.as_deref(),
                )?;
            } else {
                println!("Provide a question or use --interactive");
            }
        }
        Commands::React {
            index_name,
            query,
            llm,
            model,
            host,
            top_k,
            max_iterations,
            api_base,
            api_key,
        } => {
            cmd_react(
                &index_name,
                &query,
                top_k,
                max_iterations,
                &llm,
                &model,
                host.as_deref(),
                api_base.as_deref(),
                api_key.as_deref(),
            )?;
        }
        Commands::List => {
            cmd_list()?;
        }
        Commands::Remove { index_name, force } => {
            cmd_remove(&index_name, force)?;
        }
        Commands::Watch { index_name } => {
            cmd_watch(&index_name)?;
        }
        Commands::Serve { host, port } => {
            cmd_serve(&host, port)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

struct BuildArgs {
    index_name: String,
    docs: Vec<String>,
    embedding_model: String,
    embedding_mode: String,
    embedding_host: Option<String>,
    embedding_api_base: Option<String>,
    embedding_api_key: Option<String>,
    force: bool,
    graph_degree: usize,
    complexity: usize,
    num_threads: usize,
    compact: bool,
    recompute: bool,
    file_types: Option<String>,
    include_hidden: bool,
    doc_chunk_size: usize,
    doc_chunk_overlap: usize,
    code_chunk_size: usize,
    code_chunk_overlap: usize,
    use_ast_chunking: bool,
    ast_chunk_size: usize,
    _ast_chunk_overlap: usize,
    _ast_fallback_traditional: bool,
}

fn cmd_build(args: BuildArgs) -> Result<()> {
    use leann_core::builder::LeannBuilder;
    use leann_core::chunking;
    use leann_core::embedding::EmbeddingMode;
    use leann_core::index::DistanceMetric;

    let index_dir = indexes_dir().join(&args.index_name);
    let index_path = get_index_path(&args.index_name);

    if index_exists(&args.index_name) && !args.force {
        println!(
            "Index '{}' already exists. Use --force to rebuild.",
            args.index_name
        );
        return Ok(());
    }

    // Classify docs paths (use !is_dir for files to support named pipes/fifos
    // from process substitution like <(cat file))
    let directories: Vec<&String> = args.docs.iter().filter(|p| Path::new(p).is_dir()).collect();
    let files: Vec<&String> = args
        .docs
        .iter()
        .filter(|p| !Path::new(p).is_dir())
        .collect();

    println!("Indexing {} path(s):", args.docs.len());
    if !files.is_empty() {
        println!("  Files ({}):", files.len());
        for (i, f) in files.iter().enumerate() {
            println!(
                "    {}. {}",
                i + 1,
                std::fs::canonicalize(f)
                    .unwrap_or_else(|_| PathBuf::from(f))
                    .display()
            );
        }
    }
    if !directories.is_empty() {
        println!("  Directories ({}):", directories.len());
        for (i, d) in directories.iter().enumerate() {
            println!(
                "    {}. {}",
                i + 1,
                std::fs::canonicalize(d)
                    .unwrap_or_else(|_| PathBuf::from(d))
                    .display()
            );
        }
    }

    // Parse allowed extensions
    let allowed_extensions: Option<Vec<String>> = args.file_types.as_ref().map(|ft| {
        ft.split(',')
            .map(|e| e.trim().trim_start_matches('.').to_lowercase())
            .filter(|e| !e.is_empty())
            .collect()
    });

    // Load documents from all paths
    let mut documents = Vec::new();
    for doc_path in &args.docs {
        let p = Path::new(doc_path);
        if p.is_dir() {
            let loaded = load_documents(p, allowed_extensions.as_deref(), args.include_hidden)?;
            documents.extend(loaded);
        } else if p.exists() {
            // Handles regular files, named pipes/fifos (from process substitution
            // like <(cat file)), and other readable non-directory paths
            match leann_core::document_loaders::extract_text(p) {
                Ok(Some(content)) => {
                    documents.push((p.to_string_lossy().to_string(), content));
                }
                Ok(None) => {
                    eprintln!("Warning: no text extracted from {}", doc_path);
                }
                Err(e) => {
                    eprintln!("Warning: failed to read {}: {}", doc_path, e);
                }
            }
        } else {
            eprintln!("Warning: path not found: {}", doc_path);
        }
    }

    println!("Loaded {} documents", documents.len());
    if documents.is_empty() {
        anyhow::bail!("No documents found");
    }

    let mode = EmbeddingMode::from_str_lossy(&args.embedding_mode);
    let metric = DistanceMetric::default(); // MIPS

    let mut builder = LeannBuilder::new(&args.embedding_model, None, &args.embedding_mode)
        .with_m(args.graph_degree)
        .with_ef_construction(args.complexity)
        .with_distance_metric(metric)
        .with_compact(args.compact)
        .with_recompute(args.recompute)
        .with_num_threads(args.num_threads);

    // Chunk and add documents
    let mut total_chunks = 0;
    for (path, content) in &documents {
        let is_code = is_code_file(path);

        let chunks = if args.use_ast_chunking && is_code {
            let code_chunks = chunking::ast::chunk_code(content, path, args.ast_chunk_size);
            code_chunks.into_iter().map(|c| c.text).collect::<Vec<_>>()
        } else if is_code {
            chunking::chunk_text(content, args.code_chunk_size, args.code_chunk_overlap)
        } else {
            chunking::chunk_text(content, args.doc_chunk_size, args.doc_chunk_overlap)
        };

        for (i, chunk) in chunks.iter().enumerate() {
            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), serde_json::json!(path));
            metadata.insert("chunk_index".to_string(), serde_json::json!(i));
            builder.add_text(chunk, metadata);
            total_chunks += 1;
        }
    }

    println!("Created {} chunks", total_chunks);
    println!("Building index '{}' with hnsw backend...", args.index_name);

    // Create embedding provider
    let provider = create_embedding_provider(
        &mode,
        &args.embedding_model,
        args.embedding_host.as_deref(),
        args.embedding_api_base.as_deref(),
        args.embedding_api_key.as_deref(),
    )?;

    std::fs::create_dir_all(&index_dir)?;
    builder.build_index(&index_path, provider.as_ref())?;

    println!("Index built at {}", index_path.display());
    Ok(())
}

fn is_code_file(path: &str) -> bool {
    let code_extensions = [
        "rs", "py", "js", "jsx", "ts", "tsx", "java", "go", "c", "cpp", "cc", "cxx", "h", "hpp",
        "rb", "sh", "bash", "cs", "swift", "kt", "scala", "r", "lua", "php", "pl", "ex", "exs",
        "zig", "nim", "v", "d",
    ];
    Path::new(path)
        .extension()
        .map(|e| code_extensions.contains(&e.to_string_lossy().to_lowercase().as_str()))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn cmd_search(
    index_name: &str,
    query: &str,
    top_k: usize,
    complexity: usize,
    beam_width: usize,
    prune_ratio: f64,
    show_metadata: bool,
) -> Result<()> {
    use leann_core::searcher::{LeannSearcher, SearchConfig};

    let index_path = resolve_index_for_search(index_name)?;
    let searcher = LeannSearcher::open(&index_path)?;
    let results = searcher.search_with_params(
        query,
        top_k,
        &SearchConfig {
            complexity,
            beam_width,
            prune_ratio,
            ..Default::default()
        },
    )?;

    if results.is_empty() {
        println!("No results found.");
    } else {
        for (i, result) in results.iter().enumerate() {
            println!("\n[{}] Score: {:.4}", i + 1, result.score);
            println!("    ID: {}", result.id);
            if show_metadata
                && let Some(source) = result.metadata.get("source").and_then(|v| v.as_str())
            {
                println!("    Source: {}", source);
            }
            let preview = if result.text.len() > 200 {
                format!("{}...", &result.text[..200])
            } else {
                result.text.clone()
            };
            println!("    {}", preview);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Ask
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmd_ask(
    index_name: &str,
    question: &str,
    top_k: usize,
    llm_type: &str,
    model: &str,
    host: Option<&str>,
    api_base: Option<&str>,
    api_key: Option<&str>,
) -> Result<()> {
    use leann_core::chat::{LeannChat, LlmConfig};
    use leann_core::searcher::LeannSearcher;

    let index_path = resolve_index_for_search(index_name)?;
    let searcher = LeannSearcher::open(&index_path)?;

    let config = LlmConfig {
        llm_type: llm_type.to_string(),
        model: Some(model.to_string()),
        api_key: api_key.map(String::from),
        base_url: api_base.map(String::from),
        host: host.map(String::from),
    };

    let chat = LeannChat::new(searcher, Some(&config))?;
    let answer = chat.ask(question, top_k)?;
    println!("\n{}", answer);

    Ok(())
}

fn cmd_interactive_ask(
    index_name: &str,
    top_k: usize,
    llm_type: &str,
    model: &str,
    host: Option<&str>,
    api_base: Option<&str>,
    api_key: Option<&str>,
) -> Result<()> {
    use leann_core::chat::{LeannChat, LlmConfig};
    use leann_core::searcher::LeannSearcher;
    use std::io::{self, BufRead, Write};

    let index_path = resolve_index_for_search(index_name)?;
    let searcher = LeannSearcher::open(&index_path)?;

    let config = LlmConfig {
        llm_type: llm_type.to_string(),
        model: Some(model.to_string()),
        api_key: api_key.map(String::from),
        base_url: api_base.map(String::from),
        host: host.map(String::from),
    };

    let chat = LeannChat::new(searcher, Some(&config))?;

    println!("Interactive mode. Type 'quit' or 'exit' to stop.\n");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        io::stdin().lock().read_line(&mut line)?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" {
            println!("Goodbye!");
            break;
        }

        match chat.ask(line, top_k) {
            Ok(answer) => println!("\n{}\n", answer),
            Err(e) => eprintln!("Error: {}\n", e),
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// React
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmd_react(
    index_name: &str,
    query: &str,
    top_k: usize,
    max_iterations: usize,
    llm_type: &str,
    model: &str,
    host: Option<&str>,
    api_base: Option<&str>,
    api_key: Option<&str>,
) -> Result<()> {
    use leann_core::chat::LlmConfig;
    use leann_core::react_agent::ReActAgent;
    use leann_core::searcher::LeannSearcher;

    let index_path = resolve_index_for_search(index_name)?;
    let searcher = LeannSearcher::open(&index_path)?;

    let config = LlmConfig {
        llm_type: llm_type.to_string(),
        model: Some(model.to_string()),
        api_key: api_key.map(String::from),
        base_url: api_base.map(String::from),
        host: host.map(String::from),
    };

    let mut agent = ReActAgent::new(searcher, Some(&config), max_iterations)?;
    let answer = agent.run(query, top_k)?;
    println!("\n{}", answer);

    Ok(())
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

fn cmd_list() -> Result<()> {
    use leann_core::index::IndexMeta;

    let current_path = std::env::current_dir()?;
    println!("LEANN Indexes");
    println!("{}", "=".repeat(50));
    println!("\nCurrent Project");
    println!("   {}", current_path.display());
    println!("   {}", "-".repeat(45));

    let mut total = 0;

    // CLI-format indexes: .leann/indexes/<name>/
    let cli_indexes_dir = current_path.join(".leann").join("indexes");
    if cli_indexes_dir.exists()
        && let Ok(entries) = std::fs::read_dir(&cli_indexes_dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let meta_file = path.join("documents.leann.meta.json");
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                let status = if meta_file.exists() {
                    "OK"
                } else {
                    "incomplete"
                };

                let mut size_mb = 0.0f64;
                if let Ok(files) = std::fs::read_dir(&path) {
                    for f in files.flatten() {
                        if let Ok(meta) = f.metadata() {
                            size_mb += meta.len() as f64 / (1024.0 * 1024.0);
                        }
                    }
                }

                total += 1;
                print!("   {}. {} [{}]", total, name, status);
                if size_mb > 0.01 {
                    print!(" ({:.1} MB)", size_mb);
                }
                println!();
            }
        }
    }

    // App-format indexes: *.leann.meta.json files in the project
    for meta_file in glob_meta_files(&current_path) {
        // Skip CLI indexes
        if meta_file.starts_with(&cli_indexes_dir) {
            continue;
        }
        if let Ok(meta) = IndexMeta::load(&meta_file) {
            let display_name = meta_file
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            total += 1;
            println!(
                "   {}. {} - {} (dim={}, passages={})",
                total,
                display_name,
                meta.embedding_model,
                meta.dimensions,
                meta.total_passages.unwrap_or(0),
            );
        }
    }

    println!("\n{}", "=".repeat(50));
    if total == 0 {
        println!("Get started:");
        println!("   leann build my-docs --docs ./documents");
    } else {
        println!("Total: {} index(es)", total);
    }

    Ok(())
}

fn glob_meta_files(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.starts_with('.') && name != "node_modules" && name != "target" {
                    results.extend(glob_meta_files(&path));
                }
            } else {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.ends_with(".leann.meta.json") {
                    results.push(path);
                }
            }
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

fn cmd_remove(index_name: &str, force: bool) -> Result<()> {
    let index_dir = indexes_dir().join(index_name);

    if !index_dir.exists() {
        println!("Index '{}' not found.", index_name);
        return Ok(());
    }

    if !force {
        use std::io::{self, Write};
        print!("Remove index '{}'? [y/N] ", index_name);
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let mut removed = 0;
    if let Ok(entries) = std::fs::read_dir(&index_dir) {
        for entry in entries.flatten() {
            if entry.path().is_file() {
                std::fs::remove_file(entry.path())?;
                removed += 1;
            }
        }
    }
    std::fs::remove_dir_all(&index_dir)?;
    println!("Removed index '{}' ({} files).", index_name, removed);

    Ok(())
}

// ---------------------------------------------------------------------------
// Watch
// ---------------------------------------------------------------------------

fn cmd_watch(index_name: &str) -> Result<()> {
    let index_dir = indexes_dir().join(index_name);
    if !index_dir.exists() {
        anyhow::bail!("Index '{}' not found.", index_name);
    }

    let sync_config_path = index_dir.join("sync_roots.json");
    if !sync_config_path.exists() {
        println!(
            "Sync config not found for index '{}'. Rebuild the index to enable watch.",
            index_name
        );
        return Ok(());
    }

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&sync_config_path)?)?;
    let roots = config["roots"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if roots.is_empty() {
        println!("No sync roots found for index '{}'.", index_name);
        return Ok(());
    }

    use leann_core::sync::FileSynchronizer;

    let mut all_added = Vec::new();
    let mut all_removed = Vec::new();
    let mut all_modified = Vec::new();

    let include_extensions = config["include_extensions"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let ignore_patterns = config["ignore_patterns"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    for root in &roots {
        let root_path = Path::new(root);
        if !root_path.is_dir() {
            eprintln!("Warning: sync root not found: {}", root);
            continue;
        }
        match FileSynchronizer::new(
            root_path,
            ignore_patterns.clone(),
            include_extensions.clone(),
        ) {
            Ok(mut sync) => match sync.check_for_changes() {
                Ok((added, removed, modified)) => {
                    all_added.extend(added);
                    all_removed.extend(removed);
                    all_modified.extend(modified);
                }
                Err(e) => eprintln!("Warning: failed to check {}: {}", root, e),
            },
            Err(e) => eprintln!("Warning: failed to initialize sync for {}: {}", root, e),
        }
    }

    if all_added.is_empty() && all_removed.is_empty() && all_modified.is_empty() {
        println!("No changes detected.");
    } else {
        println!("\n=== Changes since last checkpoint ===");
        if !all_added.is_empty() {
            println!("\nadded ({}):", all_added.len());
            for f in &all_added {
                println!("  - {}", f);
            }
        }
        if !all_modified.is_empty() {
            println!("\nmodified ({}):", all_modified.len());
            for f in &all_modified {
                println!("  - {}", f);
            }
        }
        if !all_removed.is_empty() {
            println!("\nremoved ({}):", all_removed.len());
            for f in &all_removed {
                println!("  - {}", f);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Serve
// ---------------------------------------------------------------------------

fn cmd_serve(host: &str, port: u16) -> Result<()> {
    println!("Starting HTTP server on {}:{}...", host, port);
    println!("Endpoints:");
    println!("  GET  /health");
    println!("  GET  /indexes");
    println!("  POST /indexes/{{name}}/search");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_server(host.to_string(), port))?;

    Ok(())
}

async fn run_server(host: String, port: u16) -> Result<()> {
    use axum::{
        Router,
        extract::Path as AxumPath,
        http::StatusCode,
        response::Json,
        routing::{get, post},
    };
    use serde::{Deserialize, Serialize};

    #[derive(Serialize)]
    struct HealthResponse {
        status: String,
        version: String,
    }

    #[derive(Serialize)]
    struct IndexListResponse {
        indexes: Vec<String>,
    }

    #[derive(Deserialize)]
    struct SearchRequest {
        query: String,
        #[serde(default = "default_top_k")]
        top_k: usize,
    }
    fn default_top_k() -> usize {
        5
    }

    #[derive(Serialize)]
    struct SearchResultResponse {
        id: String,
        score: f64,
        text: String,
        metadata: HashMap<String, serde_json::Value>,
    }

    #[derive(Serialize)]
    struct SearchResponse {
        results: Vec<SearchResultResponse>,
    }

    async fn health() -> Json<HealthResponse> {
        Json(HealthResponse {
            status: "ok".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    async fn list_indexes() -> Json<IndexListResponse> {
        let mut indexes = Vec::new();
        let idx_dir = indexes_dir();
        if let Ok(entries) = std::fs::read_dir(&idx_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let meta = entry.path().join("documents.leann.meta.json");
                    if meta.exists() {
                        indexes.push(entry.file_name().to_string_lossy().to_string());
                    }
                }
            }
        }
        Json(IndexListResponse { indexes })
    }

    async fn search_index(
        AxumPath(name): AxumPath<String>,
        Json(request): Json<SearchRequest>,
    ) -> Result<Json<SearchResponse>, StatusCode> {
        use leann_core::searcher::LeannSearcher;

        let index_path = get_index_path(&name);
        let searcher = LeannSearcher::open(&index_path).map_err(|_| StatusCode::NOT_FOUND)?;

        let results = searcher
            .search(&request.query, request.top_k)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        Ok(Json(SearchResponse {
            results: results
                .into_iter()
                .map(|r| SearchResultResponse {
                    id: r.id,
                    score: r.score,
                    text: r.text,
                    metadata: r.metadata,
                })
                .collect(),
        }))
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/indexes", get(list_indexes))
        .route("/indexes/{name}/search", post(search_index));

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port))
        .await
        .map_err(|e| anyhow::anyhow!("binding {}:{}: {}", host, port, e))?;
    tracing::info!("LEANN HTTP server listening on {}:{}", host, port);

    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve an index name to its path for search/ask/react.
/// Tries CLI-format first (.leann/indexes/<name>/documents.leann),
/// then falls back to treating the name as a direct path.
fn resolve_index_for_search(index_name: &str) -> Result<PathBuf> {
    // 1) CLI-format
    let cli_path = get_index_path(index_name);
    let cli_meta = indexes_dir()
        .join(index_name)
        .join("documents.leann.meta.json");
    if cli_meta.exists() {
        return Ok(cli_path);
    }

    // 2) Direct path (for app-format or absolute paths)
    let direct = PathBuf::from(index_name);
    let direct_meta = direct.parent().unwrap_or(Path::new(".")).join(format!(
        "{}.meta.json",
        direct.file_name().unwrap_or_default().to_string_lossy()
    ));
    if direct_meta.exists() {
        return Ok(direct);
    }

    // 3) Treat as name in current directory
    let local_meta = PathBuf::from(format!("{}.meta.json", index_name));
    if local_meta.exists() {
        return Ok(PathBuf::from(index_name));
    }

    anyhow::bail!(
        "Index '{}' not found. Looked in:\n  - {}\n  - {}\nRun `leann list` to see available indexes.",
        index_name,
        cli_meta.display(),
        direct_meta.display(),
    )
}

/// Load documents from a directory, optionally filtering by extensions.
fn load_documents(
    dir: &Path,
    allowed_extensions: Option<&[String]>,
    include_hidden: bool,
) -> Result<Vec<(String, String)>> {
    let default_extensions = [
        "txt", "md", "rst", "rs", "py", "js", "jsx", "ts", "tsx", "java", "go", "c", "cpp", "cc",
        "cxx", "h", "hpp", "rb", "sh", "bash", "toml", "yaml", "yml", "json", "xml", "html", "htm",
        "css", "sql", "r", "lua", "php", "swift", "kt", "scala", "ex", "exs", "pdf",
    ];

    let mut documents = Vec::new();

    walk_dir(dir, include_hidden, &mut |path: &Path| {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let allowed = if let Some(exts) = allowed_extensions {
            exts.iter().any(|e| e == &ext)
        } else {
            default_extensions.contains(&ext.as_str())
        };

        if allowed {
            match leann_core::document_loaders::extract_text(path) {
                Ok(Some(content)) => {
                    documents.push((path.to_string_lossy().to_string(), content));
                }
                Ok(None) => {} // empty or unsupported
                Err(e) => {
                    tracing::debug!("Skipping {}: {}", path.display(), e);
                }
            }
        }
    });

    Ok(documents)
}

fn walk_dir(dir: &Path, include_hidden: bool, callback: &mut dyn FnMut(&Path)) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !include_hidden && name.starts_with('.') {
                    continue;
                }
                if name == "node_modules"
                    || name == "target"
                    || name == "__pycache__"
                    || name == "venv"
                    || name == ".venv"
                {
                    continue;
                }
                walk_dir(&path, include_hidden, callback);
            } else {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !include_hidden && name.starts_with('.') {
                    continue;
                }
                callback(&path);
            }
        }
    }
}

/// Create an embedding provider based on mode and options.
fn create_embedding_provider(
    mode: &leann_core::embedding::EmbeddingMode,
    model: &str,
    host: Option<&str>,
    api_base: Option<&str>,
    api_key: Option<&str>,
) -> Result<Box<dyn leann_core::embedding::EmbeddingProvider>> {
    use leann_core::embedding::EmbeddingMode;

    match mode {
        EmbeddingMode::OpenAI => {
            let provider = leann_core::embedding::openai::OpenAiEmbedding::new(
                model, api_key, api_base, None,
            )?;
            Ok(Box::new(provider))
        }
        EmbeddingMode::Ollama => {
            let provider = leann_core::embedding::ollama::OllamaEmbedding::new(model, host);
            Ok(Box::new(provider))
        }
        EmbeddingMode::Gemini => {
            let provider = leann_core::embedding::gemini::GeminiEmbedding::new(model, api_key)?;
            Ok(Box::new(provider))
        }
        _ => {
            // sentence-transformers / mlx: try OpenAI, fall back to Ollama
            if let Ok(provider) = leann_core::embedding::openai::OpenAiEmbedding::new(
                "text-embedding-3-small",
                None,
                None,
                None,
            ) {
                Ok(Box::new(provider))
            } else {
                let provider =
                    leann_core::embedding::ollama::OllamaEmbedding::new("nomic-embed-text", None);
                Ok(Box::new(provider))
            }
        }
    }
}

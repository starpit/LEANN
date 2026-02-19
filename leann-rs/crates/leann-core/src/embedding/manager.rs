use anyhow::Result;
use std::net::TcpListener;
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// Manages embedding server subprocess lifecycle.
pub struct EmbeddingServerManager {
    /// The running server process, if any.
    process: Option<Child>,
    /// Port the server is listening on.
    port: Option<u16>,
    /// Configuration signature for reuse detection.
    config_signature: Option<ConfigSignature>,
}

#[derive(Debug, Clone, PartialEq)]
struct ConfigSignature {
    model_name: String,
    embedding_mode: String,
    passages_file: String,
}

impl EmbeddingServerManager {
    pub fn new() -> Self {
        Self {
            process: None,
            port: None,
            config_signature: None,
        }
    }

    /// Start the embedding server, reusing an existing one if the config matches.
    pub fn start_server(
        &mut self,
        port: u16,
        model_name: &str,
        embedding_mode: &str,
        passages_file: &str,
        leann_binary: &str,
    ) -> Result<u16> {
        let new_sig = ConfigSignature {
            model_name: model_name.to_string(),
            embedding_mode: embedding_mode.to_string(),
            passages_file: passages_file.to_string(),
        };

        // Reuse existing server if config matches and process is alive
        if let (Some(ref mut process), Some(port), Some(ref sig)) =
            (&mut self.process, self.port, &self.config_signature)
        {
            if sig == &new_sig {
                match process.try_wait() {
                    Ok(None) => {
                        info!("Reusing existing embedding server on port {}", port);
                        return Ok(port);
                    }
                    _ => {
                        // Process died, need to restart
                    }
                }
            } else {
                // Config changed, stop existing server
                self.stop_server();
            }
        }

        // Find available port
        let actual_port = find_available_port(port)?;

        // Build command
        let mut cmd = Command::new(leann_binary);
        cmd.arg("serve-embeddings")
            .arg("--port")
            .arg(actual_port.to_string())
            .arg("--model-name")
            .arg(model_name);

        if !passages_file.is_empty() {
            cmd.arg("--passages-file").arg(passages_file);
        }
        if embedding_mode != "sentence-transformers" {
            cmd.arg("--embedding-mode").arg(embedding_mode);
        }

        info!("Starting embedding server on port {}", actual_port);
        let child = cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to start embedding server: {}", e))?;

        self.process = Some(child);
        self.port = Some(actual_port);
        self.config_signature = Some(new_sig);

        // Wait for server to be ready
        self.wait_for_ready(actual_port, Duration::from_secs(120))?;

        Ok(actual_port)
    }

    /// Stop the embedding server if running.
    pub fn stop_server(&mut self) {
        if let Some(mut process) = self.process.take() {
            info!("Terminating embedding server process");

            // Try graceful shutdown first
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                unsafe {
                    libc::kill(process.id() as i32, libc::SIGTERM);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = process.kill();
            }

            // Wait with timeout
            match wait_with_timeout(&mut process, Duration::from_secs(5)) {
                Ok(_) => info!("Server process terminated gracefully"),
                Err(_) => {
                    warn!("Server did not terminate in time, force killing");
                    let _ = process.kill();
                    let _ = process.wait();
                }
            }
        }

        self.port = None;
        self.config_signature = None;
    }

    /// Get the port the server is listening on, if running.
    pub fn port(&self) -> Option<u16> {
        self.port
    }

    /// Check if the server process is still alive.
    pub fn is_alive(&mut self) -> bool {
        if let Some(ref mut process) = self.process {
            matches!(process.try_wait(), Ok(None))
        } else {
            false
        }
    }

    fn wait_for_ready(&self, port: u16, timeout: Duration) -> Result<()> {
        let start = Instant::now();
        let check_interval = Duration::from_millis(500);

        while start.elapsed() < timeout {
            if is_port_in_use(port) {
                info!("Embedding server ready on port {}", port);
                return Ok(());
            }

            // Check if process died
            if let Some(ref process) = self.process {
                // We can't call try_wait on an immutable reference, just sleep and check port
            }

            std::thread::sleep(check_interval);
        }

        anyhow::bail!(
            "Embedding server failed to start within {} seconds",
            timeout.as_secs()
        )
    }
}

impl Default for EmbeddingServerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EmbeddingServerManager {
    fn drop(&mut self) {
        self.stop_server();
    }
}

fn find_available_port(start: u16) -> Result<u16> {
    for port in start..start + 100 {
        if TcpListener::bind(("localhost", port)).is_ok() {
            return Ok(port);
        }
    }
    anyhow::bail!(
        "No available ports found in range {}-{}",
        start,
        start + 100
    )
}

fn is_port_in_use(port: u16) -> bool {
    std::net::TcpStream::connect(("localhost", port)).is_ok()
}

fn wait_with_timeout(child: &mut Child, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(_) => return Ok(()),
            None => {
                if start.elapsed() > timeout {
                    anyhow::bail!("Timeout waiting for process");
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

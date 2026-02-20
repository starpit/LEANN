use anyhow::Result;
use tracing::info;

use crate::chat::{LlmConfig, LlmParams, LlmProvider, get_llm};
use crate::search_result::SearchResult;
use crate::searcher::LeannSearcher;

/// Simple ReAct agent for multi-turn retrieval.
///
/// Follows the pattern:
/// 1. Thought: LLM reasons about what information is needed
/// 2. Action: Performs a search query
/// 3. Observation: Gets search results
/// 4. Repeat until LLM decides it has enough information to answer
pub struct ReActAgent {
    searcher: LeannSearcher,
    llm: Box<dyn LlmProvider>,
    max_iterations: usize,
    search_history: Vec<SearchHistoryEntry>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SearchHistoryEntry {
    iteration: usize,
    thought: String,
    action: String,
    results_count: usize,
}

impl ReActAgent {
    pub fn new(
        searcher: LeannSearcher,
        llm_config: Option<&LlmConfig>,
        max_iterations: usize,
    ) -> Result<Self> {
        let config = llm_config.cloned().unwrap_or_default();
        let llm = get_llm(&config)?;

        Ok(Self {
            searcher,
            llm,
            max_iterations,
            search_history: Vec::new(),
        })
    }

    /// Run the ReAct agent to answer a question.
    pub fn run(&mut self, question: &str, top_k: usize) -> Result<String> {
        info!("Starting ReAct agent for question: {}", question);
        self.search_history.clear();
        let mut previous_observations: Vec<String> = Vec::new();
        let mut all_context: Vec<String> = Vec::new();

        for iteration in 1..=self.max_iterations {
            info!("--- Iteration {}/{} ---", iteration, self.max_iterations);

            let prompt = self.create_react_prompt(question, iteration, &previous_observations);
            let response = self.llm.ask(&prompt, &LlmParams::default())?;

            let (thought, action) = parse_llm_response(&response);
            info!("Thought: {}", thought);

            if action.is_none() {
                // Extract final answer
                let final_answer = if response.contains("Final Answer:") {
                    response.split("Final Answer:").nth(1).unwrap_or("").trim()
                } else {
                    response.trim()
                };
                info!("Final answer: {}", final_answer);
                return Ok(final_answer.to_string());
            }

            let search_query = action.unwrap();
            info!("Action: search(\"{}\")", search_query);

            let results = self.searcher.search(&search_query, top_k)?;
            let observation = format_search_results(&results);
            previous_observations.push(observation.clone());
            all_context.push(format!("Search: {}\n{}", search_query, observation));

            self.search_history.push(SearchHistoryEntry {
                iteration,
                thought,
                action: search_query,
                results_count: results.len(),
            });

            // Early stop if no results after iteration 2
            if results.is_empty() && iteration >= 2 {
                let final_prompt = format!(
                    "Based on the previous searches, provide your best answer.\n\n\
                     Question: {}\n\n\
                     Previous searches:\n{}\n\n\
                     Provide your final answer.",
                    question,
                    all_context.join("\n")
                );
                return self.llm.ask(&final_prompt, &LlmParams::default());
            }
        }

        // Max iterations reached
        let final_prompt = format!(
            "Based on all searches, provide your final answer.\n\n\
             Question: {}\n\n\
             All results:\n{}\n\n\
             Provide your final answer now.",
            question,
            all_context.join("\n")
        );
        self.llm.ask(&final_prompt, &LlmParams::default())
    }

    fn create_react_prompt(
        &self,
        question: &str,
        iteration: usize,
        previous_observations: &[String],
    ) -> String {
        let mut prompt = format!(
            "You are a helpful assistant that answers questions by searching through a knowledge base.\n\n\
             Question: {}\n\n\
             You can search the knowledge base by using the action: search(\"query\")\n\n\
             Previous observations:\n",
            question
        );

        if previous_observations.is_empty() {
            prompt.push_str("None yet.\n");
        } else {
            for (i, obs) in previous_observations.iter().enumerate() {
                prompt.push_str(&format!("\nObservation {}:\n{}\n", i + 1, obs));
            }
        }

        prompt.push_str(&format!(
            "\nCurrent iteration: {}/{}\n\n\
             Think step by step:\n\
             1. If you need more information, use search(\"your search query\")\n\
             2. If you have enough information, provide your final answer\n\n\
             Format your response as:\n\
             Thought: [your reasoning]\n\
             Action: search(\"query\") OR Final Answer: [your answer]\n",
            iteration, self.max_iterations
        ));

        prompt
    }
}

fn format_search_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let text_preview = if r.text.len() > 500 {
                format!("{}...", &r.text[..500])
            } else {
                r.text.clone()
            };
            let mut entry = format!(
                "[Result {}] (Score: {:.3})\n{}",
                i + 1,
                r.score,
                text_preview
            );
            if let Some(source) = r.metadata.get("source").and_then(|v| v.as_str()) {
                entry.push_str(&format!("\nSource: {}", source));
            }
            entry
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn parse_llm_response(response: &str) -> (String, Option<String>) {
    let mut thought = String::new();
    let mut action = None;

    // Extract thought
    if let Some(idx) = response.find("Thought:") {
        let after = &response[idx + 8..];
        if let Some(action_idx) = after.find("Action:") {
            thought = after[..action_idx].trim().to_string();
        } else if let Some(final_idx) = after.find("Final Answer:") {
            thought = after[..final_idx].trim().to_string();
        } else {
            thought = after.trim().to_string();
        }
    }

    // Check for final answer
    if response.contains("Final Answer:") {
        return (thought, None);
    }

    // Extract search action
    if let Some(idx) = response.find("Action:") {
        let after = &response[idx + 7..];
        if let Some(start) = after.find("search(\"") {
            let query_start = start + 8;
            if let Some(end) = after[query_start..].find("\")") {
                action = Some(after[query_start..query_start + end].to_string());
            }
        } else if let Some(start) = after.find("search(") {
            let query_start = start + 7;
            if let Some(end) = after[query_start..].find(')') {
                let q = after[query_start..query_start + end]
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();
                action = Some(q);
            }
        }
    }

    (thought, action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llm_response_with_action() {
        let response =
            "Thought: I need to find information about HNSW.\nAction: search(\"HNSW algorithm\")";
        let (thought, action) = parse_llm_response(response);
        assert_eq!(thought, "I need to find information about HNSW.");
        assert_eq!(action, Some("HNSW algorithm".to_string()));
    }

    #[test]
    fn test_parse_llm_response_final_answer() {
        let response = "Thought: I have enough info.\nFinal Answer: HNSW is an approximate nearest neighbor algorithm.";
        let (thought, action) = parse_llm_response(response);
        assert_eq!(thought, "I have enough info.");
        assert!(action.is_none());
    }
}

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use serde::Deserialize;

use super::{CostSource, EstimateStatus, UnifiedMessage};
use crate::{pricing::PricingService, TokenBreakdown};

#[derive(Debug, Deserialize)]
struct UsageRow {
    t: String,
    #[serde(default)]
    agent: String,
    model: String,
    #[serde(default)]
    served: String,
    #[serde(default)]
    provider: String,
    #[serde(default)]
    session: String,
    #[serde(default)]
    native_session: String,
    #[serde(rename = "in")]
    input: i64,
    out: i64,
    #[serde(default)]
    cache_read: i64,
    #[serde(default)]
    cache_write: i64,
    #[serde(default)]
    cache_write_1h: i64,
    #[serde(default)]
    reasoning: i64,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    ms: Option<i64>,
}

struct Observation {
    message: UnifiedMessage,
    response_id: Option<String>,
    local_client: String,
    matched: bool,
}

#[derive(Default)]
pub(crate) struct Reconciliation {
    observations: Vec<Observation>,
    by_response: HashMap<String, Vec<usize>>,
    pub codex_response_ids: HashMap<String, String>,
    emitted: HashSet<String>,
    unknown_local_clients: HashSet<String>,
    ambiguous_responses: HashSet<String>,
    selected: HashSet<String>,
    all_clients: bool,
    invalid_rows: usize,
    excluded_rows: usize,
    excluded_tokens: i64,
}

fn canonical_response_id(id: &str) -> &str {
    // 此 alias 只代表 Responses 經 Messages 相容層後的同一回應。
    if id.starts_with("msg_resp_") {
        &id[4..]
    } else {
        id
    }
}

fn local_response_id<'a>(
    message: &'a UnifiedMessage,
    codex: &'a HashMap<String, String>,
) -> Option<&'a str> {
    let key = message.dedup_key.as_deref()?;
    if let Some(id) = codex.get(key) {
        return Some(id);
    }
    if message.client == "claude" && !key.contains(":advisor:") {
        return key
            .strip_prefix("message:")
            .or_else(|| key.split_once(':').map(|(id, _)| id));
    }
    if let Some((_, id)) = key.split_once(":response:") {
        if matches!(
            message.client.as_str(),
            "pi" | "omp" | "senpi" | "kimchi" | "prime-agent"
        ) {
            return Some(id);
        }
    }
    if message.client == "codex" && key.starts_with("codex:usage-record:") {
        return key
            .rsplit_once(':')
            .map(|(_, id)| id)
            .filter(|id| id.starts_with("resp_"));
    }
    None
}

impl Reconciliation {
    pub(crate) fn read(paths: &[PathBuf], clients: &[String]) -> Self {
        let mut result = Self {
            selected: clients.iter().cloned().collect(),
            all_clients: clients.is_empty(),
            ..Self::default()
        };
        let mut seen_rows = HashSet::new();
        for path in paths {
            let file = match File::open(path) {
                Ok(file) => file,
                Err(_) => {
                    result.invalid_rows += 1;
                    continue;
                }
            };
            for line in BufReader::new(file).lines() {
                let Ok(line) = line else {
                    result.invalid_rows += 1;
                    break;
                };
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                    result.invalid_rows += 1;
                    continue;
                };
                let Ok(row) = serde_json::from_value::<UsageRow>(value.clone()) else {
                    result.invalid_rows += 1;
                    continue;
                };
                let Ok(timestamp) = chrono::DateTime::parse_from_rfc3339(&row.t) else {
                    result.invalid_rows += 1;
                    continue;
                };
                if [
                    row.input,
                    row.out,
                    row.cache_read,
                    row.cache_write,
                    row.cache_write_1h,
                    row.reasoning,
                ]
                .iter()
                .any(|n| *n < 0)
                {
                    result.invalid_rows += 1;
                    continue;
                }
                let id = row
                    .response_id
                    .filter(|id| !id.trim().is_empty())
                    .map(|id| canonical_response_id(&id).to_string());
                // 同 response id 並不足以排除 retry；只有整筆 metadata 相同才是副本。
                if id.is_some() && !seen_rows.insert(value.to_string()) {
                    continue;
                }
                let reasoning = row.reasoning.min(row.out);
                let model = if row.served.trim().is_empty() {
                    row.model
                } else {
                    row.served
                };
                let session = if !row.native_session.is_empty() {
                    row.native_session
                } else if !row.session.is_empty() {
                    row.session
                } else {
                    "unknown".to_string()
                };
                let provider = if row.provider.is_empty() {
                    "unknown".to_string()
                } else {
                    row.provider
                };
                let mut message = UnifiedMessage::new(
                    "magpie",
                    model,
                    provider,
                    session,
                    timestamp.timestamp_millis(),
                    TokenBreakdown {
                        input: row.input,
                        output: row.out - reasoning,
                        cache_read: row.cache_read,
                        cache_write: row.cache_write,
                        cache_write_1h: row.cache_write_1h.min(row.cache_write),
                        reasoning,
                    },
                    0.0,
                );
                message.duration_ms = row.ms;
                message.agent = (!row.agent.is_empty()).then(|| row.agent.clone());
                message.dedup_key = id.clone().map(|id| format!("magpie:response:{id}"));
                let local_client = match row.agent.as_str() {
                    "claude-code" | "claudecode" => "claude".to_string(),
                    _ => row.agent,
                };
                if let Some(id) = &id {
                    result
                        .by_response
                        .entry(id.clone())
                        .or_default()
                        .push(result.observations.len());
                }
                result.observations.push(Observation {
                    message,
                    response_id: id,
                    local_client,
                    matched: false,
                });
            }
        }
        result
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub(crate) fn reconcile(
        &mut self,
        message: &mut UnifiedMessage,
        pricing: Option<&PricingService>,
    ) -> bool {
        if message.client == "magpie" {
            return true;
        }
        let Some(id) =
            local_response_id(message, &self.codex_response_ids).map(canonical_response_id)
        else {
            self.unknown_local_clients.insert(message.client.clone());
            return true;
        };
        let Some(indices) = self.by_response.get(id) else {
            return true;
        };
        if indices.len() != 1 {
            self.ambiguous_responses.insert(id.to_string());
            return true;
        }
        let observation = &mut self.observations[indices[0]];
        observation.matched = true;
        if !self.emitted.insert(id.to_string()) {
            return false;
        }
        message.tokens = observation.message.tokens.clone();
        message.model_id = observation.message.model_id.clone();
        if observation.message.provider_id != "unknown" {
            message.provider_id = observation.message.provider_id.clone();
        }
        message.pricing_model_id = None;
        message.model_attribution_conflicted = false;
        message.cost = 0.0;
        message.cost_source = CostSource::Unknown;
        message.estimate_status = EstimateStatus::Unpriced;
        crate::apply_pricing_if_available(message, pricing);
        true
    }

    pub(crate) fn has_match(&self, message: &UnifiedMessage) -> bool {
        local_response_id(message, &self.codex_response_ids)
            .is_some_and(|id| self.emitted.contains(canonical_response_id(id)))
    }

    pub(crate) fn remaining(&mut self, pricing: Option<&PricingService>) -> Vec<UnifiedMessage> {
        let mut messages = Vec::new();
        for observation in &self.observations {
            if observation.matched {
                continue;
            }
            let selected_agent = self.all_clients
                || self.selected.contains(&observation.local_client)
                || (observation.local_client.is_empty()
                    && self.selected.iter().any(|client| client != "magpie"));
            let uncertain = selected_agent
                && (observation.local_client.is_empty()
                    || self
                        .unknown_local_clients
                        .contains(&observation.local_client)
                    || !matches!(
                        observation.local_client.as_str(),
                        "claude" | "codex" | "pi" | "omp" | "senpi" | "kimchi" | "prime-agent"
                    ));
            let ambiguous = observation
                .response_id
                .as_ref()
                .is_some_and(|id| self.ambiguous_responses.contains(id));
            let local_sources_selected =
                self.all_clients || self.selected.iter().any(|client| client != "magpie");
            if local_sources_selected
                && (observation.response_id.is_none() || uncertain || ambiguous)
            {
                self.excluded_rows += 1;
                self.excluded_tokens = self
                    .excluded_tokens
                    .saturating_add(observation.message.tokens.total());
                continue;
            }
            if self.all_clients || self.selected.contains("magpie") {
                let mut message = observation.message.clone();
                crate::apply_pricing_if_available(&mut message, pricing);
                messages.push(message);
            }
        }
        messages
    }

    pub(crate) fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.invalid_rows > 0 {
            warnings.push(format!("Warning: Magpie skipped {} unreadable or malformed usage records; usage is incomplete.", self.invalid_rows));
        }
        if self.excluded_rows > 0 {
            warnings.push(format!("Magpie: excluded {} rows / {} tokens (overlap unverified). Totals are conservative. Exact IDs: Claude Code, Codex token_usage_record, Pi-family responseId; other source identities unverified.", self.excluded_rows, self.excluded_tokens));
        }
        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn ledger(rows: &str, clients: &[&str]) -> (tempfile::NamedTempFile, Reconciliation) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(rows.as_bytes()).unwrap();
        let state = Reconciliation::read(
            &[file.path().to_path_buf()],
            &clients.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        );
        (file, state)
    }

    const ROW: &str = r#"{"t":"2026-10-18T02:50:24Z","agent":"claude","model":"alias","served":"physical-model","in":67878,"out":298,"cache_read":65187,"response_id":"msg_demo"}"#;

    #[test]
    fn exact_match_overrides_usage_and_preserves_local_metadata() {
        let (_file, mut state) = ledger(ROW, &[]);
        let mut message = UnifiedMessage::new_with_dedup(
            "claude",
            "alias",
            "google",
            "session-demo",
            1234,
            TokenBreakdown {
                input: 122042,
                ..Default::default()
            },
            2.0,
            Some("msg_demo:req_demo".into()),
        );
        message.session_title = Some("Local title".into());
        assert!(state.reconcile(&mut message, None));
        assert_eq!(message.tokens.total(), 133363);
        assert_eq!(message.session_id, "session-demo");
        assert_eq!(message.timestamp, 1234);
        assert_eq!(message.session_title.as_deref(), Some("Local title"));
        assert_eq!(message.model_id, "physical-model");
        assert_eq!(message.cost_source, CostSource::Unknown);
        assert!(!state.reconcile(&mut message, None));
        assert!(state.remaining(None).is_empty());
    }

    #[test]
    fn source_filters_keep_gateway_only_and_exclude_uncertain_overlap() {
        let (_file, mut state) = ledger(ROW, &["magpie"]);
        assert_eq!(state.remaining(None)[0].tokens.total(), 133363);
        let (_file, mut state) = ledger(ROW, &[]);
        let mut unknown = UnifiedMessage::new("claude", "m", "p", "s", 1, Default::default(), 0.0);
        assert!(state.reconcile(&mut unknown, None));
        assert!(state.remaining(None).is_empty());
        assert!(state.warnings()[0].contains("133363 tokens"));
        let (_file, mut state) = ledger(ROW, &["codex", "magpie"]);
        assert_eq!(state.remaining(None).len(), 1);
    }

    #[test]
    fn gateway_metadata_keeps_native_session_and_provider() {
        let row = ROW.replace("\"agent\":\"claude\"", "\"agent\":\"claude\",\"provider\":\"vendor\",\"session\":\"route-session\",\"native_session\":\"native-session\"");
        let (_file, mut state) = ledger(&row, &["magpie"]);
        let messages = state.remaining(None);
        assert_eq!(messages[0].session_id, "native-session");
        assert_eq!(messages[0].provider_id, "vendor");
    }

    #[test]
    fn reasoning_is_output_subset_and_cache_is_independent() {
        let (_file, mut state) = ledger(
            r#"{"t":"2026-10-18T00:00:00Z","model":"m","in":10,"out":8,"reasoning":20,"cache_read":3,"cache_write":4,"cache_write_1h":9,"response_id":"resp_demo"}"#,
            &["magpie"],
        );
        let messages = state.remaining(None);
        assert_eq!(
            messages[0].tokens,
            TokenBreakdown {
                input: 10,
                output: 0,
                reasoning: 8,
                cache_read: 3,
                cache_write: 4,
                cache_write_1h: 4
            }
        );
        assert_eq!(messages[0].tokens.total(), 25);
    }

    #[test]
    fn repeated_metadata_is_duplicate_but_same_id_retry_is_not() {
        let rows = format!("{ROW}\n{ROW}\n{}", ROW.replace("02:50:24", "02:50:25"));
        let (_file, mut state) = ledger(&rows, &["magpie"]);
        assert_eq!(state.remaining(None).len(), 2);
    }

    #[test]
    fn gateway_only_keeps_same_response_attempts_with_different_usage() {
        let rows = format!(
            "{ROW}\n{}",
            ROW.replace("\"agent\":\"claude\"", "\"agent\":\"magpie\",\"attempt\":2")
                .replace("67878", "10")
        );
        let (_file, mut state) = ledger(&rows, &["magpie"]);
        state.ambiguous_responses.insert("msg_demo".into());
        let messages = state.remaining(None);
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages.iter().map(|m| m.tokens.total()).sum::<i64>(),
            198858
        );
        assert!(state.warnings().is_empty());

        let (_file, mut mixed) = ledger(&rows, &["claude", "magpie"]);
        let mut local = UnifiedMessage::new_with_dedup(
            "claude",
            "m",
            "p",
            "s",
            1,
            Default::default(),
            0.0,
            Some("message:msg_demo".into()),
        );
        assert!(mixed.reconcile(&mut local, None));
        assert!(mixed.remaining(None).is_empty());
        assert!(mixed.warnings()[0].contains("2 rows"));
    }

    #[test]
    fn aside_length_prefixed_response_is_not_assumed_to_be_pi_identity() {
        let row = ROW.replace("\"agent\":\"claude\"", "\"agent\":\"aside\"");
        let (_file, mut state) = ledger(&row, &[]);
        let mut local = UnifiedMessage::new_with_dedup(
            "aside",
            "m",
            "p",
            "s",
            1,
            Default::default(),
            0.0,
            Some("aside:response:6:openai:msg_demo".into()),
        );
        assert!(state.reconcile(&mut local, None));
        assert!(state.remaining(None).is_empty());
        assert!(state.warnings()[0].contains("133363 tokens"));
    }

    #[test]
    fn missing_identity_and_malformed_rows_are_visible() {
        let (_file, mut state) = ledger(
            "{\"t\":\"2026-10-18T00:00:00Z\",\"model\":\"m\",\"in\":5,\"out\":2}\n{bad",
            &[],
        );
        assert!(state.remaining(None).is_empty());
        assert_eq!(state.warnings().len(), 2);
        assert!(state.warnings()[1].contains("7 tokens"));
        let (_file, mut gateway_only) = ledger(
            "{\"t\":\"2026-10-18T00:00:00Z\",\"model\":\"m\",\"in\":5,\"out\":2}",
            &["magpie"],
        );
        assert_eq!(gateway_only.remaining(None)[0].tokens.total(), 7);
        assert!(gateway_only.warnings().is_empty());
    }

    #[test]
    fn response_alias_is_limited_and_pi_uses_response_identity() {
        let row = ROW.replace("msg_demo", "msg_resp_demo");
        let (_file, mut state) = ledger(&row, &[]);
        let mut message = UnifiedMessage::new_with_dedup(
            "pi",
            "alias",
            "openai",
            "s",
            1,
            Default::default(),
            0.0,
            Some("pi:response:resp_demo".into()),
        );
        assert!(state.reconcile(&mut message, None));
        assert_eq!(message.tokens.total(), 133363);
        assert_ne!(canonical_response_id("rid_demo"), "resp_demo");
    }
}

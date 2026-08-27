use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalUsage {
    pub input_tokens: u64,
    /// True when `input_tokens` already includes cache read and cache creation
    /// input tokens. Claude-style usage leaves cached input tokens separate.
    #[serde(default, skip_serializing_if = "is_false")]
    pub input_tokens_include_cache: bool,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_creation_ephemeral_5m_tokens: u64,
    pub cache_creation_ephemeral_1h_tokens: u64,
    pub cache_read_tokens: u64,
    pub reasoning_tokens: u64,
}

impl CanonicalUsage {
    /// Cache-creation input represented by this snapshot.
    ///
    /// Claude can return both the aggregate creation count and its 5-minute /
    /// 1-hour breakdown.  The breakdown is descriptive when the aggregate is
    /// present, so summing all three fields would count the same tokens twice.
    #[must_use]
    pub fn effective_cache_creation_tokens(&self) -> u64 {
        if self.cache_creation_tokens > 0 {
            self.cache_creation_tokens
        } else {
            self.cache_creation_ephemeral_5m_tokens
                .saturating_add(self.cache_creation_ephemeral_1h_tokens)
        }
    }

    /// Cache-read and cache-creation input represented by this snapshot.
    #[must_use]
    pub fn effective_cache_input_tokens(&self) -> u64 {
        self.cache_read_tokens
            .saturating_add(self.effective_cache_creation_tokens())
    }

    /// Input token count including cache dimensions exactly once.
    #[must_use]
    pub fn inclusive_input_tokens(&self) -> u64 {
        if self.input_tokens_include_cache {
            self.input_tokens
        } else {
            self.input_tokens
                .saturating_add(self.effective_cache_input_tokens())
        }
    }

    /// Scalar token count used for realtime throughput accounting.
    ///
    /// A provider total is authoritative when its input dimension already
    /// includes cache tokens (or no cache tokens were reported). Claude-style
    /// totals exclude separately reported cache input, so those snapshots are
    /// rebuilt from inclusive input plus output. `reasoning_tokens` is a
    /// descriptive subset of output and is therefore not added a second time.
    #[must_use]
    pub fn inclusive_token_total(&self) -> u64 {
        if self.total_tokens > 0
            && (self.input_tokens_include_cache || self.effective_cache_input_tokens() == 0)
        {
            self.total_tokens
        } else {
            self.inclusive_input_tokens()
                .saturating_add(self.output_tokens)
        }
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::CanonicalUsage;

    #[test]
    fn inclusive_total_counts_separate_cache_and_not_its_breakdown_twice() {
        let usage = CanonicalUsage {
            input_tokens: 6,
            input_tokens_include_cache: false,
            output_tokens: 20,
            // Claude's total excludes the separately reported cache input.
            total_tokens: 26,
            cache_creation_tokens: 42,
            cache_creation_ephemeral_5m_tokens: 17,
            cache_creation_ephemeral_1h_tokens: 25,
            cache_read_tokens: 100,
            reasoning_tokens: 5,
        };

        assert_eq!(usage.effective_cache_creation_tokens(), 42);
        assert_eq!(usage.inclusive_input_tokens(), 148);
        assert_eq!(usage.inclusive_token_total(), 168);
    }

    #[test]
    fn inclusive_total_uses_creation_breakdown_when_aggregate_is_absent() {
        let usage = CanonicalUsage {
            input_tokens: 6,
            output_tokens: 20,
            total_tokens: 26,
            cache_creation_ephemeral_5m_tokens: 17,
            cache_creation_ephemeral_1h_tokens: 25,
            cache_read_tokens: 100,
            ..CanonicalUsage::default()
        };

        assert_eq!(usage.effective_cache_creation_tokens(), 42);
        assert_eq!(usage.inclusive_token_total(), 168);
    }

    #[test]
    fn inclusive_total_trusts_provider_total_when_input_already_includes_cache() {
        let usage = CanonicalUsage {
            input_tokens: 148,
            input_tokens_include_cache: true,
            output_tokens: 20,
            total_tokens: 168,
            cache_creation_tokens: 42,
            cache_read_tokens: 100,
            reasoning_tokens: 5,
            ..CanonicalUsage::default()
        };

        assert_eq!(usage.inclusive_input_tokens(), 148);
        assert_eq!(usage.inclusive_token_total(), 168);
    }

    #[test]
    fn inclusive_total_does_not_add_reasoning_outside_output() {
        let usage = CanonicalUsage {
            input_tokens: 10,
            output_tokens: 7,
            total_tokens: 0,
            // Canonical reasoning is a descriptive subset of output_tokens.
            reasoning_tokens: 3,
            ..CanonicalUsage::default()
        };

        assert_eq!(usage.inclusive_token_total(), 17);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CanonicalContentPart {
    ImageUrl(String),
    File {
        file_data: Option<String>,
        reference: Option<String>,
        mime_type: Option<String>,
        filename: Option<String>,
    },
    Audio {
        data: String,
        format: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CanonicalStreamEvent {
    Start,
    TextDelta(String),
    ReasoningDelta(String),
    ReasoningSummaryDone,
    ReasoningSignature(String),
    ContentPart(CanonicalContentPart),
    ImageGenerationCall {
        index: usize,
        item: Value,
    },
    OpenAiResponsesOutputItem {
        output_index: Option<usize>,
        item: Value,
        raw_event: Value,
    },
    ToolCallStart {
        index: usize,
        call_id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: usize,
        arguments: String,
    },
    ToolResultDelta {
        index: usize,
        tool_use_id: String,
        name: Option<String>,
        content: String,
    },
    UnknownEvent(Value),
    Finish {
        finish_reason: Option<String>,
        usage: Option<CanonicalUsage>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalStreamFrame {
    pub id: String,
    pub model: String,
    pub event: CanonicalStreamEvent,
}

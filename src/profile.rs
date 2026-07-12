//! Typed runtime profile manifest. This is the sole source of truth for
//! profile names, bounded allowlists, and transport effect capabilities.

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ToolProfile {
    Stable,
    Lean,
    Standard,
    Agent,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectClass {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug)]
pub struct ToolGrant {
    pub name: &'static str,
    pub effect: EffectClass,
}

const LEAN: &[ToolGrant] = &[
    ToolGrant {
        name: "sm_search_witnessed",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_replay_search",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_decide_assertion_authority",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_decide_action_authority",
        effect: EffectClass::Read,
    },
];

const AGENT: &[ToolGrant] = &[
    ToolGrant {
        name: "sm_add_fact",
        effect: EffectClass::Write,
    },
    ToolGrant {
        name: "sm_add_graph_edge",
        effect: EffectClass::Write,
    },
    ToolGrant {
        name: "sm_decide_action_authority",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_decide_assertion_authority",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_get_fact",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_get_fact_neighbors",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_get_search_receipt",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_graph_path",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_list_namespaces",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_replay_search",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_search_conversations",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_search_witnessed",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_set_provenance",
        effect: EffectClass::Write,
    },
    ToolGrant {
        name: "sm_stats",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_supersede_fact",
        effect: EffectClass::Write,
    },
    ToolGrant {
        name: "sm_update_fact",
        effect: EffectClass::Write,
    },
];

const STABLE: &[ToolGrant] = &[
    ToolGrant {
        name: "sm_search",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_search_witnessed",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_stats",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_list_namespaces",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_get_fact",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_get_fact_neighbors",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_graph_path",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_search_conversations",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_add_fact",
        effect: EffectClass::Write,
    },
    ToolGrant {
        name: "sm_supersede_fact",
        effect: EffectClass::Write,
    },
    ToolGrant {
        name: "sm_add_graph_edge",
        effect: EffectClass::Write,
    },
    ToolGrant {
        name: "sm_decide_assertion_authority",
        effect: EffectClass::Read,
    },
    ToolGrant {
        name: "sm_decide_action_authority",
        effect: EffectClass::Read,
    },
];

impl ToolProfile {
    pub fn manifest(self) -> Option<&'static [ToolGrant]> {
        match self {
            Self::Stable => Some(STABLE),
            Self::Lean | Self::Standard => Some(LEAN),
            Self::Agent => Some(AGENT),
            Self::Full => None,
        }
    }

    pub fn allows_http_write(self) -> bool {
        matches!(self, Self::Agent | Self::Full)
    }

    pub fn allows_http_maintenance(self) -> bool {
        matches!(self, Self::Full)
    }
}

impl std::fmt::Display for ToolProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_possible_value().unwrap().get_name())
    }
}

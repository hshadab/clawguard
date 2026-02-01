//! Strongly-typed action types replacing stringly-typed action names.

use std::fmt;

/// All recognized action types for guardrail models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    RunCommand,
    SendEmail,
    ReadFile,
    WriteFile,
    NetworkRequest,
}

impl ActionType {
    /// All known action types.
    pub const ALL: &'static [ActionType] = &[
        ActionType::RunCommand,
        ActionType::SendEmail,
        ActionType::ReadFile,
        ActionType::WriteFile,
        ActionType::NetworkRequest,
    ];

    /// Parse from a string, returning None for unknown actions.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "run_command" => Some(Self::RunCommand),
            "send_email" => Some(Self::SendEmail),
            "read_file" => Some(Self::ReadFile),
            "write_file" => Some(Self::WriteFile),
            "network_request" => Some(Self::NetworkRequest),
            _ => None,
        }
    }

    /// The string representation used in configs and CLI.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RunCommand => "run_command",
            Self::SendEmail => "send_email",
            Self::ReadFile => "read_file",
            Self::WriteFile => "write_file",
            Self::NetworkRequest => "network_request",
        }
    }

    /// One-hot index (0-4) for encoding into feature vectors.
    pub fn one_hot_index(&self) -> usize {
        match self {
            Self::RunCommand => 0,
            Self::SendEmail => 1,
            Self::ReadFile => 2,
            Self::WriteFile => 3,
            Self::NetworkRequest => 4,
        }
    }
}

impl fmt::Display for ActionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ActionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_opt(s).ok_or_else(|| {
            format!(
                "unknown action type '{}'. Valid types: run_command, send_email, \
                 read_file, write_file, network_request",
                s
            )
        })
    }
}

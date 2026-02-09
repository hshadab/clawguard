//! ClawHub API client for fetching skill metadata and content.
//!
//! Communicates with the ClawHub registry API to retrieve skill information
//! for safety scanning.

use eyre::{Result, WrapErr};
use serde::Deserialize;

use crate::skill::{Skill, SkillMetadata};

const DEFAULT_BASE_URL: &str = "https://clawhub.ai/api/v1";

/// Entry from the skill listing endpoint (subset of full skill data).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillListEntry {
    pub slug: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub stats: SkillStats,
    #[serde(default)]
    pub latest_version: Option<VersionInfo>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// Stats sub-object from the list endpoint.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SkillStats {
    #[serde(default)]
    pub stars: u64,
    #[serde(default)]
    pub downloads: u64,
}

/// Version info from the list endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct VersionInfo {
    #[serde(default)]
    pub version: String,
}

/// List response from `GET /skills`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SkillListResponse {
    #[serde(default)]
    items: Vec<SkillListEntry>,
    #[serde(default)]
    next_cursor: Option<String>,
}

/// Detail response from `GET /skills/{slug}`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct SkillDetailResponse {
    pub slug: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub stats: SkillStats,
    #[serde(default)]
    pub latest_version: Option<VersionInfo>,
    #[serde(default)]
    pub owner: Option<OwnerInfo>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// Owner/author info from the detail endpoint.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct OwnerInfo {
    #[serde(default)]
    pub handle: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub total_skills: u64,
}

/// Client for the ClawHub API.
pub struct ClawHubClient {
    base_url: String,
    client: reqwest::Client,
}

impl ClawHubClient {
    /// Create a new client with the default base URL.
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new client with a custom base URL.
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch a single skill by name, mapping the API response into the `Skill` struct.
    ///
    /// Fetches both the skill detail and the SKILL.md content.
    pub async fn fetch_skill(&self, name: &str, version: Option<&str>) -> Result<Skill> {
        // Fetch skill details
        let detail_url = format!("{}/skills/{}", self.base_url, name);
        let detail: SkillDetailResponse = self
            .client
            .get(&detail_url)
            .send()
            .await
            .wrap_err_with(|| format!("Failed to fetch skill detail: {}", name))?
            .error_for_status()
            .wrap_err_with(|| format!("HTTP error fetching skill: {}", name))?
            .json()
            .await
            .wrap_err("Failed to parse skill detail JSON")?;

        // Fetch SKILL.md content
        let mut content_url = format!("{}/skills/{}/file?path=SKILL.md", self.base_url, name);
        if let Some(v) = version {
            content_url.push_str(&format!("&version={}", v));
        }

        let skill_md = self
            .client
            .get(&content_url)
            .send()
            .await
            .wrap_err("Failed to fetch SKILL.md")?
            .text()
            .await
            .unwrap_or_default();

        let owner = detail.owner.unwrap_or_default();
        let version_str = version
            .map(|v| v.to_string())
            .or_else(|| detail.latest_version.map(|v| v.version))
            .unwrap_or_else(|| "0.0.0".to_string());

        Ok(Skill {
            name: detail.slug,
            version: version_str,
            author: owner.handle.clone(),
            description: detail.summary,
            skill_md,
            scripts: Vec::new(),
            metadata: SkillMetadata {
                stars: detail.stats.stars,
                downloads: detail.stats.downloads,
                created_at: detail.created_at.unwrap_or_default(),
                updated_at: detail.updated_at.unwrap_or_default(),
                author_account_created: owner.created_at.unwrap_or_default(),
                author_total_skills: owner.total_skills,
            },
            files: vec!["SKILL.md".to_string()],
        })
    }

    /// Fetch a paginated list of skills.
    ///
    /// Returns a tuple of (entries, next_cursor). Pass the cursor to the next call
    /// to fetch subsequent pages.
    pub async fn fetch_skill_list(
        &self,
        cursor: Option<&str>,
        limit: u32,
    ) -> Result<(Vec<SkillListEntry>, Option<String>)> {
        let mut url = format!(
            "{}/skills?limit={}&sort=updated",
            self.base_url, limit
        );
        if let Some(c) = cursor {
            url.push_str(&format!("&cursor={}", c));
        }

        let resp: SkillListResponse = self
            .client
            .get(&url)
            .send()
            .await
            .wrap_err("Failed to fetch skill list")?
            .error_for_status()
            .wrap_err("HTTP error fetching skill list")?
            .json()
            .await
            .wrap_err("Failed to parse skill list JSON")?;

        Ok((resp.items, resp.next_cursor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_list_response_deserialization() {
        let json = r#"{
            "items": [
                {
                    "slug": "test-skill",
                    "displayName": "Test Skill",
                    "summary": "A test skill",
                    "stats": { "stars": 10, "downloads": 500 },
                    "latestVersion": { "version": "1.2.3" },
                    "createdAt": "2025-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z"
                }
            ],
            "nextCursor": "abc123"
        }"#;

        let resp: SkillListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.items.len(), 1);
        assert_eq!(resp.items[0].slug, "test-skill");
        assert_eq!(resp.items[0].display_name, "Test Skill");
        assert_eq!(resp.items[0].stats.stars, 10);
        assert_eq!(resp.items[0].stats.downloads, 500);
        assert_eq!(
            resp.items[0].latest_version.as_ref().unwrap().version,
            "1.2.3"
        );
        assert_eq!(resp.next_cursor, Some("abc123".to_string()));
    }

    #[test]
    fn test_skill_detail_response_deserialization() {
        let json = r#"{
            "slug": "my-skill",
            "displayName": "My Skill",
            "summary": "Does things",
            "stats": { "stars": 42, "downloads": 1234 },
            "latestVersion": { "version": "2.0.0" },
            "owner": {
                "handle": "dev123",
                "displayName": "Dev 123",
                "createdAt": "2024-06-01T00:00:00Z",
                "totalSkills": 5
            },
            "createdAt": "2025-03-01T00:00:00Z",
            "updatedAt": "2026-02-01T00:00:00Z"
        }"#;

        let detail: SkillDetailResponse = serde_json::from_str(json).unwrap();
        assert_eq!(detail.slug, "my-skill");
        assert_eq!(detail.stats.stars, 42);
        let owner = detail.owner.unwrap();
        assert_eq!(owner.handle, "dev123");
        assert_eq!(owner.total_skills, 5);
    }

    #[test]
    fn test_skill_list_empty_response() {
        let json = r#"{ "items": [] }"#;
        let resp: SkillListResponse = serde_json::from_str(json).unwrap();
        assert!(resp.items.is_empty());
        assert!(resp.next_cursor.is_none());
    }

    #[test]
    fn test_skill_list_entry_missing_optional_fields() {
        let json = r#"{
            "items": [{ "slug": "minimal" }],
            "nextCursor": null
        }"#;
        let resp: SkillListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.items[0].slug, "minimal");
        assert_eq!(resp.items[0].display_name, "");
        assert_eq!(resp.items[0].stats.stars, 0);
    }
}

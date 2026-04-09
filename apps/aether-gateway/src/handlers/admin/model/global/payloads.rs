use super::super::super::shared::json_string_list;
use super::super::payloads::timestamp_or_now;
use super::helpers::{admin_global_models_now_unix_secs, build_admin_global_model_price_range};
use crate::handlers::admin::request::AdminAppState;
use aether_data_contracts::repository::global_models::{
    AdminGlobalModelListQuery, StoredAdminGlobalModel,
};
use serde_json::json;

pub(crate) fn build_admin_global_model_response(
    global_model: &StoredAdminGlobalModel,
    now_unix_secs: u64,
) -> serde_json::Value {
    json!({
        "id": &global_model.id,
        "name": &global_model.name,
        "display_name": &global_model.display_name,
        "is_active": global_model.is_active,
        "default_price_per_request": global_model.default_price_per_request,
        "default_tiered_pricing": global_model.default_tiered_pricing.clone(),
        "supported_capabilities": json_string_list(global_model.supported_capabilities.as_ref()),
        "config": global_model.config.clone(),
        "provider_count": global_model.provider_count,
        "active_provider_count": global_model.active_provider_count,
        "usage_count": global_model.usage_count,
        "created_at": timestamp_or_now(global_model.created_at_unix_secs, now_unix_secs),
        "updated_at": timestamp_or_now(global_model.updated_at_unix_secs, now_unix_secs),
    })
}

pub(crate) async fn build_admin_global_models_payload(
    state: &AdminAppState<'_>,
    skip: usize,
    limit: usize,
    is_active: Option<bool>,
    search: Option<String>,
) -> Option<serde_json::Value> {
    if !state.has_global_model_data_reader() {
        return None;
    }
    let page = state
        .list_admin_global_models(&AdminGlobalModelListQuery {
            offset: skip,
            limit,
            is_active,
            search,
        })
        .await
        .ok()?;
    let now_unix_secs = admin_global_models_now_unix_secs();
    let mut models = page.items;
    models.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut payload_models = Vec::with_capacity(models.len());
    for model in models {
        payload_models.push(build_admin_global_model_response(&model, now_unix_secs));
    }
    Some(json!({
        "models": payload_models,
        "total": page.total,
    }))
}

pub(crate) async fn build_admin_global_model_payload(
    state: &AdminAppState<'_>,
    global_model_id: &str,
) -> Option<serde_json::Value> {
    if !state.has_global_model_data_reader() {
        return None;
    }
    let model = state
        .get_admin_global_model_by_id(global_model_id)
        .await
        .ok()??;
    let provider_models = state
        .list_admin_provider_models_by_global_model_id(&model.id)
        .await
        .ok()
        .unwrap_or_default();
    let now_unix_secs = admin_global_models_now_unix_secs();
    let total_models = provider_models.len();
    let mut payload = build_admin_global_model_response(&model, now_unix_secs);
    if let Some(object) = payload.as_object_mut() {
        object.insert("total_models".to_string(), json!(total_models));
        object.insert("total_providers".to_string(), json!(model.provider_count));
        object.insert(
            "price_range".to_string(),
            build_admin_global_model_price_range(&model, &provider_models),
        );
    }
    Some(payload)
}

#[cfg(test)]
mod tests {
    use super::build_admin_global_model_response;
    use aether_data_contracts::repository::global_models::StoredAdminGlobalModel;
    use serde_json::json;

    #[test]
    fn admin_global_model_response_uses_stored_stats_and_usage_count() {
        let global_model = StoredAdminGlobalModel::new(
            "global-minimax".to_string(),
            "MiniMax-M2.7".to_string(),
            "MiniMax-M2.7".to_string(),
            true,
            None,
            Some(json!({
                "tiers": [{
                    "up_to": null,
                    "input_price_per_1m": 0.3,
                    "output_price_per_1m": 1.2,
                }]
            })),
            Some(json!(["cache_1h"])),
            Some(json!({
                "family": "minimax",
                "streaming": true,
            })),
            3,
            3,
            1,
            Some(1_711_000_000),
            Some(1_711_000_100),
        )
        .expect("global model should build");

        let payload = build_admin_global_model_response(&global_model, 1_711_000_100);

        assert_eq!(payload["provider_count"], 3);
        assert_eq!(payload["active_provider_count"], 3);
        assert_eq!(payload["usage_count"], 1);
    }
}

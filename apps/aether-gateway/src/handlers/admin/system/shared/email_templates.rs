use crate::handlers::shared::{
    admin_email_template_definition, admin_email_template_html_key,
    admin_email_template_subject_key, read_admin_email_template_payload,
    render_admin_email_template_html, system_config_string,
};
use crate::{AppState, GatewayError};
use axum::body::Bytes;
use axum::http;
use serde_json::json;

pub(crate) async fn build_admin_email_templates_payload(
    state: &AppState,
) -> Result<serde_json::Value, GatewayError> {
    let mut templates = Vec::new();
    for template_type in ["verification", "password_reset"] {
        if let Some(payload) = read_admin_email_template_payload(state, template_type).await? {
            let mut payload = payload;
            if let Some(object) = payload.as_object_mut() {
                object.remove("default_subject");
                object.remove("default_html");
            }
            templates.push(payload);
        }
    }

    Ok(json!({ "templates": templates }))
}

pub(crate) async fn build_admin_email_template_payload(
    state: &AppState,
    template_type: &str,
) -> Result<Result<serde_json::Value, (http::StatusCode, serde_json::Value)>, GatewayError> {
    let Some(payload) = read_admin_email_template_payload(state, template_type).await? else {
        return Ok(Err((
            http::StatusCode::NOT_FOUND,
            json!({ "detail": format!("模板类型 '{template_type}' 不存在") }),
        )));
    };
    Ok(Ok(payload))
}

pub(crate) async fn apply_admin_email_template_update(
    state: &AppState,
    template_type: &str,
    request_body: &Bytes,
) -> Result<Result<serde_json::Value, (http::StatusCode, serde_json::Value)>, GatewayError> {
    let Some(definition) = admin_email_template_definition(template_type) else {
        return Ok(Err((
            http::StatusCode::NOT_FOUND,
            json!({ "detail": format!("模板类型 '{template_type}' 不存在") }),
        )));
    };
    let payload = match serde_json::from_slice::<serde_json::Value>(request_body) {
        Ok(serde_json::Value::Object(payload)) => payload,
        _ => {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    };
    let subject = match payload.get("subject") {
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(serde_json::Value::Null) | None => None,
        Some(_) => {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    };
    let html = match payload.get("html") {
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(serde_json::Value::Null) | None => None,
        Some(_) => {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    };

    if subject.is_none() && html.is_none() {
        return Ok(Err((
            http::StatusCode::BAD_REQUEST,
            json!({ "detail": "请提供 subject 或 html" }),
        )));
    }

    let subject_key = admin_email_template_subject_key(definition.template_type);
    let html_key = admin_email_template_html_key(definition.template_type);

    if let Some(subject) = subject {
        if subject.is_empty() {
            let _ = state.delete_system_config_value(&subject_key).await?;
        } else {
            let _ = state
                .upsert_system_config_json_value(&subject_key, &json!(subject), None)
                .await?;
        }
    }

    if let Some(html) = html {
        if html.is_empty() {
            let _ = state.delete_system_config_value(&html_key).await?;
        } else {
            let _ = state
                .upsert_system_config_json_value(&html_key, &json!(html), None)
                .await?;
        }
    }

    Ok(Ok(json!({ "message": "模板保存成功" })))
}

pub(crate) async fn preview_admin_email_template(
    state: &AppState,
    template_type: &str,
    request_body: Option<&Bytes>,
) -> Result<Result<serde_json::Value, (http::StatusCode, serde_json::Value)>, GatewayError> {
    let Some(definition) = admin_email_template_definition(template_type) else {
        return Ok(Err((
            http::StatusCode::NOT_FOUND,
            json!({ "detail": format!("模板类型 '{template_type}' 不存在") }),
        )));
    };

    let payload = match request_body {
        Some(bytes) => match serde_json::from_slice::<serde_json::Value>(bytes) {
            Ok(serde_json::Value::Object(payload)) => payload,
            Ok(serde_json::Value::Null) => serde_json::Map::new(),
            _ => {
                return Ok(Err((
                    http::StatusCode::BAD_REQUEST,
                    json!({ "detail": "请求数据验证失败" }),
                )));
            }
        },
        None => serde_json::Map::new(),
    };

    let resolved = read_admin_email_template_payload(state, definition.template_type)
        .await?
        .expect("validated template type should exist");
    let resolved_html = resolved["html"].as_str().unwrap_or(definition.default_html);
    let html = payload
        .get("html")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(resolved_html);

    let email_app_name = state
        .read_system_config_json_value("email_app_name")
        .await?;
    let smtp_from_name = state
        .read_system_config_json_value("smtp_from_name")
        .await?;
    let app_name = system_config_string(email_app_name.as_ref())
        .or_else(|| system_config_string(smtp_from_name.as_ref()))
        .unwrap_or_else(|| "Aether".to_string());

    let mut defaults = std::collections::BTreeMap::new();
    defaults.insert("app_name".to_string(), app_name);
    defaults.insert("code".to_string(), "123456".to_string());
    defaults.insert("expire_minutes".to_string(), "30".to_string());
    defaults.insert("email".to_string(), "example@example.com".to_string());
    defaults.insert(
        "reset_link".to_string(),
        "https://example.com/reset?token=abc123".to_string(),
    );

    let preview_variables = definition
        .variables
        .iter()
        .map(|key| {
            let value = payload
                .get(*key)
                .map(|value| match value {
                    serde_json::Value::String(value) => value.clone(),
                    serde_json::Value::Null => "None".to_string(),
                    _ => value.to_string(),
                })
                .or_else(|| defaults.get(*key).cloned())
                .unwrap_or_else(|| format!("{{{{{key}}}}}"));
            ((*key).to_string(), value)
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    let rendered_html = render_admin_email_template_html(html, &preview_variables)?;

    Ok(Ok(json!({
        "html": rendered_html,
        "variables": preview_variables,
    })))
}

pub(crate) async fn reset_admin_email_template(
    state: &AppState,
    template_type: &str,
) -> Result<Result<serde_json::Value, (http::StatusCode, serde_json::Value)>, GatewayError> {
    let Some(definition) = admin_email_template_definition(template_type) else {
        return Ok(Err((
            http::StatusCode::NOT_FOUND,
            json!({ "detail": format!("模板类型 '{template_type}' 不存在") }),
        )));
    };

    let _ = state
        .delete_system_config_value(&admin_email_template_subject_key(definition.template_type))
        .await?;
    let _ = state
        .delete_system_config_value(&admin_email_template_html_key(definition.template_type))
        .await?;

    Ok(Ok(json!({
        "message": "模板已重置为默认值",
        "template": {
            "type": definition.template_type,
            "name": definition.name,
            "subject": definition.default_subject,
            "html": definition.default_html,
        }
    })))
}

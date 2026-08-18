# Ark 素材库

Aether 提供火山方舟素材库的原生 Action API 兼容入口，同时为控制台提供按用户隔离的资源 API。素材库用于管理图片、视频、音频及真人验证生成的 `LivenessFace` 素材组；已激活的素材可通过 `asset://<本地素材 ID>` 引用到 Seedance 视频请求中。

## Provider 配置

素材库端点使用 API 格式 `doubao:asset_library`，Key 必须启用能力 `ark_asset_library`。Aether 到上游支持三种认证模式：

| `auth_type` | 配置 | 上游认证 |
| --- | --- | --- |
| `volc_aksk` | `auth_config.access_key_id`、`secret_access_key`，可选 `security_token`、`region`、`service`、`account_id`、`project` | 火山 SignV4 |
| `bearer` | `api_key`，并在 `auth_config` 配置 `account_id`、`project` | `Authorization: Bearer ...` |
| `api_key` | `api_key`，并在 `auth_config` 配置 `account_id`、`project`；可指定 `api_key_header` | `X-Api-Key` 或 `Api-Key` |

AK/SK 只保存在加密的 Provider Key 配置中，不会进入素材记录、响应、日志或前端。素材创建后会粘滞原 Provider、素材端点和素材 Key；后续读取、更新、删除和预览不会故障转移。跨素材与视频端点引用则通过显式 `account_id` 和 `project` 证明属于同一 Ark 账号与项目。

## 客户端认证

用户到 Aether 使用 Aether API Key，三种载体均支持；控制台浏览器请求也支持现有的 Aether 登录会话 JWT：

```http
Authorization: Bearer <AETHER_API_KEY>
X-Api-Key: <AETHER_API_KEY>
Api-Key: <AETHER_API_KEY>
```

不得通过 query 传递 Key。若同时提供多个不同凭据，Aether 返回 `400`；客户端凭据不会透传上游。

## 原生 Action API

原生入口为 `POST /?Action=<Action>&Version=2024-01-01`。另提供等价别名：

```text
POST /v3/asset-library/<Action>
POST /v3/asset-library          # Action 可放 JSON body
```

支持的 Action：

- `CreateAssetGroup`、`ListAssetGroups`、`GetAssetGroup`、`UpdateAssetGroup`、`DeleteAssetGroup`
- `CreateAsset`、`ListAssets`、`GetAsset`、`UpdateAsset`、`DeleteAsset`
- `CreateVisualValidateSession`、`GetVisualValidateResult`

根入口严格要求 `Version=2024-01-01`。Action 冲突、非法 JSON 或非对象 body 返回标准 Ark 错误 envelope。响应中的 `GroupId`、`AssetId` 均为 Aether 本地 ID；Aether 在上游调用前映射回原账号的官方 ID，避免共享 Provider Key 导致跨用户枚举。

示例：

```bash
curl 'https://aether.example/?Action=CreateAssetGroup&Version=2024-01-01' \
  -H 'Authorization: Bearer <AETHER_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"Name":"角色素材","GroupType":"AIGC"}'

curl 'https://aether.example/?Action=CreateAsset&Version=2024-01-01' \
  -H 'X-Api-Key: <AETHER_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"GroupId":"agrp-...","AssetType":"Image","URL":"https://example.com/face.jpg","Name":"角色正面"}'
```

素材列表始终在本地按调用者 `user_id` 隔离、分页和计数，不会把共享上游账号的全量列表透传给用户。删除素材组会同时软删除本地组内素材。

## 控制台资源 API

用户接口位于 `/api/material-assets`，管理员接口位于 `/api/admin/material-assets`。管理员必须先指定目标 `user_id`，资源 ID 操作也会校验该 owner。

- `GET/POST /groups`
- `GET/PATCH/DELETE /groups/{id}`
- `GET /assets`
- `POST /assets/url`
- `GET/PATCH/DELETE /assets/{id}`
- `GET /assets/{id}/preview`
- `POST /verification-sessions`
- `GET /verification-sessions/{id}`

官方 CreateAsset 为 URL 创建，因此当前不提供本地文件直传；`POST /assets/upload` 返回不支持。预览使用鉴权流式代理，支持 Range；上游短期签名 URL 不持久化、不返回给浏览器，并经过 HTTPS、DNS 公网地址固定、禁代理和禁重定向检查。

## 视频引用

素材状态必须为 `Active`。素材端点与视频端点可以使用不同的端点和 Key，但必须属于同一 Provider，并配置完全一致的 `account_id` 与 `project`。Aether 只扫描视频请求根对象的 `content` 字段，在最终 body 规则执行后、上游鉴权签名前完成引用替换：

```json
{
  "model": "Doubao-Seedance-2.0",
  "content": [
    {"type": "text", "text": "保持角色一致，走向镜头"},
    {"type": "image_url", "image_url": {"url": "asset://asset-..."}}
  ]
}
```

跨用户、非 `Active`、已删除、账号或项目不一致的引用都会在本地拒绝，不会把本地 ID 发往上游。

## 真人验证

`CreateVisualValidateSession` 返回的 BytedToken 仅加密保存，控制台只获取本地 session ID 和 H5 地址。轮询成功后，上游 `GroupId` 会投影为本地 `LivenessFace` 素材组 ID；已过期会话返回 `410`，不会继续访问上游。

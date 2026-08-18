# Ark 素材库

Aether 提供火山方舟素材库的原生 Action API 兼容入口，同时为控制台提供按用户隔离的资源 API。素材库用于管理素材组、URL 图片素材及真人验证生成的 `LivenessFace` 素材组；已激活的素材可通过 `asset://<本地素材 ID>` 引用到 Seedance 视频请求中。

## Provider 配置

素材库端点使用 API 格式 `doubao:asset_library`，Key 必须启用能力 `ark_asset_library`。Aether 到上游支持三种认证模式：

在管理端进入“提供商 → 提供商详情 → 密钥 → 添加密钥”，在“上游认证”中选择 API Key、Bearer Token 或 Volcengine AK/SK。Region、Service、Security Token 与 API Key 请求头位于同一表单的“高级认证参数”中。

Endpoint 的 `base_url` 配置服务地址，`custom_path` 配置素材库路径。Aether 会在最终地址上追加当前操作的 `Action` 与固定版本 `Version=2024-01-01`，不要把这两个参数写死在 `custom_path` 中；路径末尾 `/` 应与上游文档保持一致，避免上游用 307 重定向 POST 请求。

| 上游 | `base_url` | `custom_path` | 最终请求示例 |
| --- | --- | --- | --- |
| 火山方舟 | `https://ark.cn-beijing.volcengineapi.com` | `/` | `POST /?Action=ListAssetGroups&Version=2024-01-01` |
| K23 Seedance | `https://ai.k23.cn` | `/seedance/assets/` | `POST /seedance/assets/?Action=ListAssetGroups&Version=2024-01-01` |

| `auth_type` | 配置 | 上游认证 |
| --- | --- | --- |
| `volc_aksk` | `auth_config.access_key_id`、`secret_access_key`，可选 `security_token`、`region`、`service` | 火山 SignV4 |
| `bearer` | `api_key` | `Authorization: Bearer ...` |
| `api_key` | `api_key`；可指定 `api_key_header` | `X-Api-Key` 或 `Api-Key` |

AK/SK 只保存在加密的 Provider Key 配置中，不会进入素材记录、响应、日志或前端。素材归属于 Provider，并以 Provider 与上游 ID 组成唯一身份；不同 Provider 可安全使用相同的上游 ID。素材库 Provider Key 不接受 `account_id`、`account_binding`、`project` 及其别名，直接调用管理 API 也会返回校验错误。素材创建后会粘滞原 Provider、素材端点和素材 Key，后续读取、更新、删除和预览不会故障转移。

## 客户端认证

普通 Aether API Key 继续支持三种请求头载体；一枚 Key 可任选其中一种使用。控制台浏览器请求也支持现有的 Aether 登录会话 JWT：

```http
Authorization: Bearer <AETHER_API_KEY>
X-Api-Key: <AETHER_API_KEY>
Api-Key: <AETHER_API_KEY>
```

不得通过 query 传递 Key。若同时提供多个不同凭据，Aether 返回 `400`；客户端凭据不会透传上游。

用户也可以在“我的 API 密钥 → 创建 API 密钥”中选择“火山引擎 AK/SK”。Aether 会生成独立的 Access Key ID 与 Secret Access Key，并复用同一套 API Key 策略、禁用状态、过期时间、IP 规则、Provider/API 格式权限、RPM、并发和计费归属。SK 仅在创建成功时显示一次，列表和详情只保留 AK；AK/SK 不支持完整密钥找回、CLI 安装或 CC Switch 导入。

用户 AK/SK 可用于原生 Ark 素材 Action API 和 `/api/material-assets/**` 用户资源 API，包括返回的相对预览地址。请求采用火山 HMAC-SHA256 规范，签名必须覆盖实际 method、path、query、Host、`Content-Type`、`X-Date`、`X-Content-Sha256` 和原始 body bytes，Credential scope 使用 `cn-beijing/ark/request`。Aether 会校验请求时间窗和实际 body hash；AK/SK 不能与 `X-Api-Key` 或 `Api-Key` 混用。

## 原生 Action API

面向客户端的原生入口为 `POST /?Action=<Action>&Version=2024-01-01`。根入口必须在 query 中提供完全匹配的 `Action` 与 `Version`；Aether 转发上游时也使用相同的 Action 名称和版本。另提供等价别名：

```text
POST /v3/asset-library/<Action>
POST /v3/asset-library          # Action 可放 JSON body
```

支持的 Action：

- `CreateAssetGroup`、`ListAssetGroups`、`GetAssetGroup`、`UpdateAssetGroup`、`DeleteAssetGroup`
- `CreateAsset`、`ListAssets`、`GetAsset`、`UpdateAsset`、`DeleteAsset`
- `CreateVisualValidateSession`、`GetVisualValidateResult`

Action 冲突、非法 JSON 或非对象 body 返回标准 Ark 错误 envelope。请求 body 使用火山/K23 的官方字段：

| Action | 关键请求字段 | 关键响应字段 |
| --- | --- | --- |
| `CreateAssetGroup` | `Name`、`GroupType=AIGC`，可选 `Description` | `Result.Id` |
| `GetAssetGroup`、`UpdateAssetGroup`、`DeleteAssetGroup` | `Id`；更新时可带 `Name`、`Description` | 资源信息、`Result.Id` 或空结果 |
| `ListAssetGroups` | `Filter.GroupType`，可选 `Filter.GroupIds` 数组、`Filter.Name`、分页和排序字段 | `Result.TotalCount`、`Items`、`PageNumber`、`PageSize` |
| `CreateAsset` | `GroupId`、`URL`、`AssetType=Image`，可选 `Name` | `Result.Id` |
| `GetAsset`、`UpdateAsset`、`DeleteAsset` | `Id`；更新时可带 `Name` | 资源信息、`Result.Id` 或空结果 |
| `ListAssets` | 可选 `Filter.GroupIds`、`Filter.Statuses` 数组、`Filter.GroupType`、`Filter.Name`、分页和排序字段 | `Result.TotalCount`、`Items`、`PageNumber`、`PageSize` |
| `CreateVisualValidateSession` | `CallbackURL` | 顶层 `BytedToken`、`H5Link`、`CallbackURL` |
| `GetVisualValidateResult` | `BytedToken` | 顶层 `GroupId` |

单资源查询、更新、删除统一使用 `Id`，只有 `CreateAsset` 使用 `GroupId` 指定父素材组。为兼容旧客户端，入口仍可识别 `GroupId`、`AssetId` 等旧别名，但发往上游的字段会规范化为上述官方字段。列表请求中的 `GroupIds`、`Statuses` 必须是数组，规范列表总数字段为 `TotalCount`；`PageSize` 范围为 1–100，`SortBy` 仅支持 `CreateTime`、`UpdateTime`，`SortOrder` 仅支持 `Asc`、`Desc`。

Aether 以 Provider 作为素材隔离和路由边界，不持久化 Ark 项目维度。为兼容默认项目客户端，`ProjectName` 可省略或填写 `default`，其他值会在访问上游前被拒绝；默认值不会写入上游请求。

原生响应中的 `Id`（以及兼容别名）和真人验证结果中的 `GroupId` 均为 Aether 本地 ID。Aether 在访问上游前将其映射回绑定 Provider 的上游 ID，并在返回前投影为本地 ID，避免共享 Provider Key 导致跨用户枚举；上游 ID 不应当直接作为后续 Aether 请求参数。

素材响应的 `URL` 是相对于 Aether 服务地址的鉴权预览路径，不是上游临时签名 URL。客户端应以 Aether origin 解析该路径，并使用普通 API Key 三种请求头之一或用户 AK/SK 对预览 GET 重新鉴权；预览代理支持 Range 请求，且不会透出 Provider 凭据。

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

素材列表始终在本地按调用者 `user_id` 隔离、过滤、分页和计数，不会把共享上游账号的全量列表透传给用户。删除素材组会同时软删除本地组内素材。

### 基础连接测试

Provider 密钥旁的“素材库基础连通性测试”会用选中的 Endpoint 和已保存密钥调用一次 `ListAssetGroups`，请求固定为第一页、每页一条并过滤 `GroupType=AIGC`。它用于验证 `base_url`/`custom_path` 拼接、`Action`/`Version`、认证、网络连通性及列表响应结构。

该测试不会创建、更新或删除素材，也不会验证素材 URL 可访问性、真人验证 H5/回调/轮询、预览或视频引用。测试成功仅表示基础列表调用可用，不代表所有 Action 已完成端到端验证。

### 上游错误映射

Aether 会保留脱敏后的火山 `RequestId`，并把常见上游错误映射为可操作的错误码。REST 响应字段为 `code`、`detail`、`request_id`；原生 Action API 返回对应的 Ark `ResponseMetadata.Error` envelope。上游原始错误正文、凭据和内部字段不会返回客户端。

| 上游错误 | Aether 错误码 | 含义 |
| --- | --- | --- |
| `SubscriptionRequired` | `SubscriptionRequired` | 当前火山账号尚未开通素材库所需套餐 |
| `SignatureDoesNotMatch` 等签名错误 | `SignatureDoesNotMatch` | 检查 AK/SK、Region、Service 和系统时间 |
| `InvalidAccessKeyId`、安全令牌或请求时间错误 | `InvalidCredentials` | Provider 凭据无效或已过期 |
| `AccessDenied`、`PermissionDenied` | `AccessDenied` | Provider 账号缺少对应操作权限 |
| `Throttling`、`RateLimitExceeded` | `RateLimitExceeded` | 火山侧限流，请稍后重试 |

其他火山业务错误会保留其脱敏后的上游错误码，错误消息统一为安全描述，便于定位参数问题且不泄露上游内部信息。

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

素材状态必须为 `Active`。素材端点与视频端点可以使用不同的端点和 Key，但必须属于同一 Provider。Aether 只扫描视频请求根对象的 `content` 字段，在最终 body 规则执行后、上游鉴权签名前完成引用替换：

```json
{
  "model": "Doubao-Seedance-2.0",
  "content": [
    {"type": "text", "text": "保持角色一致，走向镜头"},
    {"type": "image_url", "image_url": {"url": "asset://asset-..."}}
  ]
}
```

跨用户、非 `Active`、已删除或 Provider 不一致的引用都会在本地拒绝，不会把本地 ID 发往上游。

## 真人验证

原生流程先调用 `CreateVisualValidateSession` 并传入 `CallbackURL`，再打开响应中的 `H5Link` 完成验证；随后使用 `BytedToken` 轮询 `GetVisualValidateResult`，直到返回 `GroupId`。虽然部分文档把 `CallbackURL` 标记为可选，K23 当前实际协议会在缺失时返回 `MissingParameter.CallbackURL`，因此 Aether 将其作为必填字段校验。两个真人验证 Action 都保持 K23 的顶层响应格式，不额外包裹 `ResponseMetadata` 或 `Result`。

验证成功后，Aether 会使用同一 Provider、Endpoint 和 Key 分页读取该 `LivenessFace` 组内的图片素材，生成稳定的本地素材 ID，再将上游 `GroupId` 投影为本地组 ID。只有组内素材全部同步成功后会话才标记为成功；中途失败保持可重试，已导入项会幂等复用。上游素材 URL 不会持久化，预览时由 Aether 即时刷新并安全代理。

控制台 REST 创建接口使用 `callback_url`，查询接口使用本地 session ID；`BytedToken` 仅加密保存，不在 REST 响应中暴露。会话尚未完成时可继续查询，已过期会话返回 `410` 且不再访问上游。

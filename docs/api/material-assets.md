# Ark 素材库

Aether 提供与火山方舟 `2024-01-01` 素材库协议对齐的原生 Action API，并为控制台提供按用户隔离的 REST API。

对外协议中的素材组 ID、素材 ID 和素材 URL 均采用方舟上游返回值：素材组 ID 形如 `group-*`，素材 ID 形如 `asset-*`；数据库内部主键不会出现在原生 API 或控制台 REST 响应中。`GetAsset`、`ListAssets` 返回的 `URL` 是方舟临时访问地址，不是 Aether 预览路径。

## 官方协议基线

- Base URL：`https://ark.cn-beijing.volcengineapi.com/`
- Method：`POST`
- Version：`2024-01-01`
- Region：`cn-beijing`
- Service：`ark`
- 官方鉴权：Volcengine Access Key（AK/SK），HMAC-SHA256
- `ProjectName`：默认 `default`，非默认项目必须原样传入，并与目标资源所属项目一致

火山官方素材库接口仅支持 Access Key 鉴权。Aether Provider 中的 Bearer Token 与 API Key 是面向 K23 等兼容中转服务的扩展，不属于火山官方鉴权协议。

## Provider 配置

素材库 Endpoint 使用 API 格式 `doubao:asset_library`，Provider Key 必须启用 `ark_asset_library` 能力。管理端路径为“提供商 → 提供商详情 → 密钥 → 添加/编辑密钥”。

| `auth_type` | 配置 | 用途 |
| --- | --- | --- |
| `volc_aksk` | `access_key_id`、`secret_access_key`；可选 `security_token`、`region`、`service` | 火山官方 AK/SK |
| `bearer` | `api_key` | Aether 兼容中转扩展：`Authorization: Bearer ...` |
| `api_key` | `api_key`；可选 `api_key_header` | Aether 兼容中转扩展：`X-Api-Key` 或 `Api-Key` |

Endpoint 的 `base_url` 配置服务地址，`custom_path` 配置素材库路径。Aether 在最终地址追加当前 `Action` 与 `Version=2024-01-01`，不要把它们写死在 `custom_path` 中。

| 上游 | `base_url` | `custom_path` | 最终请求示例 |
| --- | --- | --- | --- |
| 火山方舟 | `https://ark.cn-beijing.volcengineapi.com` | `/` | `POST /?Action=ListAssetGroups&Version=2024-01-01` |
| K23 Seedance | `https://ai.k23.cn` | `/seedance/assets/` | `POST /seedance/assets/?Action=ListAssetGroups&Version=2024-01-01` |

素材以用户所有权、Provider、Endpoint、Provider Key 和官方资源 ID 作为路由身份。资源创建成功后粘滞到原 Provider、Endpoint 和 Provider Key；后续查询、更新、删除和预览使用同一上游身份，不进行跨 Provider 故障转移。`ProjectName` 是对外兼容 Ark 协议的逻辑字段，不作为上游响应的授权边界。

## Aether 客户端鉴权

### Aether API Key

一枚 Aether API Key 可任选以下一个请求头使用：

```http
Authorization: Bearer <AETHER_API_KEY>
X-Api-Key: <AETHER_API_KEY>
Api-Key: <AETHER_API_KEY>
```

不得通过 query 传递 Key。若同时提供多个不同凭据，Aether 返回 `400`。客户端凭据不会透传给上游。

### 用户 AK/SK

用户可以在“我的 API 密钥 → 创建 API 密钥”中选择“火山引擎 AK/SK”。Aether 生成独立 AK 与 SK，并复用同一套用户、权限、过期时间、IP、RPM、并发和计费规则。SK 仅在创建时显示一次。

用户 AK/SK 可调用原生素材 Action API 和 `/api/material-assets/**`。签名覆盖实际 method、path、query、Host、`Content-Type`、`X-Date`、`X-Content-Sha256` 和原始 body bytes；Credential scope 为 `cn-beijing/ark/request`。AK/SK 不得与 Aether API Key 请求头混用。

> 上述两种客户端鉴权都是 Aether 入口能力；直接请求火山官方 Ark 时仍只能使用火山 AK/SK。

## 原生 Action API

推荐入口：

```text
POST /?Action=<Action>&Version=2024-01-01
```

Aether 也保留以下兼容入口：

```text
POST /v3/asset-library/<Action>
POST /v3/asset-library          # Action 放在 JSON body
```

### 12 个 Action

| Action | 官方请求字段 | `Result` 关键字段 | 官方文档 |
| --- | --- | --- | --- |
| `CreateAssetGroup` | `Name` 必填；`Description`、`GroupType` 可选；`ProjectName` 默认 `default` | `Id` | [创建素材资产组合](https://www.volcengine.com/docs/82379/2318270) |
| `ListAssetGroups` | `Filter`、`Filter.GroupType`、`PageNumber`、`PageSize` 必填；支持 `Filter.GroupIds`、`Filter.Name`、`SortBy`、`SortOrder`、`ProjectName` | `TotalCount`、`Items`、`PageNumber`、`PageSize` | [查询素材资产组合列表](https://www.volcengine.com/docs/82379/2318272) |
| `GetAssetGroup` | `Id` 必填；`ProjectName` 默认 `default` | 素材组完整信息 | [查询素材资产组合信息](https://www.volcengine.com/docs/82379/2318275) |
| `UpdateAssetGroup` | `Id` 必填；可更新 `Name`、`Description`；`ProjectName` 默认 `default` | `Id` | [更新素材资产组合信息](https://www.volcengine.com/docs/82379/2318276) |
| `DeleteAssetGroup` | `Id` 必填；`ProjectName` 默认 `default` | `{}` | [删除素材资产组](https://www.volcengine.com/docs/82379/2341606) |
| `CreateAsset` | `GroupId`、`URL`、`AssetType` 必填；`Name` 可选；`ProjectName` 默认 `default` | `Id` | [创建素材资产](https://www.volcengine.com/docs/82379/2318271) |
| `ListAssets` | 支持 `Filter.GroupIds`、`Filter.GroupType`、`Filter.Name`、`Filter.Statuses`、分页、排序和 `ProjectName` | `TotalCount`、`Items`、`PageNumber`、`PageSize` | [查询素材资产列表](https://www.volcengine.com/docs/82379/2318273) |
| `GetAsset` | `Id` 必填；`ProjectName` 默认 `default` | 素材完整信息，含临时 `URL` | [查询素材资产信息](https://www.volcengine.com/docs/82379/2318274) |
| `UpdateAsset` | `Id` 必填；可更新 `Name`；`ProjectName` 默认 `default` | `Id` | [更新素材资产信息](https://www.volcengine.com/docs/82379/2318277) |
| `DeleteAsset` | `Id` 必填；`ProjectName` 默认 `default` | `{}` | [删除素材资产](https://www.volcengine.com/docs/82379/2318278) |
| `CreateVisualValidateSession` | `CallbackURL` 必填；`ProjectName` 默认 `default` | `BytedToken`、`H5Link`、`CallbackURL` | [拉起真人认证 H5](https://www.volcengine.com/docs/82379/2333587) |
| `GetVisualValidateResult` | `BytedToken` 必填；`ProjectName` 默认 `default` | `GroupId` | [获取真人 Asset Group ID](https://www.volcengine.com/docs/82379/2333588) |

单资源查询、更新和删除统一使用 `Id`；只有 `CreateAsset` 使用 `GroupId` 指定父素材组。`ProjectName` 是官方资源路由字段，不会被 Aether 丢弃；创建素材时必须与父素材组的项目一致。

### 标准响应 envelope

全部 12 个 Action（包括两个真人验证 Action）成功时都返回标准方舟结构：

```json
{
  "ResponseMetadata": {
    "RequestId": "20240514212750A1B2C3D4E5F6789ABC",
    "Action": "CreateAsset",
    "Version": "2024-01-01",
    "Service": "ark",
    "Region": "cn-beijing"
  },
  "Result": {
    "Id": "asset-20240514212750-a3f8k"
  }
}
```

错误也使用 `ResponseMetadata.Error`，并保留脱敏后的上游 `RequestId`。Aether 不再为真人验证返回非官方的顶层 `BytedToken`、`H5Link` 或 `GroupId`。

## CreateAsset：图片、视频和音频

`CreateAsset` 是异步接口。创建成功只表示已获得官方 `asset-*` ID；素材通常先处于 `Processing`，完成后变为 `Active`，失败为 `Failed`。视频处理时间通常更长。

`AssetType` 支持三种官方取值：

| `AssetType` | 格式与主要限制 |
| --- | --- |
| `Image` | jpeg、png、webp、bmp、tiff、gif、heic、heif；单张小于 30 MB |
| `Video` | mp4、mov；480p/720p/1080p/4K；2–30 秒；不超过 200 MB；24–60 FPS |
| `Audio` | wav、mp3；2–30 秒；不超过 15 MB |

三种类型均只支持公共可访问 URL，不支持 Base64 或文件 body。控制台会要求明确选择图片、视频或音频。作为入口安全约束，Aether 只接受不含用户信息、且不直接指向私网或保留地址的公网 HTTPS URL；这是 Aether 的 SSRF 防护扩展。

```bash
curl 'https://aether.example/?Action=CreateAsset&Version=2024-01-01' \
  -H 'X-Api-Key: <AETHER_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "GroupId": "group-20240514212750-b5e9m",
    "URL": "https://example.com/reference.mp4",
    "Name": "角色动作参考",
    "AssetType": "Video",
    "ProjectName": "default"
  }'
```

## 官方 ID 与 URL

- `CreateAssetGroup` 返回的 `Result.Id` 原样作为后续 Group `Id` 或 `CreateAsset.GroupId`。
- `CreateAsset` 返回的 `Result.Id` 原样作为后续 Asset `Id`。
- `ListAssetGroups`、`GetAssetGroup`、`ListAssets`、`GetAsset` 返回同一套官方 ID，不会替换为 Aether 数据库主键。
- 原生 `GetAssetGroup`、`GetAsset` 在完成用户所有权、粘滞 Provider 和官方 `Id`/`GroupId` 校验后返回上游 `Result`。其中 `ProjectName` 统一映射为请求资源在 Aether 中绑定的逻辑项目，其余官方字段不使用本地缓存重新拼装。
- `ListAssets.Items[].URL` 与 `GetAsset.Result.URL` 是方舟公共访问地址，官方有效期为 12 小时；每次查询都应以最新值为准。
- Aether 内部 ID 仅用于本地关系和审计，绝不作为原生协议响应字段，也不能作为 REST 路径或 `asset://` 请求引用。

控制台 REST 的字段含义：

| 字段 | 含义 |
| --- | --- |
| `id` | 官方素材 ID（`asset-*`） |
| `group_id` | 官方素材组 ID（`group-*`） |
| `url` | 当前方舟素材 URL；可能过期并在刷新后变化 |
| `uri` | 视频请求引用：`asset://<官方素材 ID>` |
| `preview_url` | Aether 鉴权预览代理；不是官方素材 URL，也不得提交给方舟作为素材地址 |

## 列表、分页与过滤

`ListAssetGroups` 使用官方请求结构：

```json
{
  "Filter": {
    "GroupIds": ["group-20240514212750-b5e9m"],
    "GroupType": "AIGC",
    "Name": "demo-portrait-group"
  },
  "PageNumber": 1,
  "PageSize": 10,
  "SortBy": "CreateTime",
  "SortOrder": "Desc",
  "ProjectName": "default"
}
```

`ListAssets.Filter.Statuses` 支持 `Active`、`Processing`、`Failed`。`PageSize` 最大 100；`SortOrder` 为 `Asc` 或 `Desc`。Aether 会按当前用户及其粘滞 Provider/Endpoint/Provider Key 请求上游，并只接受已绑定的官方 `Id`/`GroupId`；原生列表数据以实时上游响应为准，`ProjectName` 统一映射为 Aether 逻辑项目。这样无需信任中转返回的项目名称，也能避免共享 Provider Key 导致跨用户资源泄漏。

## 控制台 REST API

用户接口位于 `/api/material-assets`，管理员接口位于 `/api/admin/material-assets`。管理员请求必须指定目标 `user_id`，并对资源 owner 做二次校验。

- `GET/POST /groups`
- `GET/PATCH/DELETE /groups/{official_group_id}`
- `GET /assets`
- `POST /assets/url`
- `GET/PATCH/DELETE /assets/{official_asset_id}`
- `GET /assets/{official_asset_id}/preview`
- `POST /verification-sessions`
- `GET /verification-sessions/{session_id}`

REST 是控制台适配层，不改变官方资源身份。创建、列表、详情和更新响应中的 `id`、`group_id`、`url` 均来自方舟；`preview_url` 是单独标记的 Aether 能力。官方只支持 URL 创建，因此 `/assets/upload` 不提供 Base64 或本地文件直传。

## 视频引用

素材必须处于 `Active`。请求使用官方素材 ID：

```json
{
  "model": "Doubao-Seedance-2.0",
  "content": [
    {"type": "text", "text": "保持角色一致，走向镜头"},
    {
      "type": "video_url",
      "video_url": {"url": "asset://asset-20240514212750-a3f8k"},
      "role": "reference_video"
    }
  ]
}
```

图片、视频、音频分别使用 `image_url`、`video_url`、`audio_url`。跨用户、非 `Active`、已删除或 Provider 不一致的引用会在本地拒绝；素材库 Endpoint/Key 与视频 Endpoint/Key 可以不同，路由边界按 Provider 区分。

## 真人验证

1. 调用 `CreateVisualValidateSession`，传入公网可访问的 `CallbackURL` 和对应 `ProjectName`。
2. 从 `Result.H5Link` 打开验证页面，并安全保存 `Result.BytedToken`。Token 有效期为 30 分钟且只能认证一次。
3. 回调参数 `resultCode=10000` 表示认证通过。
4. 使用同一 `BytedToken` 和 `ProjectName` 调用 `GetVisualValidateResult`。
5. 从标准响应 `Result.GroupId` 获取官方 `LivenessFace` 素材组 ID。

Aether 控制台 REST 会加密保存 `BytedToken`，不向浏览器重复暴露；验证成功后按同一 Provider 与项目同步官方素材组。成功结果必须包含官方 `GroupId`，否则按上游协议错误处理，不能仅凭状态码伪造成功。

## Provider 连接测试

Provider Key 旁的“素材库基础连通性测试”使用所选 Endpoint 与 Key 调用 `ListAssetGroups`，验证地址拼接、`Action`、`Version`、认证、网络连通性和标准响应结构。该测试不创建、更新或删除资源，也不能替代 12 个 Action 的端到端验证。

## 错误映射

REST 错误字段为 `code`、`detail`、`request_id`；原生 Action API 返回官方 `ResponseMetadata.Error` envelope。Aether 不返回 Provider 密钥或未经脱敏的上游正文。

| 上游错误 | Aether 错误码 | 建议 |
| --- | --- | --- |
| `MissingParameter.*` | 保留原错误码 | 补充缺少的官方请求字段 |
| `SubscriptionRequired` | `SubscriptionRequired` | 检查火山账号套餐是否开通 |
| `SignatureDoesNotMatch` | `SignatureDoesNotMatch` | 检查 AK/SK、Region、Service、系统时间和签名 body |
| `InvalidAccessKeyId`、安全令牌或时间窗口错误 | `InvalidCredentials` | 检查 Provider 凭据与有效期 |
| `AccessDenied`、`PermissionDenied` | `AccessDenied` | 检查账号权限与项目访问权 |
| `Throttling`、`RateLimitExceeded` | `RateLimitExceeded` | 降低请求频率后重试 |

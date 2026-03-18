import type { OAuthOrganizationInfo } from '@/api/endpoints/types/provider'

type OAuthIdentityDisplayValue = {
  oauth_account_id?: string | null
  oauth_account_name?: string | null
  oauth_account_user_id?: string | null
  oauth_organizations?: OAuthOrganizationInfo[] | null
} | null | undefined

function formatOAuthIdentityShort(
  value: string | null | undefined,
  head = 8,
  tail = 6,
): string {
  const normalized = String(value || '').trim()
  if (!normalized) return ''
  if (normalized.length <= head + tail + 3) return normalized
  return `${normalized.slice(0, head)}...${normalized.slice(-tail)}`
}

function readStr(raw: unknown): string {
  return typeof raw === 'string' ? raw.trim() : ''
}

function formatOAuthAccountBadge(accountId: string): string {
  return accountId.slice(0, 8)
}

function getPrimaryOAuthOrganization(
  value: OAuthIdentityDisplayValue,
): { id: string; title: string } | null {
  const organizations: OAuthOrganizationInfo[] = Array.isArray(value?.oauth_organizations)
    ? value.oauth_organizations
    : []
  let firstWithId: OAuthOrganizationInfo | null = null

  for (let index = 0; index < organizations.length; index += 1) {
    const org = organizations[index]
    if (typeof org?.id !== 'string' || !org.id.trim()) continue
    if (!firstWithId) firstWithId = org
    if (org.is_default) {
      firstWithId = org
      break
    }
  }

  if (!firstWithId?.id) return null

  return {
    id: firstWithId.id.trim(),
    title: typeof firstWithId.title === 'string' ? firstWithId.title.trim() : '',
  }
}

export function getOAuthOrgBadge(
  value: OAuthIdentityDisplayValue,
): { id: string; label: string; title: string } | null {
  const org = getPrimaryOAuthOrganization(value)

  const accountId = readStr(value?.oauth_account_id)
  const accountName = readStr(value?.oauth_account_name)
  const accountUserId = readStr(value?.oauth_account_user_id)

  const badgeId = accountId || org?.id || ''
  const label = accountName
    || (accountId ? formatOAuthAccountBadge(accountId) : '')
    || (org?.id ? `org:${formatOAuthIdentityShort(org.id, 6, 4)}` : '')
  if (!badgeId || !label) return null

  const titleParts = [
    accountName ? `name: ${accountName}` : '',
    accountId ? `account_id: ${accountId}` : '',
    accountUserId ? `account_user_id: ${accountUserId}` : '',
    org?.id ? `org_id: ${org.id}` : '',
    org?.title ? `org_title: ${org.title}` : '',
  ].filter(Boolean)

  return {
    id: badgeId,
    label,
    title: titleParts.join(' | '),
  }
}

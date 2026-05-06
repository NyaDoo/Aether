import apiClient from './client'

export interface LicenseStatus {
  licensed: boolean
  demo_mode: boolean
  mode: 'licensed' | 'unlicensed' | 'expired' | 'invalid'
  reason: string | null
  license_id: string | null
  customer: string | null
  edition: string | null
  expires_at: string | null
  issued_at: string | null
  features: string[]
  limits: Record<string, unknown>
  instance_id: string | null
}

export const licenseApi = {
  async getStatus(): Promise<LicenseStatus> {
    const response = await apiClient.get<LicenseStatus>('/api/license/status')
    return response.data
  },

  async activate(license: string | Record<string, unknown>): Promise<LicenseStatus> {
    const response = await apiClient.post<LicenseStatus>('/api/license/activate', { license })
    return response.data
  }
}

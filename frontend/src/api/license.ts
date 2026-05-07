import apiClient from './client'

export interface LicenseStatus {
  licensed: boolean
  demo_mode: boolean
  mode: 'licensed' | 'unlicensed'
  reason: string | null
  license_id: string | null
  customer: string | null
  edition: string | null
  expires_at: string | null
  issued_at: string | null
  features: string[]
  limits: Record<string, unknown>
  instance_id: string | null
  machine_fingerprint: string | null
  can_deactivate: boolean
}

export interface LicenseMachineBinding {
  fingerprint: string
  fingerprint_version: string
}

export const licenseApi = {
  async getStatus(): Promise<LicenseStatus> {
    const response = await apiClient.get<LicenseStatus>('/api/license/status')
    return response.data
  },

  async getMachineBinding(): Promise<LicenseMachineBinding> {
    const response = await apiClient.get<LicenseMachineBinding>('/api/license/machine')
    return response.data
  },

  async activate(license: string | Record<string, unknown>): Promise<LicenseStatus> {
    const response = await apiClient.post<LicenseStatus>('/api/license/activate', { license })
    return response.data
  },

  async deactivate(): Promise<LicenseStatus> {
    const response = await apiClient.delete<LicenseStatus>('/api/license/activate')
    return response.data
  }
}

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
classifier="${script_dir}/classify-release.sh"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/aether-release-classifier.XXXXXX")"
trap 'rm -rf "${test_root}"' EXIT

sha="0123456789abcdef0123456789abcdef01234567"

assert_output() {
    local output="$1"
    local expected="$2"
    if ! grep -Fqx "${expected}" "${output}"; then
        echo "Missing expected output '${expected}' in ${output}:" >&2
        sed 's/^/  /' "${output}" >&2
        exit 1
    fi
}

run_case() {
    local name="$1"
    local event="$2"
    local ref_type="$3"
    local ref_name="$4"
    local manual_publish="$5"
    shift 5

    local output="${test_root}/${name}.out"
    GITHUB_OUTPUT="${output}" \
    GITHUB_EVENT_NAME="${event}" \
    GITHUB_REF_TYPE="${ref_type}" \
    GITHUB_REF_NAME="${ref_name}" \
    GITHUB_SHA="${sha}" \
    AETHER_MANUAL_PUBLISH="${manual_publish}" \
        bash "${classifier}"

    local expected
    for expected in "$@"; do
        assert_output "${output}" "${expected}"
    done
}

run_case synvoe-push push branch synvoe false \
    "publish=true" \
    "github_release=false" \
    "version_tag=synvoe-0123456789ab" \
    "source_ref=${sha}" \
    "branch_release=true" \
    "tag_release=false" \
    "is_synvoe=true"

run_case synvoe-manual-build workflow_dispatch branch synvoe false \
    "publish=false" \
    "version_tag=synvoe-0123456789ab" \
    "branch_release=true"

run_case synvoe-manual-publish workflow_dispatch branch synvoe true \
    "publish=true" \
    "github_release=false" \
    "branch_release=true"

run_case main-manual-publish workflow_dispatch branch main true \
    "publish=false" \
    "github_release=false" \
    "version_tag=snapshot-0123456789ab" \
    "branch_release=false" \
    "tag_release=false"

run_case stable-tag push tag v1.2.3 false \
    "publish=true" \
    "github_release=true" \
    "version_tag=v1.2.3" \
    "prerelease=false" \
    "make_latest=true" \
    "tag_release=true" \
    "is_synvoe=false"

run_case synvoe-tag push tag v1.2.3-synvoe.4 false \
    "publish=true" \
    "github_release=true" \
    "version_tag=v1.2.3-synvoe.4" \
    "prerelease=true" \
    "make_latest=false" \
    "tag_release=true" \
    "is_synvoe=true"

run_case synvoe-tag-manual-build workflow_dispatch tag v1.2.3-synvoe.4 false \
    "publish=false" \
    "github_release=false" \
    "version_tag=v1.2.3-synvoe.4" \
    "tag_release=true"

invalid_output="${test_root}/invalid-tag.out"
if GITHUB_OUTPUT="${invalid_output}" \
   GITHUB_EVENT_NAME="push" \
   GITHUB_REF_TYPE="tag" \
   GITHUB_REF_NAME="synvoe-v1.2.3" \
   GITHUB_SHA="${sha}" \
       bash "${classifier}"; then
    echo "Invalid release tag was accepted" >&2
    exit 1
fi

for invalid_tag in v01.2.3 v1.2.3-synvoe.04; do
    invalid_output="${test_root}/${invalid_tag}.out"
    if GITHUB_OUTPUT="${invalid_output}" \
       GITHUB_EVENT_NAME="push" \
       GITHUB_REF_TYPE="tag" \
       GITHUB_REF_NAME="${invalid_tag}" \
       GITHUB_SHA="${sha}" \
           bash "${classifier}" >/dev/null 2>&1; then
        echo "Non-SemVer release tag was accepted: ${invalid_tag}" >&2
        exit 1
    fi
done

assert_file_contains() {
    local path="$1"
    local expected="$2"
    if ! grep -Fq "${expected}" "${repo_root}/${path}"; then
        echo "Missing release contract '${expected}' in ${path}" >&2
        exit 1
    fi
}

assert_file_contains "install.sh" 'REPO="${AETHER_REPO:-NyaDoo/Aether}"'
assert_file_contains "install.sh" 'SOURCE_REF="${AETHER_SOURCE_REF:-synvoe}"'
assert_file_contains "install.sh" 'IMAGE_REPO="${AETHER_IMAGE_REPO:-ghcr.io/nyadoo/aether}"'
assert_file_contains "install.sh" 'VERSION="synvoe"'
assert_file_contains "docker-compose.yml" 'image: ${APP_IMAGE:-ghcr.io/nyadoo/aether:synvoe}'
assert_file_contains "docker-compose.single-node.yml" 'image: ${APP_IMAGE:-ghcr.io/nyadoo/aether:synvoe}'

compose_dir="${test_root}/compose"
install_log="${test_root}/install.log"
if ! AETHER_LANG=en \
     AETHER_INSTALL_MODE=compose-single-node \
     AETHER_COMPOSE_DIR="${compose_dir}" \
     ADMIN_PASSWORD='CiContract!123456789' \
         bash "${repo_root}/install.sh" --skip-start >"${install_log}" 2>&1; then
    cat "${install_log}" >&2
    exit 1
fi
if ! grep -Fqx 'APP_IMAGE=ghcr.io/nyadoo/aether:synvoe' "${compose_dir}/.env"; then
    echo "Compose installer did not select the synvoe image:" >&2
    grep '^APP_IMAGE=' "${compose_dir}/.env" >&2 || true
    exit 1
fi

echo "Release contract tests passed."

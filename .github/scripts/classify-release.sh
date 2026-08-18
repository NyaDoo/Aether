#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"
: "${GITHUB_REF_TYPE:?GITHUB_REF_TYPE is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"

manual_publish="${AETHER_MANUAL_PUBLISH:-false}"
short_sha="${GITHUB_SHA:0:12}"

publish="false"
github_release="false"
version_tag="snapshot-${short_sha}"
source_ref="${GITHUB_SHA}"
prerelease="false"
make_latest="false"
branch_release="false"
tag_release="false"
is_synvoe="false"

if [[ "${GITHUB_REF_TYPE}" == "tag" ]]; then
    tag="${GITHUB_REF_NAME}"
    if [[ ! "${tag}" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-(beta|rc|synvoe)\.(0|[1-9][0-9]*))?$ ]]; then
        echo "Unsupported release tag: ${tag}" >&2
        echo "Expected vX.Y.Z, vX.Y.Z-beta.N, vX.Y.Z-rc.N, or vX.Y.Z-synvoe.N." >&2
        exit 1
    fi

    version_tag="${tag}"
    source_ref="${tag}"
    tag_release="true"

    if [[ "${tag}" == *-* ]]; then
        prerelease="true"
    else
        make_latest="true"
    fi

    if [[ "${tag}" == *-synvoe.* ]]; then
        is_synvoe="true"
    fi

    if [[ "${GITHUB_EVENT_NAME}" == "push" || ( "${GITHUB_EVENT_NAME}" == "workflow_dispatch" && "${manual_publish}" == "true" ) ]]; then
        publish="true"
        github_release="true"
    fi
elif [[ "${GITHUB_REF_TYPE}" == "branch" && "${GITHUB_REF_NAME}" == "synvoe" ]]; then
    version_tag="synvoe-${short_sha}"
    source_ref="${GITHUB_SHA}"
    prerelease="true"
    branch_release="true"
    is_synvoe="true"

    if [[ "${GITHUB_EVENT_NAME}" == "push" || ( "${GITHUB_EVENT_NAME}" == "workflow_dispatch" && "${manual_publish}" == "true" ) ]]; then
        publish="true"
    fi
fi

{
    echo "publish=${publish}"
    echo "github_release=${github_release}"
    echo "version_tag=${version_tag}"
    echo "source_ref=${source_ref}"
    echo "prerelease=${prerelease}"
    echo "make_latest=${make_latest}"
    echo "branch_release=${branch_release}"
    echo "tag_release=${tag_release}"
    echo "is_synvoe=${is_synvoe}"
} >> "${GITHUB_OUTPUT}"

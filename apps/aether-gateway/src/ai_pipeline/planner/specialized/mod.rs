//! Non-matrix AI surfaces such as files and video.

pub(crate) mod files;
pub(crate) mod video;

pub(crate) use self::files::{
    maybe_build_stream_local_gemini_files_decision_payload,
    maybe_build_sync_local_gemini_files_decision_payload,
};
pub(crate) use self::video::maybe_build_sync_local_video_decision_payload;

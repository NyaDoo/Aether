use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use aes::cipher::{block_padding::Pkcs7, BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use base64::engine::general_purpose::{STANDARD, URL_SAFE, URL_SAFE_NO_PAD};
use base64::Engine as _;
use cbc::{Decryptor, Encryptor};
use hmac::{Hmac, Mac};
use pbkdf2::pbkdf2_hmac;
use sha2::{Digest, Sha256};
use uuid::Uuid;

const FERNET_VERSION: u8 = 0x80;
const HMAC_SIZE: usize = 32;
const IV_SIZE: usize = 16;
const SIGNING_KEY_SIZE: usize = 16;
const ENCRYPTION_KEY_SIZE: usize = 16;
const MIN_TOKEN_SIZE: usize = 1 + 8 + IV_SIZE + HMAC_SIZE;
const PBKDF2_ITERATIONS: u32 = 100_000;
const MAX_CACHED_DERIVED_KEYS: usize = 16;

pub const APP_SALT_SEED: &[u8] = b"aether-v1";
pub const APP_SALT_HEX: &str = "8797080a7a4b45b4810e934d1af36261";
pub const DEVELOPMENT_ENCRYPTION_KEY: &str = "dev-encryption-key-do-not-use-in-production";

static RAW_FERNET_KEY_CACHE: LazyLock<Mutex<HashMap<Box<str>, [u8; 32]>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

type Aes128CbcDec = Decryptor<aes::Aes128>;
type Aes128CbcEnc = Encryptor<aes::Aes128>;
type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, thiserror::Error)]
pub enum PythonFernetError {
    #[error("invalid Python Fernet outer base64 payload")]
    InvalidOuterBase64,
    #[error("invalid Python Fernet inner base64 payload")]
    InvalidInnerBase64,
    #[error("invalid Python Fernet token structure")]
    InvalidTokenStructure,
    #[error("unsupported Python Fernet token version: {0:#x}")]
    UnsupportedTokenVersion(u8),
    #[error("invalid Python Fernet token signature")]
    InvalidTokenSignature,
    #[error("invalid Python Fernet token padding")]
    InvalidPadding,
    #[error("invalid Python Fernet plaintext utf-8")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
}

#[derive(Debug, Clone)]
pub struct PythonFernetCompat {
    signing_key: [u8; SIGNING_KEY_SIZE],
    encryption_key: [u8; ENCRYPTION_KEY_SIZE],
}

impl PythonFernetCompat {
    pub fn from_secret(secret: &str) -> Self {
        let raw_key = raw_fernet_key(secret);
        Self::from_raw_key(raw_key)
    }

    pub fn decrypt_ciphertext(&self, ciphertext: &str) -> Result<String, PythonFernetError> {
        if ciphertext.is_empty() {
            return Ok(String::new());
        }

        let outer =
            decode_urlsafe(ciphertext).map_err(|_| PythonFernetError::InvalidOuterBase64)?;
        let inner =
            decode_urlsafe_bytes(&outer).map_err(|_| PythonFernetError::InvalidInnerBase64)?;
        let plaintext = self.decrypt_token_bytes(&inner)?;
        String::from_utf8(plaintext).map_err(PythonFernetError::InvalidUtf8)
    }

    pub fn encrypt_plaintext(&self, plaintext: &str) -> Result<String, PythonFernetError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.encrypt_token(plaintext, timestamp, *Uuid::new_v4().as_bytes())
    }

    fn from_raw_key(raw_key: [u8; 32]) -> Self {
        let mut signing_key = [0u8; SIGNING_KEY_SIZE];
        let mut encryption_key = [0u8; ENCRYPTION_KEY_SIZE];
        signing_key.copy_from_slice(&raw_key[..SIGNING_KEY_SIZE]);
        encryption_key.copy_from_slice(&raw_key[SIGNING_KEY_SIZE..]);
        Self {
            signing_key,
            encryption_key,
        }
    }

    fn decrypt_token_bytes(&self, token: &[u8]) -> Result<Vec<u8>, PythonFernetError> {
        if token.len() < MIN_TOKEN_SIZE {
            return Err(PythonFernetError::InvalidTokenStructure);
        }
        if token[0] != FERNET_VERSION {
            return Err(PythonFernetError::UnsupportedTokenVersion(token[0]));
        }

        let signed_len = token.len() - HMAC_SIZE;
        let (signed, signature) = token.split_at(signed_len);

        let mut mac = HmacSha256::new_from_slice(&self.signing_key)
            .map_err(|_| PythonFernetError::InvalidTokenSignature)?;
        mac.update(signed);
        mac.verify_slice(signature)
            .map_err(|_| PythonFernetError::InvalidTokenSignature)?;

        let iv_offset = 1 + 8;
        let ciphertext_offset = iv_offset + IV_SIZE;
        let iv = &token[iv_offset..ciphertext_offset];
        let mut ciphertext = token[ciphertext_offset..signed_len].to_vec();
        let plaintext = Aes128CbcDec::new((&self.encryption_key).into(), iv.into())
            .decrypt_padded_mut::<Pkcs7>(&mut ciphertext)
            .map_err(|_| PythonFernetError::InvalidPadding)?;
        Ok(plaintext.to_vec())
    }

    fn encrypt_token(
        &self,
        plaintext: &str,
        timestamp: u64,
        iv: [u8; IV_SIZE],
    ) -> Result<String, PythonFernetError> {
        let plaintext = plaintext.as_bytes();
        let mut padded = vec![0u8; plaintext.len() + IV_SIZE];
        padded[..plaintext.len()].copy_from_slice(plaintext);
        let ciphertext = Aes128CbcEnc::new((&self.encryption_key).into(), (&iv).into())
            .encrypt_padded_mut::<Pkcs7>(&mut padded, plaintext.len())
            .map_err(|_| PythonFernetError::InvalidPadding)?
            .to_vec();

        let mut signed = Vec::with_capacity(1 + 8 + IV_SIZE + ciphertext.len() + HMAC_SIZE);
        signed.push(FERNET_VERSION);
        signed.extend_from_slice(&timestamp.to_be_bytes());
        signed.extend_from_slice(&iv);
        signed.extend_from_slice(&ciphertext);

        let mut mac = HmacSha256::new_from_slice(&self.signing_key)
            .map_err(|_| PythonFernetError::InvalidTokenSignature)?;
        mac.update(&signed);
        let signature = mac.finalize().into_bytes();
        signed.extend_from_slice(&signature);

        let inner = URL_SAFE.encode(signed);
        Ok(URL_SAFE.encode(inner.as_bytes()))
    }
}

pub fn derive_python_fernet_key(secret: &str) -> String {
    URL_SAFE.encode(raw_fernet_key(secret))
}

pub fn decrypt_python_fernet_ciphertext(
    secret: &str,
    ciphertext: &str,
) -> Result<String, PythonFernetError> {
    PythonFernetCompat::from_secret(secret).decrypt_ciphertext(ciphertext)
}

pub fn looks_like_python_fernet_ciphertext(ciphertext: &str) -> bool {
    let ciphertext = ciphertext.trim();
    if ciphertext.is_empty() {
        return false;
    }

    let Ok(outer) = decode_urlsafe(ciphertext) else {
        return false;
    };
    let Ok(inner) = decode_urlsafe_bytes(&outer) else {
        return false;
    };

    inner.len() >= MIN_TOKEN_SIZE && inner.first().copied() == Some(FERNET_VERSION)
}

pub fn encrypt_python_fernet_plaintext(
    secret: &str,
    plaintext: &str,
) -> Result<String, PythonFernetError> {
    PythonFernetCompat::from_secret(secret).encrypt_plaintext(plaintext)
}

pub fn warm_python_fernet_secret(secret: &str) {
    let _ = raw_fernet_key(secret);
}

fn raw_fernet_key(secret: &str) -> [u8; 32] {
    if let Ok(raw_key) = decode_direct_fernet_key(secret) {
        return raw_key;
    }

    if let Some(raw_key) = RAW_FERNET_KEY_CACHE
        .lock()
        .expect("raw fernet key cache should lock")
        .get(secret)
        .copied()
    {
        return raw_key;
    }

    let mut salt = [0u8; 16];
    salt.copy_from_slice(&Sha256::digest(APP_SALT_SEED)[..16]);

    let mut raw_key = [0u8; 32];
    pbkdf2_hmac::<Sha256>(secret.as_bytes(), &salt, PBKDF2_ITERATIONS, &mut raw_key);

    let mut cache = RAW_FERNET_KEY_CACHE
        .lock()
        .expect("raw fernet key cache should lock");
    if cache.len() >= MAX_CACHED_DERIVED_KEYS && !cache.contains_key(secret) {
        cache.clear();
    }
    cache.insert(secret.into(), raw_key);
    raw_key
}

fn decode_direct_fernet_key(secret: &str) -> Result<[u8; 32], PythonFernetError> {
    let decoded = URL_SAFE
        .decode(secret)
        .or_else(|_| STANDARD.decode(secret))
        .map_err(|_| PythonFernetError::InvalidInnerBase64)?;
    let raw_key: [u8; 32] = decoded
        .as_slice()
        .try_into()
        .map_err(|_| PythonFernetError::InvalidTokenStructure)?;
    Ok(raw_key)
}

fn decode_urlsafe(value: &str) -> Result<Vec<u8>, base64::DecodeError> {
    URL_SAFE
        .decode(value)
        .or_else(|_| URL_SAFE_NO_PAD.decode(value))
}

fn decode_urlsafe_bytes(value: &[u8]) -> Result<Vec<u8>, base64::DecodeError> {
    URL_SAFE
        .decode(value)
        .or_else(|_| URL_SAFE_NO_PAD.decode(value))
}

#[cfg(test)]
mod tests {
    use super::{
        decrypt_python_fernet_ciphertext, derive_python_fernet_key,
        encrypt_python_fernet_plaintext, looks_like_python_fernet_ciphertext, PythonFernetCompat,
        PythonFernetError, APP_SALT_HEX, DEVELOPMENT_ENCRYPTION_KEY,
    };

    #[test]
    fn derives_python_pbkdf2_key_for_development_secret() {
        assert_eq!(APP_SALT_HEX, "8797080a7a4b45b4810e934d1af36261");
        assert_eq!(
            derive_python_fernet_key(DEVELOPMENT_ENCRYPTION_KEY),
            "qGVbbzTSey8Hi1DRtS6wkb2jL33pRBHXTQW-GO6qne0="
        );
    }

    #[test]
    fn passes_through_existing_fernet_key_secret() {
        let direct_key = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=";
        assert_eq!(derive_python_fernet_key(direct_key), direct_key);
    }

    #[test]
    fn decrypts_legacy_python_ciphertext_with_direct_fernet_key() {
        let direct_key = "h0Wzfv1ieDOgsmGELpKEV7qH8QpFP+lRnPW4pI0g4/M=";
        let ciphertext = "Z0FBQUFBQnA1YjlRNXowblFSVWYyVzdOT1VQbXBsQWY3RHF5UTFkaGtwbmsweWkyVTVLdExFSGYwSVE2aUNMOXhzdkNCSTlOUTRlOEFEbFlTN25jc0xFQ2xkWEpQaEkxTzhGMGhrUXFsQnlUN3JRRUdOcndWelU9";
        let plaintext = decrypt_python_fernet_ciphertext(direct_key, ciphertext)
            .expect("legacy ciphertext should decrypt");

        assert_eq!(plaintext, "sk-sanitized-regression-token");
    }

    #[test]
    fn treats_unpadded_direct_key_like_python_pbkdf2_secret() {
        let unpadded_direct_key = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY";
        assert_eq!(
            derive_python_fernet_key(unpadded_direct_key),
            "cI8mUtZz6AfpTnBy9xP48Wcp7k_r9h6jJ8jtUoc30cY="
        );
    }

    #[test]
    fn decrypts_python_compatible_outer_wrapped_ciphertext() {
        let crypto = PythonFernetCompat::from_secret(DEVELOPMENT_ENCRYPTION_KEY);
        let ciphertext = crypto
            .encrypt_token(
                "{\"api_key\":\"sk-test\",\"provider\":\"openai\"}",
                1_710_000_000,
                *b"fixed-fernet-iv!",
            )
            .expect("ciphertext should build");

        let plaintext = decrypt_python_fernet_ciphertext(DEVELOPMENT_ENCRYPTION_KEY, &ciphertext)
            .expect("ciphertext should decrypt");

        assert_eq!(
            plaintext,
            "{\"api_key\":\"sk-test\",\"provider\":\"openai\"}"
        );
    }

    #[test]
    fn detects_python_fernet_ciphertext_shape() {
        let ciphertext = encrypt_python_fernet_plaintext(DEVELOPMENT_ENCRYPTION_KEY, "sk-test")
            .expect("ciphertext should build");

        assert!(looks_like_python_fernet_ciphertext(&ciphertext));
        assert!(!looks_like_python_fernet_ciphertext("sk-plaintext-openai"));
        assert!(!looks_like_python_fernet_ciphertext(
            r#"{"headers":{"x-account-id":"acc-1"}}"#
        ));
    }

    #[test]
    fn rejects_tampered_signature() {
        let crypto = PythonFernetCompat::from_secret(DEVELOPMENT_ENCRYPTION_KEY);
        let mut ciphertext = crypto
            .encrypt_token("secret", 1_710_000_000, *b"fixed-fernet-iv!")
            .expect("ciphertext should build");
        ciphertext.replace_range(ciphertext.len() - 2.., "AA");

        let err = decrypt_python_fernet_ciphertext(DEVELOPMENT_ENCRYPTION_KEY, &ciphertext)
            .expect_err("tampered ciphertext should fail");
        assert!(matches!(
            err,
            PythonFernetError::InvalidInnerBase64
                | PythonFernetError::InvalidTokenSignature
                | PythonFernetError::InvalidPadding
        ));
    }

    #[test]
    fn encrypt_and_decrypt_round_trip() {
        let ciphertext =
            encrypt_python_fernet_plaintext(DEVELOPMENT_ENCRYPTION_KEY, "sk-live-openai")
                .expect("ciphertext should build");
        let plaintext = decrypt_python_fernet_ciphertext(DEVELOPMENT_ENCRYPTION_KEY, &ciphertext)
            .expect("ciphertext should decrypt");
        assert_eq!(plaintext, "sk-live-openai");
    }
}

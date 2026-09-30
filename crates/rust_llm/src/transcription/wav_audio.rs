//! Port of `lib/ruby_llm/transcription/wav_audio.rb` (`RubyLLM::Transcription::WavAudio`): the
//! RIFF/WAVE reader WebSocket transcription uses to learn the PCM format and send raw samples.

use crate::error::{Error, Result};

/// The format and sample data of a WAV file.
#[derive(Debug, Clone)]
pub(crate) struct WavAudio {
    pub(crate) data: Vec<u8>,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) bits_per_sample: u16,
    /// The WAVE format tag: 1 is PCM, 6 A-law, 7 mu-law.
    pub(crate) encoding: u16,
}

const UNSPECIFIED: u32 = 0xFFFF_FFFF;

fn invalid(message: &str) -> Error {
    Error::Argument(message.into())
}

fn u32_at(content: &[u8], offset: usize) -> Option<u32> {
    let bytes = content.get(offset..offset + 4)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

impl WavAudio {
    /// `WavAudio.new(content)`.
    pub(crate) fn new(content: &[u8]) -> Result<WavAudio> {
        let content = wav_content(content)?;
        let mut format: Option<(u16, u16, u32, u16)> = None;
        let mut data: Option<Vec<u8>> = None;
        // `parse_chunks`.
        let len = content.len() as u64;
        let mut offset: u64 = 12;
        while offset + 8 <= len {
            let at = offset as usize;
            let name = &content[at..at + 4];
            let mut length = u64::from(u32_at(content, at + 4).unwrap_or(0));
            if name == b"data" && length == u64::from(UNSPECIFIED) {
                length = len - offset - 8;
            }
            // `validate_chunk_length`: the body must hold every byte the header declares.
            let start = offset + 8;
            if start + length > len {
                return Err(invalid("WAV file contains a truncated chunk"));
            }
            let body = &content[start as usize..(start + length) as usize];
            if name == b"fmt " {
                format = Some(parse_format(body)?);
            }
            if name == b"data" {
                data = Some(body.to_vec());
            }
            offset += 8 + length + (length % 2);
        }
        if offset != len {
            return Err(invalid("WAV file contains a truncated chunk or padding"));
        }
        match (data, format) {
            (Some(data), Some((encoding, channels, sample_rate, bits_per_sample))) => {
                Ok(WavAudio {
                    data,
                    sample_rate,
                    channels,
                    bits_per_sample,
                    encoding,
                })
            }
            _ => Err(invalid(
                "WAV file must contain audio data and format information",
            )),
        }
    }

    /// `duration`: seconds of audio, from the byte rate the format implies.
    pub(crate) fn duration(&self) -> f64 {
        let byte_rate = u64::from(self.sample_rate)
            * u64::from(self.channels)
            * u64::from(self.bits_per_sample)
            / 8;
        self.data.len() as f64 / byte_rate as f64
    }
}

/// `wav_content`: the RIFF container, cut to its declared size unless the size is unspecified.
fn wav_content(content: &[u8]) -> Result<&[u8]> {
    if !content.starts_with(b"RIFF") || content.get(8..12) != Some(b"WAVE") {
        return Err(invalid(
            "This streaming transcription endpoint requires a WAV file",
        ));
    }
    let declared = u32_at(content, 4).unwrap_or(0);
    if declared == UNSPECIFIED {
        return Ok(content);
    }
    let size = declared as usize + 8;
    if content.len() < size {
        return Err(invalid("WAV file is truncated"));
    }
    Ok(&content[..size])
}

/// `parse_format`: the `vvVVvv` fields of a `fmt ` chunk.
fn parse_format(data: &[u8]) -> Result<(u16, u16, u32, u16)> {
    if data.len() < 16 {
        return Err(invalid("WAV file contains an invalid audio format"));
    }
    let u16_at = |i: usize| u16::from_le_bytes([data[i], data[i + 1]]);
    let encoding = u16_at(0);
    let channels = u16_at(2);
    let sample_rate = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let bits_per_sample = u16_at(14);
    if channels == 0 || sample_rate == 0 || bits_per_sample == 0 {
        return Err(invalid("WAV file contains an invalid audio format"));
    }
    Ok((encoding, channels, sample_rate, bits_per_sample))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(chunks: &[u8], declared_size: Option<u32>) -> Vec<u8> {
        let mut content = b"WAVE".to_vec();
        content.extend_from_slice(chunks);
        let size = declared_size.unwrap_or(content.len() as u32);
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&content);
        out
    }

    fn chunk(name: &[u8], data: &[u8]) -> Vec<u8> {
        let mut out = name.to_vec();
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            out.push(0);
        }
        out
    }

    fn format() -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&24_000u32.to_le_bytes());
        fmt.extend_from_slice(&48_000u32.to_le_bytes());
        fmt.extend_from_slice(&2u16.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());
        chunk(b"fmt ", &fmt)
    }

    fn message(result: Result<WavAudio>) -> String {
        match result {
            Err(Error::Argument(message)) => message,
            other => panic!("expected an argument error, got {other:?}"),
        }
    }

    // spec: transcription/wav_audio_spec.rb:17 reads PCM format and audio after padded metadata chunks
    #[test]
    fn reads_pcm_format_and_audio_after_padded_metadata_chunks() {
        let chunks = [chunk(b"JUNK", b"x"), format(), chunk(b"data", b"\x00\x01")].concat();
        let audio = WavAudio::new(&wav(&chunks, None)).unwrap();

        assert_eq!(audio.data, b"\x00\x01");
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.bits_per_sample, 16);
        assert_eq!(audio.encoding, 1);
        assert_eq!(audio.duration(), 1.0 / 24_000.0);
    }

    // spec: transcription/wav_audio_spec.rb:28 reads WAV recordings with unspecified RIFF and data lengths
    #[test]
    fn reads_wav_recordings_with_unspecified_riff_and_data_lengths() {
        let chunks = [
            format(),
            b"data".to_vec(),
            UNSPECIFIED.to_le_bytes().to_vec(),
            b"\x00\x01".to_vec(),
        ]
        .concat();
        let audio = WavAudio::new(&wav(&chunks, Some(UNSPECIFIED))).unwrap();

        assert_eq!(audio.data, b"\x00\x01");
    }

    // spec: transcription/wav_audio_spec.rb:34 rejects truncated data and missing odd-byte padding
    #[test]
    fn rejects_truncated_data_and_missing_odd_byte_padding() {
        let truncated = [
            format(),
            b"data".to_vec(),
            4u32.to_le_bytes().to_vec(),
            b"ab".to_vec(),
        ];
        assert!(
            message(WavAudio::new(&wav(&truncated.concat(), None))).contains("truncated chunk")
        );

        let unpadded = [
            format(),
            b"JUNK".to_vec(),
            1u32.to_le_bytes().to_vec(),
            b"x".to_vec(),
        ];
        assert!(message(WavAudio::new(&wav(&unpadded.concat(), None))).contains("padding"));
    }

    // spec: transcription/wav_audio_spec.rb:41 rejects an incomplete RIFF container, short format or missing audio
    #[test]
    fn rejects_an_incomplete_riff_container_short_format_or_missing_audio() {
        assert!(message(WavAudio::new(&wav(&format(), Some(1000)))).contains("truncated"));
        assert!(
            message(WavAudio::new(&wav(&chunk(b"fmt ", b"x"), None)))
                .contains("invalid audio format")
        );
        assert!(message(WavAudio::new(&wav(&format(), None))).contains("must contain audio"));
    }

    #[test]
    fn rejects_content_that_is_not_a_wav_file() {
        assert!(message(WavAudio::new(b"ID3 not a wave file")).contains("requires a WAV file"));
    }

    #[test]
    fn reads_the_ruby_fixture_as_mono_24khz_pcm() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ruby.wav"
        ))
        .unwrap();
        let audio = WavAudio::new(&bytes).unwrap();
        assert_eq!(
            (
                audio.encoding,
                audio.channels,
                audio.sample_rate,
                audio.bits_per_sample
            ),
            (1, 1, 24_000, 16)
        );
        assert!((audio.duration() - 3.7).abs() < 1e-9);
    }
}

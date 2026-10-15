//! Detect how the active terminal renders East Asian Ambiguous characters.
//!
//! The probe runs only after raw mode and the alternate screen are active. It
//! writes one ambiguous-width scalar at a known location, asks the terminal for
//! its cursor position (DSR), and erases the probe before the first frame.

use std::io::{self, IsTerminal, Write};
#[cfg(any(unix, test))]
use std::time::{Duration, Instant};

use crossterm::event::KeyEvent;
#[cfg(any(unix, test))]
use crossterm::event::{KeyCode, KeyModifiers};

#[cfg(any(unix, test))]
const PROBE_TIMEOUT: Duration = Duration::from_millis(225);
const PROBE_PREFIX: &[u8] = b"\x1b[1;1H\x1b[2K";
const PROBE_GLYPH: &[u8] = "·".as_bytes();
#[cfg(any(unix, test))]
const DSR_QUERY: &[u8] = b"\x1b[6n";
const PROBE_CLEANUP: &[u8] = b"\x1b[1;1H\x1b[2K\x1b[1;1H";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum AmbientWidth {
    Narrow,
    Wide,
    #[default]
    Unknown,
}

#[derive(Debug, Default)]
pub(crate) struct ProbeResult {
    pub width: AmbientWidth,
    pub pending_keys: Vec<KeyEvent>,
}

/// Probe the real terminal attached to stdin/stdout.
///
/// Any unsupported terminal, I/O error, or missing/invalid DSR reply degrades
/// to `Unknown`. The deadline is deliberately short so a terminal multiplexer
/// which does not pass DSR replies through cannot delay startup noticeably.
pub(crate) fn probe_ambient_width(output: &mut impl Write) -> ProbeResult {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return ProbeResult::default();
    }

    probe_tty(output).unwrap_or_default()
}

#[cfg(unix)]
fn probe_tty(output: &mut impl Write) -> io::Result<ProbeResult> {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    // Use the controlling terminal rather than assuming fd 0 is suitable for
    // direct reads. The IsTerminal checks above still ensure redirected TUI
    // launches do not emit a probe at all.
    let tty = OpenOptions::new().read(true).open("/dev/tty")?;
    let nonblocking = NonblockingGuard::new(tty.as_raw_fd())?;

    let query_result = output
        .write_all(PROBE_PREFIX)
        .and_then(|()| output.write_all(PROBE_GLYPH))
        .and_then(|()| output.write_all(DSR_QUERY))
        .and_then(|()| output.flush());

    let read = if query_result.is_ok() {
        read_dsr_with_timeout(PROBE_TIMEOUT, |remaining| {
            poll_and_read(nonblocking.fd, remaining)
        })
    } else {
        DsrRead::default()
    };

    // Always remove the probe, even when the terminal ignored DSR or returned
    // an unexpected position. A full TUI frame has not been drawn yet.
    let cleanup_result = output
        .write_all(PROBE_CLEANUP)
        .and_then(|()| output.flush());
    query_result?;
    cleanup_result?;

    Ok(ProbeResult {
        width: classify_position(read.position),
        pending_keys: decode_pending_keys(&read.pending),
    })
}

#[cfg(unix)]
struct NonblockingGuard {
    fd: std::os::fd::RawFd,
    original_flags: libc::c_int,
}

#[cfg(unix)]
impl NonblockingGuard {
    fn new(fd: std::os::fd::RawFd) -> io::Result<Self> {
        // SAFETY: `fd` is owned by the live File in `probe_tty`. fcntl neither
        // takes ownership nor retains the pointer/value passed to it.
        let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if original_flags == -1 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: same live descriptor; flags are the existing valid flags plus
        // O_NONBLOCK and are restored when this guard drops.
        if unsafe { libc::fcntl(fd, libc::F_SETFL, original_flags | libc::O_NONBLOCK) } == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { fd, original_flags })
    }
}

#[cfg(unix)]
impl Drop for NonblockingGuard {
    fn drop(&mut self) {
        // SAFETY: the File outlives this guard because it is declared first in
        // `probe_tty`. Restoration is best effort on an already-failing path.
        let _ = unsafe { libc::fcntl(self.fd, libc::F_SETFL, self.original_flags) };
    }
}

#[cfg(unix)]
fn poll_and_read(fd: std::os::fd::RawFd, remaining: Duration) -> io::Result<Option<Vec<u8>>> {
    let timeout_ms = remaining.as_millis().clamp(1, i32::MAX as u128) as i32;
    let mut descriptor = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };

    // SAFETY: `descriptor` is a valid one-element pollfd array for the
    // duration of this call. `poll` does not retain its pointer.
    let ready = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
    if ready > 0 {
        let mut bytes = [0_u8; 64];
        // SAFETY: `bytes` is writable for its full length and `fd` is a
        // live nonblocking controlling-terminal descriptor.
        let count = unsafe { libc::read(fd, bytes.as_mut_ptr().cast(), bytes.len()) };
        if count > 0 {
            return Ok(Some(bytes[..count as usize].to_vec()));
        }
        if count == 0 {
            return Ok(None);
        }
        return Err(io::Error::last_os_error());
    }
    if ready == 0 {
        return Ok(None);
    }
    // Do not restart with the original timeout after EINTR: doing so could
    // exceed the probe's absolute deadline under repeated signals.
    Err(io::Error::last_os_error())
}

#[cfg(windows)]
fn probe_tty(output: &mut impl Write) -> io::Result<ProbeResult> {
    let query_result = output
        .write_all(PROBE_PREFIX)
        .and_then(|()| output.write_all(PROBE_GLYPH))
        .and_then(|()| output.flush());

    // Windows consoles expose the cursor position synchronously through the
    // console API, so crossterm does not need an input-side DSR round trip.
    let position = if query_result.is_ok() {
        crossterm::cursor::position()
            .ok()
            .map(|(column, row)| (row.saturating_add(1), column.saturating_add(1)))
    } else {
        None
    };

    let cleanup_result = output
        .write_all(PROBE_CLEANUP)
        .and_then(|()| output.flush());
    query_result?;
    cleanup_result?;
    Ok(ProbeResult {
        width: classify_position(position),
        pending_keys: Vec::new(),
    })
}

#[cfg(not(any(unix, windows)))]
fn probe_tty(_output: &mut impl Write) -> io::Result<ProbeResult> {
    Ok(ProbeResult::default())
}

fn classify_position(position: Option<(u16, u16)>) -> AmbientWidth {
    match position {
        Some((1, 2)) => AmbientWidth::Narrow,
        Some((1, 3)) => AmbientWidth::Wide,
        _ => AmbientWidth::Unknown,
    }
}

#[cfg(any(unix, test))]
#[derive(Debug, Default, PartialEq, Eq)]
struct DsrRead {
    position: Option<(u16, u16)>,
    pending: Vec<u8>,
}

#[cfg(any(unix, test))]
fn read_dsr_with_timeout<F>(timeout: Duration, mut next_chunk: F) -> DsrRead
where
    F: FnMut(Duration) -> io::Result<Option<Vec<u8>>>,
{
    let Some(deadline) = Instant::now().checked_add(timeout) else {
        return DsrRead::default();
    };
    let mut parser = DsrParser::default();

    loop {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return DsrRead {
                position: None,
                pending: parser.pending,
            };
        };
        if remaining.is_zero() {
            return DsrRead {
                position: None,
                pending: parser.pending,
            };
        }

        match next_chunk(remaining) {
            Ok(Some(bytes)) => {
                if let Some((position, response_range)) = parser.push(&bytes) {
                    parser.pending.drain(response_range);
                    return DsrRead {
                        position: Some(position),
                        pending: parser.pending,
                    };
                }
            }
            Ok(None) | Err(_) => {
                return DsrRead {
                    position: None,
                    pending: parser.pending,
                };
            }
        }
    }
}

#[cfg(any(unix, test))]
#[derive(Default)]
struct DsrParser {
    pending: Vec<u8>,
}

#[cfg(any(unix, test))]
impl DsrParser {
    fn push(&mut self, bytes: &[u8]) -> Option<((u16, u16), std::ops::Range<usize>)> {
        self.pending.extend_from_slice(bytes);
        find_dsr(&self.pending)
    }
}

#[cfg(test)]
fn parse_dsr(bytes: &[u8]) -> Option<(u16, u16)> {
    find_dsr(bytes).map(|(position, _)| position)
}

#[cfg(any(unix, test))]
fn find_dsr(bytes: &[u8]) -> Option<((u16, u16), std::ops::Range<usize>)> {
    for start in 0..bytes.len().saturating_sub(1) {
        if bytes.get(start..start + 2) != Some(b"\x1b[") {
            continue;
        }

        let mut cursor = start + 2;
        let row_start = cursor;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == row_start || bytes.get(cursor) != Some(&b';') {
            continue;
        }
        let Some(row) = parse_u16_ascii(&bytes[row_start..cursor]) else {
            continue;
        };

        cursor += 1;
        let column_start = cursor;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == column_start || bytes.get(cursor) != Some(&b'R') {
            continue;
        }
        let Some(column) = parse_u16_ascii(&bytes[column_start..cursor]) else {
            continue;
        };
        return Some(((row, column), start..cursor + 1));
    }
    None
}

#[cfg(any(unix, test))]
fn parse_u16_ascii(bytes: &[u8]) -> Option<u16> {
    bytes.iter().try_fold(0_u16, |value, byte| {
        value
            .checked_mul(10)?
            .checked_add(u16::from(byte.checked_sub(b'0')?))
    })
}

/// Convert non-DSR bytes read during the probe back into the key events the
/// event loop would otherwise have received. The probe is short, but startup
/// typing must not disappear merely because a cursor report shared its read.
#[cfg(any(unix, test))]
fn decode_pending_keys(bytes: &[u8]) -> Vec<KeyEvent> {
    let mut keys = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\x1b' {
            if let Some((key, consumed)) = decode_escape_key(&bytes[index..]) {
                if let Some(key) = key {
                    keys.push(key);
                }
                index += consumed;
                continue;
            }
            keys.push(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
            index += 1;
            continue;
        }

        let byte = bytes[index];
        let event = match byte {
            b'\r' | b'\n' => Some(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            b'\t' => Some(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
            0x7f | 0x08 => Some(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE)),
            1..=26 => Some(KeyEvent::new(
                KeyCode::Char(char::from(b'a' + byte - 1)),
                KeyModifiers::CONTROL,
            )),
            0x20..=0x7e => Some(KeyEvent::new(
                KeyCode::Char(char::from(byte)),
                KeyModifiers::NONE,
            )),
            _ => None,
        };
        if let Some(event) = event {
            keys.push(event);
            index += 1;
            continue;
        }

        let tail = &bytes[index..];
        match std::str::from_utf8(tail) {
            Ok(text) => {
                for ch in text.chars() {
                    keys.push(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
                }
                break;
            }
            Err(error) if error.valid_up_to() > 0 => {
                let valid = &tail[..error.valid_up_to()];
                let text = std::str::from_utf8(valid)
                    .expect("valid_up_to always identifies a valid UTF-8 prefix");
                for ch in text.chars() {
                    keys.push(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
                }
                index += error.valid_up_to();
            }
            Err(error) => {
                let skip = error.error_len().unwrap_or(1).min(tail.len());
                index += skip;
            }
        }
    }
    keys
}

/// Decode one escape-prefixed key. `None` as the event means a terminal control
/// wrapper (currently bracketed-paste start/end) that should be removed while
/// preserving the text between its two markers.
#[cfg(any(unix, test))]
fn decode_escape_key(bytes: &[u8]) -> Option<(Option<KeyEvent>, usize)> {
    let simple = |code, consumed| Some((Some(KeyEvent::new(code, KeyModifiers::NONE)), consumed));
    match bytes {
        [0x1b, b'[', b'A', ..] | [0x1b, b'O', b'A', ..] => simple(KeyCode::Up, 3),
        [0x1b, b'[', b'B', ..] | [0x1b, b'O', b'B', ..] => simple(KeyCode::Down, 3),
        [0x1b, b'[', b'C', ..] | [0x1b, b'O', b'C', ..] => simple(KeyCode::Right, 3),
        [0x1b, b'[', b'D', ..] | [0x1b, b'O', b'D', ..] => simple(KeyCode::Left, 3),
        [0x1b, b'[', b'H', ..] | [0x1b, b'O', b'H', ..] => simple(KeyCode::Home, 3),
        [0x1b, b'[', b'F', ..] | [0x1b, b'O', b'F', ..] => simple(KeyCode::End, 3),
        [0x1b, b'[', b'2', b'~', ..] => simple(KeyCode::Insert, 4),
        [0x1b, b'[', b'3', b'~', ..] => simple(KeyCode::Delete, 4),
        [0x1b, b'[', b'5', b'~', ..] => simple(KeyCode::PageUp, 4),
        [0x1b, b'[', b'6', b'~', ..] => simple(KeyCode::PageDown, 4),
        [0x1b, b'[', b'2', b'0', b'0', b'~', ..] | [0x1b, b'[', b'2', b'0', b'1', b'~', ..] => {
            Some((None, 6))
        }
        [0x1b, byte, ..] if byte.is_ascii_graphic() => Some((
            Some(KeyEvent::new(
                KeyCode::Char(char::from(*byte)),
                KeyModifiers::ALT,
            )),
            2,
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cursor_position_report_amid_noise() {
        assert_eq!(parse_dsr(b"key\x1b[12;34Rtail"), Some((12, 34)));
    }

    #[test]
    fn parses_fragmented_cursor_position_report() {
        let mut parser = DsrParser::default();
        assert_eq!(parser.push(b"\x1b[1;"), None);
        assert_eq!(parser.push(b"3R"), Some(((1, 3), 0..6)));
    }

    #[test]
    fn rejects_malformed_or_overflowing_cursor_reports() {
        assert_eq!(parse_dsr(b"\x1b[;2R"), None);
        assert_eq!(parse_dsr(b"\x1b[1;xR"), None);
        assert_eq!(parse_dsr(b"\x1b[999999;2R"), None);
        assert_eq!(parse_dsr(b"\x1b[999999;2Rnoise\x1b[1;2R"), Some((1, 2)));
    }

    #[test]
    fn preserves_keys_surrounding_the_cursor_report() {
        let mut chunks = vec![b"q\x1b[1;".to_vec(), b"2R\x1b[B".to_vec()].into_iter();
        let read = read_dsr_with_timeout(Duration::from_millis(20), |_| Ok(chunks.next()));
        assert_eq!(read.position, Some((1, 2)));
        let keys = decode_pending_keys(&read.pending);
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].code, KeyCode::Char('q'));
        assert_eq!(keys[1].code, KeyCode::Down);
    }

    #[test]
    fn preserves_control_utf8_alt_and_pasted_text() {
        let keys = decode_pending_keys(b"\x03\x1bx\x1b[200~h\xc3\xa9\x1b[201~");
        assert_eq!(
            keys[0],
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
        );
        assert_eq!(
            keys[1],
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::ALT)
        );
        assert_eq!(keys[2].code, KeyCode::Char('h'));
        assert_eq!(keys[3].code, KeyCode::Char('é'));
    }

    #[test]
    fn recognizes_narrow_and_wide_probe_positions() {
        assert_eq!(classify_position(Some((1, 2))), AmbientWidth::Narrow);
        assert_eq!(classify_position(Some((1, 3))), AmbientWidth::Wide);
        assert_eq!(classify_position(Some((2, 3))), AmbientWidth::Unknown);
    }

    #[test]
    fn timeout_is_forwarded_as_a_strict_bound() {
        let timeout = Duration::from_millis(225);
        let mut observed = None;
        let result = read_dsr_with_timeout(timeout, |remaining| {
            observed = Some(remaining);
            Ok(None)
        });

        assert_eq!(result, DsrRead::default());
        assert!(observed.is_some_and(|remaining| remaining <= timeout));
    }

    #[test]
    fn incomplete_replies_stop_at_the_absolute_timeout() {
        let started = Instant::now();
        let mut polls = 0;
        let result = read_dsr_with_timeout(Duration::from_millis(2), |_| {
            polls += 1;
            Ok(Some(Vec::new()))
        });

        assert_eq!(result, DsrRead::default());
        assert!(polls > 0);
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn zero_timeout_does_not_attempt_a_read() {
        let mut called = false;
        let result = read_dsr_with_timeout(Duration::ZERO, |_| {
            called = true;
            Ok(None)
        });

        assert_eq!(result, DsrRead::default());
        assert!(!called);
    }
}

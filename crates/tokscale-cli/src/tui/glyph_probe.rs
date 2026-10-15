//! Detect how the active terminal renders East Asian Ambiguous characters.
//!
//! The probe runs only after raw mode and the alternate screen are active. It
//! writes one ambiguous-width scalar at a known location, asks the terminal for
//! its cursor position (DSR), and erases the probe before the first frame.

use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

const PROBE_TIMEOUT: Duration = Duration::from_millis(225);
const PROBE_PREFIX: &[u8] = b"\x1b[1;1H\x1b[2K";
const PROBE_GLYPH: &[u8] = "·".as_bytes();
const DSR_QUERY: &[u8] = b"\x1b[6n";
const PROBE_CLEANUP: &[u8] = b"\x1b[1;1H\x1b[2K\x1b[1;1H";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum AmbientWidth {
    Narrow,
    Wide,
    #[default]
    Unknown,
}

/// Probe the real terminal attached to stdin/stdout.
///
/// Any unsupported terminal, I/O error, or missing/invalid DSR reply degrades
/// to `Unknown`. The deadline is deliberately short so a terminal multiplexer
/// which does not pass DSR replies through cannot delay startup noticeably.
pub(crate) fn probe_ambient_width(output: &mut impl Write) -> AmbientWidth {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return AmbientWidth::Unknown;
    }

    probe_tty(output).unwrap_or(AmbientWidth::Unknown)
}

#[cfg(unix)]
fn probe_tty(output: &mut impl Write) -> io::Result<AmbientWidth> {
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

    let position = if query_result.is_ok() {
        read_dsr_with_timeout(PROBE_TIMEOUT, |remaining| {
            poll_and_read(nonblocking.fd, remaining)
        })
    } else {
        None
    };

    // Always remove the probe, even when the terminal ignored DSR or returned
    // an unexpected position. A full TUI frame has not been drawn yet.
    let cleanup_result = output
        .write_all(PROBE_CLEANUP)
        .and_then(|()| output.flush());
    query_result?;
    cleanup_result?;

    Ok(classify_position(position))
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
fn probe_tty(output: &mut impl Write) -> io::Result<AmbientWidth> {
    let query_result = output
        .write_all(PROBE_PREFIX)
        .and_then(|()| output.write_all(PROBE_GLYPH))
        .and_then(|()| output.flush());

    // Windows consoles expose the cursor position synchronously through the
    // console API, so crossterm does not need an input-side DSR round trip.
    let position = query_result.as_ref().ok().and_then(|()| {
        crossterm::cursor::position()
            .ok()
            .map(|(column, row)| (row.saturating_add(1), column.saturating_add(1)))
    });

    let cleanup_result = output
        .write_all(PROBE_CLEANUP)
        .and_then(|()| output.flush());
    query_result?;
    cleanup_result?;
    Ok(classify_position(position))
}

#[cfg(not(any(unix, windows)))]
fn probe_tty(_output: &mut impl Write) -> io::Result<AmbientWidth> {
    Ok(AmbientWidth::Unknown)
}

fn classify_position(position: Option<(u16, u16)>) -> AmbientWidth {
    match position {
        Some((1, 2)) => AmbientWidth::Narrow,
        Some((1, 3)) => AmbientWidth::Wide,
        _ => AmbientWidth::Unknown,
    }
}

fn read_dsr_with_timeout<F>(timeout: Duration, mut next_chunk: F) -> Option<(u16, u16)>
where
    F: FnMut(Duration) -> io::Result<Option<Vec<u8>>>,
{
    let deadline = Instant::now().checked_add(timeout)?;
    let mut parser = DsrParser::default();

    loop {
        let remaining = deadline.checked_duration_since(Instant::now())?;
        if remaining.is_zero() {
            return None;
        }

        match next_chunk(remaining) {
            Ok(Some(bytes)) => {
                if let Some(position) = parser.push(&bytes) {
                    return Some(position);
                }
            }
            Ok(None) | Err(_) => return None,
        }
    }
}

#[derive(Default)]
struct DsrParser {
    pending: Vec<u8>,
}

impl DsrParser {
    fn push(&mut self, bytes: &[u8]) -> Option<(u16, u16)> {
        self.pending.extend_from_slice(bytes);
        // A DSR reply is tiny. Bound noise from keys or unrelated terminal
        // replies while retaining enough tail for a fragmented response.
        if self.pending.len() > 128 {
            self.pending.drain(..self.pending.len() - 128);
        }
        parse_dsr(&self.pending)
    }
}

fn parse_dsr(bytes: &[u8]) -> Option<(u16, u16)> {
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
        return Some((row, column));
    }
    None
}

fn parse_u16_ascii(bytes: &[u8]) -> Option<u16> {
    bytes.iter().try_fold(0_u16, |value, byte| {
        value
            .checked_mul(10)?
            .checked_add(u16::from(byte.checked_sub(b'0')?))
    })
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
        assert_eq!(parser.push(b"3R"), Some((1, 3)));
    }

    #[test]
    fn rejects_malformed_or_overflowing_cursor_reports() {
        assert_eq!(parse_dsr(b"\x1b[;2R"), None);
        assert_eq!(parse_dsr(b"\x1b[1;xR"), None);
        assert_eq!(parse_dsr(b"\x1b[999999;2R"), None);
        assert_eq!(parse_dsr(b"\x1b[999999;2Rnoise\x1b[1;2R"), Some((1, 2)));
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

        assert_eq!(result, None);
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

        assert_eq!(result, None);
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

        assert_eq!(result, None);
        assert!(!called);
    }
}

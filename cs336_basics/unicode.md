# Unicode written answers

## unicode1

**(a)** `chr(0)` returns the Unicode null character, U+0000.

**(b)** Its representation is `'\x00'`, while printing emits the actual null character, which normally has no visible glyph.

**(c)** A null character remains part of a Python string and does not terminate it, so `"this is a test" + chr(0) + "string"` normally looks like `this is a teststring` when printed, although its representation exposes the embedded `\x00`.

## unicode2

**(a)** UTF-8 preserves ASCII as single bytes and usually uses less space for ASCII-heavy text than UTF-16 or UTF-32, shortening the initial byte sequences used for training. It is also byte-oriented and independent of endianness; it is not always the smallest encoding, since many CJK characters take three bytes in UTF-8 but only two in UTF-16.

**(b)** The input `b'\xc3\xa9'` represents `é`, but decoding each byte separately raises `UnicodeDecodeError` because a multibyte UTF-8 character must be decoded as a complete sequence.

**(c)** `b'\xc3\x28'` is invalid UTF-8 because `0xC3` requires a continuation byte in the range `0x80`–`0xBF`, whereas `0x28` is outside that range.

//! Large, aligned, huge-page-backed buffers for state vectors.
//!
//! A 30-qubit state is 8–16 GiB. Allocating it with `mmap` lets us ask the
//! kernel for transparent huge pages (2 MiB instead of 4 KiB: 512x fewer TLB
//! entries for the same memory) and pre-fault the pages in parallel, so the
//! first gate does not pay for millions of page faults on one thread.

use std::ptr::NonNull;
use std::slice;

use rayon::prelude::*;

/// Alignment and rounding unit of every buffer: one transparent huge page.
pub const HUGE_PAGE: usize = 2 << 20;

/// A zero-initialised, 2 MiB-aligned buffer of plain-old-data values.
pub struct Buffer<T: Copy + Send + Sync> {
    ptr: NonNull<T>,
    len: usize,
    mapped_bytes: usize,
}

// SAFETY: the buffer owns its mapping exclusively, like a Vec<T>.
unsafe impl<T: Copy + Send + Sync> Send for Buffer<T> {}
unsafe impl<T: Copy + Send + Sync> Sync for Buffer<T> {}

impl<T: Copy + Send + Sync> Buffer<T> {
    /// Allocate `len` zeroed elements, aligned to [`HUGE_PAGE`], and fault the
    /// pages in from the current rayon pool so they are spread over its
    /// threads.
    ///
    /// `T` must be a type for which all-zero bytes is a valid value (floats,
    /// integers and arrays of them).
    pub fn zeroed(len: usize) -> std::io::Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| std::io::Error::other("buffer size overflows usize"))?;
        let mapped_bytes = bytes.max(1).div_ceil(HUGE_PAGE) * HUGE_PAGE;
        let aligned = allocate(mapped_bytes)?;
        let buffer = Buffer {
            ptr: NonNull::new(aligned as *mut T).expect("allocation returned null"),
            len,
            mapped_bytes,
        };
        buffer.first_touch();
        Ok(buffer)
    }

    /// Write one byte per 4 KiB page from the rayon pool. Fresh anonymous pages
    /// are already zero, so this only commits them, in parallel.
    fn first_touch(&self) {
        let base = self.ptr.as_ptr() as usize;
        let pages = self.mapped_bytes / 4096;
        (0..pages)
            .into_par_iter()
            .with_min_len(512)
            .for_each(|page| {
                // SAFETY: each page is inside the mapping; volatile keeps the
                // store, and writing 0 over 0 preserves the zeroed contents.
                unsafe { std::ptr::write_volatile((base + page * 4096) as *mut u8, 0) };
            });
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the mapping holds `len` initialised (zeroed or written) values.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, and `&mut self` guarantees exclusive access.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T: Copy + Send + Sync> Drop for Buffer<T> {
    fn drop(&mut self) {
        // SAFETY: releasing exactly the range `allocate` returned.
        unsafe { release(self.ptr.as_ptr() as *mut u8, self.mapped_bytes) };
    }
}

/// `bytes` (a multiple of [`HUGE_PAGE`]) of zeroed memory aligned to
/// [`HUGE_PAGE`]: an anonymous mapping on Unix, with transparent huge pages
/// requested on Linux.
#[cfg(unix)]
fn allocate(bytes: usize) -> std::io::Result<*mut u8> {
    // Over-allocate by one huge page so the start can be aligned to 2 MiB;
    // mmap only guarantees 4 KiB alignment.
    let request = bytes + HUGE_PAGE;
    #[cfg(target_os = "linux")]
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
    #[cfg(not(target_os = "linux"))]
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
    // SAFETY: anonymous private mapping with no address hint.
    let raw = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            request,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    let start = raw as usize;
    let aligned = start.next_multiple_of(HUGE_PAGE);
    // Return the unaligned head and the unused tail to the kernel.
    // SAFETY: both ranges lie inside the mapping created above.
    unsafe {
        if aligned > start {
            libc::munmap(raw, aligned - start);
        }
        let tail = start + request - (aligned + bytes);
        if tail > 0 {
            libc::munmap((aligned + bytes) as *mut libc::c_void, tail);
        }
        // Advice only: failure (e.g. THP disabled) is harmless.
        #[cfg(target_os = "linux")]
        libc::madvise(aligned as *mut libc::c_void, bytes, libc::MADV_HUGEPAGE);
    }
    Ok(aligned as *mut u8)
}

/// Release memory from [`allocate`].
///
/// # Safety
/// `ptr` and `bytes` must come from one call to `allocate`.
#[cfg(unix)]
unsafe fn release(ptr: *mut u8, bytes: usize) {
    // SAFETY: the caller passes exactly the range `allocate` kept.
    unsafe { libc::munmap(ptr as *mut libc::c_void, bytes) };
}

#[cfg(not(unix))]
fn layout(bytes: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(bytes, HUGE_PAGE).expect("valid layout")
}

/// `bytes` of zeroed memory aligned to [`HUGE_PAGE`], from the global
/// allocator (systems without `mmap`).
#[cfg(not(unix))]
fn allocate(bytes: usize) -> std::io::Result<*mut u8> {
    // SAFETY: the layout has a nonzero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout(bytes)) };
    if ptr.is_null() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            "out of memory",
        ));
    }
    Ok(ptr)
}

/// Release memory from [`allocate`].
///
/// # Safety
/// `ptr` and `bytes` must come from one call to `allocate`.
#[cfg(not(unix))]
unsafe fn release(ptr: *mut u8, bytes: usize) {
    // SAFETY: allocated with the same layout.
    unsafe { std::alloc::dealloc(ptr, layout(bytes)) };
}

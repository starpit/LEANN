use ndarray::Array2;

// ── L2 squared distance ─────────────────────────────────────────────

/// Compute L2 squared distance between two slices.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { l2_distance_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { l2_distance_avx2(a, b) }
        } else {
            l2_distance_scalar(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        l2_distance_scalar(a, b)
    }
}

#[inline]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn l2_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks16 = n / 16;
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // 4 independent accumulators to break FMA dependency chain
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for i in 0..chunks16 {
        let offset = i * 16;
        let va0 = vld1q_f32(pa.add(offset));
        let vb0 = vld1q_f32(pb.add(offset));
        let diff0 = vsubq_f32(va0, vb0);
        sum0 = vfmaq_f32(sum0, diff0, diff0);

        let va1 = vld1q_f32(pa.add(offset + 4));
        let vb1 = vld1q_f32(pb.add(offset + 4));
        let diff1 = vsubq_f32(va1, vb1);
        sum1 = vfmaq_f32(sum1, diff1, diff1);

        let va2 = vld1q_f32(pa.add(offset + 8));
        let vb2 = vld1q_f32(pb.add(offset + 8));
        let diff2 = vsubq_f32(va2, vb2);
        sum2 = vfmaq_f32(sum2, diff2, diff2);

        let va3 = vld1q_f32(pa.add(offset + 12));
        let vb3 = vld1q_f32(pb.add(offset + 12));
        let diff3 = vsubq_f32(va3, vb3);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
    }

    // Combine accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    let mut result = vaddvq_f32(sum0);

    // Handle remainder with single-accumulator 4-float chunks
    let start16 = chunks16 * 16;
    let remaining = n - start16;
    let chunks4 = remaining / 4;

    let mut sum_tail = vdupq_n_f32(0.0);
    for i in 0..chunks4 {
        let offset = start16 + i * 4;
        let va = vld1q_f32(pa.add(offset));
        let vb = vld1q_f32(pb.add(offset));
        let diff = vsubq_f32(va, vb);
        sum_tail = vfmaq_f32(sum_tail, diff, diff);
    }
    result += vaddvq_f32(sum_tail);

    // Scalar remainder
    let start_scalar = start16 + chunks4 * 4;
    for i in start_scalar..n {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks32 = n / 32;
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // 4 independent accumulators to break FMA dependency chain
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in 0..chunks32 {
        let offset = i * 32;
        let va0 = _mm256_loadu_ps(pa.add(offset));
        let vb0 = _mm256_loadu_ps(pb.add(offset));
        let diff0 = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        let va1 = _mm256_loadu_ps(pa.add(offset + 8));
        let vb1 = _mm256_loadu_ps(pb.add(offset + 8));
        let diff1 = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        let va2 = _mm256_loadu_ps(pa.add(offset + 16));
        let vb2 = _mm256_loadu_ps(pb.add(offset + 16));
        let diff2 = _mm256_sub_ps(va2, vb2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        let va3 = _mm256_loadu_ps(pa.add(offset + 24));
        let vb3 = _mm256_loadu_ps(pb.add(offset + 24));
        let diff3 = _mm256_sub_ps(va3, vb3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    // Horizontal sum of 8 floats
    let hi = _mm256_extractf128_ps(sum0, 1);
    let lo = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(sums2);

    // Handle remainder with single-accumulator 8-float chunks
    let start32 = chunks32 * 32;
    let remaining = n - start32;
    let chunks8 = remaining / 8;

    let mut sum_tail = _mm256_setzero_ps();
    for i in 0..chunks8 {
        let offset = start32 + i * 8;
        let va = _mm256_loadu_ps(pa.add(offset));
        let vb = _mm256_loadu_ps(pb.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum_tail = _mm256_fmadd_ps(diff, diff, sum_tail);
    }
    let hi_t = _mm256_extractf128_ps(sum_tail, 1);
    let lo_t = _mm256_castps256_ps128(sum_tail);
    let sum128_t = _mm_add_ps(lo_t, hi_t);
    let shuf_t = _mm_movehdup_ps(sum128_t);
    let sums_t = _mm_add_ps(sum128_t, shuf_t);
    let shuf2_t = _mm_movehl_ps(sums_t, sums_t);
    let sums2_t = _mm_add_ss(sums_t, shuf2_t);
    result += _mm_cvtss_f32(sums2_t);

    // Scalar remainder
    let start_scalar = start32 + chunks8 * 8;
    for i in start_scalar..n {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
}

// ── Inner product distance (negated dot product) ─────────────────────

/// Compute negated inner product distance (lower = more similar).
#[inline]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { inner_product_distance_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { inner_product_distance_avx2(a, b) }
        } else {
            inner_product_distance_scalar(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        inner_product_distance_scalar(a, b)
    }
}

#[inline]
fn inner_product_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn inner_product_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks16 = n / 16;
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // 4 independent accumulators to break FMA dependency chain
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for i in 0..chunks16 {
        let offset = i * 16;
        let va0 = vld1q_f32(pa.add(offset));
        let vb0 = vld1q_f32(pb.add(offset));
        sum0 = vfmaq_f32(sum0, va0, vb0);

        let va1 = vld1q_f32(pa.add(offset + 4));
        let vb1 = vld1q_f32(pb.add(offset + 4));
        sum1 = vfmaq_f32(sum1, va1, vb1);

        let va2 = vld1q_f32(pa.add(offset + 8));
        let vb2 = vld1q_f32(pb.add(offset + 8));
        sum2 = vfmaq_f32(sum2, va2, vb2);

        let va3 = vld1q_f32(pa.add(offset + 12));
        let vb3 = vld1q_f32(pb.add(offset + 12));
        sum3 = vfmaq_f32(sum3, va3, vb3);
    }

    // Combine accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    let mut result = vaddvq_f32(sum0);

    // Handle remainder with single-accumulator 4-float chunks
    let start16 = chunks16 * 16;
    let remaining = n - start16;
    let chunks4 = remaining / 4;

    let mut sum_tail = vdupq_n_f32(0.0);
    for i in 0..chunks4 {
        let offset = start16 + i * 4;
        let va = vld1q_f32(pa.add(offset));
        let vb = vld1q_f32(pb.add(offset));
        sum_tail = vfmaq_f32(sum_tail, va, vb);
    }
    result += vaddvq_f32(sum_tail);

    // Scalar remainder
    let start_scalar = start16 + chunks4 * 4;
    for i in start_scalar..n {
        result += a[i] * b[i];
    }

    -result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn inner_product_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks32 = n / 32;
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // 4 independent accumulators to break FMA dependency chain
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in 0..chunks32 {
        let offset = i * 32;
        let va0 = _mm256_loadu_ps(pa.add(offset));
        let vb0 = _mm256_loadu_ps(pb.add(offset));
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        let va1 = _mm256_loadu_ps(pa.add(offset + 8));
        let vb1 = _mm256_loadu_ps(pb.add(offset + 8));
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

        let va2 = _mm256_loadu_ps(pa.add(offset + 16));
        let vb2 = _mm256_loadu_ps(pb.add(offset + 16));
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

        let va3 = _mm256_loadu_ps(pa.add(offset + 24));
        let vb3 = _mm256_loadu_ps(pb.add(offset + 24));
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
    }

    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum0, 1);
    let lo = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(sums2);

    // Handle remainder with single-accumulator 8-float chunks
    let start32 = chunks32 * 32;
    let remaining = n - start32;
    let chunks8 = remaining / 8;

    let mut sum_tail = _mm256_setzero_ps();
    for i in 0..chunks8 {
        let offset = start32 + i * 8;
        let va = _mm256_loadu_ps(pa.add(offset));
        let vb = _mm256_loadu_ps(pb.add(offset));
        sum_tail = _mm256_fmadd_ps(va, vb, sum_tail);
    }
    let hi_t = _mm256_extractf128_ps(sum_tail, 1);
    let lo_t = _mm256_castps256_ps128(sum_tail);
    let sum128_t = _mm_add_ps(lo_t, hi_t);
    let shuf_t = _mm_movehdup_ps(sum128_t);
    let sums_t = _mm_add_ps(sum128_t, shuf_t);
    let shuf2_t = _mm_movehl_ps(sums_t, sums_t);
    let sums2_t = _mm_add_ss(sums_t, shuf2_t);
    result += _mm_cvtss_f32(sums2_t);

    // Scalar remainder
    let start_scalar = start32 + chunks8 * 8;
    for i in start_scalar..n {
        result += a[i] * b[i];
    }

    -result
}

// ── Batched 4x L2 squared distance ───────────────────────────────────

/// Compute L2 squared distance from one query to 4 database vectors simultaneously.
/// The query is loaded once and reused, reducing memory bandwidth by 4x for the query.
#[inline]
pub fn l2_distance_batch_4(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(query.len(), y0.len());
    debug_assert_eq!(query.len(), y1.len());
    debug_assert_eq!(query.len(), y2.len());
    debug_assert_eq!(query.len(), y3.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { l2_distance_batch_4_neon(query, y0, y1, y2, y3) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { l2_distance_batch_4_avx2(query, y0, y1, y2, y3) }
        } else {
            l2_distance_batch_4_scalar(query, y0, y1, y2, y3)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        l2_distance_batch_4_scalar(query, y0, y1, y2, y3)
    }
}

#[inline]
fn l2_distance_batch_4_scalar(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    let mut d0: f32 = 0.0;
    let mut d1: f32 = 0.0;
    let mut d2: f32 = 0.0;
    let mut d3: f32 = 0.0;
    for i in 0..query.len() {
        let q = query[i];
        let diff0 = q - y0[i];
        d0 += diff0 * diff0;
        let diff1 = q - y1[i];
        d1 += diff1 * diff1;
        let diff2 = q - y2[i];
        d2 += diff2 * diff2;
        let diff3 = q - y3[i];
        d3 += diff3 * diff3;
    }
    [d0, d1, d2, d3]
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn l2_distance_batch_4_neon(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let n = query.len();
    let chunks4 = n / 4;
    let pq = query.as_ptr();
    let p0 = y0.as_ptr();
    let p1 = y1.as_ptr();
    let p2 = y2.as_ptr();
    let p3 = y3.as_ptr();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for i in 0..chunks4 {
        let offset = i * 4;
        let vq = vld1q_f32(pq.add(offset));

        let vy0 = vld1q_f32(p0.add(offset));
        let diff0 = vsubq_f32(vq, vy0);
        sum0 = vfmaq_f32(sum0, diff0, diff0);

        let vy1 = vld1q_f32(p1.add(offset));
        let diff1 = vsubq_f32(vq, vy1);
        sum1 = vfmaq_f32(sum1, diff1, diff1);

        let vy2 = vld1q_f32(p2.add(offset));
        let diff2 = vsubq_f32(vq, vy2);
        sum2 = vfmaq_f32(sum2, diff2, diff2);

        let vy3 = vld1q_f32(p3.add(offset));
        let diff3 = vsubq_f32(vq, vy3);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
    }

    let mut r0 = vaddvq_f32(sum0);
    let mut r1 = vaddvq_f32(sum1);
    let mut r2 = vaddvq_f32(sum2);
    let mut r3 = vaddvq_f32(sum3);

    // Scalar remainder
    let start = chunks4 * 4;
    for i in start..n {
        let q = *pq.add(i);
        let d0 = q - *p0.add(i);
        r0 += d0 * d0;
        let d1 = q - *p1.add(i);
        r1 += d1 * d1;
        let d2 = q - *p2.add(i);
        r2 += d2 * d2;
        let d3 = q - *p3.add(i);
        r3 += d3 * d3;
    }

    [r0, r1, r2, r3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn l2_distance_batch_4_avx2(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let n = query.len();
    let chunks8 = n / 8;
    let pq = query.as_ptr();
    let p0 = y0.as_ptr();
    let p1 = y1.as_ptr();
    let p2 = y2.as_ptr();
    let p3 = y3.as_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in 0..chunks8 {
        let offset = i * 8;
        let vq = _mm256_loadu_ps(pq.add(offset));

        let vy0 = _mm256_loadu_ps(p0.add(offset));
        let diff0 = _mm256_sub_ps(vq, vy0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        let vy1 = _mm256_loadu_ps(p1.add(offset));
        let diff1 = _mm256_sub_ps(vq, vy1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        let vy2 = _mm256_loadu_ps(p2.add(offset));
        let diff2 = _mm256_sub_ps(vq, vy2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        let vy3 = _mm256_loadu_ps(p3.add(offset));
        let diff3 = _mm256_sub_ps(vq, vy3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    #[inline(always)]
    unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }

    let mut r0 = hsum_avx2(sum0);
    let mut r1 = hsum_avx2(sum1);
    let mut r2 = hsum_avx2(sum2);
    let mut r3 = hsum_avx2(sum3);

    // Scalar remainder
    let start = chunks8 * 8;
    for i in start..n {
        let q = *pq.add(i);
        let d0 = q - *p0.add(i);
        r0 += d0 * d0;
        let d1 = q - *p1.add(i);
        r1 += d1 * d1;
        let d2 = q - *p2.add(i);
        r2 += d2 * d2;
        let d3 = q - *p3.add(i);
        r3 += d3 * d3;
    }

    [r0, r1, r2, r3]
}

// ── Batched 4x inner product distance ────────────────────────────────

/// Compute negated inner product distance from one query to 4 database vectors simultaneously.
#[inline]
pub fn inner_product_distance_batch_4(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(query.len(), y0.len());
    debug_assert_eq!(query.len(), y1.len());
    debug_assert_eq!(query.len(), y2.len());
    debug_assert_eq!(query.len(), y3.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { inner_product_distance_batch_4_neon(query, y0, y1, y2, y3) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { inner_product_distance_batch_4_avx2(query, y0, y1, y2, y3) }
        } else {
            inner_product_distance_batch_4_scalar(query, y0, y1, y2, y3)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        inner_product_distance_batch_4_scalar(query, y0, y1, y2, y3)
    }
}

#[inline]
fn inner_product_distance_batch_4_scalar(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    let mut d0: f32 = 0.0;
    let mut d1: f32 = 0.0;
    let mut d2: f32 = 0.0;
    let mut d3: f32 = 0.0;
    for i in 0..query.len() {
        let q = query[i];
        d0 += q * y0[i];
        d1 += q * y1[i];
        d2 += q * y2[i];
        d3 += q * y3[i];
    }
    [-d0, -d1, -d2, -d3]
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn inner_product_distance_batch_4_neon(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let n = query.len();
    let chunks4 = n / 4;
    let pq = query.as_ptr();
    let p0 = y0.as_ptr();
    let p1 = y1.as_ptr();
    let p2 = y2.as_ptr();
    let p3 = y3.as_ptr();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for i in 0..chunks4 {
        let offset = i * 4;
        let vq = vld1q_f32(pq.add(offset));

        let vy0 = vld1q_f32(p0.add(offset));
        sum0 = vfmaq_f32(sum0, vq, vy0);

        let vy1 = vld1q_f32(p1.add(offset));
        sum1 = vfmaq_f32(sum1, vq, vy1);

        let vy2 = vld1q_f32(p2.add(offset));
        sum2 = vfmaq_f32(sum2, vq, vy2);

        let vy3 = vld1q_f32(p3.add(offset));
        sum3 = vfmaq_f32(sum3, vq, vy3);
    }

    let mut r0 = vaddvq_f32(sum0);
    let mut r1 = vaddvq_f32(sum1);
    let mut r2 = vaddvq_f32(sum2);
    let mut r3 = vaddvq_f32(sum3);

    // Scalar remainder
    let start = chunks4 * 4;
    for i in start..n {
        let q = *pq.add(i);
        r0 += q * *p0.add(i);
        r1 += q * *p1.add(i);
        r2 += q * *p2.add(i);
        r3 += q * *p3.add(i);
    }

    [-r0, -r1, -r2, -r3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn inner_product_distance_batch_4_avx2(
    query: &[f32],
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let n = query.len();
    let chunks8 = n / 8;
    let pq = query.as_ptr();
    let p0 = y0.as_ptr();
    let p1 = y1.as_ptr();
    let p2 = y2.as_ptr();
    let p3 = y3.as_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in 0..chunks8 {
        let offset = i * 8;
        let vq = _mm256_loadu_ps(pq.add(offset));

        let vy0 = _mm256_loadu_ps(p0.add(offset));
        sum0 = _mm256_fmadd_ps(vq, vy0, sum0);

        let vy1 = _mm256_loadu_ps(p1.add(offset));
        sum1 = _mm256_fmadd_ps(vq, vy1, sum1);

        let vy2 = _mm256_loadu_ps(p2.add(offset));
        sum2 = _mm256_fmadd_ps(vq, vy2, sum2);

        let vy3 = _mm256_loadu_ps(p3.add(offset));
        sum3 = _mm256_fmadd_ps(vq, vy3, sum3);
    }

    #[inline(always)]
    unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }

    let mut r0 = hsum_avx2(sum0);
    let mut r1 = hsum_avx2(sum1);
    let mut r2 = hsum_avx2(sum2);
    let mut r3 = hsum_avx2(sum3);

    // Scalar remainder
    let start = chunks8 * 8;
    for i in start..n {
        let q = *pq.add(i);
        r0 += q * *p0.add(i);
        r1 += q * *p1.add(i);
        r2 += q * *p2.add(i);
        r3 += q * *p3.add(i);
    }

    [-r0, -r1, -r2, -r3]
}

// ── L2 normalization ─────────────────────────────────────────────────

/// Normalize rows of a matrix to unit L2 norm in-place.
pub fn normalize_l2_inplace(data: &mut Array2<f32>) {
    for mut row in data.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|x| x / norm);
        }
    }
}

// ── Visited list (generation-based) ──────────────────────────────────

/// A generation-based visited tracker that avoids O(n) allocation per reset.
///
/// Instead of allocating a new `vec![false; n]` for each search/level,
/// we increment a generation counter. A node is "visited" if
/// `visited[node] == generation`.
pub struct VisitedList {
    visited: Vec<u32>,
    generation: u32,
}

impl VisitedList {
    /// Create a new visited list for `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            visited: vec![0; n],
            generation: 0,
        }
    }

    /// Reset the visited state (O(1) instead of O(n)).
    /// On overflow, falls back to clearing the whole vector.
    #[inline]
    pub fn reset(&mut self) {
        if self.generation == u32::MAX {
            self.visited.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Number of nodes this visited list can track.
    #[inline]
    pub fn len(&self) -> usize {
        self.visited.len()
    }

    /// Mark a node as visited.
    ///
    /// # Safety
    /// Caller must ensure `node < self.len()`. Use the assert at search
    /// entry points to establish this invariant for all graph node IDs.
    #[inline(always)]
    pub fn set(&mut self, node: usize) {
        debug_assert!(node < self.visited.len());
        unsafe {
            *self.visited.get_unchecked_mut(node) = self.generation;
        }
    }

    /// Check if a node has been visited in the current generation.
    ///
    /// # Safety
    /// Caller must ensure `node < self.len()`. Use the assert at search
    /// entry points to establish this invariant for all graph node IDs.
    #[inline(always)]
    pub fn is_visited(&self, node: usize) -> bool {
        debug_assert!(node < self.visited.len());
        unsafe { *self.visited.get_unchecked(node) == self.generation }
    }

    /// Check if a node is unvisited and mark it as visited in one operation.
    /// Returns `true` if the node was **newly visited** (was not visited before).
    ///
    /// # Safety
    /// Caller must ensure `node < self.len()`.
    #[inline(always)]
    pub fn check_and_set(&mut self, node: usize) -> bool {
        debug_assert!(node < self.visited.len());
        unsafe {
            let entry = self.visited.get_unchecked_mut(node);
            if *entry == self.generation {
                false
            } else {
                *entry = self.generation;
                true
            }
        }
    }

    /// Issue a hardware prefetch for the visited table entry of `node`.
    /// This brings the cache line into L1 before the actual check_and_set.
    #[inline(always)]
    pub fn prefetch(&self, node: usize) {
        debug_assert!(node < self.visited.len());
        unsafe {
            let ptr = self.visited.as_ptr().add(node) as *const u8;
            #[cfg(target_arch = "aarch64")]
            {
                // PRFM PLDL1KEEP - prefetch for load into L1 cache
                std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
            }
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                let _ = ptr; // no-op on unsupported architectures
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        let result = l2_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "L2: {result} vs {expected}"
        );
    }

    #[test]
    fn test_l2_distance_non_aligned() {
        // Test with length not a multiple of 4 or 8
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        let result = l2_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "L2 non-aligned: {result} vs {expected}"
        );
    }

    #[test]
    fn test_inner_product_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected: f32 = -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let result = inner_product_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "IP: {result} vs {expected}"
        );
    }

    #[test]
    fn test_inner_product_distance_non_aligned() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let result = inner_product_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "IP non-aligned: {result} vs {expected}"
        );
    }

    #[test]
    fn test_l2_distance_large() {
        // Test with typical embedding dimension (384)
        let a: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| (384 - i) as f32 * 0.01).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        let result = l2_distance(&a, &b);
        assert!(
            (result - expected).abs() / expected < 1e-5,
            "L2 large: {result} vs {expected}"
        );
    }

    #[test]
    fn test_visited_list() {
        let mut vl = VisitedList::new(10);
        vl.reset();

        assert!(!vl.is_visited(0));
        assert!(!vl.is_visited(5));

        vl.set(3);
        vl.set(7);

        assert!(!vl.is_visited(0));
        assert!(vl.is_visited(3));
        assert!(vl.is_visited(7));

        // Reset and verify all are unvisited
        vl.reset();
        assert!(!vl.is_visited(3));
        assert!(!vl.is_visited(7));

        // Set new nodes
        vl.set(1);
        assert!(vl.is_visited(1));
        assert!(!vl.is_visited(3));
    }

    #[test]
    fn test_l2_distance_batch_4() {
        let q: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let y0: Vec<f32> = (0..384).map(|i| (384 - i) as f32 * 0.01).collect();
        let y1: Vec<f32> = (0..384).map(|i| (i * 2 % 384) as f32 * 0.01).collect();
        let y2: Vec<f32> = (0..384).map(|i| (i + 50) as f32 * 0.01).collect();
        let y3: Vec<f32> = (0..384).map(|_| 0.5).collect();

        let batch = l2_distance_batch_4(&q, &y0, &y1, &y2, &y3);
        let single = [
            l2_distance(&q, &y0),
            l2_distance(&q, &y1),
            l2_distance(&q, &y2),
            l2_distance(&q, &y3),
        ];

        for i in 0..4 {
            assert!(
                (batch[i] - single[i]).abs() / single[i].max(1e-10) < 1e-5,
                "L2 batch[{i}]: {} vs {}",
                batch[i],
                single[i]
            );
        }
    }

    #[test]
    fn test_l2_distance_batch_4_small() {
        let q = vec![1.0, 2.0, 3.0];
        let y0 = vec![4.0, 5.0, 6.0];
        let y1 = vec![0.0, 0.0, 0.0];
        let y2 = vec![1.0, 2.0, 3.0];
        let y3 = vec![2.0, 3.0, 4.0];

        let batch = l2_distance_batch_4(&q, &y0, &y1, &y2, &y3);
        assert!((batch[0] - 27.0).abs() < 1e-5);
        assert!((batch[1] - 14.0).abs() < 1e-5);
        assert!((batch[2] - 0.0).abs() < 1e-5);
        assert!((batch[3] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_inner_product_distance_batch_4() {
        let q: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let y0: Vec<f32> = (0..384).map(|i| (384 - i) as f32 * 0.01).collect();
        let y1: Vec<f32> = (0..384).map(|i| (i * 2 % 384) as f32 * 0.01).collect();
        let y2: Vec<f32> = (0..384).map(|i| (i + 50) as f32 * 0.01).collect();
        let y3: Vec<f32> = (0..384).map(|_| 0.5).collect();

        let batch = inner_product_distance_batch_4(&q, &y0, &y1, &y2, &y3);
        let single = [
            inner_product_distance(&q, &y0),
            inner_product_distance(&q, &y1),
            inner_product_distance(&q, &y2),
            inner_product_distance(&q, &y3),
        ];

        for i in 0..4 {
            assert!(
                (batch[i] - single[i]).abs() / single[i].abs().max(1e-10) < 1e-5,
                "IP batch[{i}]: {} vs {}",
                batch[i],
                single[i]
            );
        }
    }

    #[test]
    fn test_visited_list_check_and_set() {
        let mut vl = VisitedList::new(10);
        vl.reset();

        // First check_and_set should return true (newly visited)
        assert!(vl.check_and_set(3));
        // Second should return false (already visited)
        assert!(!vl.check_and_set(3));
        // Different node should return true
        assert!(vl.check_and_set(7));

        // After reset, should be true again
        vl.reset();
        assert!(vl.check_and_set(3));
    }

    #[test]
    fn test_visited_list_prefetch() {
        let vl = VisitedList::new(100);
        // Prefetch should not panic
        vl.prefetch(0);
        vl.prefetch(50);
        vl.prefetch(99);
    }

    #[test]
    fn test_normalize_l2() {
        let mut data = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0]).unwrap();
        normalize_l2_inplace(&mut data);

        // Row 0: [3/5, 4/5, 0]
        assert!((data[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((data[[0, 1]] - 0.8).abs() < 1e-6);
        assert!((data[[0, 2]] - 0.0).abs() < 1e-6);

        // Row 1: [0, 0, 1]
        assert!((data[[1, 0]] - 0.0).abs() < 1e-6);
        assert!((data[[1, 1]] - 0.0).abs() < 1e-6);
        assert!((data[[1, 2]] - 1.0).abs() < 1e-6);
    }
}

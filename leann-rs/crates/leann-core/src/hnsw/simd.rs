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

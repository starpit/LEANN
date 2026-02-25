use anyhow::{Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};

use super::graph::*;

/// Read a little-endian value from a reader.
fn read_le<T: Copy + Default, R: Read>(reader: &mut R) -> Result<T> {
    let size = std::mem::size_of::<T>();
    let mut buf = [0u8; 8]; // stack buffer, covers all primitive types
    reader
        .read_exact(&mut buf[..size])
        .context("unexpected EOF reading struct")?;
    Ok(unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const T) })
}

/// Write a little-endian value to a writer.
fn write_le<T: Copy, W: Write>(writer: &mut W, val: T) -> Result<()> {
    let size = std::mem::size_of::<T>();
    let ptr = &val as *const T as *const u8;
    let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };
    writer.write_all(bytes)?;
    Ok(())
}

/// Read a vector: 8-byte count (u64) followed by count elements.
fn read_vec<T: Copy + Default, R: Read>(reader: &mut R) -> Result<Vec<T>> {
    let count: u64 = read_le(reader)?;
    let count = count as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    let elem_size = std::mem::size_of::<T>();
    let total_bytes = count.checked_mul(elem_size).with_context(|| {
        format!(
            "vector size overflow: {} elements x {} bytes/elem exceeds usize",
            count, elem_size
        )
    })?;
    // Cap allocation at 4 GB — any legitimate HNSW index stays well
    // below this; a larger value means the file is corrupt/truncated.
    const MAX_ALLOC: usize = 4 << 30;
    if total_bytes > MAX_ALLOC {
        anyhow::bail!(
            "vector allocation too large: {} bytes ({} elements x {} bytes/elem)",
            total_bytes,
            count,
            elem_size
        );
    }
    let mut result = vec![T::default(); count];
    // Read directly into the typed buffer, avoiding a separate byte allocation + copy.
    let byte_slice =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, total_bytes) };
    reader.read_exact(byte_slice).with_context(|| {
        format!(
            "reading vector: expected {} bytes ({} elements)",
            total_bytes, count
        )
    })?;
    Ok(result)
}

/// Write a vector: 8-byte count (u64) followed by elements.
fn write_vec<T: Copy, W: Write>(writer: &mut W, data: &[T]) -> Result<()> {
    write_le(writer, data.len() as u64)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    writer.write_all(bytes)?;
    Ok(())
}

/// Read an HNSW index from a FAISS-format file.
///
/// Supports both standard (non-compact) and compact (CSR) formats,
/// as written by the Python `convert_to_csr.py` or the custom FAISS fork.
pub fn read_hnsw_index<R: Read + Seek>(reader: &mut R) -> Result<HnswGraph> {
    // Read IndexHNSW header
    let index_fourcc: u32 = read_le(reader)?;
    if index_fourcc != FOURCC_HNSW_FLAT {
        anyhow::bail!(
            "Unexpected HNSW FourCC: 0x{:08x}, expected 0x{:08x} (IHNf)",
            index_fourcc,
            FOURCC_HNSW_FLAT
        );
    }

    let d: i32 = read_le(reader)?;
    let _ntotal: i64 = read_le(reader)?;
    let _dummy1: i64 = read_le(reader)?;
    let _dummy2: i64 = read_le(reader)?;
    let is_trained: bool = read_le::<u8, _>(reader)? != 0;
    let _ = is_trained;
    let metric_type: i32 = read_le(reader)?;
    let metric_arg: f32 = if metric_type > 1 {
        read_le(reader)?
    } else {
        0.0
    };

    // Read HNSW struct vectors
    let assign_probas: Vec<f64> = read_vec(reader)?;
    let cum_nneighbor_per_level: Vec<i32> = read_vec(reader)?;
    let levels: Vec<i32> = read_vec(reader)?;

    let ntotal = levels.len();

    // Probe for compact format flag.
    //
    // The compact format prefixes a single 0x01 byte before level_ptr.
    // Standard FAISS format has no such byte — the offsets vector follows
    // directly.  A single-byte probe is ambiguous: if the offsets count's
    // LSB (little-endian) happens to be 0x01, we'd false-positive.
    //
    // To disambiguate: after reading the flag byte, peek at the next u64
    // (level_ptr count).  In a valid compact graph, level_ptr has
    // sum(levels[i]+1) entries — bounded by ntotal * (max_level+1).
    // If the count is wildly out of range, we misdetected and should
    // fall back to standard format.
    let pos_before_compact = reader.stream_position()?;

    let is_compact = match read_le::<u8, _>(reader) {
        Ok(1) => {
            // Peek at the would-be level_ptr count to validate.
            // In a valid compact graph, level_ptr has exactly
            // sum(levels[i] + 1) entries.  We already parsed the
            // levels array, so we can compute the expected count.
            let expected_level_ptr_count: u64 = levels.iter().map(|&l| (l as u64) + 1).sum();
            let pos_after_flag = reader.stream_position()?;
            match read_le::<u64, _>(reader) {
                Ok(level_ptr_count) if level_ptr_count == expected_level_ptr_count => {
                    // Matches — seek back so read_vec re-reads the count.
                    reader.seek(SeekFrom::Start(pos_after_flag))?;
                    true
                }
                Ok(_) => {
                    // Mismatch — the 0x01 byte was the LSB of the
                    // standard offsets count, not a compact flag.
                    reader.seek(SeekFrom::Start(pos_before_compact))?;
                    false
                }
                Err(_) => {
                    reader.seek(SeekFrom::Start(pos_before_compact))?;
                    false
                }
            }
        }
        Ok(_) => {
            reader.seek(SeekFrom::Start(pos_before_compact))?;
            false
        }
        Err(_) => {
            reader.seek(SeekFrom::Start(pos_before_compact))?;
            false
        }
    };

    if is_compact {
        // Read compact format
        let level_ptr: Vec<u64> = read_vec(reader)?;
        let node_offsets: Vec<u64> = read_vec(reader)?;

        let entry_point: i32 = read_le(reader)?;
        let max_level: i32 = read_le(reader)?;
        let ef_construction: i32 = read_le(reader)?;
        let ef_search: i32 = read_le(reader)?;
        let _dummy_upper_beam: i32 = read_le(reader)?;

        let storage_fourcc: u32 = read_le(reader)?;

        // Read compact neighbors data
        let neighbors: Vec<i32> = read_vec(reader)?;

        // Read remaining storage data
        let mut storage_data = Vec::new();
        reader.read_to_end(&mut storage_data)?;

        let vector_storage = if storage_fourcc == FOURCC_NULL || storage_data.is_empty() {
            VectorStorage::Null
        } else {
            VectorStorage::Raw {
                fourcc: storage_fourcc,
                data: storage_data,
            }
        };

        let config = HnswConfig {
            m: (cum_nneighbor_per_level.first().copied().unwrap_or(64) / 2) as usize,
            ef_construction: ef_construction as usize,
            ef_search: ef_search as usize,
            distance_metric: if metric_type == 0 {
                crate::index::DistanceMetric::L2
            } else {
                crate::index::DistanceMetric::Mips
            },
            is_compact: true,
            is_recompute: matches!(vector_storage, VectorStorage::Null),
            seed: None,
        };

        Ok(HnswGraph {
            ntotal,
            dimensions: d as usize,
            entry_point,
            max_level,
            levels,
            assign_probas,
            cum_nneighbor_per_level,
            config,
            metric_type,
            metric_arg,
            storage: GraphStorage::Compact {
                level_ptr,
                node_offsets,
                neighbors,
            },
            vector_storage,
        })
    } else {
        // Standard (non-compact) format
        //
        // Some FAISS variants prepend a 0x00 padding byte before the
        // offsets vector.  Rather than guessing from one byte, try
        // reading without the extra byte first; if the offsets count
        // looks wrong, try again from +1.
        let pos_standard = reader.stream_position()?;

        // Peek at the would-be offsets count at the current position.
        let offsets_count_raw: u64 = read_le(reader)?;
        reader.seek(SeekFrom::Start(pos_standard))?;

        // For a valid standard HNSW, offsets has ntotal or ntotal+1 entries.
        let plausible = offsets_count_raw == ntotal as u64
            || offsets_count_raw == (ntotal + 1) as u64;

        if !plausible {
            // Try skipping one padding byte.
            let alt_pos = pos_standard + 1;
            reader.seek(SeekFrom::Start(alt_pos))?;
            let alt_count: u64 = read_le(reader)?;
            let alt_plausible =
                alt_count == ntotal as u64 || alt_count == (ntotal + 1) as u64;
            if alt_plausible {
                reader.seek(SeekFrom::Start(alt_pos))?;
            } else {
                // Neither interpretation has a plausible count —
                // proceed from the original position and let
                // downstream reads produce a clear error.
                reader.seek(SeekFrom::Start(pos_standard))?;
            }
        }

        let offsets: Vec<u64> = read_vec(reader)?;
        let neighbors: Vec<i32> = read_vec(reader)?;

        let entry_point: i32 = read_le(reader)?;
        let max_level: i32 = read_le(reader)?;
        let ef_construction: i32 = read_le(reader)?;
        let ef_search: i32 = read_le(reader)?;
        let _dummy_upper_beam: i32 = read_le(reader)?;

        // Try to read storage section
        let (storage_fourcc, storage_data) = match read_le::<u32, _>(reader) {
            Ok(fourcc) => {
                let mut data = Vec::new();
                reader.read_to_end(&mut data)?;
                (fourcc, data)
            }
            Err(_) => (FOURCC_NULL, Vec::new()),
        };

        let vector_storage = if storage_fourcc == FOURCC_NULL || storage_data.is_empty() {
            VectorStorage::Null
        } else {
            VectorStorage::Raw {
                fourcc: storage_fourcc,
                data: storage_data,
            }
        };

        let config = HnswConfig {
            m: (cum_nneighbor_per_level.first().copied().unwrap_or(64) / 2) as usize,
            ef_construction: ef_construction as usize,
            ef_search: ef_search as usize,
            distance_metric: if metric_type == 0 {
                crate::index::DistanceMetric::L2
            } else {
                crate::index::DistanceMetric::Mips
            },
            is_compact: false,
            is_recompute: matches!(vector_storage, VectorStorage::Null),
            seed: None,
        };

        Ok(HnswGraph {
            ntotal,
            dimensions: d as usize,
            entry_point,
            max_level,
            levels,
            assign_probas,
            cum_nneighbor_per_level,
            config,
            metric_type,
            metric_arg,
            storage: GraphStorage::Standard { offsets, neighbors },
            vector_storage,
        })
    }
}

/// Write an HNSW index in compact (CSR) format.
pub fn write_hnsw_compact<W: Write>(writer: &mut W, graph: &HnswGraph) -> Result<()> {
    let (level_ptr, node_offsets, neighbors) = match &graph.storage {
        GraphStorage::Compact {
            level_ptr,
            node_offsets,
            neighbors,
        } => (level_ptr, node_offsets, neighbors),
        _ => anyhow::bail!("Cannot write compact format from non-compact storage"),
    };

    // Write IndexHNSW header
    write_le(writer, FOURCC_HNSW_FLAT)?;
    write_le(writer, graph.dimensions as i32)?;
    write_le(writer, graph.ntotal as i64)?;
    write_le(writer, 0i64)?; // dummy1
    write_le(writer, 0i64)?; // dummy2
    write_le(writer, 1u8)?; // is_trained
    write_le(writer, graph.metric_type)?;
    if graph.metric_type > 1 {
        write_le(writer, graph.metric_arg)?;
    }

    // Write HNSW struct vectors
    write_vec(writer, &graph.assign_probas)?;
    write_vec(writer, &graph.cum_nneighbor_per_level)?;
    write_vec(writer, &graph.levels)?;

    // Compact flag
    write_le(writer, 1u8)?; // storage_is_compact = true

    // Write compact data
    write_vec(writer, level_ptr)?;
    write_vec(writer, node_offsets)?;

    // Write scalar parameters
    write_le(writer, graph.entry_point)?;
    write_le(writer, graph.max_level)?;
    write_le(writer, graph.config.ef_construction as i32)?;
    write_le(writer, graph.config.ef_search as i32)?;
    write_le(writer, 1i32)?; // dummy_upper_beam

    // Write storage fourcc
    let (storage_fourcc, storage_data) = match &graph.vector_storage {
        VectorStorage::Null => (FOURCC_NULL, &[][..]),
        VectorStorage::Raw { fourcc, data } => (*fourcc, data.as_slice()),
    };
    write_le(writer, storage_fourcc)?;

    // Write compact neighbors
    write_vec(writer, neighbors)?;

    // Write storage data if not null
    if storage_fourcc != FOURCC_NULL && !storage_data.is_empty() {
        writer.write_all(storage_data)?;
    }

    Ok(())
}

/// Write an HNSW index in standard (non-compact) format.
pub fn write_hnsw_standard<W: Write>(writer: &mut W, graph: &HnswGraph) -> Result<()> {
    let (offsets, neighbors) = match &graph.storage {
        GraphStorage::Standard { offsets, neighbors } => (offsets, neighbors),
        _ => anyhow::bail!("Cannot write standard format from compact storage"),
    };

    // Write IndexHNSW header
    write_le(writer, FOURCC_HNSW_FLAT)?;
    write_le(writer, graph.dimensions as i32)?;
    write_le(writer, graph.ntotal as i64)?;
    write_le(writer, 0i64)?; // dummy1
    write_le(writer, 0i64)?; // dummy2
    write_le(writer, 1u8)?; // is_trained
    write_le(writer, graph.metric_type)?;
    if graph.metric_type > 1 {
        write_le(writer, graph.metric_arg)?;
    }

    // Write HNSW struct vectors
    write_vec(writer, &graph.assign_probas)?;
    write_vec(writer, &graph.cum_nneighbor_per_level)?;
    write_vec(writer, &graph.levels)?;

    // Write standard adjacency data
    write_vec(writer, offsets)?;
    write_vec(writer, neighbors)?;

    // Write scalar parameters
    write_le(writer, graph.entry_point)?;
    write_le(writer, graph.max_level)?;
    write_le(writer, graph.config.ef_construction as i32)?;
    write_le(writer, graph.config.ef_search as i32)?;
    write_le(writer, 1i32)?; // dummy_upper_beam

    // Write storage
    let (storage_fourcc, storage_data) = match &graph.vector_storage {
        VectorStorage::Null => (FOURCC_NULL, &[][..]),
        VectorStorage::Raw { fourcc, data } => (*fourcc, data.as_slice()),
    };
    write_le(writer, storage_fourcc)?;
    if storage_fourcc != FOURCC_NULL && !storage_data.is_empty() {
        writer.write_all(storage_data)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_test_graph() -> HnswGraph {
        // Minimal compact graph for testing I/O
        HnswGraph {
            ntotal: 2,
            dimensions: 4,
            entry_point: 0,
            max_level: 0,
            levels: vec![1, 1],
            assign_probas: vec![1.0],
            cum_nneighbor_per_level: vec![4],
            config: HnswConfig {
                m: 2,
                ef_construction: 16,
                ef_search: 16,
                distance_metric: crate::index::DistanceMetric::Mips,
                is_compact: true,
                is_recompute: true,
                seed: None,
            },
            metric_type: 1,
            metric_arg: 0.0,
            storage: GraphStorage::Compact {
                level_ptr: vec![0, 1, 1, 2],
                node_offsets: vec![0, 2, 4],
                neighbors: vec![1, 0],
            },
            vector_storage: VectorStorage::Null,
        }
    }

    #[test]
    fn test_compact_roundtrip() {
        let graph = make_test_graph();

        let mut buf = Vec::new();
        write_hnsw_compact(&mut buf, &graph).unwrap();

        let mut cursor = Cursor::new(buf);
        let loaded = read_hnsw_index(&mut cursor).unwrap();

        assert_eq!(loaded.ntotal, 2);
        assert_eq!(loaded.dimensions, 4);
        assert_eq!(loaded.entry_point, 0);
        assert!(loaded.is_compact());
        assert!(loaded.is_pruned());
    }

    /// Regression test: a standard FAISS index whose offsets count has
    /// LSB == 0x01 used to be misdetected as compact format, causing
    /// a multiply-with-overflow panic in read_vec.
    #[test]
    fn test_standard_format_not_misdetected_as_compact() {
        // Build a standard-format graph where offsets has exactly 257
        // elements (257 % 256 == 1), so the first byte of its u64
        // count is 0x01 — the same value as the compact flag.
        let ntotal = 256; // 256 nodes → offsets.len() == 257 (one per node + sentinel)
        let m = 4;
        let cum = vec![2 * m as i32]; // one level, 2*M neighbors per node
        let levels = vec![1i32; ntotal];

        // Build offsets: each node gets a block of 2*M neighbor slots.
        let neighbors_per_node = 2 * m;
        let mut offsets = Vec::with_capacity(ntotal + 1);
        for i in 0..=ntotal {
            offsets.push((i * neighbors_per_node) as u64);
        }
        assert_eq!(offsets.len(), 257); // LSB of 257u64 == 0x01

        let total_neighbors = ntotal * neighbors_per_node;
        let neighbors = vec![-1i32; total_neighbors]; // all empty

        let graph = HnswGraph {
            ntotal,
            dimensions: 4,
            entry_point: 0,
            max_level: 0,
            levels,
            assign_probas: vec![1.0],
            cum_nneighbor_per_level: cum,
            config: HnswConfig {
                m,
                ef_construction: 16,
                ef_search: 16,
                distance_metric: crate::index::DistanceMetric::L2,
                is_compact: false,
                is_recompute: false,
                seed: None,
            },
            metric_type: 0,
            metric_arg: 0.0,
            storage: GraphStorage::Standard { offsets, neighbors },
            vector_storage: VectorStorage::Null,
        };

        let mut buf = Vec::new();
        write_hnsw_standard(&mut buf, &graph).unwrap();

        let mut cursor = Cursor::new(buf);
        let loaded = read_hnsw_index(&mut cursor).unwrap();

        assert_eq!(loaded.ntotal, ntotal);
        assert!(
            !loaded.is_compact(),
            "should be detected as standard, not compact"
        );
    }
}

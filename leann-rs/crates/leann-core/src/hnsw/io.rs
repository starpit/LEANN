use anyhow::{Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};

use super::graph::*;

/// Read a little-endian value from a reader.
fn read_le<T: Copy + Default, R: Read>(reader: &mut R) -> Result<T> {
    let size = std::mem::size_of::<T>();
    let mut buf = vec![0u8; size];
    reader
        .read_exact(&mut buf)
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
    let total_bytes = count * elem_size;
    let mut buf = vec![0u8; total_bytes];
    reader.read_exact(&mut buf).with_context(|| {
        format!(
            "reading vector: expected {} bytes ({} elements)",
            total_bytes, count
        )
    })?;
    let mut result = vec![T::default(); count];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), result.as_mut_ptr() as *mut u8, total_bytes);
    }
    Ok(result)
}

/// Write a vector: 8-byte count (u64) followed by elements.
fn write_vec<T: Copy, W: Write>(writer: &mut W, data: &[T]) -> Result<()> {
    write_le(writer, data.len() as u64)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
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
    let ntotal: i64 = read_le(reader)?;
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

    // Probe for compact format flag
    let pos_before_compact = reader.stream_position()?;

    let is_compact = match read_le::<u8, _>(reader) {
        Ok(1) => true,
        Ok(0) => {
            reader.seek(SeekFrom::Start(pos_before_compact))?;
            false
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
        // Handle potential extra byte
        let pos_before_probe = reader.stream_position()?;
        match read_le::<u8, _>(reader) {
            Ok(0x00) => { /* consumed extra byte */ }
            Ok(_) => {
                reader.seek(SeekFrom::Start(pos_before_probe))?;
            }
            Err(_) => {
                reader.seek(SeekFrom::Start(pos_before_probe))?;
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
}

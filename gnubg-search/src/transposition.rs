use gnubg_sys::PositionKey;

/// Zobrist-style hash for a GNU Backgammon position key.
pub fn zobrist_hash(key: &PositionKey) -> u64 {
    // Deterministic SplitMix64 expansion over every byte/index pair. This keeps
    // the hash table independent from the eval cache hash while avoiding a large
    // static random table for the compact 10-byte old-position key.
    let mut hash = 0x9e37_79b9_7f4a_7c15_u64;
    for (idx, byte) in key.0.iter().copied().enumerate() {
        let seed = (u64::from(byte) << 8) ^ idx as u64 ^ 0xa5a5_a5a5_a5a5_a5a5;
        hash ^= splitmix64(seed.wrapping_add((idx as u64) << 32));
        hash = hash.rotate_left(7).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    hash ^ (hash >> 32)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTEntry {
    pub key_hash: u64,
    pub depth: u8,
    pub flag: TTFlag,
    pub eval: f32,
    pub best_move_idx: u16,
    pub age: u8,
}

#[derive(Debug)]
pub struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    age: u8,
    mask: usize,
    hits: u64,
    lookups: u64,
}

impl TranspositionTable {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);
        Self {
            entries,
            age: 0,
            mask: capacity - 1,
            hits: 0,
            lookups: 0,
        }
    }

    pub fn lookup(&mut self, hash: u64, depth: u8, _ply: u8) -> Option<(f32, TTFlag, u16)> {
        self.lookups += 1;
        let index = self.index(hash);
        let entry = self.entries[index].as_mut()?;
        if entry.key_hash == hash && entry.depth >= depth {
            entry.age = self.age;
            self.hits += 1;
            Some((entry.eval, entry.flag, entry.best_move_idx))
        } else {
            None
        }
    }

    pub fn store(
        &mut self,
        hash: u64,
        depth: u8,
        flag: TTFlag,
        eval: f32,
        best_move_idx: u16,
        _ply: u8,
    ) {
        let index = self.index(hash);
        match self.entries[index].as_mut() {
            Some(entry) if entry.key_hash == hash && depth < entry.depth => {
                entry.age = self.age;
            }
            Some(entry) if depth < entry.depth => {
                entry.age = self.age;
            }
            _ => {
                self.entries[index] = Some(TTEntry {
                    key_hash: hash,
                    depth,
                    flag,
                    eval,
                    best_move_idx,
                    age: self.age,
                });
            }
        }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.hits = 0;
        self.lookups = 0;
    }

    pub fn new_search(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    pub const fn stats(&self) -> (u64, u64) {
        (self.hits, self.lookups)
    }

    fn index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_hits_at_sufficient_depth() {
        let mut tt = TranspositionTable::new(3);
        let hash = 0x1234;
        tt.store(hash, 4, TTFlag::Exact, 0.25, 7, 0);

        assert_eq!(tt.lookup(hash, 3, 0), Some((0.25, TTFlag::Exact, 7)));
        assert_eq!(tt.stats(), (1, 1));
        assert_eq!(tt.lookup(hash, 5, 0), None);
        assert_eq!(tt.stats(), (1, 2));
    }

    #[test]
    fn depth_preferred_replacement() {
        let mut tt = TranspositionTable::new(2);
        let hash = 0x55;
        tt.store(hash, 4, TTFlag::Exact, 1.0, 1, 0);
        tt.store(hash, 2, TTFlag::LowerBound, 2.0, 2, 0);
        assert_eq!(tt.lookup(hash, 4, 0), Some((1.0, TTFlag::Exact, 1)));

        tt.store(hash, 5, TTFlag::UpperBound, 3.0, 3, 0);
        assert_eq!(tt.lookup(hash, 5, 0), Some((3.0, TTFlag::UpperBound, 3)));
    }

    #[test]
    fn clear_resets_entries_and_stats() {
        let mut tt = TranspositionTable::new(8);
        tt.store(1, 1, TTFlag::Exact, 0.0, 0, 0);
        assert!(tt.lookup(1, 1, 0).is_some());
        tt.clear();
        assert_eq!(tt.lookup(1, 1, 0), None);
        assert_eq!(tt.stats(), (0, 1));
    }

    #[test]
    fn zobrist_hash_is_deterministic() {
        let key = PositionKey([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(zobrist_hash(&key), zobrist_hash(&key));
        assert_ne!(zobrist_hash(&key), zobrist_hash(&PositionKey([0; 10])));
    }
}

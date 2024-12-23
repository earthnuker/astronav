use std::collections::BTreeSet;

use bitvec_simd::BitVec;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use nohash_hasher::IntSet;
use rand::{thread_rng, Rng};
use roaring::RoaringBitmap;
use rustc_hash::FxHashSet;

fn rand_u32() -> u32 {
    let mut rng = thread_rng();
    rng.gen()
}

fn bench_sets(c: &mut Criterion) {
    let mut g = c.benchmark_group("sets");
    let mut set = IntSet::default();
    g.bench_function("IntSet_set", |b| {
        b.iter_batched(rand_u32, |v| set.insert(v), BatchSize::SmallInput);
    });
    g.bench_function("IntSet_get", |b| {
        b.iter_batched(rand_u32, |v| set.get(&v), BatchSize::SmallInput);
    });
    let mut set = BTreeSet::default();
    g.bench_function("BTreeSet_set", |b| {
        b.iter_batched(rand_u32, |v| set.insert(v), BatchSize::SmallInput);
    });
    g.bench_function("BTreeSet_get", |b| {
        b.iter_batched(rand_u32, |v| set.get(&v), BatchSize::SmallInput);
    });
    let mut set = FxHashSet::default();
    g.bench_function("FxHashSet_set", |b| {
        b.iter_batched(rand_u32, |v| set.insert(v), BatchSize::SmallInput);
    });
    g.bench_function("FxHashSet_get", |b| {
        b.iter_batched(rand_u32, |v| set.get(&v), BatchSize::SmallInput);
    });
    let mut set = RoaringBitmap::default();
    g.bench_function("RoaringBitmap_set", |b| {
        b.iter_batched(rand_u32, |v| set.insert(v), BatchSize::SmallInput);
    });
    g.bench_function("RoaringBitmap_get", |b| {
        b.iter_batched(rand_u32, |v| set.contains(v), BatchSize::SmallInput);
    });

    let mut set = BitVec::zeros(10_000_000);
    g.bench_function("BitVec_set", |b| {
        b.iter_batched(
            || rand_u32() % 10_000_000,
            |v| set.set(v as usize, true),
            BatchSize::SmallInput,
        );
    });
    g.bench_function("BitVec_get", |b| {
        b.iter_batched(
            || rand_u32() % 10_000_000,
            |v| set.get(v as usize),
            BatchSize::SmallInput,
        );
    });
    g.finish();
}

criterion_group! {
    name = map_benches;
    config = Criterion::default().significance_level(0.1).sample_size(1000);
    targets = bench_sets
}

criterion_main!(map_benches);

// fn main() {}

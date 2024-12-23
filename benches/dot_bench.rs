use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use rand_distr::StandardNormal;

fn rand_v3() -> [f32; 3] {
    let mut rng = rand::thread_rng();
    [
        rng.sample(StandardNormal),
        rng.sample(StandardNormal),
        rng.sample(StandardNormal),
    ]
}

fn arand() -> f32 {
    let mut rng = rand::thread_rng();
    rng.sample::<f32, _>(StandardNormal).abs()
}

fn veclen(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dist2(p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

fn dist2_unrolled(p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    [
        ((p1[0] - p2[0]) * (p1[0] - p2[0])),
        ((p1[1] - p2[1]) * (p1[1] - p2[1])),
        ((p1[2] - p2[2]) * (p1[2] - p2[2])),
    ]
    .into_iter()
    .sum()
}

fn dist2_unrolled_2(p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    ((p1[0] - p2[0]) * (p1[0] - p2[0]))
        + ((p1[1] - p2[1]) * (p1[1] - p2[1]))
        + ((p1[2] - p2[2]) * (p1[2] - p2[2]))
}

fn dist(p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    dist2(p1, p2).sqrt()
}

/// Dot product (cosine of angle) between two 3D vectors
fn ndot_vec_dist(u: &[f32; 3], v: &[f32; 3]) -> f32 {
    let z: [f32; 3] = [0.0; 3];
    let lm = dist(u, &z) * dist(v, &z);
    ((u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])) / lm
}

/// Dot product (cosine of angle) between two 3D vectors
fn ndot_vec_len(u: &[f32; 3], v: &[f32; 3]) -> f32 {
    let lm = veclen(u) * veclen(v);
    ((u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])) / lm
}

fn ndot_iter(u: &[f32; 3], v: &[f32; 3]) -> f32 {
    let mut l_u = 0.0;
    let mut l_v = 0.0;
    let mut l_s = 0.0;
    for (u, v) in u.iter().zip(v.iter()) {
        l_s += u * v;
        l_u += u * u;
        l_v += v * v;
    }
    l_s / (l_u * l_v).sqrt()
}

fn bench_ndot(c: &mut Criterion) {
    let mut g = c.benchmark_group("ndot");
    g.bench_function("vec_dist", |b| {
        b.iter_batched(
            || (rand_v3(), rand_v3()),
            |(v1, v2)| ndot_vec_dist(&v1, &v2),
            BatchSize::SmallInput,
        );
    });
    g.bench_function("vec_len", |b| {
        b.iter_batched(
            || (rand_v3(), rand_v3()),
            |(v1, v2)| ndot_vec_len(&v1, &v2),
            BatchSize::SmallInput,
        );
    });
    g.bench_function("iter", |b| {
        b.iter_batched(
            || (rand_v3(), rand_v3()),
            |(v1, v2)| ndot_iter(&v1, &v2),
            BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_dist(c: &mut Criterion) {
    let mut g = c.benchmark_group("dist");
    g.bench_function("dist2", |b| {
        b.iter_batched(
            || (rand_v3(), rand_v3()),
            |(v1, v2)| dist2(&v1, &v2),
            BatchSize::SmallInput,
        );
    });
    g.bench_function("dist2_unrolled", |b| {
        b.iter_batched(
            || (rand_v3(), rand_v3()),
            |(v1, v2)| dist2_unrolled(&v1, &v2),
            BatchSize::SmallInput,
        );
    });
    g.bench_function("dist2_unrolled_2", |b| {
        b.iter_batched(
            || (rand_v3(), rand_v3()),
            |(v1, v2)| dist2_unrolled_2(&v1, &v2),
            BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn vsub(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn h_old(node: &[f32; 3], m: f32, goal: &[f32; 3], r: f32) -> f32 {
    (dist(node, goal) - (r * m)).max(0.0)
}

fn h_new(node: &[f32; 3], next: &[f32; 3], goal: &[f32; 3]) -> f32 {
    -ndot_iter(&vsub(node, goal), &vsub(node, next)).acos()
}

fn bench_new_heur(c: &mut Criterion) {
    c.bench_function("old_heuristic", |b| {
        b.iter_batched(
            || (rand_v3(), arand(), rand_v3(), arand()),
            |(node, m, goal, range)| h_old(&node, m, &goal, range),
            BatchSize::SmallInput,
        );
    });

    c.bench_function("new_heuristic", |b| {
        b.iter_batched(
            || (rand_v3(), rand_v3(), rand_v3()),
            |(v1, v2, v3)| h_new(&v1, &v2, &v3),
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(dist_benches, bench_dist);
criterion_group!(ndot_benches, bench_ndot);
criterion_group!(heur_benches, bench_new_heur);
criterion_main!(dist_benches);

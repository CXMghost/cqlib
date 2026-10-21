// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Complex-matrix utilities for the Block-ZXZ decomposition.
//!
//! Mirrors the `numpy`/`scipy` matrix operations of the Python project.
//! `ndarray::Array2<Complex64>` is the primary matrix type (consistent with
//! the rest of the crate); `faer` is only borrowed for diagonalization and
//! singular-value decomposition.

use ndarray::{Array2, s};
use num_complex::Complex64;

/// Complex matrix alias.
pub type Cm = Array2<Complex64>;

/// Builds the `n x n` identity matrix.
pub fn eye(n: usize) -> Cm {
    Array2::eye(n)
}

/// Matrix product `A @ B`.
#[inline]
pub fn mat_mul(a: &Cm, b: &Cm) -> Cm {
    a.dot(b)
}

/// Conjugate transpose `A^H`.
#[inline]
pub fn adjoint(a: &Cm) -> Cm {
    a.t().mapv(|z| z.conj())
}

/// Builds a diagonal matrix from a list of diagonal values.
pub fn diagonal_from_values(values: &[Complex64]) -> Cm {
    let n = values.len();
    let mut out = Array2::<Complex64>::zeros((n, n));
    for (i, &v) in values.iter().enumerate() {
        out[[i, i]] = v;
    }
    out
}

/// Builds the block-diagonal matrix `diag(top_left, bottom_right)`.
pub fn block_diag(top_left: &Cm, bottom_right: &Cm) -> Cm {
    let td = top_left.dim().0;
    let bd = bottom_right.dim().0;
    let n = td + bd;
    let mut out = Array2::<Complex64>::zeros((n, n));
    out.slice_mut(s![..td, ..td]).assign(top_left);
    out.slice_mut(s![td.., td..]).assign(bottom_right);
    out
}

/// Maximum elementwise absolute difference (used for allclose checks).
pub fn max_abs_diff(a: &Cm, b: &Cm) -> f64 {
    let mut m = 0.0f64;
    for ((i, j), &av) in a.indexed_iter() {
        let d = (av - b[[i, j]]).norm();
        if d > m {
            m = d;
        }
    }
    m
}

/// Returns whether `A` is unitary within the `atol` tolerance.
pub fn is_unitary(a: &Cm, atol: f64) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    let n = a.nrows();
    max_abs_diff(&mat_mul(&adjoint(a), a), &eye(n)) <= atol
}

/// Returns whether `A` is diagonal within the `atol` tolerance.
pub fn is_diagonal(a: &Cm, atol: f64) -> bool {
    let (r, c) = a.dim();
    for i in 0..r {
        for j in 0..c {
            if i != j && a[[i, j]].norm() > atol {
                return false;
            }
        }
    }
    true
}

// ---- faer conversions ----

/// Copies an ndarray matrix into a faer matrix.
pub fn to_faer(a: &Cm) -> faer::Mat<Complex64> {
    let (r, c) = a.dim();
    faer::Mat::<Complex64>::from_fn(r, c, |i, j| a[[i, j]])
}

/// Copies a faer matrix reference into an ndarray matrix.
pub fn from_faer(m: &faer::MatRef<'_, Complex64>) -> Cm {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

// ---- Numeric decompositions (mirroring numpy.linalg.svd / scipy.linalg.schur) ----

/// Polar decomposition through the SVD: returns `(S, U_polar)` with
/// `matrix ~ S @ U_polar`.
///
/// Mirrors the Python `polar_from_svd`: `X = U Sigma V^H`,
/// `S = U Sigma U^H`, `U_polar = U V^H`.
pub fn polar_from_svd(matrix: &Cm) -> (Cm, Cm) {
    let fm = to_faer(matrix);
    let svd = fm.svd().expect("SVD failed");
    let u = from_faer(&svd.U());
    let v = from_faer(&svd.V());
    let s_vals: Vec<f64> = fm.singular_values().expect("singular values failed");
    let sigma = diagonal_from_real(&s_vals);
    let s_factor = mat_mul(&mat_mul(&u, &sigma), &adjoint(&u));
    let u_polar = mat_mul(&u, &adjoint(&v));
    (s_factor, u_polar)
}

/// Eigen (Schur-style) diagonalization of a complex unitary matrix.
///
/// Returns `(Q, D)` with `matrix ~ Q D Q^H` and `D` the diagonal matrix of
/// eigenvalues. Mirrors the Python `schur_unitary_diagonalize`
/// (`scipy.linalg.schur` with `output="complex"`).
///
/// Degenerate-spectrum repair: faer's complex EVD does not guarantee that
/// eigenvectors of a repeated eigenvalue are mutually orthogonal (a normal
/// matrix is theoretically unitarily diagonalizable). Columns are grouped
/// by eigenvalue clusters and re-orthonormalized within each group by a
/// modified Gram-Schmidt pass; an orthonormal basis of a degenerate
/// subspace still spans the same eigenspace, so the diagonalization
/// property is preserved, and subspaces of distinct eigenvalues stay
/// (numerically) orthogonal by normality.
pub fn schur_unitary_diagonalize(matrix: &Cm) -> (Cm, Cm) {
    let fm = to_faer(matrix);
    let eig = fm.eigen().expect("EVD failed");
    let q_raw = from_faer(&eig.U());
    let evals: Vec<Complex64> = fm.eigenvalues().expect("eigenvalues failed");
    let q = reorthogonalize_degenerate_eigenvectors(&q_raw, &evals);
    let d = diagonal_from_values(&evals);
    (q, d)
}

/// Re-orthonormalizes eigenvectors inside degenerate subspaces (see the
/// documentation of [`schur_unitary_diagonalize`]).
///
/// Eigenvectors of the same eigenvalue cluster are processed by a modified
/// Gram-Schmidt pass over the complex inner product; orthogonality across
/// clusters is naturally preserved by the spectral property of normal
/// matrices. Exact degeneracies on the unit circle jitter at the `~1e-15`
/// level in floating point while distinct eigenvalues are usually separated
/// by `O(0.1)`, so a `1e-7` clustering tolerance is safe. The function does
/// not move any eigenvector out of its eigenspace, so `M = Q D Q^H` still
/// holds.
fn reorthogonalize_degenerate_eigenvectors(q: &Cm, evals: &[Complex64]) -> Cm {
    let n = q.ncols();
    debug_assert_eq!(q.nrows(), n);
    debug_assert_eq!(evals.len(), n);

    const CLUSTER_TOL: f64 = 1e-7;

    // Clustering: columns of the same eigenvalue (exact degeneracy on the
    // unit circle plus floating-point jitter) join the same group.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for col in 0..n {
        let mut placed = false;
        for group in groups.iter_mut() {
            if (evals[group[0]] - evals[col]).norm() <= CLUSTER_TOL {
                group.push(col);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![col]);
        }
    }

    let mut out = q.clone();
    for group in &groups {
        if group.len() <= 1 {
            continue;
        }
        // Modified Gram-Schmidt within the group; the subspace is unchanged.
        for (pos, &col) in group.iter().enumerate() {
            let mut v: Vec<Complex64> = (0..q.nrows()).map(|r| out[[r, col]]).collect();
            for &prev in group.iter().take(pos) {
                let mut proj = Complex64::new(0.0, 0.0);
                for r in 0..q.nrows() {
                    proj += out[[r, prev]].conj() * v[r];
                }
                for r in 0..q.nrows() {
                    v[r] -= proj * out[[r, prev]];
                }
            }
            let norm: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for r in 0..q.nrows() {
                    out[[r, col]] = v[r] / norm;
                }
            }
            // A tiny norm keeps the original vector: faer guarantees it is a
            // non-zero eigenvector, and near-linear dependence within the
            // group cannot be repaired by GS but is no worse than before.
        }
    }
    out
}

/// Builds a complex diagonal matrix from real values (zero imaginary parts).
fn diagonal_from_real(values: &[f64]) -> Cm {
    let v: Vec<Complex64> = values.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    diagonal_from_values(&v)
}

/// Builds a diagonal unitary square root from unit-modulus eigenvalues.
///
/// Mirrors the Python `build_unitary_diagonal_sqrt`:
/// `normalized = e / |e|`, `phase = exp(0.5j * arg(normalized))`.
pub fn build_unitary_diagonal_sqrt(eigenvalues: &Cm) -> Cm {
    let n = eigenvalues.nrows();
    let mut d = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        let e = eigenvalues[[i, i]];
        let mag = e.norm();
        let normalized = if mag > 0.0 {
            e / mag
        } else {
            Complex64::new(1.0, 0.0)
        };
        // sqrt(e^{i theta}) = e^{i theta / 2}
        d[[i, i]] = normalized.sqrt();
    }
    d
}

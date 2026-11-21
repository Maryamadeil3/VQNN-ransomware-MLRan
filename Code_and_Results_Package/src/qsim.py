"""Exact (noise-free) state-vector simulation of the VQNN, vectorised over samples.

Circuit (n qubits, L layers), the single definition used everywhere in the paper:
  |0..0> -> prod_q RY(pi*x_q) -> [ prod_q RZ(phi_lq) RY(theta_lq) -> CNOT ring q->q+1 (mod n) ] x L
  output  f(x) = <Z_0>,   P(ransomware | x) = (1 - f(x)) / 2
Options (ablations): entangle=False removes the CNOT ring; rz=False removes the RZ gates.
`build_qiskit` constructs the identical circuit in Qiskit; `crosscheck` compares both.
Run `python qsim.py` to repeat all consistency checks against Qiskit / Qiskit Aer.
"""
import numpy as np

def n_params(n, L, rz=True): return n * L * (2 if rz else 1)

def _apply_1q(psi, U, q):               # psi: (N, 2, ..., 2); U: (2,2) or (N,2,2)
    psi = np.moveaxis(psi, q + 1, -1)
    psi = psi @ U.T if U.ndim == 2 else np.einsum("n...j,nij->n...i", psi, U)
    return np.moveaxis(psi, -1, q + 1)

def _cnot(psi, c, t):
    psi = np.moveaxis(psi, (c + 1, t + 1), (-2, -1)).copy()
    psi[..., 1, :] = psi[..., 1, ::-1]
    return np.moveaxis(psi, (-2, -1), (c + 1, t + 1))

def _ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2); return np.array([[c, -s], [s, c]], dtype=complex)

def _rz(a):
    return np.array([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]])

def encode(X):
    """RY(pi x_q) on every qubit of |0..0>: product state, shape (N, 2**n)."""
    N, n = X.shape
    psi = np.ones((N,) + (1,) * n, dtype=complex)
    for q in range(n):
        v = np.stack([np.cos(np.pi * X[:, q] / 2), np.sin(np.pi * X[:, q] / 2)], 1)
        psi = psi * v.reshape((N,) + (1,) * q + (2,) + (1,) * (n - q - 1))
    return psi.reshape(N, -1)

def unitary(params, n, L=2, entangle=True, rz=True):
    """The variational block as a 2**n x 2**n matrix (it does not depend on the sample)."""
    psi = np.eye(2 ** n, dtype=complex).reshape((2 ** n,) + (2,) * n)   # batch of basis states
    p = np.asarray(params).reshape(L, n, 2 if rz else 1)
    for l in range(L):
        for q in range(n):
            psi = _apply_1q(psi, _ry(p[l, q, 0]), q)
            if rz: psi = _apply_1q(psi, _rz(p[l, q, 1]), q)
        if entangle and n > 1:
            for q in range(n): psi = _cnot(psi, q, (q + 1) % n)
    return psi.reshape(2 ** n, -1).T                                    # column j = U|j>

_cache = {}
def expect_z0(X, params, L=2, entangle=True, rz=True):
    """X: (N, n) in [0,1]; returns <Z_0> for every sample, shape (N,). Exact, no shots."""
    N, n = X.shape
    key = (id(X), X.shape)
    if key not in _cache:
        if len(_cache) > 64: _cache.clear()
        _cache[key] = (X, encode(X).real.copy())                                   # keep X alive so id() stays unique
    U = unitary(params, n, L, entangle, rz)
    z = np.r_[np.ones(2 ** (n - 1)), -np.ones(2 ** (n - 1))]            # Z on qubit 0 (most significant index)
    M = ((U.conj().T * z) @ U).real                                     # encoded states are real -> Re(U^dag Z U) suffices
    E = _cache[key][1]
    return ((E @ M) * E).sum(1)

def prob(X, params, **kw): return (1 - expect_z0(X, params, **kw)) / 2

def build_qiskit(x, params, L=2, entangle=True, rz=True):
    from qiskit import QuantumCircuit
    n = len(x); qc = QuantumCircuit(n)
    for q in range(n): qc.ry(np.pi * x[q], q)
    p = np.asarray(params).reshape(L, n, 2 if rz else 1)
    for l in range(L):
        for q in range(n):
            qc.ry(p[l, q, 0], q)
            if rz: qc.rz(p[l, q, 1], q)
        if entangle and n > 1:
            for q in range(n): qc.cx(q, (q + 1) % n)
    return qc

def crosscheck(n=4, L=2, trials=20, seed=0):
    from qiskit.quantum_info import Statevector, SparsePauliOp
    r = np.random.default_rng(seed); worst = 0
    for kw in ({}, {"entangle": False}, {"rz": False}):
        for _ in range(trials):
            x = r.random(n); th = r.uniform(0, 2 * np.pi, n_params(n, L, kw.get("rz", True)))
            z = Statevector(build_qiskit(x, th, L, **kw)).expectation_value(
                SparsePauliOp("I" * (n - 1) + "Z")).real
            worst = max(worst, abs(z - expect_z0(x[None], th, L, **kw)[0]))
    return worst

# ---------------------------------------------------------------------------------------------
# Noisy evaluation (density matrices). Used only to TEST an already trained circuit under
# hardware-like noise; training is always exact. Noise model:
#   * single-qubit depolarizing channel with probability p1 after every single-qubit gate
#   * two-qubit depolarizing channel with probability p2 after every CNOT
#   * symmetric readout error e on the measured qubit
#   * finite sampling: <Z0> estimated from `shots` measurements (shots=None -> exact expectation)
# `qiskit_crosscheck_noisy` compares this implementation with Qiskit Aer's density-matrix simulator.
_P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]

def _op_on(n, ops):
    """Kronecker product placing 2x2 operators `ops` {qubit: op} on n qubits (qubit 0 = most significant)."""
    M = np.array([[1.0 + 0j]])
    for q in range(n):
        M = np.kron(M, ops.get(q, np.eye(2)))
    return M

def _gate_list(params, n, L, entangle, rz):
    p = np.asarray(params).reshape(L, n, 2 if rz else 1); g = []
    for l in range(L):
        for q in range(n):
            g.append(("1q", q, _ry(p[l, q, 0])))
            if rz: g.append(("1q", q, _rz(p[l, q, 1])))
        if entangle and n > 1:
            for q in range(n): g.append(("cx", q, (q + 1) % n))
    return g

def _cnot_matrix(n, c, t):
    d = 2 ** n; M = np.zeros((d, d))
    for i in range(d):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[c]: bits[t] ^= 1
        M[sum(b << (n - 1 - k) for k, b in enumerate(bits)), i] = 1
    return M

def expect_z0_noisy(X, params, L=2, entangle=True, rz=True, p1=0.0, p2=0.0, readout=0.0, shots=None, rng=None):
    N, n = X.shape; d = 2 ** n
    psi = encode(X)                                             # (N, d) encoded product states
    rho = np.einsum("ni,nj->nij", psi, psi.conj())
    pauli1 = {q: [_op_on(n, {q: P}) for P in _P[1:]] for q in range(n)}
    def dep1(rho, q):
        if p1 <= 0: return rho
        return (1 - p1) * rho + p1 / 3 * sum(P @ rho @ P.conj().T for P in pauli1[q])
    def dep2(rho, a, b):
        if p2 <= 0: return rho
        acc = 0
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0: continue
                P = _op_on(n, {a: _P[i], b: _P[j]}); acc = acc + P @ rho @ P.conj().T
        return (1 - p2) * rho + p2 / 15 * acc
    for q in range(n):                                          # noise after the encoding rotations
        rho = dep1(rho, q)
    for kind, a, b in _gate_list(params, n, L, entangle, rz):
        if kind == "1q":
            U = _op_on(n, {a: b}); rho = U @ rho @ U.conj().T; rho = dep1(rho, a)
        else:
            U = _cnot_matrix(n, a, b); rho = U @ rho @ U.T; rho = dep2(rho, a, b)
    p0 = np.real(np.einsum("nii->ni", rho))[:, : d // 2].sum(1)  # P(qubit 0 = 0)
    p0 = p0 * (1 - readout) + (1 - p0) * readout
    if shots:
        rng = rng or np.random.default_rng(0)
        p0 = rng.binomial(shots, np.clip(p0, 0, 1)) / shots
    return 2 * p0 - 1

def prob_noisy(X, params, **kw): return (1 - expect_z0_noisy(X, params, **kw)) / 2

def qiskit_crosscheck_noisy(n=4, L=2, p1=0.01, p2=0.05, trials=5, seed=1):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit import transpile
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(4 / 3 * p1, 1), ["ry", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(16 / 15 * p2, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    r = np.random.default_rng(seed); worst = 0
    for _ in range(trials):
        x = r.random(n); th = r.uniform(0, 2 * np.pi, n_params(n, L))
        qc = build_qiskit(x, th, L); qc.save_density_matrix()
        rho = np.asarray(sim.run(transpile(qc, sim, optimization_level=0)).result().data()["density_matrix"])
        p0 = sum(np.real(rho[i, i]) for i in range(2 ** n) if (i & 1) == 0)   # Qiskit: qubit 0 = least significant bit
        worst = max(worst, abs((2 * p0 - 1) - expect_z0_noisy(x[None], th, L, p1=p1, p2=p2)[0]))
    return worst


if __name__ == "__main__":
    print("max |numpy - Qiskit statevector|, 4 qubits:", crosscheck(4), " 8 qubits:", crosscheck(8, trials=5))
    X = np.random.default_rng(0).random((50, 4)); th = np.random.default_rng(1).uniform(0, 2 * np.pi, 16)
    print("max |exact - density matrix without noise|:", np.abs(expect_z0(X, th) - expect_z0_noisy(X, th)).max())
    try:
        print("max |density matrix - Qiskit Aer| with noise:", qiskit_crosscheck_noisy())
    except ImportError:
        print("qiskit-aer not installed: noisy cross-check skipped")
    print(build_qiskit([.1, .2, .3, .4], np.arange(16) / 10).draw())

// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cqlib_c.h"

static void test_basic_circuit(void) {
    CCircuit* circuit = circuit_new(2);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 2);
    assert(circuit_num_operations(circuit) == 0);

    assert(circuit_h(circuit, 0) == 0);
    assert(circuit_x(circuit, 1) == 0);
    assert(circuit_y(circuit, 0) == 0);
    assert(circuit_z(circuit, 1) == 0);
    assert(circuit_rx(circuit, 0, 0.25) == 0);
    assert(circuit_ry(circuit, 1, 0.5) == 0);
    assert(circuit_rz(circuit, 0, 0.75) == 0);
    assert(circuit_cx(circuit, 0, 1) == 0);
    assert(circuit_cz(circuit, 1, 0) == 0);
    assert(circuit_measure(circuit, 0) == 0);
    assert(circuit_reset(circuit, 1) == 0);

    assert(circuit_num_operations(circuit) == 11);
    assert(circuit_validate(circuit) == 0);
    circuit_free(circuit);
}

static void test_errors(void) {
    assert(circuit_num_qubits(NULL) == 0);
    assert(circuit_num_operations(NULL) == 0);
    assert(circuit_num_parameters(NULL) == 0);
    assert(circuit_validate(NULL) == -1);
    assert(circuit_h(NULL, 0) == -1);
    circuit_free(NULL);

    CCircuit* circuit = circuit_new(1);
    assert(circuit_h(circuit, 1) == -2);
    assert(circuit_cx(circuit, 0, 1) == -2);
    circuit_free(circuit);
}

static void test_symbolic_parameters(void) {
    CParameter* theta = param_parse("theta");
    CParameter* phi = param_parse("phi");
    assert(theta != NULL);
    assert(phi != NULL);

    const char* bindings = "theta:0.5,phi:1.25";
    assert(fabs(param_evaluate(theta, bindings) - 0.5) < 1e-12);
    assert(fabs(param_evaluate(phi, bindings) - 1.25) < 1e-12);

    CCircuit* circuit = circuit_new(2);
    assert(circuit_rx_param(circuit, 0, theta) == 0);
    assert(circuit_ry_param(circuit, 1, phi) == 0);
    assert(circuit_rz_param(circuit, 0, theta) == 0);
    assert(circuit_cx(circuit, 0, 1) == 0);
    assert(circuit_num_operations(circuit) == 4);
    assert(circuit_num_parameters(circuit) == 2);

    CCircuit* assigned = circuit_assign_params(circuit, bindings);
    assert(assigned != NULL);
    assert(circuit_num_operations(assigned) == 4);
    assert(circuit_num_parameters(assigned) == 0);
    assert(circuit_validate(assigned) == 0);

    circuit_free(assigned);
    circuit_free(circuit);
    param_free(theta);
    param_free(phi);
}

static void test_extended_gates(void) {
    CCircuit* circuit = circuit_new(3);
    assert(circuit != NULL);

    /* Section 2.1.1: single-qubit no-param gates */
    assert(circuit_i(circuit, 0) == 0);
    assert(circuit_s(circuit, 1) == 0);
    assert(circuit_sdg(circuit, 2) == 0);
    assert(circuit_t(circuit, 0) == 0);
    assert(circuit_tdg(circuit, 1) == 0);
    assert(circuit_x2p(circuit, 2) == 0);
    assert(circuit_x2m(circuit, 0) == 0);
    assert(circuit_y2p(circuit, 1) == 0);
    assert(circuit_y2m(circuit, 2) == 0);

    /* Section 2.1.2: single-qubit param gates (numeric) */
    assert(circuit_phase(circuit, 0, 0.5) == 0);
    assert(circuit_u(circuit, 1, 0.1, 0.2, 0.3) == 0);
    assert(circuit_xy(circuit, 2, 0.4) == 0);
    assert(circuit_xy2p(circuit, 0, 0.5) == 0);
    assert(circuit_xy2m(circuit, 1, 0.6) == 0);
    assert(circuit_rxy(circuit, 2, 0.7, 0.8) == 0);

    /* Section 2.1.3: two-qubit no-param gates */
    assert(circuit_cy(circuit, 0, 1) == 0);
    assert(circuit_swap(circuit, 1, 2) == 0);

    /* Section 2.1.4: two-qubit param gates (numeric) */
    assert(circuit_rxx(circuit, 0, 1, 0.5) == 0);
    assert(circuit_ryy(circuit, 1, 2, 0.5) == 0);
    assert(circuit_rzz(circuit, 0, 2, 0.5) == 0);
    assert(circuit_rzx(circuit, 1, 0, 0.5) == 0);
    assert(circuit_crx(circuit, 0, 1, 0.5) == 0);
    assert(circuit_cry(circuit, 1, 2, 0.5) == 0);
    assert(circuit_crz(circuit, 0, 2, 0.5) == 0);
    assert(circuit_fsim(circuit, 1, 0, 0.3, 0.7) == 0);

    /* Section 2.1.5: three-qubit gate */
    assert(circuit_ccx(circuit, 0, 1, 2) == 0);

    /* Section 2.1.6: barrier */
    uint32_t barrier_qubits[] = {0, 1};
    assert(circuit_barrier(circuit, barrier_qubits, 2) == 0);
    assert(circuit_barrier(circuit, NULL, 0) == 0);

    /* Section 2.1.7: properties */
    assert(circuit_width(circuit) == 3);
    assert(circuit_qubits_len(circuit) == 3);
    uint32_t qbuf[3];
    assert(circuit_qubits(circuit, qbuf, 3) == 3);
    assert(qbuf[0] == 0 && qbuf[1] == 1 && qbuf[2] == 2);
    assert(circuit_depth(circuit, false) > 0);
    assert(circuit_remove_operation(circuit, 0) == 0);

    /* NaN rejection */
    assert(circuit_u(circuit, 0, NAN, 0.0, 0.0) == -3);
    assert(circuit_i(circuit, 10) == -2);

    /* Inverse and decompose */
    CCircuit* inv = circuit_inverse(circuit);
    assert(inv != NULL);
    circuit_free(inv);

    CCircuit* dec = circuit_decompose(circuit);
    assert(dec != NULL);
    circuit_free(dec);

    /* to_matrix */
    size_t mlen = circuit_to_matrix_len(circuit, NULL, 0);
    assert(mlen == 64); /* 2^3 * 2^3 = 64 */
    double* mbuf = (double*)malloc(mlen * 2 * sizeof(double));
    assert(circuit_to_matrix(circuit, NULL, 0, mbuf, mlen * 2) == mlen);
    free(mbuf);

    /* global phase */
    assert(circuit_set_global_phase(circuit, 0.5) == 0);
    assert(circuit_set_global_phase(circuit, NAN) == -3);

    circuit_free(circuit);
}

static void test_circuit_gate_operation_matrix(void) {
    const uint32_t orders[][2] = {{0, 1}, {1, 0}, {7, 2}};
    const uint32_t targets[] = {0, 1};
    for (size_t order = 0; order < 3; ++order) {
        for (int parameterized = 0; parameterized < 2; ++parameterized) {
            CCircuit* body = circuit_new(0);
            assert(body != NULL);
            assert(circuit_add_qubits(body, orders[order], 2) == 0);
            /* Build on dense ids, then map into the explicit storage order. */
            CCircuit* definition = circuit_new(2);
            assert(definition != NULL);
            CParameter* theta = param_parse("theta");
            assert(theta != NULL);
            if (parameterized) {
                assert(circuit_rx_param(definition, 0, theta) == 0);
            } else {
                assert(circuit_x(definition, 0) == 0);
            }
            assert(circuit_compose(body, definition, orders[order], 2) == 0);
            circuit_free(definition);
            assert(circuit_set_global_phase(body, 0.13) == 0);
            CCircuitGate* gate = circuit_to_gate(body, "first_gate");
            assert(gate != NULL);
            for (size_t application = 0; application < 2; ++application) {
                double angle = application == 0 ? 0.37 : -0.61;
                CParameter* argument = param_from_double(angle);
                const CParameter* arguments[] = {argument};
                assert(argument != NULL);
                CCircuit* circuit = circuit_new(2);
                assert(circuit != NULL);
                assert(circuit_circuit_gate(circuit, gate, targets, 2,
                                            parameterized ? arguments : NULL,
                                            parameterized ? 1 : 0) == 0);
                /* CParameter arguments are stored as parameter-table indices.
                 * Resolve them in the owning circuit before extracting the raw
                 * DAG Operation, which has no access to that parameter table.
                 * The gate definition still contains theta; matrix() must bind it. */
                CCircuit* resolved = circuit_assign_params(circuit, "");
                assert(resolved != NULL && circuit_num_parameters(resolved) == 0);
                circuit_free(circuit);
                circuit = resolved;
                CCircuitDag* dag = circuit_dag_from_circuit(circuit);
                assert(dag != NULL && circuit_dag_op_nodes_len(dag) == 1);
                uint32_t node;
                assert(circuit_dag_op_nodes(dag, &node, 1) == 0);
                COperation* operation = circuit_dag_operation(dag, node);
                assert(operation != NULL);
                size_t rows = 0, cols = 0;
                assert(operation_matrix_len(operation) == 16);
                assert(operation_matrix_dims(operation, &rows, &cols) == 0);
                assert(rows == 4 && cols == 4);
                double local[32], full[32];
                assert(operation_matrix(operation, local, 32) == 16);
                assert(circuit_to_matrix(circuit, NULL, 0, full, 32) == 16);
                for (size_t row = 0; row < 4; ++row) {
                    for (size_t col = 0; col < 4; ++col) {
                        for (int local_basis = 0; local_basis < 2; ++local_basis) {
                            size_t flip = local_basis ? 2 : 1;
                            double re = 0, im = 0;
                            if (parameterized) {
                                if (row == col) re = cos(angle / 2);
                                if (row == (col ^ flip)) im = -sin(angle / 2);
                            } else if (row == (col ^ flip)) {
                                re = 1;
                            }
                            double expected_re = re * cos(0.13) - im * sin(0.13);
                            double expected_im = re * sin(0.13) + im * cos(0.13);
                            const double* matrix = local_basis ? local : full;
                            size_t index = 2 * (row * 4 + col);
                            assert(fabs(matrix[index] - expected_re) < 1e-12);
                            assert(fabs(matrix[index + 1] - expected_im) < 1e-12);
                        }
                    }
                }
                operation_free(operation);
                circuit_dag_free(dag);
                circuit_free(circuit);
                param_free(argument);
            }
            circuit_gate_free(gate);
            param_free(theta);
            circuit_free(body);
        }
    }
}

int main(void) {
    test_basic_circuit();
    test_errors();
    test_symbolic_parameters();
    test_extended_gates();
    test_circuit_gate_operation_matrix();
    printf("binding-c circuit tests passed\n");
    return 0;
}

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Generates quantum volume circuits
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info.random import random_unitary

MAX_INT = np.iinfo(np.int32).max-1


def qv_circuits(qubit_list=None, trials=1, seed=None):
    """
    Return a list of square quantum volume circuits (depth=width)

    The qubit_list is specified as a list integers. For each integer, circuits
    with the given number of qubits are generated.
    Args:
        qubit_list (list): List of integers giving number of qubits to incliude.
        trials (int): Number of random iterations
        seed (int): Seed used in random number generator.

    Returns:
        qv_circs: List of lists of circuits for the qv sequences
    """
    if seed is None:
        seed = np.random.randint(MAX_INT)
    rnd = np.random.RandomState(seed)  # pylint: disable=no-member
    qv_seeds = rnd.randint(MAX_INT, size=len(qubit_list))
    qv_circs = []
    for idx, qub_int in enumerate(qubit_list):
        qv_gen = QuantumVolumeGenerator(qub_int, seed=qv_seeds[idx])
        qv_circs.append(qv_gen(trials))
    return qv_circs

class QuantumVolumeGenerator():
    """A Quantum Volume circuit generator.
    """
    def __init__(self, qubits, seed=None):
        """A generator for quantum volume circuits.

        Generates a collection of quantum circuits with names
        ``QV{volume}_{seed}+{offset}``, where volume is :math:`2^{qubits}`,
        ``seed`` is the seed used in the random number generator, and
        ``offset`` is the number of times the generator was called; the first
        call has ``offset=0``.

        Parameters:
            qubits (int): Dimension of QV circuit.
            seed (int): Optional seed at which to start generator

        Example:

        .. jupyter-execute::

            from alexandria.quantum_volume import QuantumVolumeGenerator
            qv_gen = QuantumVolumeGenerator(4, seed=9876)

            qv16_circs = qv_gen(5)
            for circ in qv16_circs:
                print(circ.name)

        """
        if seed is None:
            seed = np.random.randint(MAX_INT)
        self.seed = seed
        self.rnd = np.random.RandomState(self.seed)  # pylint: disable=no-member
        self.qubits = qubits
        qv_dim = 2**qubits
        self.circ_name = 'QV{}_{}'.format(qv_dim, self.seed)
        self.count = 0

    def __call__(self, samples=None):
        """Creates a collection of Quantum Volume circuits.

        Parameters:
            samples (int): Number of circuits to generate.

        Returns:
            list: A list of QuantumCircuits.
        """
        if samples is None:
            samples = 1
        out = []
        for _ in range(samples):
            qc_name = self.circ_name + '+{}'.format(self.count)
            qc = QuantumCircuit(self.qubits, name=qc_name)
            for _ in range(self.qubits):
                # Generate uniformly random permutation Pj of [0...n-1]
                perm = self.rnd.permutation(self.qubits)
                # For each pair p in Pj, generate Haar random SU(4)
                for k in range(int(self.qubits/2)):
                    U = random_unitary(4, seed=self.rnd.randint(MAX_INT))
                    pair = int(perm[2*k]), int(perm[2*k+1])
                    qc.append(U, [pair[0], pair[1]])
            out.append(qc)
            self.count += 1
        return out

    def __next__(self):
        return self.__call__()[0]

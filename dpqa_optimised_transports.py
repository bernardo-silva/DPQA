""" DPQA Simplified """
from collections import defaultdict
import json
import time
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Mapping, Sequence

from pysat.card import CardEnc
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from z3 import (
    Solver,
    Optimize,
    sat,
    Bool,
    BoolRef,
    Int,
    ArithRef,
    Not,
    Or,
    And,
    Implies,
    is_true,
    Sum,
    Abs,
)

PYSAT_ENCODING = 2  # default choice: sequential counter

@dataclass
class Circuit:
    """class to encode a quantum circuit."""

    circuit: QuantumCircuit
    num_qubits: int
    dag: DAGCircuit
    gates: list[DAGOpNode]
    num_gates: int
    num_mq_gates: int
    gates_qubits: list[tuple[int, ...]]
    gates_indexes: dict[int, list[int]]

    @staticmethod
    def from_qiskit_circuit(circuit: QuantumCircuit) -> "Circuit":
        """Construct a Circuit from a QuantumCircuit."""
        num_qubits = circuit.num_qubits
        dag = circuit_to_dag(circuit)
        gates = dag.op_nodes()
        gates_qubits = [node_qubits(circuit, node) for node in gates]
        num_gates = len(gates)
        num_mq_gates = len([gate for gate in gates_qubits if len(gate) > 1])

        gates_indexes = defaultdict(list)
        for q in range(num_qubits):
            for t, qubits in enumerate(gates_qubits):
                if q in qubits:
                    gates_indexes[q].append(t)

        return Circuit(
            circuit,
            num_qubits,
            dag,
            gates,
            num_gates,
            num_mq_gates,
            gates_qubits,
            gates_indexes,
        )


def node_qubits(circ: QuantumCircuit, node: DAGOpNode) -> tuple[int, ...]:
    """Get the qubits of a DAGOpNode."""
    return tuple(sorted(map(lambda q: circ.find_bit(q).index, node.qargs)))


def split_circuit(circ: QuantumCircuit, window_size: int = 5) -> list[QuantumCircuit]:
    """Split a circuit into smaller circuits."""
    dag = circuit_to_dag(circ)
    stages: list[list[DAGOpNode]] = []
    for layer in dag.layers():
        gates = [gate for gate in layer["graph"].op_nodes() if gate.op.num_qubits >= 2]
        stages.append(gates)

    gate_windows = [[gate for stage in stages[i : i + window_size] for gate in stage]
                    for i in range(0, len(stages), window_size)]

    circuits = []
    for gates in gate_windows:
        new_circ = QuantumCircuit.copy_empty_like(circ)
        for gate in gates:
            new_circ.append(gate.op, gate.qargs)
        circuits.append(new_circ)
    return circuits

@dataclass
class Architecture:
    """class to encode the architecture of the quantum device."""

    n_x: int = 0
    n_y: int = 0
    n_c: int = 0
    n_r: int = 0


@dataclass
class DPQASettings:
    """class to encode the settings of the compilation problem."""

    name: str = ""
    directory: str = ""
    minimize_distance: bool = False
    verbose: bool = False
    all_aod: bool = False
    no_transfer: bool = False
    optimal_ratio: float = 0
    row_per_site: int = 3
    cardinality_encoding: str = "pysat"


class DPQAOptimisedTransports:
    """class to encode the compilation problem to SMT and solves using Z3."""

    def __init__(
        self,
        name: str,
        directory: str = "",
        bounds: tuple[int, int, int, int] = (16, 16, 16, 16),
        minimize_distance: bool = False,
        verbose: bool = False,
        all_aod: bool = False,
        no_transfer: bool = False,
    ):
        self.solver: Solver = Solver()
        self.satisfiable: bool = False

        self.circuit: Circuit | None = None
        self.architecture: Architecture = Architecture(*bounds)

        self.settings = DPQASettings(
            name,
            directory,
            minimize_distance,
            verbose,
            all_aod,
            no_transfer,
        )

        self.result_json = {"name": name, "layers": []}

    def add_metadata(self, metadata: Mapping[str, Any]):
        self.result_json = {}
        for key, val in metadata.items():
            self.result_json[key] = val

    def write_settings_json(self):
        pass
        # self.result_json["sat"] = self.satisfiable
        # # self.result_json["n_t"] = self.num_transports
        # self.result_json["n_q"] = self.circuit.num_qubits
        # self.result_json["all_aod"] = self.settings.all_aod
        # self.result_json["no_transfer"] = self.settings.no_transfer
        # self.result_json["n_c"] = self.architecture.n_c
        # self.result_json["n_r"] = self.architecture.n_r
        # self.result_json["n_x"] = self.architecture.n_x
        # self.result_json["n_y"] = self.architecture.n_y
        # self.result_json["row_per_site"] = self.settings.row_per_site
        # self.result_json["n_g"] = self.circuit.num_gates
        # self.result_json["g_q"] = self.gates
        # self.result_json["g_s"] = self.gate_names

    def _constraint_all_aod(self, num_stage: int, a: Sequence[Sequence[BoolRef]]):
        """All qubits on AODs"""
        if self.settings.all_aod:
            for q, s in product(range(self.circuit.num_qubits), range(num_stage)):
                self.solver.add(a[q][s])

    def _constraint_no_transfer(self, num_stage: int, a: Sequence[Sequence[BoolRef]]):
        """No transfer from AOD to SLM and vice versa"""
        if self.settings.no_transfer:
            for q, s in product(range(self.circuit.num_gates), range(num_stage)):
                self.solver.add(a[q][s] == a[q][0])

    def _constraint_var_bounds(
        self,
        num_stage: int,
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
    ):
        """Bounds on the variables"""
        for q in range(self.circuit.num_qubits):
            for s in range(1, num_stage):
                # starting from s=1 since the values with s=0 are loaded
                self.solver.add(x[q][s] >= 0)
                self.solver.add(x[q][s] < self.architecture.n_x)
                self.solver.add(y[q][s] >= 0)
                self.solver.add(y[q][s] < self.architecture.n_y)
            for s in range(num_stage):
                # starting from s=0 since the solver finds these values
                self.solver.add(c[q][s] >= 0)
                self.solver.add(c[q][s] < self.architecture.n_c)
                self.solver.add(r[q][s] >= 0)
                self.solver.add(r[q][s] < self.architecture.n_r)

    def _constraint_fixed_slm(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
    ):
        """SLMs do not move"""
        for q in range(self.circuit.num_qubits):
            for s in range(num_stage - 1):
                self.solver.add(Implies(Not(a[q][s]), x[q][s] == x[q][s + 1]))
                self.solver.add(Implies(Not(a[q][s]), y[q][s] == y[q][s + 1]))

    def _constraint_aod_move_together(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
    ):
        """AODs move together"""
        for q in range(self.circuit.num_qubits):
            for s in range(num_stage - 1):
                self.solver.add(Implies(a[q][s], c[q][s + 1] == c[q][s]))
                self.solver.add(Implies(a[q][s], r[q][s + 1] == r[q][s]))
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                for s in range(num_stage - 1):
                    self.solver.add(
                        Implies(
                            And(a[q0][s], a[q1][s], c[q0][s] == c[q1][s]),
                            x[q0][s + 1] == x[q1][s + 1],
                        )
                    )
                    self.solver.add(
                        Implies(
                            And(a[q0][s], a[q1][s], r[q0][s] == r[q1][s]),
                            y[q0][s + 1] == y[q1][s + 1],
                        )
                    )

    def _constraint_aod_order_from_slm(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
    ):
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            for s in range(num_stage - 1):
                if q0 == q1:
                    continue

                self.solver.add(
                    Implies(
                        And(a[q0][s], a[q1][s], c[q0][s] < c[q1][s]),
                        x[q0][s + 1] <= x[q1][s + 1],
                    )
                )
                self.solver.add(
                    Implies(
                        And(a[q0][s], a[q1][s], r[q0][s] < r[q1][s]),
                        y[q0][s + 1] <= y[q1][s + 1],
                    )
                )

    def _constraint_slm_order_from_aod(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
    ):
        # row/col constraints when atom transfer from SLM to AOD
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            for s in range(num_stage):
                if q0 == q1:
                    continue
                self.solver.add(
                    Implies(
                        And(a[q0][s], a[q1][s], x[q0][s] < x[q1][s]),
                        c[q0][s] < c[q1][s],
                    )
                )
                self.solver.add(
                    Implies(
                        And(a[q0][s], a[q1][s], y[q0][s] < y[q1][s]),
                        r[q0][s] < r[q1][s],
                    )
                )

    def _constraint_aod_crowding(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
    ):
        # not too many AOD columns/rows can be together, default 3
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            for s in range(num_stage - 1):
                if q0 == q1:
                    continue
                self.solver.add(
                    Implies(
                        And(
                            a[q0][s],
                            a[q1][s],
                            c[q0][s] - c[q1][s] > self.settings.row_per_site - 1,
                        ),
                        x[q0][s + 1] > x[q1][s + 1],
                    )
                )
                self.solver.add(
                    Implies(
                        And(
                            a[q0][s],
                            a[q1][s],
                            r[q0][s] - r[q1][s] > self.settings.row_per_site - 1,
                        ),
                        y[q0][s + 1] > y[q1][s + 1],
                    )
                )

    def _constraint_aod_crowding_init(
        self,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
    ):
        # not too many AOD cols/rows can be together, default 3, for init stage
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            if q0 == q1:
                continue
            self.solver.add(
                Implies(
                    And(
                        a[q0][0],
                        a[q1][0],
                        c[q0][0] - c[q1][0] > self.settings.row_per_site - 1,
                    ),
                    x[q0][0] > x[q1][0],
                )
            )
            self.solver.add(
                Implies(
                    And(
                        a[q0][0],
                        a[q1][0],
                        r[q0][0] - r[q1][0] > self.settings.row_per_site - 1,
                    ),
                    y[q0][0] > y[q1][0],
                )
            )

    def _constraint_site_crowding(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
    ):
        """Two atoms cannot be in the same AOD site or SLM site. Removed pure_graph condition."""

        # bound number of atoms in each site, needed if not double counting
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                for s in range(num_stage):
                    # Two atoms cannot be in the same AOD site
                    self.solver.add(
                        Implies(
                            And(a[q0][s], a[q1][s]),
                            Or(c[q0][s] != c[q1][s], r[q0][s] != r[q1][s]),
                        )
                    )
                    # Two atoms cannot be in the same SLM site
                    self.solver.add(
                        Implies(
                            And(Not(a[q0][s]), Not(a[q1][s])),
                            Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]),
                        )
                    )

    def _constraint_no_swap(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
    ):
        # no atom transfer if two atoms meet
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                for s in range(1, num_stage):
                    self.solver.add(
                        Implies(
                            And(x[q0][s] == x[q1][s], y[q0][s] == y[q1][s]),
                            And(a[q0][s] == a[q0][s - 1], a[q1][s] == a[q1][s - 1]),
                        )
                    )

    def _solver_init(self, num_stage: int = 2):
        # define the variables and add the constraints that do not depend on
        # the gates to execute. return the variable arrays a, c, r, x, y
        if self.circuit is None:
            raise ValueError("Circuit is not set")

        # self.dpqa = Solver()
        self.solver = Optimize()

        # variables
        a = [
            [Bool(f"a_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.circuit.num_qubits)
        ]
        # for col and row, the data does not matter if atom in SLM
        c = [
            [Int(f"c_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.circuit.num_qubits)
        ]
        r = [
            [Int(f"r_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.circuit.num_qubits)
        ]
        x = [
            [Int(f"x_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.circuit.num_qubits)
        ]
        y = [
            [Int(f"y_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.circuit.num_qubits)
        ]

        # if self.settings.cardinality_encoding == "z3atleast":
        #     self.dpqa = Then(
        #         "simplify", "solve-eqs", "card2bv", "bit-blast", "aig", "sat"
        #     ).solver()

        # Non-circuit dependent constraints
        self.constraint_aod_order_from_prev(x, y, c, r, a)
        self._constraint_all_aod(num_stage, a)
        self._constraint_no_transfer(num_stage, a)
        self._constraint_var_bounds(num_stage, x, y, c, r)

        self._constraint_fixed_slm(num_stage, a, x, y)
        self._constraint_aod_move_together(num_stage, a, x, y, c, r)
        self._constraint_aod_order_from_slm(num_stage, a, x, y, c, r)
        self._constraint_slm_order_from_aod(num_stage, a, x, y, c, r)
        self._constraint_aod_crowding(num_stage, a, x, y, c, r)
        self._constraint_aod_crowding_init(a, x, y, c, r)
        self._constraint_site_crowding(num_stage, a, x, y, c, r)
        self._constraint_no_swap(num_stage, a, x, y)

        return a, c, r, x, y

    def constraint_aod_order_from_prev(
        self,
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
        a: Sequence[Sequence[BoolRef]],
    ):
        if len(self.result_json["layers"]) <= 0:
            return

        variables = self.result_json["layers"][-1]["qubits"]

        for q in range(self.circuit.num_qubits):
            # load location info
            if "x" in variables[q]:
                self.solver.add(x[q][0] == variables[q]["x"])
            if "y" in variables[q]:
                self.solver.add(y[q][0] == variables[q]["y"])
            if "c" in variables[q]:
                self.solver.add(c[q][0] == variables[q]["c"])
            if "r" in variables[q]:
                self.solver.add(r[q][0] == variables[q]["r"])
            if "a" in variables[q]:
                self.solver.add(a[q][0] == bool(variables[q]["a"]))
            if variables[q]["a"] == 1:
                self.solver.add(c[q][0] == variables[q]["c"])
                self.solver.add(r[q][0] == variables[q]["r"])
        # virtually putting everything down to acillary SLMs
        # let solver pick some qubits to AOD, so we don't set a_q,0
        # we also don't set c_q,0 and r_q,0, but enforce ordering when
        # two qubits are both in AOD last round, i.e., don't swap
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                if variables[q0]["a"] != 1 or variables[q1]["a"] != 1:
                    continue

                if variables[q0]["x"] == variables[q1]["x"]:
                    if variables[q0]["c"] < variables[q1]["c"]:
                        self.solver.add(c[q0][0] < c[q1][0])
                    if variables[q0]["c"] > variables[q1]["c"]:
                        self.solver.add(c[q0][0] > c[q1][0])
                    if variables[q0]["c"] == variables[q1]["c"]:
                        self.solver.add(c[q0][0] == c[q1][0])
                if variables[q0]["y"] == variables[q1]["y"]:
                    if variables[q0]["r"] < variables[q1]["r"]:
                        self.solver.add(r[q0][0] < r[q1][0])
                    if variables[q0]["r"] > variables[q1]["r"]:
                        self.solver.add(r[q0][0] > r[q1][0])
                    if variables[q0]["r"] == variables[q1]["r"]:
                        self.solver.add(r[q0][0] == r[q1][0])

    def constraint_dependency_collision(
        self,
        t: Sequence[ArithRef],
    ):
        """Enforce gates that depend on each other to be in order"""

        for (t_g0, gate0), (t_g1, gate1) in combinations(zip(t, self.circuit.gates), 2):
            if gate0 in self.circuit.dag.predecessors(gate1):
                self.solver.add(t_g0 < t_g1)

            elif gate1 in self.circuit.dag.predecessors(gate0):
                self.solver.add(t_g1 < t_g0)

    def constraint_connectivity(
        self,
        num_stage: int,
        t: Sequence[ArithRef],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
    ):
        """Qubits must be in the same site when interacting"""
        for t_g, gate_qubits in zip(t, self.circuit.gates_qubits):
            for s in range(1, num_stage):  # since stage 0 is 'trash'
                # if len(self.gates[g]) == 2:
                for q0, q1 in combinations(gate_qubits, 2):
                    self.solver.add(Implies(t_g == s, x[q0][s] == x[q1][s]))
                    self.solver.add(Implies(t_g == s, y[q0][s] == y[q1][s]))

    def constraint_interaction_exactness(
        self,
        num_stage: int,
        t: Sequence[ArithRef],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
    ):
        qubits = range(self.circuit.num_qubits)
        for q0, q1 in combinations(qubits, 2):
            gates_where_interact = [
                gate
                for gate in self.circuit.gates_indexes[q0]
                if q1 in self.circuit.gates_qubits[gate]
            ]
            # If the qubits interact, when they are in the same site, they must be in the same gate
            if gates_where_interact:
                for s in range(1, num_stage):
                    self.solver.add(
                        Implies(
                            And(x[q0][s] == x[q1][s], y[q0][s] == y[q1][s]),
                            Or(*[t[g] == s for g in gates_where_interact]),
                        )
                    )
            # If the qubits never interact, they must be in different sites
            else:
                for s in range(1, num_stage):
                    self.solver.add(Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]))

                # If the qubits never interact, they must be in different sites
                # qubits = set((q0, q1))
                # if all(not qubits.issubset(set(gate_qubits)) for gate_qubits
                # in self.circuit.gates_qubits):
                #     (self.dpqa).add(Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]))
                # If the qubits interact, when they are in the same site, they
                # must be in the same gate
                # else:
                #     (self.dpqa).add(
                #         Implies(
                #             And(x[q0][s] == x[q1][s], y[q0][s] == y[q1][s]),
                #             Or(*[t[g] == s for g in self.circuit.gates_indexes[(q0, q1)]])
                #         )
                #     )

    def constraint_gate_batch(
        self,
        num_stage: int,
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
        a: Sequence[Sequence[BoolRef]],
    ) -> list[ArithRef]:
        """define the scheduling variables of gates, t. Return t
        add the constraints related to the gates to execute """

        num_gate = self.circuit.num_gates
        t = [Int(f"t_g{g}") for g in range(num_gate)]

        for t_g in t:
            self.solver.add(t_g < num_stage)
            self.solver.add(t_g >= 0)

        self.constraint_dependency_collision(t)
        self.constraint_connectivity(num_stage, t, x, y)
        self.constraint_interaction_exactness(num_stage, t, x, y)

        return t

    def constraint_gate_card_pysat(
        self,
        num_gate: int,
        num_stage: int,
        bound_gate: int,
        t: Sequence[ArithRef],
    ):
        offset = num_gate - 1

        # since stage0 is 'trash', the total number of variables to
        # enforce cardinality is (#stage-1)*#gate. the index starts from 1
        # thus the +1
        numvar = (num_stage - 1) * num_gate + 1

        ancillary = {}
        # get the CNF encoding the cardinality constraint
        cnf = CardEnc.atleast(
            lits=range(1, numvar),
            top_id=numvar - 1,
            bound=bound_gate,
            encoding=PYSAT_ENCODING,
        )
        for conj in cnf:
            or_list = []
            for i in conj:
                val = abs(i)
                idx = val + offset
                if i in range(1, numvar):
                    or_list.append(t[idx % num_gate] == (idx // num_gate))
                elif i in range(-numvar + 1, 0):
                    or_list.append(Not(t[idx % num_gate] == (idx // num_gate)))
                else:
                    if val not in ancillary:
                        ancillary[val] = Bool(f"anx_{val}")
                    if i < 0:
                        or_list.append(Not(ancillary[val]))
                    else:
                        or_list.append(ancillary[val])
            self.solver.add(Or(*or_list))

    def constraint_gate_card(
        self,
        bound_gate: int,
        num_stage: int,
        t: Sequence[ArithRef],
    ):
        # add the cardinality constraints on the number of gates

        method = self.settings.cardinality_encoding
        num_gate = self.circuit.num_gates
        if method == "summation":
            # (self.dpqa).add(sum([If(t[g] == s, 1, 0) for g in range(num_gate)
            #                     for s in range(1, num_stage)]) >= bound_gate)
            raise ValueError()
        if method == "z3atleast":
            # tmp = [(t[g] == s) for g in range(num_gate)
            #        for s in range(1, num_stage)]
            # (self.dpqa).add(AtLeast(*tmp, bound_gate))
            raise ValueError()
        if method == "pysat":
            self.constraint_gate_card_pysat(num_gate, num_stage, bound_gate, t)
        else:
            raise ValueError("cardinality method unknown")

    def read_partial_solution(
        self,
        s: int,
        model: Any,
        a: Sequence[Sequence[BoolRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
    ):
        real_s = len(self.result_json["layers"])
        if real_s == 0 and s == 0:
            real_s = -1
        if self.settings.verbose:
            print(f"        stage {real_s}:")
        layer = {}
        layer["qubits"] = []
        for q in range(self.circuit.num_qubits):
            layer["qubits"].append(
                {
                    "id": q,
                    "a": 1 if is_true(model[a[q][s]]) else 0,
                    "x": model[x[q][s]].as_long(),
                    "y": model[y[q][s]].as_long(),
                    "c": model[c[q][s]].as_long(),
                    "r": model[r[q][s]].as_long(),
                }
            )
            if self.settings.verbose:
                if is_true(model[a[q][s]]):
                    print(
                        f"        q_{q} is at ({model[x[q][s]].as_long()}, "
                        f"{model[y[q][s]].as_long()})"
                        f" AOD c_{model[c[q][s]].as_long()},"
                        f" r_{model[r[q][s]].as_long()}"
                    )
                else:
                    print(
                        f"        q_{q} is at ({model[x[q][s]].as_long()}, "
                        f"{model[y[q][s]].as_long()}) SLM"
                        f" c_{model[c[q][s]].as_long()},"
                        f" r_{model[r[q][s]].as_long()}"
                    )
        return layer

    def process_partial_solution(
        self,
        num_stage: int,
        a: Sequence[Sequence[BoolRef]],
        c: Sequence[Sequence[ArithRef]],
        r: Sequence[Sequence[ArithRef]],
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
        t: Sequence[ArithRef],
    ):
        model = self.solver.model()

        for s in range(num_stage):
            layer = self.read_partial_solution(s, model, a, c, r, x, y)

            if s == 0 and self.result_json["layers"]:
                # if there is a 'previous last stage' and we're at the first
                # stage of the partial solution, we write over the last a/c/r's
                for q in range(self.circuit.num_qubits):
                    self.result_json["layers"][-1]["qubits"][q]["a"] = (
                        1 if is_true(model[a[q][s]]) else 0
                    )
                    self.result_json["layers"][-1]["qubits"][q]["c"] = model[
                        c[q][s]
                    ].as_long()
                    self.result_json["layers"][-1]["qubits"][q]["r"] = model[
                        r[q][s]
                    ].as_long()

            if s > 0:
                # otherwise, only append the x/y/z/c/r data when we're not at
                # the first stage of the partial solution. The first stage is
                # 'trash' and also its x/y are loaded from pervious solutions.
                layer["gates"] = []
                gates_done = []
                for g in range(self.circuit.num_gates):
                    if model[t[g]].as_long() == s:
                        qubits = self.circuit.gates_qubits[g]
                        if self.settings.verbose:
                            print(f"        CZ(q_{qubits[0]}," f" q_{qubits[1]})")
                        layer["gates"].append(
                            {
                                f"q{n}": qubit
                                for n, qubit in enumerate(qubits)
                                # "id": self.circuit.gates_indexes[qubits],
                                # "q0": qubits[0],
                                # "q1": qubits[1],
                            }
                        )
                        gates_done.append(g)

                self.result_json["layers"].append(layer)

    def minimize_distance(
        self,
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
    ):
        xdist = Sum([Abs(x1 - x0) for xq in x for x0, x1 in zip(xq, xq[1:])])
        ydist = Sum([Abs(y1 - y0) for yq in y for y0, y1 in zip(yq, yq[1:])])
        self.solver.minimize(xdist + ydist)

    def minimize_movement(
        self,
        x: Sequence[Sequence[ArithRef]],
        y: Sequence[Sequence[ArithRef]],
    ):
        movements = Sum(
            [
                Or(x0 != x1, y0 != y1)
                for xq, yq in zip(x, y)
                for x0, x1, y0, y1 in zip(xq, xq[1:], yq, yq[1:])
            ]
        )

        self.solver.minimize(movements)

    def solve_optimal(self, step: int):
        """optimal solving with step steps"""

        if self.circuit is None:
            raise ValueError("circuit is not set")

        print(f"optimal solving with {step} step")
        bound_gate = self.circuit.num_gates

        a, c, r, x, y = self._solver_init(step + 1)
        t = self.constraint_gate_batch(step + 1, x, y, c, r, a)
        self.constraint_gate_card(bound_gate, step + 1, t)
        if self.settings.minimize_distance:
            # self.minimize_distance(x, y)
            self.minimize_movement(x, y)

        try:
            solved_batch_gates = self.solver.check() == sat
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            return

        while not solved_batch_gates:
            print(f"    no solution, step={step} too small")
            step += 1
            if step > self.circuit.num_mq_gates + 1:
                print("No solution found")
                raise ValueError("No solution found")
            a, c, r, x, y = self._solver_init(step + 1)  # self.dpqa is cleaned
            t = self.constraint_gate_batch(step + 1, x, y, c, r, a)
            # if self.settings.verbose:
            #     print(self.gates)
            self.constraint_gate_card(bound_gate, step + 1, t)

            if self.settings.minimize_distance:
                # self.minimize_distance(x, y)
                self.minimize_movement(x, y)
            try:
                solved_batch_gates = self.solver.check() == sat
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                return

        print(f"    found solution with {bound_gate} gates in {step} step")
        self.process_partial_solution(step + 1, a, c, r, x, y, t)

    def solve(self, start: int = 1, save_results: bool = True):
        self.write_settings_json()
        t_s = time.time()
        step = start  # compile for 1 step, or 2 stages each time

        self.solve_optimal(step)

        if "duration" not in self.result_json:
            self.result_json["duration"] = 0
        self.result_json["timestamp"] = str(time.time())
        self.result_json["duration"] += time.time() - t_s
        self.result_json["n_t"] = len(self.result_json["layers"])
        print(f"runtime {self.result_json['duration']}")

        if save_results:
            path = self.settings.directory + f"{self.result_json['name']}.json"
            with open(path, "w", encoding="utf-8") as file:
                json.dump(self.result_json, file)

    def compile(
        self,
        circuit: QuantumCircuit,
        optimal: bool = True,
        window_size: int = 5,
        start: int = 1,
    ) -> dict:
        """Compile a qiskit circuit"""
        if optimal:
            self.circuit = Circuit.from_qiskit_circuit(circuit)
            self.solve(save_results=False, start=start)
        else:
            for circ in split_circuit(circuit, window_size):
                self.circuit = Circuit.from_qiskit_circuit(circ)
                self.solve(save_results=False, start=start)

        return self.result_json


# unit tests for the constraints




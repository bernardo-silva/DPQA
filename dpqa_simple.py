""" DPQA Simplified """
from collections import defaultdict
import json
import time
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Set

from networkx import Graph, max_weight_matching
from pysat.card import CardEnc
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from z3 import *

PYSAT_ENCODING = 2  # default choice: sequential counter


# @dataclass
# class QuantumGate:
#     """class to encode a quantum gate."""
#
#     name: str
#     qubits: Tuple[int]
#     ancestor: "QuantumGate" | None = None


@dataclass
class Circuit:
    """class to encode a quantum circuit."""

    circuit: QuantumCircuit
    num_qubits: int
    dag: DAGCircuit
    gates: List[DAGOpNode]
    num_gates: int
    gates_qubits: List[Tuple[int, int]]
    # gates_indexes: Dict[Tuple[int, int], List[int]]
    gates_indexes: Dict[int, List[int]]

    @staticmethod
    def from_qiskit_circuit(circuit: QuantumCircuit) -> "Circuit":
        """Construct a Circuit from a QuantumCircuit."""
        num_qubits = circuit.num_qubits
        dag = circuit_to_dag(circuit)
        gates = dag.op_nodes()
        num_gates = len(gates)
        gates_qubits =[node_qubits(circuit, node) for node in gates]

        gates_indexes = defaultdict(list)
        for q in range(num_qubits):
            for t, qubits in enumerate(gates_qubits):
                if q in qubits:
                    gates_indexes[q].append(t)

        return Circuit(
            circuit, num_qubits, dag, gates, num_gates, gates_qubits, gates_indexes
        )


def node_qubits(circ: QuantumCircuit, node: DAGOpNode) -> Tuple[int, int]:
    """Get the qubits of a DAGOpNode."""
    return tuple(sorted(map(lambda q: circ.find_bit(q).index, node.qargs)))


# def op_node_to_gate(circ: QuantumCircuit, node: DAGOpNode) -> QuantumGate:
#     """Convert a DAGOpNode to a QuantumGate."""
#     qubits = tuple(map(lambda q: circ.find_bit(q).index, node.qargs))
#
#     return QuantumGate(
#         node.name,
#         qubits
#         None,
#     )
# def circuit_to_gates(circ: QuantumCircuit) -> List[QuantumGate]:
#     dag = circuit_to_dag(circ)


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
    verbose: bool = False
    all_aod: bool = False
    no_transfer: bool = False
    optimal_ratio: float = 0
    row_per_site: int = 3
    cardinality_encoding: str = "pysat"


class DPQA_Simple:
    """class to encode the compilation problem to SMT and solves using Z3."""

    def __init__(
        self,
        name: str,
        directory: str = "",
        bounds: Tuple[int, int, int, int] = (16, 16, 16, 16),
        verbose: bool = False,
        all_aod: bool = False,
        no_transfer: bool = False,
    ):
        self.dpqa: Solver = Solver()
        self.satisfiable: bool = False

        self.circuit: Circuit | None = None
        self.architecture: Architecture = Architecture(*bounds)

        self.num_transports: int = 1

        self.settings = DPQASettings(
            name,
            directory,
            verbose,
            all_aod,
            no_transfer,
        )

        self.result_json = {"name": name, "layers": []}

    def set_depth(self, depth: int):
        self.num_transports = depth

    def add_metadata(self, metadata: Mapping[str, Any]):
        self.result_json = {}
        for k, v in metadata.items():
            self.result_json[k] = v

    def write_settings_json(self):
        pass
        # self.result_json["sat"] = self.satisfiable
        # self.result_json["n_t"] = self.num_transports
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

    def constraint_all_aod(self, num_stage: int, a: Sequence[Sequence[Any]]):
        """All qubits on AODs"""
        if self.settings.all_aod:
            for q, s in product(range(self.circuit.num_gates), range(num_stage)):
                (self.dpqa).add(a[q][s])

    def constraint_no_transfer(self, num_stage: int, a: Sequence[Sequence[Any]]):
        """No transfer from AOD to SLM and vice versa"""
        if self.settings.no_transfer:
            for q, s in product(range(self.circuit.num_gates), range(num_stage)):
                (self.dpqa).add(a[q][s] == a[q][0])

    def constraint_var_bounds(
        self,
        num_stage: int,
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        """Bounds on the variables"""
        for q in range(self.circuit.num_qubits):
            for s in range(1, num_stage):
                # starting from s=1 since the values with s=0 are loaded
                (self.dpqa).add(x[q][s] >= 0)
                (self.dpqa).add(x[q][s] < self.architecture.n_x)
                (self.dpqa).add(y[q][s] >= 0)
                (self.dpqa).add(y[q][s] < self.architecture.n_y)
            for s in range(num_stage):
                # starting from s=0 since the solver finds these values
                (self.dpqa).add(c[q][s] >= 0)
                (self.dpqa).add(c[q][s] < self.architecture.n_c)
                (self.dpqa).add(r[q][s] >= 0)
                (self.dpqa).add(r[q][s] < self.architecture.n_r)

    def constraint_fixed_slm(
        self,
        num_stage: int,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):
        """SLMs do not move"""
        for q in range(self.circuit.num_qubits):
            for s in range(num_stage - 1):
                (self.dpqa).add(Implies(Not(a[q][s]), x[q][s] == x[q][s + 1]))
                (self.dpqa).add(Implies(Not(a[q][s]), y[q][s] == y[q][s + 1]))

    def constraint_aod_move_together(
        self,
        num_stage: int,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        """AODs move together"""
        for q in range(self.circuit.num_qubits):
            for s in range(num_stage - 1):
                (self.dpqa).add(Implies(a[q][s], c[q][s + 1] == c[q][s]))
                (self.dpqa).add(Implies(a[q][s], r[q][s + 1] == r[q][s]))
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                for s in range(num_stage - 1):
                    (self.dpqa).add(
                        Implies(
                            And(a[q0][s], a[q1][s], c[q0][s] == c[q1][s]),
                            x[q0][s + 1] == x[q1][s + 1],
                        )
                    )
                    (self.dpqa).add(
                        Implies(
                            And(a[q0][s], a[q1][s], r[q0][s] == r[q1][s]),
                            y[q0][s + 1] == y[q1][s + 1],
                        )
                    )

    def constraint_aod_order_from_slm(
        self,
        num_stage: int,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            for s in range(num_stage - 1):
                if q0 == q1:
                    continue

                (self.dpqa).add(
                    Implies(
                        And(a[q0][s], a[q1][s], c[q0][s] < c[q1][s]),
                        x[q0][s + 1] <= x[q1][s + 1],
                    )
                )
                (self.dpqa).add(
                    Implies(
                        And(a[q0][s], a[q1][s], r[q0][s] < r[q1][s]),
                        y[q0][s + 1] <= y[q1][s + 1],
                    )
                )

    def constraint_slm_order_from_aod(
        self,
        num_stage: int,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        # row/col constraints when atom transfer from SLM to AOD
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            for s in range(num_stage):
                if q0 == q1:
                    continue
                (self.dpqa).add(
                    Implies(
                        And(a[q0][s], a[q1][s], x[q0][s] < x[q1][s]),
                        c[q0][s] < c[q1][s],
                    )
                )
                (self.dpqa).add(
                    Implies(
                        And(a[q0][s], a[q1][s], y[q0][s] < y[q1][s]),
                        r[q0][s] < r[q1][s],
                    )
                )

    def constraint_aod_crowding(
        self,
        num_stage: int,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        # not too many AOD columns/rows can be together, default 3
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            for s in range(num_stage - 1):
                if q0 == q1:
                    continue
                (self.dpqa).add(
                    Implies(
                        And(
                            a[q0][s],
                            a[q1][s],
                            c[q0][s] - c[q1][s] > self.settings.row_per_site - 1,
                        ),
                        x[q0][s + 1] > x[q1][s + 1],
                    )
                )
                (self.dpqa).add(
                    Implies(
                        And(
                            a[q0][s],
                            a[q1][s],
                            r[q0][s] - r[q1][s] > self.settings.row_per_site - 1,
                        ),
                        y[q0][s + 1] > y[q1][s + 1],
                    )
                )

    def constraint_aod_crowding_init(
        self,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        # not too many AOD cols/rows can be together, default 3, for init stage
        for q0, q1 in product(
            range(self.circuit.num_qubits), range(self.circuit.num_qubits)
        ):
            if q0 == q1:
                continue
            (self.dpqa).add(
                Implies(
                    And(
                        a[q0][0],
                        a[q1][0],
                        c[q0][0] - c[q1][0] > self.settings.row_per_site - 1,
                    ),
                    x[q0][0] > x[q1][0],
                )
            )
            (self.dpqa).add(
                Implies(
                    And(
                        a[q0][0],
                        a[q1][0],
                        r[q0][0] - r[q1][0] > self.settings.row_per_site - 1,
                    ),
                    y[q0][0] > y[q1][0],
                )
            )

    def constraint_site_crowding(
        self,
        num_stage: int,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        """Two atoms cannot be in the same AOD site or SLM site. Removed pure_graph condition."""

        # bound number of atoms in each site, needed if not double counting
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                for s in range(num_stage):
                    # Two atoms cannot be in the same AOD site
                    (self.dpqa).add(
                        Implies(
                            And(a[q0][s], a[q1][s]),
                            Or(c[q0][s] != c[q1][s], r[q0][s] != r[q1][s]),
                        )
                    )
                    # Two atoms cannot be in the same SLM site
                    (self.dpqa).add(
                        Implies(
                            And(Not(a[q0][s]), Not(a[q1][s])),
                            Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]),
                        )
                    )

    def constraint_no_swap(
        self,
        num_stage: int,
        a: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):
        # no atom transfer if two atoms meet
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                for s in range(1, num_stage):
                    (self.dpqa).add(
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
        self.dpqa = Optimize()

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
        self.constraint_all_aod(num_stage, a)
        self.constraint_no_transfer(num_stage, a)
        self.constraint_var_bounds(num_stage, x, y, c, r)

        self.constraint_fixed_slm(num_stage, a, x, y)
        self.constraint_aod_move_together(num_stage, a, x, y, c, r)
        self.constraint_aod_order_from_slm(num_stage, a, x, y, c, r)
        self.constraint_slm_order_from_aod(num_stage, a, x, y, c, r)
        self.constraint_aod_crowding(num_stage, a, x, y, c, r)
        self.constraint_aod_crowding_init(a, x, y, c, r)
        self.constraint_site_crowding(num_stage, a, x, y, c, r)
        self.constraint_no_swap(num_stage, a, x, y)

        return a, c, r, x, y

    def constraint_aod_order_from_prev(
        self,
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        if len(self.result_json["layers"]) <= 0:
            return

        variables = self.result_json["layers"][-1]["qubits"]

        for q in range(self.circuit.num_qubits):
            # load location info
            if "x" in variables[q]:
                (self.dpqa).add(x[q][0] == variables[q]["x"])
            if "y" in variables[q]:
                (self.dpqa).add(y[q][0] == variables[q]["y"])
        # virtually putting everything down to acillary SLMs
        # let solver pick some qubits to AOD, so we don't set a_q,0
        # we also don't set c_q,0 and r_q,0, but enforce ordering when
        # two qubits are both in AOD last round, i.e., don't swap
        for q0 in range(self.circuit.num_qubits):
            for q1 in range(q0 + 1, self.circuit.num_qubits):
                if variables[q0]["a"] == 1 and variables[q1]["a"] == 1:
                    if variables[q0]["x"] == variables[q1]["x"]:
                        if variables[q0]["c"] < variables[q1]["c"]:
                            (self.dpqa).add(c[q0][0] < c[q1][0])
                        if variables[q0]["c"] > variables[q1]["c"]:
                            (self.dpqa).add(c[q0][0] > c[q1][0])
                        if variables[q0]["c"] == variables[q1]["c"]:
                            (self.dpqa).add(c[q0][0] == c[q1][0])
                    if variables[q0]["y"] == variables[q1]["y"]:
                        if variables[q0]["r"] < variables[q1]["r"]:
                            (self.dpqa).add(r[q0][0] < r[q1][0])
                        if variables[q0]["r"] > variables[q1]["r"]:
                            (self.dpqa).add(r[q0][0] > r[q1][0])
                        if variables[q0]["r"] == variables[q1]["r"]:
                            (self.dpqa).add(r[q0][0] == r[q1][0])

    def constraint_dependency_collision(
        self,
        t: Sequence[Any],
    ):
        """Enforce gates that depend on each other to be in order"""

        for (t_g0, gate0), (t_g1, gate1) in combinations(zip(t, self.circuit.gates), 2):
            if gate0 in self.circuit.dag.predecessors(gate1):
                self.dpqa.add(t_g0 < t_g1)

            elif gate1 in self.circuit.dag.predecessors(gate0):
                (self.dpqa).add(t_g1 < t_g0)

    def constraint_connectivity(
        self,
        num_gate: int,
        num_stage: int,
        t: Sequence[Any],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):
        """Qubits must be in the same site when interacting"""
        for t_g, gate_qubits in zip(t, self.circuit.gates_qubits):
            for s in range(1, num_stage):  # since stage 0 is 'trash'
                # if len(self.gates[g]) == 2:
                for q0, q1 in combinations(gate_qubits, 2):
                    (self.dpqa).add(Implies(t_g == s, x[q0][s] == x[q1][s]))
                    (self.dpqa).add(Implies(t_g == s, y[q0][s] == y[q1][s]))

    def constraint_interaction_exactness(
        self,
        num_stage: int,
        t: Sequence[Any],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):
        # TODO: Not right
        qubits = range(self.circuit.num_qubits)
        for q0, q1 in combinations(qubits, 2):
            gates_where_interact = [gate for gate in self.circuit.gates_indexes[q0] if q1 in self.circuit.gates_qubits[gate]]
            # If the qubits interact, when they are in the same site, they must be in the same gate
            if gates_where_interact:
                for s in range(1, num_stage):
                    (self.dpqa).add(
                        Implies(
                            And(x[q0][s] == x[q1][s], y[q0][s] == y[q1][s]),
                            Or(*[t[g] == s for g in gates_where_interact]) 
                        )
                    )
            # If the qubits never interact, they must be in different sites
            else:
                for s in range(1, num_stage):
                    (self.dpqa).add(Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]))


                # If the qubits never interact, they must be in different sites
                # qubits = set((q0, q1))
                # if all(not qubits.issubset(set(gate_qubits)) for gate_qubits in self.circuit.gates_qubits):
                #     (self.dpqa).add(Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]))
                # # If the qubits interact, when they are in the same site, they must be in the same gate
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
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ) -> List[Int]:
        """define thebscheduling variables of gates, t. Return t
        add the constraints related to the gates to execute
        Done
        """

        num_gate = self.circuit.num_gates
        t = [Int(f"t_g{g}") for g in range(num_gate)]

        self.constraint_aod_order_from_prev(x, y, c, r)
        for t_g in t:
            (self.dpqa).add(t_g < num_stage)
            (self.dpqa).add(t_g >= 0)

        self.constraint_dependency_collision(t)
        self.constraint_connectivity(num_gate, num_stage, t, x, y)
        self.constraint_interaction_exactness(num_stage, t, x, y)

        return t

    def constraint_gate_card_pysat(
        self,
        num_gate: int,
        num_stage: int,
        bound_gate: int,
        t: Sequence[Any],
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
                    if val not in ancillary.keys():
                        ancillary[val] = Bool("anx_{}".format(val))
                    if i < 0:
                        or_list.append(Not(ancillary[val]))
                    else:
                        or_list.append(ancillary[val])
            (self.dpqa).add(Or(*or_list))

    def constraint_gate_card(
        self,
        bound_gate: int,
        num_stage: int,
        t: Sequence[Any],
    ):
        # add the cardinality constraints on the number of gates

        method = self.settings.cardinality_encoding
        num_gate = self.circuit.num_gates
        if method == "summation":
            # (self.dpqa).add(sum([If(t[g] == s, 1, 0) for g in range(num_gate)
            #                     for s in range(1, num_stage)]) >= bound_gate)
            raise ValueError()
        elif method == "z3atleast":
            # tmp = [(t[g] == s) for g in range(num_gate)
            #        for s in range(1, num_stage)]
            # (self.dpqa).add(AtLeast(*tmp, bound_gate))
            raise ValueError()
        elif method == "pysat":
            self.constraint_gate_card_pysat(num_gate, num_stage, bound_gate, t)
        else:
            raise ValueError("cardinality method unknown")

    def read_partial_solution(
        self,
        s: int,
        model: Any,
        a: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
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
            if is_true(model[a[q][s]]):
                if self.settings.verbose:
                    print(
                        f"        q_{q} is at ({model[x[q][s]].as_long()}, "
                        f"{model[y[q][s]].as_long()})"
                        f" AOD c_{model[c[q][s]].as_long()},"
                        f" r_{model[r[q][s]].as_long()}"
                    )
            else:
                if self.settings.verbose:
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
        a: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        t: Sequence[Any],
    ):
        model = (self.dpqa).model()

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
                            print(
                                f"        CZ(q_{qubits[0]},"
                                f" q_{qubits[1]})"
                            )
                        layer["gates"].append(
                            {
                                f"q{n}": qubit for n, qubit in enumerate(qubits)
                                # "id": self.circuit.gates_indexes[qubits],
                                # "q0": qubits[0],
                                # "q1": qubits[1],
                            }
                        )
                        gates_done.append(g)

                self.result_json["layers"].append(layer)

    def minimize_distance(
        self,
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):
        xdist = Sum([Abs(x1 - x0) for xq in x for x0, x1 in zip(xq, xq[1:])])
        ydist = Sum([Abs(y1 - y0) for yq in y for y0, y1 in zip(yq, yq[1:])])
        self.dpqa.minimize(xdist + ydist)

        # self.remove_gates(gates_done)

    def hybrid_strategy(self):
        """default strategy for hybrid solving: if n_q <30, use optimal solving
        i.e., optimal_ratio=1 with no transfer; if n_q >= 30, last 5% optimal"""
        if not self.settings.optimal_ratio:
            self.settings.optimal_ratio = 1 if self.circuit.num_qubits < 30 else 0.05
        if self.settings.optimal_ratio == 1 and self.circuit.num_qubits < 30:
            self.settings.no_transfer = True

    def solve_greedy(self, step: int):
        print(f"greedy solving with {step} step")
        a, c, r, x, y = self._solver_init(step + 1)
        total_g_q = len(self.gates)
        t_curr = 1

        while len(self.gates) > self.settings.optimal_ratio * total_g_q:
            print(f"gate batch {t_curr}")

            (self.dpqa).push()  # gate related constraints
            t = self.constraint_gate_batch(step + 1, c, r, x, y)

            G = Graph()
            G.add_edges_from(self.gates)
            bound_gate = len(max_weight_matching(G))

            (self.dpqa).push()  # gate bound
            self.constraint_gate_card(bound_gate, step + 1, t)

            solved_batch_gates = (self.dpqa).check() == sat

            while not solved_batch_gates:
                print(f"    no solution, bound_gate={bound_gate} too large")
                (self.dpqa).pop()  # pop to reduce gate bound
                bound_gate -= 1
                if bound_gate <= 0:
                    raise ValueError("gate card should > 0")

                (self.dpqa).push()  # new gate bound
                self.constraint_gate_card(bound_gate, step + 1, t)

                solved_batch_gates = (self.dpqa).check() == sat

            print(f"    found solution with {bound_gate} gates in {step} step")
            self.process_partial_solution(step + 1, a, c, r, x, y, t)
            (self.dpqa).pop()  # the gate bound constraints for solved batch
            t_curr += 1
            (self.dpqa).pop()  # the gate related constraints for solved batch

    def solve_optimal(self, step: int):
        """optimal solving with step steps"""

        if self.circuit is None:
            raise ValueError("circuit is not set")

        print(f"optimal solving with {step} step")
        bound_gate = self.circuit.num_gates

        a, c, r, x, y = self._solver_init(step + 1)
        t = self.constraint_gate_batch(step + 1, c, r, x, y)
        self.constraint_gate_card(bound_gate, step + 1, t)
        self.minimize_distance(x, y)

        solved_batch_gates = (self.dpqa).check() == sat

        while not solved_batch_gates:
            print(f"    no solution, step={step} too small")
            step += 1
            if step > self.circuit.num_gates + 1:
                print("No solution found")
                return
            a, c, r, x, y = self._solver_init(step + 1)  # self.dpqa is cleaned
            t = self.constraint_gate_batch(step + 1, c, r, x, y)
            # if self.settings.verbose:
            #     print(self.gates)
            self.constraint_gate_card(bound_gate, step + 1, t)

            self.minimize_distance(x, y)
            solved_batch_gates = (self.dpqa).check() == sat


        print(f"    found solution with {bound_gate} gates in {step} step")
        self.process_partial_solution(step + 1, a, c, r, x, y, t)

    def solve(self, optimal: bool = False, save_results: bool = True):
        self.write_settings_json()
        t_s = time.time()
        step = 1  # compile for 1 step, or 2 stages each time
        total_g_q = self.circuit.num_gates

        if not optimal:
            self.solve_greedy(step)
            if len(self.gates) > 0:
                print(f"final {len(self.gates)/total_g_q*100} percent")
                self.solve_optimal(step)
        else:
            self.solve_optimal(step)

        self.result_json["timestamp"] = str(time.time())
        self.result_json["duration"] = str(time.time() - t_s)
        self.result_json["n_t"] = len(self.result_json["layers"])
        print(f"runtime {self.result_json['duration']}")

        if save_results:
            path = self.settings.directory + f"{self.result_json['name']}.json"
            with open(path, "w", encoding="utf-8") as file:
                json.dump(self.result_json, file)

    def compile(self, circuit: QuantumCircuit) -> dict:
        self.circuit = Circuit.from_qiskit_circuit(circuit)

        self.solve(optimal=True, save_results=False)

        return self.result_json

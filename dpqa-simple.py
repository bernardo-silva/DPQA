""" DPQA Simplified """
import json
import time
from typing import Any, Dict, Sequence, Tuple, Mapping, List
from itertools import combinations, product
from dataclasses import dataclass

from networkx import Graph, max_weight_matching
from z3 import And, Bool, Implies, Int, Not, Or, Solver, Then, is_true, sat
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


PYSAT_ENCODING = 2  # default choice: sequential counter



@dataclass
class Architecture:
    """class to encode the architecture of the quantum device."""
    n_x: int = 0
    n_y: int = 0
    n_c: int = 0
    n_r: int = 0

@dataclass
class DPQA_Settings:
    """class to encode the settings of the compilation problem."""
    name: str = ""
    directory: str = ""
    verbose: bool = False
    all_commutable: bool = False
    all_aod: bool = False
    no_transfer: bool = False
    pure_graph: bool = False
    optimal_ratio: float = 0
    row_per_site: int = 3
    cardinality_encoding: str = "pysat"


class DPQA_Simple:
    """class to encode the compilation problem to SMT and solves using Z3."""

    def __init__(self,
                 name: str,
                 directory: str = "",
                 bounds: Tuple[int, int, int, int] = (16, 16, 16, 16),
                 print_detail: bool = False,
                 all_commutable: bool = False,
                 all_aod: bool = False,
                 no_transfer: bool = False,
                 pure_graph: bool = False,
                 ):

        self.num_transports: int = 1
        self.num_qubits: int = 0
        self.num_gates: int = 0
        self.gates: Sequence[Sequence[int]] = tuple()
        self.gate_names: Sequence[str] = tuple()
        # self.dependencies: Sequence[Sequence[int]] = tuple()
        # self.collisions: Sequence[Sequence[int]] = tuple()
        # self.gate_index_original: Dict[Sequence[int], int] = {}
        # self.gate_index: Dict[Sequence[int], int] = {}

        self.architecture = Architecture(*bounds)
        self.satisfiable: bool = False

        self.settings = DPQA_Settings(name, directory, print_detail, all_commutable,
                                      all_aod, no_transfer, pure_graph)


        self.result_json = {"name": name, "layers": []}

    def set_depth(self, depth: int):
        self.num_transports = depth

    def add_metadata(self, metadata: Mapping[str, Any]):
        self.result_json = {}
        for k, v in metadata.items():
            self.result_json[k] = v

    def write_settings_json(self):
        self.result_json["sat"] = self.satisfiable
        self.result_json["n_t"] = self.num_transports
        self.result_json["n_q"] = self.num_qubits
        self.result_json["all_commutable"] = self.settings.all_commutable
        self.result_json["all_aod"] = self.settings.all_aod
        self.result_json["no_transfer"] = self.settings.no_transfer
        self.result_json["pure_graph"] = self.settings.pure_graph
        self.result_json["n_c"] = self.architecture.n_c
        self.result_json["n_r"] = self.architecture.n_r
        self.result_json["n_x"] = self.architecture.n_x
        self.result_json["n_y"] = self.architecture.n_y
        self.result_json["row_per_site"] = self.settings.row_per_site
        self.result_json["n_g"] = self.num_gates
        self.result_json["g_q"] = self.gates
        self.result_json["g_s"] = self.gate_names

    def constraint_all_aod(self, num_stage: int, a: Sequence[Sequence[Any]]):
        """ All qubits on AODs """
        if self.settings.all_aod:
            for q, s in product(range(self.num_gates), range(num_stage)):
                (self.dpqa).add(a[q][s])

    def constraint_no_transfer(self, num_stage: int, a: Sequence[Sequence[Any]]):
        """ No transfer from AOD to SLM and vice versa """
        if self.settings.no_transfer:
            for q, s in product(range(self.num_gates), range(num_stage)):
                (self.dpqa).add(a[q][s] == a[q][0])

    def constraint_var_bounds(
        self,
        num_stage: int,
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
    ):
        """ Bounds on the variables """
        for q in range(self.num_qubits):
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
        """ SLMs do not move """
        for q in range(self.num_qubits):
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
        """ AODs move together """
        for q in range(self.num_qubits):
            for s in range(num_stage - 1):
                (self.dpqa).add(Implies(a[q][s], c[q][s + 1] == c[q][s]))
                (self.dpqa).add(Implies(a[q][s], r[q][s + 1] == r[q][s]))
        for q0 in range(self.num_qubits):
            for q1 in range(q0 + 1, self.num_qubits):
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
        for q0, q1 in product(range(self.num_qubits), range(self.num_qubits)):
            for s in range(num_stage - 1):
                if q0 != q1:
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
        for q in range(self.num_qubits):
            for qq in range(self.num_qubits):
                for s in range(num_stage):
                    if q != qq:
                        (self.dpqa).add(
                            Implies(
                                And(a[q][s], a[qq][s], x[q][s] < x[qq][s]),
                                c[q][s] < c[qq][s],
                            )
                        )
                        (self.dpqa).add(
                            Implies(
                                And(a[q][s], a[qq][s], y[q][s] < y[qq][s]),
                                r[q][s] < r[qq][s],
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
        for q in range(self.num_qubits):
            for qq in range(self.num_qubits):
                for s in range(num_stage - 1):
                    if q != qq:
                        (self.dpqa).add(
                            Implies(
                                And(
                                    a[q][s],
                                    a[qq][s],
                                    c[q][s] - c[qq][s] > self.row_per_site - 1,
                                ),
                                x[q][s + 1] > x[qq][s + 1],
                            )
                        )
                        (self.dpqa).add(
                            Implies(
                                And(
                                    a[q][s],
                                    a[qq][s],
                                    r[q][s] - r[qq][s] > self.row_per_site - 1,
                                ),
                                y[q][s + 1] > y[qq][s + 1],
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
        for q in range(self.num_qubits):
            for qq in range(self.num_qubits):
                if q != qq:
                    (self.dpqa).add(
                        Implies(
                            And(
                                a[q][0],
                                a[qq][0],
                                c[q][0] - c[qq][0] > self.row_per_site - 1,
                            ),
                            x[q][0] > x[qq][0],
                        )
                    )
                    (self.dpqa).add(
                        Implies(
                            And(
                                a[q][0],
                                a[qq][0],
                                r[q][0] - r[qq][0] > self.row_per_site - 1,
                            ),
                            y[q][0] > y[qq][0],
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

        # if self.pure_graph:
        # bound number of atoms in each site, needed if not double counting
        for q0 in range(self.num_qubits):
            for q1 in range(q0 + 1, self.num_qubits):
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
        for q0 in range(self.num_qubits):
            for q1 in range(q0 + 1, self.num_qubits):
                for s in range(1, num_stage):
                    (self.dpqa).add(
                        Implies(
                            And(x[q0][s] == x[q1][s], y[q0][s] == y[q1][s]),
                            And(a[q0][s] == a[q0][s - 1], a[q1][s] == a[q1][s - 1]),
                        )
                    )

    def solver_init(self, num_stage: int = 2):
        # define the variables and add the constraints that do not depend on
        # the gates to execute. return the variable arrays a, c, r, x, y

        # variables
        a = [
            [Bool(f"a_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.num_qubits)
        ]
        # for col and row, the data does not matter if atom in SLM
        c = [
            [Int(f"c_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.num_qubits)
        ]
        r = [
            [Int(f"r_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.num_qubits)
        ]
        x = [
            [Int(f"x_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.num_qubits)
        ]
        y = [
            [Int(f"y_q{q}_t{t}") for t in range(num_stage)]
            for q in range(self.num_qubits)
        ]

        (self.dpqa) = Solver()
        if self.cardenc == "z3atleast":
            (self.dpqa) = Then(
                "simplify", "solve-eqs", "card2bv", "bit-blast", "aig", "sat"
            ).solver()

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

        for q in range(self.num_qubits):
            # load location info
            if "x" in variables[q]:
                (self.dpqa).add(x[q][0] == variables[q]["x"])
            if "y" in variables[q]:
                (self.dpqa).add(y[q][0] == variables[q]["y"])
        # virtually putting everything down to acillary SLMs
        # let solver pick some qubits to AOD, so we don't set a_q,0
        # we also don't set c_q,0 and r_q,0, but enforce ordering when
        # two qubits are both in AOD last round, i.e., don't swap
        for q0 in range(self.num_qubits):
            for q1 in range(q0 + 1, self.num_qubits):
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
        raise NotImplementedError

    def constraint_connectivity(
        self,
        num_gate: int,
        num_stage: int,
        t: Sequence[Any],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):
        for g in range(num_gate):
            for s in range(1, num_stage):  # since stage 0 is 'trash'
                if len(self.gates[g]) == 2:
                    q0 = self.gates[g][0]
                    q1 = self.gates[g][1]
                    (self.dpqa).add(Implies(t[g] == s, x[q0][s] == x[q1][s]))
                    (self.dpqa).add(Implies(t[g] == s, y[q0][s] == y[q1][s]))

    def constraint_interaction_exactness(
        self,
        num_stage: int,
        t: Sequence[Any],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):

        # TODO: Not right
        for q0, q1 in combinations(range(self.num_qubits), r=2):
            for s in range(1, num_stage):
                if (q0, q1) in self.gates or (q1, q0) in self.gates:
                    (self.dpqa).add(Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]))
                else:
                    (self.dpqa).add(
                        Implies(
                            And(x[q0][s] == x[q1][s], y[q0][s] == y[q1][s]),
                            t[self.gate_index[(q0, q1)]] == s,
                        )
                    )

    def constraint_gate_batch(
        self,
        num_stage: int,
        c: Sequence[Sequence[Any]],
        r: Sequence[Sequence[Any]],
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[Any]],
    ):
        # define the scheduling variables of gates, t. Return t
        # add the constraints related to the gates to execute

        num_gate = len(self.gates)
        t = [Int(f"t_g{g}") for g in range(num_gate)]

        self.constraint_aod_order_from_prev(x, y, c, r)
        for g in range(num_gate):
            (self.dpqa).add(t[g] < num_stage)
            (self.dpqa).add(t[g] >= 0)

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
        from pysat.card import CardEnc

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

        method = self.cardenc
        num_gate = len(self.gates)
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
        for q in range(self.num_qubits):
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

    """how to stitch partial solution: (suppose there are 3 stages or 2
        steps in each partial solution). a/c/r_s variables govern the move
        from stage s to s+1, so a/c/r for the last stage doesn't matter.
        
                                        partial solutions:

        full solution:                    ----------- x/y_0
                                            | a/c/r_0 |
        ----------- x/y_0  <-----------   ----------- x/y_1
        | a/c/r_0 |  <-----------------   | a/c/r_1 |
        ----------- x/y_1  <-----------   ----------- x/y_2  ----
        | a/c/r_1 |  <------------.       | a/c/r_2 |           |
        ----------- x/y_2  <-----. \                            |
        | a/c/r_2 |  <---------.  \ \                           | 
        ----------- x/y_3       \  \ \    ----------- x/y_0  <---
        | a/c/r_3 |              \  \ \-  | a/c/r_0 |
                                    \  \--  ----------- x/y_1
                                    \----  | a/c/r_1 |
                                            ...
        
        just above, we prepared a Dict `layer` that looks like
                    ----------- x/y_s
                    | a/c/r_s |

        if s=0, we need to overwrite the a/c/r of the last stage of the
        full solution, because they are copied over from the last stage of
        the previous partial solution, and those values are not useful.
        
        except for this case, just append `layer` to the full solution.
        """

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
                for q in range(self.num_qubits):
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
                for g in range(len(self.gates)):
                    if model[t[g]].as_long() == s:
                        if self.settings.verbose:
                            print(
                                f"        CZ(q_{self.gates[g][0]},"
                                f" q_{self.gates[g][1]})"
                            )
                        layer["gates"].append(
                            {
                                "id": self.gate_index_original[
                                    (self.gates[g][0], self.gates[g][1])
                                ],
                                "q0": self.gates[g][0],
                                "q1": self.gates[g][1],
                            }
                        )
                        gates_done.append(g)

                self.result_json["layers"].append(layer)

        self.remove_gates(gates_done)

    def hybrid_strategy(self):
        # default strategy for hybrid solving: if n_q <30, use optimal solving
        # i.e., optimal_ratio=1 with no transfer; if n_q >= 30, last 5% optimal
        if not self.optimal_ratio:
            self.set_optimal_ratio(1 if self.num_qubits < 30 else 0.05)
        if self.optimal_ratio == 1 and self.num_qubits < 30:
            self.set_no_transfer()

    def solve_greedy(self, step: int):
        print(f"greedy solving with {step} step")
        a, c, r, x, y = self.solver_init(step + 1)
        total_g_q = len(self.gates)
        t_curr = 1

        while len(self.gates) > self.optimal_ratio * total_g_q:
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
        print(f"optimal solving with {step} step")
        bound_gate = len(self.gates)

        a, c, r, x, y = self.solver_init(step + 1)
        t = self.constraint_gate_batch(step + 1, c, r, x, y)
        self.constraint_gate_card(bound_gate, step + 1, t)

        solved_batch_gates = (self.dpqa).check() == sat

        while not solved_batch_gates:
            print(f"    no solution, step={step} too small")
            step += 1
            a, c, r, x, y = self.solver_init(step + 1)  # self.dpqa is cleaned
            t = self.constraint_gate_batch(step + 1, c, r, x, y)
            if self.settings.verbose:
                print(self.gates)
            self.constraint_gate_card(bound_gate, step + 1, t)

            solved_batch_gates = (self.dpqa).check() == sat

        print(f"    found solution with {bound_gate} gates in {step} step")
        self.process_partial_solution(step + 1, a, c, r, x, y, t)

    def solve(self, save_results: bool = True):
        self.write_settings_json()
        t_s = time.time()
        step = 1  # compile for 1 step, or 2 stages each time
        total_g_q = len(self.gates)

        self.solve_greedy(step)
        if len(self.gates) > 0:
            print(f"final {len(self.gates)/total_g_q*100} percent")
            self.solve_optimal(step)

        self.result_json["timestamp"] = str(time.time())
        self.result_json["duration"] = str(time.time() - t_s)
        self.result_json["n_t"] = len(self.result_json["layers"])
        print(f"runtime {self.result_json['duration']}")

        if save_results:
            with open(self.settings.directory + f"{self.result_json['name']}.json", "w") as f:
                json.dump(self.result_json, f)

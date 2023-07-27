from typing import Sequence
from solve import DPQA

def dpqa_solve(circuit: Sequence[Sequence[int]],
               architecture: Sequence[int] = [16, 16, 16, 16],
               all_aod: bool = False,
               filename: str = "",
               directory: str = './results/smt/',
               print_detail: bool = False) -> dict:
    """ Run DPQA on a given circuit. """

    dpqa = DPQA(
        filename,
        dir=directory,
        print_detail=print_detail
    )

    dpqa.set_architecture(architecture)
    dpqa.set_program(circuit)

    dpqa.setPureGraph()
    dpqa.setCommutation()

    if all_aod:
        dpqa.setAOD()
    
    dpqa.hybrid_strategy()
    dpqa.solve()

    return dpqa.result_json
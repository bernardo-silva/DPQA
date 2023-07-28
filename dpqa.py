from typing import Sequence, Tuple
from .solve import DPQA

def dpqa_solve(circuit: Sequence[Sequence[int]],
               architecture: Tuple[int, int, int, int] = [16, 16, 16, 16],
               save_results: bool = False,
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

    dpqa.set_pure_graph()
    dpqa.set_commutation()

    if all_aod:
        dpqa.set_all_AOD()
    
    dpqa.hybrid_strategy()
    dpqa.solve(save_results=save_results)

    return dpqa.result_json

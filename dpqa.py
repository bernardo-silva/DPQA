from typing import Sequence, Tuple
from .solve import DPQA
from .dpqa_simple import DPQA_Simple

def dpqa_solve(circuit: Sequence[Sequence[int]],
               architecture: Tuple[int, int, int, int] = [16, 16, 16, 16],
               save_results: bool = False,
               all_aod: bool = False,
               filename: str = "",
               directory: str = './results/smt/',
               verbose: bool = False) -> dict:
    """ Run DPQA on a given circuit. """

    dpqa = DPQA_Simple(name=filename, directory=directory, verbose=verbose)


    if all_aod:
        dpqa.set_all_AOD()

    return dpqa.result_json

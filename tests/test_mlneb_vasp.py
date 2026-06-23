import os
import shutil
import unittest
from pathlib import Path

from .functions import get_endstructures


def vasp_tests_enabled():
    return os.environ.get("CATLEARN_RUN_VASP_TESTS", "0") == "1"


def vasp_command_available():
    return bool(
        os.environ.get("ASE_VASP_COMMAND")
        or os.environ.get("VASP_COMMAND")
        or os.environ.get("VASP_SCRIPT")
    )


def vasp_pp_available():
    return bool(os.environ.get("VASP_PP_PATH"))


def make_vasp_compatible(atoms):
    from ase.calculators.emt import EMT

    atoms = atoms.copy()
    atoms.pbc = True
    atoms.calc = EMT()
    atoms.get_potential_energy()
    atoms.get_forces()
    return atoms


class TestMLNEBVasp(unittest.TestCase):
    """
    VASP smoke tests for MLNEB.

    These tests are skipped by default because they launch external VASP
    calculations. Enable them with:

        CATLEARN_RUN_VASP_TESTS=1 python -m unittest tests.test_mlneb_vasp

    Required environment:
        ASE_VASP_COMMAND or VASP_COMMAND or VASP_SCRIPT
        VASP_PP_PATH
    """

    def setUp(self):
        self.workdir = Path("vasp_mlneb_test").resolve()
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir()

    def tearDown(self):
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

    def make_vasp_calc(self, label):
        from ase.calculators.vasp import Vasp

        return Vasp(
            directory=str(self.workdir / label),
            xc=os.environ.get("CATLEARN_VASP_XC", "PBE"),
            encut=float(os.environ.get("CATLEARN_VASP_ENCUT", "250")),
            kpts=(1, 1, 1),
            gamma=True,
            ismear=0,
            sigma=0.05,
            ediff=1e-4,
            nelm=60,
            ibrion=-1,
            nsw=0,
            lreal="Auto",
            lwave=False,
            lcharg=False,
        )

    def test_vasp_calculator_uses_serial_exact_evaluation(self):
        "Test that CatLearn detects ASE VASP as a rank-0 exact evaluator."
        try:
            self.make_vasp_calc("detect")
        except Exception as exc:
            self.skipTest(f"ASE VASP calculator unavailable: {exc}")

        from catlearn.activelearning.activelearning import ActiveLearning

        calc = self.make_vasp_calc("detect")
        self.assertTrue(ActiveLearning.is_serial_external_calculator(calc))

    @unittest.skipUnless(
        vasp_tests_enabled(),
        "Set CATLEARN_RUN_VASP_TESTS=1 to launch external VASP jobs.",
    )
    @unittest.skipUnless(
        vasp_command_available(),
        "Set ASE_VASP_COMMAND, VASP_COMMAND, or VASP_SCRIPT.",
    )
    @unittest.skipUnless(
        vasp_pp_available(),
        "Set VASP_PP_PATH so ASE can build POTCAR files.",
    )
    def test_mlneb_vasp_single_exact_evaluation(self):
        """
        Run a minimal MLNEB/VASP smoke test.

        This intentionally checks only that a VASP-backed exact evaluation can
        be triggered and stored. It does not try to converge a production NEB.
        """
        from catlearn.activelearning.mlneb import MLNEB

        initial, final = get_endstructures()
        initial = make_vasp_compatible(initial)
        final = make_vasp_compatible(final)

        mlneb = MLNEB(
            start=initial,
            end=final,
            ase_calc=self.make_vasp_calc("exact"),
            neb_interpolation="linear",
            n_images=5,
            unc_convergence=0.10,
            use_restart=False,
            check_unc=True,
            verbose=False,
            local_opt_kwargs=dict(logfile=None),
            parallel_run=True,
            parallel_eval=False,
            scale_fmax=1.0,
            seed=1,
        )

        atoms = make_vasp_compatible(initial)
        mlneb.evaluate(atoms)

        self.assertGreaterEqual(mlneb.get_training_set_size(), 3)
        self.assertIn("energy", mlneb.candidate.calc.results)
        self.assertIn("forces", mlneb.candidate.calc.results)


if __name__ == "__main__":
    unittest.main()
